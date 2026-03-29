"""
HGNN トレーニングスクリプト
Bridge Importance Scoring MVP v1.3

HeteroData を用いた橋梁閉鎖時の影響度（間接被害）予測モデルのトレーニング
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
import numpy as np
import pandas as pd
import yaml
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from hetero_data_converter import HeteroGraphConverter
from hgnn_model import create_model

__version__ = "1.3.0"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HGNNTrainer:
    """HGNN トレーニングクラス"""
    
    def __init__(
        self,
        data,
        config: Dict,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            data: HeteroData
            config: 設定辞書
            device: デバイス（CPU or CUDA）
        """
        self.data = data
        self.config = config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # データを device に移動
        self.data = self.data.to(self.device)

        # Log1p ターゲット変換
        self.log1p_target = config.get('hgnn', {}).get('use_log1p_target', False)
        if self.log1p_target:
            self.data['bridge'].y = torch.log1p(self.data['bridge'].y)
            logger.info("Applied log1p transform to target variable")

        # サンプル重みの計算（高スコア橋に大きな重み）
        self.use_weighted_loss = config.get('hgnn', {}).get('use_weighted_loss', False)
        self.sample_weights = self._compute_sample_weights()
        self.use_edge_attr = bool(config.get('hgnn', {}).get('use_edge_attr', False))

        logger.info(f"Using device: {self.device}")
        logger.info(f"HeteroData: {self.data}")

    def _compute_sample_weights(self) -> torch.Tensor:
        """ターゲット値に比例したサンプル重みを計算（学習空間で計算）"""
        hcfg = self.config.get('hgnn', {})
        scheme = hcfg.get('weight_scheme', 'linear')
        alpha = hcfg.get('weight_alpha', 2.0)
        y = self.data['bridge'].y.squeeze().float()  # log1p済みのターゲット

        if scheme == 'quantile':
            q_edges = hcfg.get('quantile_edges', [0.5, 0.8, 0.9])
            q_weights = hcfg.get('quantile_weights', [1.0, 1.5, 2.5, 4.0])
            if len(q_weights) != len(q_edges) + 1:
                logger.warning("Invalid quantile_weights length; fallback to linear weighting")
                scheme = 'linear'
            else:
                y_np = y.detach().cpu().numpy()
                thresholds = np.quantile(y_np, q_edges)
                w_np = np.full_like(y_np, fill_value=float(q_weights[0]), dtype=float)
                for i, th in enumerate(thresholds):
                    w_np[y_np >= th] = float(q_weights[i + 1])
                weights = torch.tensor(w_np, dtype=torch.float32, device=y.device)

        if scheme != 'quantile':
            # y を [0, 1] に正規化して重みを計算: weight = 1 + alpha * y_norm
            y_min, y_max = y.min(), y.max()
            if (y_max - y_min).item() > 1e-8:
                y_norm = (y - y_min) / (y_max - y_min)
            else:
                y_norm = torch.zeros_like(y)
            weights = 1.0 + alpha * y_norm

        weights = weights / weights.mean()  # 平均1.0に正規化
        logger.info(f"Sample weights: min={weights.min():.3f}, max={weights.max():.3f}, "
                    f"mean={weights.mean():.3f} (scheme={scheme}, alpha={alpha})")
        return weights  # shape [N]

    def get_target_name(self) -> str:
        """ターゲット変数名を取得"""
        return getattr(self.data['bridge'], 'target_name', 'target')
    
    def prepare_data_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.15, random_state: int = 42):
        """
        Train/Validation/Test 分割を作成
        
        Args:
            train_ratio: トレーニングデータの割合
            val_ratio: 検証データの割合
            random_state: ランダムシード
        """
        logger.info("Preparing data splits...")
        
        num_bridges = self.data['bridge'].x.shape[0]
        indices = np.arange(num_bridges)
        
        # Train/Temp 分割
        train_idx, temp_idx = train_test_split(
            indices, 
            train_size=train_ratio,
            random_state=random_state
        )
        
        # Validation/Test 分割
        val_size = val_ratio / (1 - train_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_size,
            random_state=random_state
        )
        
        # マスクを作成
        self.data['bridge'].train_mask = torch.zeros(num_bridges, dtype=torch.bool)
        self.data['bridge'].val_mask = torch.zeros(num_bridges, dtype=torch.bool)
        self.data['bridge'].test_mask = torch.zeros(num_bridges, dtype=torch.bool)
        
        self.data['bridge'].train_mask[train_idx] = True
        self.data['bridge'].val_mask[val_idx] = True
        self.data['bridge'].test_mask[test_idx] = True
        
        logger.info(f"Train: {len(train_idx)} bridges")
        logger.info(f"Validation: {len(val_idx)} bridges")
        logger.info(f"Test: {len(test_idx)} bridges")
        
        return train_idx, val_idx, test_idx
    
    def train(
        self,
        model,
        num_epochs: int = 100,
        lr: float = 0.001,
        weight_decay: float = 5e-4,
        patience: int = 20,
        save_path: Optional[str] = None
    ):
        """
        モデルのトレーニング
        
        Args:
            model: HGNN モデル
            num_epochs: エポック数
            lr: 学習率
            weight_decay: 重み減衰
            patience: Early stopping の patience
            save_path: モデル保存パス
        
        Returns:
            (model, history, training_info) のタプル
        """
        logger.info("Starting training...")
        
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = self.config.get('hgnn', {}).get('loss_function', 'mse')
        criterion = torch.nn.HuberLoss(delta=1.0) if loss_fn == 'huber' else torch.nn.MSELoss()
        huber_delta = 1.0  # element-wise 計算用
        
        best_val_loss = float('inf')
        best_epoch = 0
        best_time_sec = 0.0
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }

        t0 = time.perf_counter()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            edge_attr_dict = {
                et: self.data[et].edge_attr for et in self.data.edge_types if hasattr(self.data[et], 'edge_attr')
            } if self.use_edge_attr else None
            out = model(self.data.x_dict, self.data.edge_index_dict, edge_attr_dict=edge_attr_dict)
            
            # Loss（train mask のみ）
            train_mask = self.data['bridge'].train_mask
            if self.use_weighted_loss:
                # element-wise Huber loss × sample weights
                pred_t = out[train_mask]
                true_t = self.data['bridge'].y[train_mask]
                elem_loss = F.huber_loss(pred_t, true_t, reduction='none', delta=huber_delta)
                w = self.sample_weights[train_mask].unsqueeze(1)
                loss = (elem_loss * w).mean()
            else:
                loss = criterion(out[train_mask], self.data['bridge'].y[train_mask])
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                out = model(self.data.x_dict, self.data.edge_index_dict, edge_attr_dict=edge_attr_dict)
                
                # Train metrics
                train_pred = out[train_mask].cpu().numpy()
                train_true = self.data['bridge'].y[train_mask].cpu().numpy()
                train_mae = mean_absolute_error(train_true, train_pred)
                
                # Validation metrics
                val_mask = self.data['bridge'].val_mask
                val_pred = out[val_mask].cpu().numpy()
                val_true = self.data['bridge'].y[val_mask].cpu().numpy()
                val_loss = criterion(out[val_mask], self.data['bridge'].y[val_mask]).item()
                val_mae = mean_absolute_error(val_true, val_pred)
                
                # History
                history['train_loss'].append(loss.item())
                history['val_loss'].append(val_loss)
                history['train_mae'].append(train_mae)
                history['val_mae'].append(val_mae)
            
            # Logging
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs} | "
                           f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | "
                           f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_time_sec = time.perf_counter() - t0
                patience_counter = 0
                
                # Save best model
                if save_path:
                    torch.save(model.state_dict(), save_path)
                    logger.info(f"Best model saved to {save_path} (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info("Training completed!")
        
        # Load best model
        if save_path and Path(save_path).exists():
            model.load_state_dict(torch.load(save_path))
            logger.info(f"Best model loaded from {save_path}")

        total_time_sec = time.perf_counter() - t0
        if self.device.type == 'cuda':
            peak_gpu_memory_mb = float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))
        else:
            peak_gpu_memory_mb = 0.0

        training_info = {
            'best_epoch': int(best_epoch),
            'best_time_sec': float(best_time_sec),
            'total_epochs': int(len(history['train_loss'])),
            'total_time_sec': float(total_time_sec),
            'peak_gpu_memory_mb': float(peak_gpu_memory_mb),
        }

        logger.info("Training efficiency:")
        logger.info(f"  Best epoch: {training_info['best_epoch']}")
        logger.info(f"  Time to best epoch: {training_info['best_time_sec']:.2f}s")
        logger.info(f"  Total epochs: {training_info['total_epochs']}")
        logger.info(f"  Total training time: {training_info['total_time_sec']:.2f}s")
        logger.info(f"  Peak GPU memory: {training_info['peak_gpu_memory_mb']:.2f} MB")

        return model, history, training_info
    
    def evaluate(self, model) -> Dict:
        """
        テストデータでモデルを評価
        
        Returns:
            評価メトリクスの辞書
        """
        logger.info("Evaluating model on test set...")
        
        model.eval()
        with torch.no_grad():
            edge_attr_dict = {
                et: self.data[et].edge_attr for et in self.data.edge_types if hasattr(self.data[et], 'edge_attr')
            } if self.use_edge_attr else None
            out = model(self.data.x_dict, self.data.edge_index_dict, edge_attr_dict=edge_attr_dict)
            
            test_mask = self.data['bridge'].test_mask
            test_pred = out[test_mask].cpu().numpy()
            test_true = self.data['bridge'].y[test_mask].cpu().numpy()

            # Log1p 逆変換
            if self.log1p_target:
                test_pred = np.expm1(test_pred)
                test_true = np.expm1(test_true)
            
            # Metrics
            mse = mean_squared_error(test_true, test_pred)
            mae = mean_absolute_error(test_true, test_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(test_true, test_pred)

            # Top-k Recall（高影響橋検出性能）
            y_true_flat = test_true.flatten()
            y_pred_flat = test_pred.flatten()
            k = int(min(20, len(y_true_flat)))
            if k > 0:
                top_true_idx = np.argpartition(-y_true_flat, k - 1)[:k]
                top_pred_idx = np.argpartition(-y_pred_flat, k - 1)[:k]
                topk_recall = len(set(top_true_idx).intersection(set(top_pred_idx))) / float(k)
            else:
                topk_recall = 0.0
            
            # 予測範囲のチェック
            pred_min, pred_max = test_pred.min(), test_pred.max()
            true_min, true_max = test_true.min(), test_true.max()
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'top20_recall': float(topk_recall),
                'pred_range': (pred_min, pred_max),
                'true_range': (true_min, true_max)
            }
            
            logger.info("Test Metrics:")
            logger.info(f"  MSE: {mse:.4f}")
            logger.info(f"  MAE: {mae:.4f}")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  R²: {r2:.4f}")
            logger.info(f"  Top-20 Recall: {topk_recall:.4f}")
            logger.info(f"  Prediction range: [{pred_min:.2f}, {pred_max:.2f}]")
            logger.info(f"  True range: [{true_min:.2f}, {true_max:.2f}]")
            
            return metrics
    
    def plot_training_history(self, history: Dict, save_path: Optional[str] = None):
        """トレーニング履歴をプロット"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE
        axes[1].plot(history['train_mae'], label='Train MAE')
        axes[1].plot(history['val_mae'], label='Val MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Training and Validation MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.close()
    
    def plot_predictions(self, model, save_path: Optional[str] = None):
        """予測値 vs 真値のプロット"""
        target_name = self.get_target_name()

        model.eval()
        with torch.no_grad():
            edge_attr_dict = {
                et: self.data[et].edge_attr for et in self.data.edge_types if hasattr(self.data[et], 'edge_attr')
            } if self.use_edge_attr else None
            out = model(self.data.x_dict, self.data.edge_index_dict, edge_attr_dict=edge_attr_dict)
            
            test_mask = self.data['bridge'].test_mask
            test_pred = out[test_mask].cpu().numpy().flatten()
            test_true = self.data['bridge'].y[test_mask].cpu().numpy().flatten()

            # Log1p 逆変換
            if self.log1p_target:
                test_pred = np.expm1(test_pred)
                test_true = np.expm1(test_true)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(test_true, test_pred, alpha=0.5, s=30)
        
        # 対角線（理想的な予測）
        min_val = min(test_true.min(), test_pred.min())
        max_val = max(test_true.max(), test_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel(f'True {target_name}')
        ax.set_ylabel(f'Predicted {target_name}')
        ax.set_title(f'HGNN Predictions vs Ground Truth ({target_name})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # R² を表示
        r2 = r2_score(test_true, test_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Predictions plot saved to {save_path}")
        
        plt.close()


def main():
    """メイン実行関数"""
    logger.info(f"Bridge Closure Impact HGNN Training v{__version__}")
    logger.info("=" * 80)
    
    # 設定ファイルの読み込み
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    hgnn_cfg = config.get('hgnn', {})
    experiment_name = hgnn_cfg.get('experiment_name', None)
    if experiment_name:
        output_dir = Path(f"output/{experiment_name}")
    else:
        use_weighted = hgnn_cfg.get('use_weighted_loss', False)
        output_dir = Path("output/hgnn_training_v1_3_weighted" if use_weighted else "output/hgnn_training_v1_3")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # HeteroData の読み込み
    hetero_data_path = "output/bridge_importance/heterogeneous_graph_heterodata.pt"
    
    if not Path(hetero_data_path).exists():
        logger.error(f"HeteroData file not found: {hetero_data_path}")
        logger.info("Please run the data conversion first.")
        return
    
    logger.info(f"Loading HeteroData from {hetero_data_path}...")
    data = torch.load(hetero_data_path)
    
    # Trainer の初期化
    trainer = HGNNTrainer(data, config)
    target_name = trainer.get_target_name()
    logger.info(f"Target column: {target_name}")
    
    # データ分割
    train_idx, val_idx, test_idx = trainer.prepare_data_splits(
        train_ratio=0.7,
        val_ratio=0.15,
        random_state=42
    )
    
    # モデルの作成
    model_config = config.get('hgnn', {})
    model = create_model(
        data,
        model_type=model_config.get('model_type', 'standard'),
        hidden_channels=model_config.get('hidden_channels', 64),
        num_layers=model_config.get('num_layers', 2),
        conv_type=model_config.get('conv_type', 'GAT'),
        dropout=model_config.get('dropout', 0.2),
        heads=model_config.get('heads', 4),
        use_edge_attr=model_config.get('use_edge_attr', False)
    )
    
    logger.info(f"Model created: {model.__class__.__name__}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # トレーニング
    model_save_path = output_dir / "best_hgnn_model.pt"
    model, history, training_info = trainer.train(
        model,
        num_epochs=model_config.get('num_epochs', 100),
        lr=model_config.get('learning_rate', 0.001),
        weight_decay=model_config.get('weight_decay', 5e-4),
        patience=model_config.get('patience', 20),
        save_path=str(model_save_path)
    )
    
    # 評価
    metrics = trainer.evaluate(model)
    metrics.update(training_info)
    
    # 結果の可視化
    trainer.plot_training_history(history, save_path=output_dir / "training_history.png")
    trainer.plot_predictions(model, save_path=output_dir / "predictions_vs_truth.png")
    
    # メトリクスの保存
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / "test_metrics.csv", index=False)
    logger.info(f"Test metrics saved to {output_dir / 'test_metrics.csv'}")
    
    # 履歴の保存
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    logger.info(f"Training history saved to {output_dir / 'training_history.csv'}")
    
    # ターゲット変数の統計情報を保存
    target_stats = {
        'target_variable': target_name,
        'min': float(data['bridge'].y.min()),
        'max': float(data['bridge'].y.max()),
        'mean': float(data['bridge'].y.mean()),
        'std': float(data['bridge'].y.std())
    }
    import json
    with open(output_dir / "target_stats.json", 'w') as f:
        json.dump(target_stats, f, indent=2)
    logger.info(
        f"Target statistics ({target_name}): "
        f"mean={target_stats['mean']:.6f}, std={target_stats['std']:.6f}"
    )
    
    logger.info("=" * 80)
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
