"""
HGNN 学習結果 可視化スクリプト
Bridge Importance Scoring MVP v1.3

生成される図:
  1. 学習曲線（Loss / MAE）
  2. 予測値 vs 真値散布図（全橋梁・テストセット別）
  3. 誤差ヒストグラム
  4. ターゲット変数の分布（真値 vs 予測値）
  5. 上位橋梁ランキング（真値 vs 予測値）
  6. 残差プロット
"""

import torch
import numpy as np
import pandas as pd
import yaml
import json
import argparse
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from hgnn_model import create_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── 定数 ────────────────────────────────────────────────
HETERO_PT    = Path("output/bridge_importance/heterogeneous_graph_heterodata.pt")

sns.set_style("whitegrid")
PALETTE = sns.color_palette("muted")


# ── ヘルパー ─────────────────────────────────────────────
def get_splits(n, train_ratio=0.7, val_ratio=0.15, random_state=42):
    idx = np.arange(n)
    train_idx, temp_idx = train_test_split(idx, train_size=train_ratio, random_state=random_state)
    val_size = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(temp_idx, train_size=val_size, random_state=random_state)
    return train_idx, val_idx, test_idx


def load_predictions(config, hetero_pt: Path, model_pt: Path):
    """HeteroData + best_model で全橋梁の予測値を取得"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load(str(hetero_pt), weights_only=False)
    data = data.to(device)

    model_cfg = config.get("hgnn", {})
    model = create_model(
        data,
        model_type=model_cfg.get("model_type", "standard"),
        hidden_channels=model_cfg.get("hidden_channels", 64),
        num_layers=model_cfg.get("num_layers", 2),
        conv_type=model_cfg.get("conv_type", "GAT"),
        dropout=0.0,
        heads=model_cfg.get("heads", 4),
    )
    model.load_state_dict(torch.load(str(model_pt), weights_only=False))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)

    # y_true はオリジナルスケール; 予測はlog1p空間なのでexpm1で戻す
    y_true = data["bridge"].y.cpu().numpy().flatten().astype(np.float64)
    y_pred_log = out.cpu().numpy().flatten().astype(np.float64)
    y_pred = np.expm1(np.clip(y_pred_log, -10, 20))   # overflow防止

    # 分割インデックスを再現
    n = len(y_true)
    train_idx, val_idx, test_idx = get_splits(n)
    mask = {"train": train_idx, "val": val_idx, "test": test_idx}

    return y_true, y_pred, mask


# ── 図1: 学習曲線 ─────────────────────────────────────────
def plot_training_curves(hist: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (col_train, col_val, ylabel, title) in zip(
        axes,
        [
            ("train_loss", "val_loss", "Loss (Huber)", "Training & Validation Loss"),
            ("train_mae",  "val_mae",  "MAE (log1p space)", "Training & Validation MAE"),
        ],
    ):
        ax.plot(hist[col_train], label="Train", color=PALETTE[0], lw=1.8)
        ax.plot(hist[col_val],   label="Val",   color=PALETTE[1], lw=1.8)
        best_ep = hist[col_val].idxmin()
        ax.axvline(best_ep, color="gray", ls="--", lw=1, alpha=0.7,
                   label=f"Best epoch={best_ep+1}")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)

    plt.suptitle("HGNN Learning Curves (v1.3 — log1p target, Huber Loss, bridge↔street edges)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path}")


# ── 図2: 予測 vs 真値（全体 + テストハイライト）────────────────
def plot_pred_vs_true(y_true, y_pred, mask, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (split, color, title) in zip(
        axes,
        [
            ("test",  PALETTE[2], "Test Set"),
            (None,    PALETTE[0], "All Bridges"),
        ],
    ):
        if split:
            idx = mask[split]
            yt, yp = y_true[idx], y_pred[idx]
        else:
            yt, yp = y_true, y_pred

        ax.scatter(yt, yp, alpha=0.45, s=22, color=color, edgecolors="none")

        lo = min(yt.min(), yp.min()) - 0.5
        hi = max(yt.max(), yp.max()) + 1
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.6, label="Perfect prediction")

        r2  = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        ax.text(0.04, 0.95,
                f"R² = {r2:.3f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}",
                transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6), fontsize=10)

        ax.set_xlabel("True indirect_damage_score", fontsize=11)
        ax.set_ylabel("Predicted indirect_damage_score", fontsize=11)
        ax.set_title(f"Predictions vs Ground Truth — {title}  (n={len(yt)})", fontsize=11)
        ax.legend(fontsize=10)

    plt.suptitle("HGNN v1.3: indirect_damage_score Predictions", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path}")


# ── 図3: 誤差ヒストグラム ───────────────────────────────────
def plot_error_distribution(y_true, y_pred, mask, save_path: Path):
    test_idx = mask["test"]
    errors = y_pred[test_idx] - y_true[test_idx]
    rel_errors = errors / (y_true[test_idx] + 1e-6) * 100  # %

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(errors, bins=30, color=PALETTE[3], edgecolor="white", alpha=0.85)
    axes[0].axvline(0, color="red", ls="--", lw=1.5)
    axes[0].axvline(errors.mean(), color="orange", ls="-", lw=1.5,
                    label=f"Mean = {errors.mean():.2f}")
    axes[0].set_xlabel("Prediction Error (Pred − True)", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Absolute Error Distribution (Test set)", fontsize=12)
    axes[0].legend(fontsize=10)

    # 相対誤差 (外れ値クリップ)
    rel_clipped = np.clip(rel_errors, -200, 200)
    axes[1].hist(rel_clipped, bins=30, color=PALETTE[4], edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="red", ls="--", lw=1.5)
    axes[1].set_xlabel("Relative Error (%)", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Relative Error Distribution (Test set, clipped ±200%)", fontsize=12)

    plt.suptitle("HGNN v1.3: Error Analysis", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path}")


# ── 図4: ターゲット変数の分布比較 ──────────────────────────────
def plot_target_distribution(y_true, y_pred, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 全体分布（0超のみ）
    yt_nonzero = y_true[y_true > 0]
    yp_nonzero = y_pred[y_pred > 0]

    axes[0].hist(yt_nonzero, bins=40, alpha=0.65, color=PALETTE[0], label="True", density=True)
    axes[0].hist(yp_nonzero, bins=40, alpha=0.65, color=PALETTE[1], label="Predicted", density=True)
    axes[0].set_xlabel("indirect_damage_score", fontsize=11)
    axes[0].set_ylabel("Density", fontsize=11)
    axes[0].set_title("Score Distribution (excl. zero, all bridges)", fontsize=11)
    axes[0].legend(fontsize=10)

    # log スケール
    axes[1].hist(np.log1p(yt_nonzero), bins=40, alpha=0.65, color=PALETTE[0],
                 label="True (log1p)", density=True)
    axes[1].hist(np.log1p(yp_nonzero), bins=40, alpha=0.65, color=PALETTE[1],
                 label="Predicted (log1p)", density=True)
    axes[1].set_xlabel("log1p(indirect_damage_score)", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)
    axes[1].set_title("Log-transformed Score Distribution (excl. zero)", fontsize=11)
    axes[1].legend(fontsize=10)

    plt.suptitle("HGNN v1.3: Target Variable Distribution", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path}")


# ── 図5: 上位橋梁ランキング（真値 vs 予測値） ──────────────────
def plot_top_bridges_ranking(y_true, y_pred, save_path: Path, top_n=30):
    # 真値上位N橋
    top_true_idx = np.argsort(y_true)[::-1][:top_n]
    yt = y_true[top_true_idx]
    yp = y_pred[top_true_idx]
    labels = [f"B{i+1}" for i in range(top_n)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    x = np.arange(top_n)
    w = 0.38
    axes[0].bar(x - w/2, yt, w, label="True",      color=PALETTE[0], alpha=0.85)
    axes[0].bar(x + w/2, yp, w, label="Predicted", color=PALETTE[1], alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, fontsize=8)
    axes[0].set_ylabel("indirect_damage_score", fontsize=11)
    axes[0].set_title(f"Top {top_n} Bridges by True Score — True vs Predicted", fontsize=12)
    axes[0].legend(fontsize=10)

    # 真値と予測値の順位相関
    rank_true = pd.Series(y_true).rank(ascending=False).values
    rank_pred = pd.Series(y_pred).rank(ascending=False).values
    # top_n橋に絞る
    axes[1].scatter(rank_true[top_true_idx], rank_pred[top_true_idx],
                    s=50, color=PALETTE[2], alpha=0.75)
    for i, idx in enumerate(top_true_idx):
        axes[1].annotate(labels[i], (rank_true[idx], rank_pred[idx]),
                         fontsize=6, alpha=0.6)
    axes[1].set_xlabel("True Rank", fontsize=11)
    axes[1].set_ylabel("Predicted Rank", fontsize=11)
    axes[1].set_title(f"Rank Comparison for Top {top_n} Bridges (lower = more important)", fontsize=12)
    axes[1].invert_xaxis()
    axes[1].invert_yaxis()

    plt.suptitle("HGNN v1.3: High-Impact Bridge Detection", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path}")


# ── 図6: 残差プロット ───────────────────────────────────────
def plot_residuals(y_true, y_pred, mask, save_path: Path):
    test_idx = mask["test"]
    yt = y_true[test_idx]
    yp = y_pred[test_idx]
    residuals = yp - yt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 残差 vs 真値
    axes[0].scatter(yt, residuals, alpha=0.5, s=25, color=PALETTE[0], edgecolors="none")
    axes[0].axhline(0, color="red", ls="--", lw=1.5)
    axes[0].axhline(residuals.mean(), color="orange", ls="-", lw=1.2,
                    label=f"Mean residual = {residuals.mean():.2f}")
    axes[0].set_xlabel("True indirect_damage_score", fontsize=11)
    axes[0].set_ylabel("Residual (Pred − True)", fontsize=11)
    axes[0].set_title("Residuals vs True Values (Test set)", fontsize=12)
    axes[0].legend(fontsize=10)

    # Q-Q plot（残差の正規性確認）
    from scipy import stats
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    axes[1].plot(osm, osr, "o", ms=4, alpha=0.5, color=PALETTE[1])
    xline = np.array([osm[0], osm[-1]])
    axes[1].plot(xline, slope * xline + intercept, "r-", lw=1.5)
    axes[1].set_xlabel("Theoretical Quantiles", fontsize=11)
    axes[1].set_ylabel("Sample Quantiles", fontsize=11)
    axes[1].set_title(f"Q-Q Plot of Residuals  (r={r:.3f})", fontsize=12)

    plt.suptitle("HGNN v1.3: Residual Analysis (Test set)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path}")


# ── 図7: メトリクスサマリー ─────────────────────────────────
def plot_metrics_summary(metrics: dict, hist: pd.DataFrame, save_path: Path, baseline_metrics: dict = None):
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1])

    # --- 左: メトリクスバー ---
    ax_bar = fig.add_subplot(gs[0])
    metric_names  = ["R²", "MAE", "RMSE"]
    metric_values = [metrics["r2"], metrics["mae"], metrics["rmse"]]
    colors = [PALETTE[2] if v > 0 else "tomato" for v in metric_values]
    bars = ax_bar.bar(metric_names, metric_values, color=colors, alpha=0.85, width=0.5)
    for bar, val in zip(bars, metric_values):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax_bar.set_title("Test Set Metrics", fontsize=13)
    ax_bar.set_ylabel("Value", fontsize=11)
    ax_bar.axhline(0, color="gray", lw=0.8)

    # --- 右: テキストサマリー ---
    ax_txt = fig.add_subplot(gs[1])
    ax_txt.axis("off")
    if baseline_metrics is not None:
        delta_r2 = metrics["r2"] - baseline_metrics["r2"]
        delta_mae_pct = (metrics["mae"] - baseline_metrics["mae"]) / max(baseline_metrics["mae"], 1e-8) * 100.0
        improve_text = (
            f"  Improvements (vs baseline):\n"
            f"    R²: {baseline_metrics['r2']:+.3f} → {metrics['r2']:+.3f}  ({delta_r2:+.3f})\n"
            f"    MAE: {baseline_metrics['mae']:.2f} → {metrics['mae']:.2f}  ({delta_mae_pct:+.1f}%)\n"
        )
    else:
        improve_text = ""

    summary = (
        f"  Model: BridgeImportanceHGNN\n"
        f"  Version: v1.3.0\n"
        f"  Target: indirect_damage_score\n"
        f"\n"
        f"  ▸ Epochs trained: {len(hist)}\n"
        f"  ▸ Best Val Loss: {hist['val_loss'].min():.4f}\n"
        f"\n"
        f"  Test Metrics:\n"
        f"    R²   = {metrics['r2']:.4f}\n"
        f"    MAE  = {metrics['mae']:.4f}\n"
        f"    RMSE = {metrics['rmse']:.4f}\n"
        f"    MSE  = {metrics['mse']:.4f}\n"
        f"\n"
        f"{improve_text}"
        f"  Applied:\n"
        f"    ✓ log1p target transform\n"
        f"    ✓ bridge→street edges added\n"
        f"    ✓ Huber Loss (delta=1.0)\n"
        f"    ✓ weighted loss (high-score emphasis)\n"
        f"    ✓ 300 epochs / patience=50"
    )
    ax_txt.text(0.03, 0.97, summary, transform=ax_txt.transAxes,
                va="top", fontsize=9.5, family="monospace",
                bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.7))

    plt.suptitle("HGNN v1.3 — Performance Summary", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path}")


# ── メイン ───────────────────────────────────────────────
def main():
    logger.info("HGNN Result Visualization v1.3")
    logger.info("=" * 70)

    parser = argparse.ArgumentParser(description="Visualize HGNN training results")
    parser.add_argument(
        "--result-dir",
        type=str,
        default="output/hgnn_training_v1_3_weighted",
        help="Directory containing training outputs (best_hgnn_model.pt, training_history.csv, test_metrics.csv)",
    )
    parser.add_argument(
        "--baseline-metrics",
        type=str,
        default="output/hgnn_training_v1_3/test_metrics.csv",
        help="Optional baseline metrics csv for comparison",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    model_pt = result_dir / "best_hgnn_model.pt"
    hist_csv = result_dir / "training_history.csv"
    metrics_csv = result_dir / "test_metrics.csv"
    viz_dir = result_dir / "visualization"
    viz_dir.mkdir(parents=True, exist_ok=True)

    if not model_pt.exists() or not hist_csv.exists() or not metrics_csv.exists():
        raise FileNotFoundError(f"Required files are missing in result_dir: {result_dir}")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    hist    = pd.read_csv(hist_csv)
    metrics = pd.read_csv(metrics_csv).iloc[0].to_dict()
    # pred_range / true_range 列を除いて数値のみ抽出
    num_metrics = {k: float(v) for k, v in metrics.items()
                   if k in ("mse", "mae", "rmse", "r2")}

    baseline_num_metrics = None
    baseline_path = Path(args.baseline_metrics)
    if baseline_path.exists():
        baseline_row = pd.read_csv(baseline_path).iloc[0].to_dict()
        baseline_num_metrics = {k: float(v) for k, v in baseline_row.items()
                                if k in ("mse", "mae", "rmse", "r2")}

    logger.info("Loading model and generating full predictions...")
    y_true, y_pred, mask = load_predictions(config, HETERO_PT, model_pt)
    logger.info(f"y_true range: [{y_true.min():.2f}, {y_true.max():.2f}]")
    logger.info(f"y_pred range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")

    # 各図を生成
    plot_training_curves(hist,
        viz_dir / "fig1_training_curves.png")

    plot_pred_vs_true(y_true, y_pred, mask,
        viz_dir / "fig2_pred_vs_true.png")

    plot_error_distribution(y_true, y_pred, mask,
        viz_dir / "fig3_error_distribution.png")

    plot_target_distribution(y_true, y_pred,
        viz_dir / "fig4_target_distribution.png")

    plot_top_bridges_ranking(y_true, y_pred,
        viz_dir / "fig5_top_bridges_ranking.png")

    plot_residuals(y_true, y_pred, mask,
        viz_dir / "fig6_residuals.png")

    plot_metrics_summary(num_metrics, hist,
        viz_dir / "fig7_metrics_summary.png", baseline_num_metrics)

    logger.info("=" * 70)
    logger.info(f"All figures saved to: {viz_dir}")
    logger.info("Visualization completed!")


if __name__ == "__main__":
    main()
