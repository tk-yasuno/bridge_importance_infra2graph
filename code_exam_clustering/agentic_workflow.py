# -*- coding: utf-8 -*-
"""
Agenticクラスタリングワークフロー: v0.8
自己評価と改善を繰り返す賢いクラスタリングシステム
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import config
from cluster_evaluator import ClusterEvaluator, DimensionalityReductionEvaluator, compare_methods
from alternative_methods import AlternativeClusteringMethods, AlternativeDimensionalityReduction

class AgenticClusteringWorkflow:
    """Agenticクラスタリングのメインワークフロー"""
    
    def __init__(self, df, feature_cols):
        """
        Parameters:
        -----------
        df : DataFrame
            前処理済みデータ
        feature_cols : list
            使用する特徴量のカラム名リスト
        """
        self.df = df
        self.feature_cols = feature_cols
        self.X = df[feature_cols].fillna(0)
        self.X_scaled = None
        self.scaler = None
        
        # 結果を保存
        self.clustering_results = {}
        self.evaluation_results = {}
        self.dimensionality_results = {}
        self.dim_evaluation_results = {}
        
        # 最終的な選択
        self.best_clustering_method = None
        self.best_clustering_labels = None
        self.best_dim_reduction_method = None
        self.best_dim_reduction_embedding = None
        
        # 改善履歴
        self.improvement_log = []
    
    def run(self, quality_threshold=60.0, overlap_threshold=0.5):
        """
        Agenticワークフローを実行
        
        Parameters:
        -----------
        quality_threshold : float
            クラスタリング品質の閾値（0-100）
        overlap_threshold : float
            次元削減のオーバーラップ閾値（0-1）
        """
        print("\n" + "="*70)
        print("🤖 Agenticクラスタリングワークフロー v0.8")
        print("="*70)
        
        # ステップ1: データ標準化
        self._standardize_data()
        
        # ステップ2: 初回クラスタリング（KMeans）
        self._log("【ラウンド1】初回クラスタリング（KMeans）")
        initial_labels = self._initial_clustering()
        
        # ステップ3: クラスタリング品質評価
        self._log("【評価1】クラスタリング品質評価")
        evaluator = ClusterEvaluator(self.X_scaled, initial_labels)
        initial_scores = evaluator.evaluate_all()
        self.evaluation_results['KMeans'] = initial_scores
        
        # ステップ4: 改善が必要か判定
        needs_improvement = evaluator.needs_improvement(quality_threshold)
        
        if needs_improvement:
            # ステップ5: 代替クラスタリング手法の実行
            self._log("【ラウンド2】代替クラスタリング手法の試行")
            self._try_alternative_clustering()
            
            # ステップ6: 最適手法の選択
            self._log("【選択】最適クラスタリング手法の決定")
            self._select_best_clustering()
        else:
            self.best_clustering_method = 'KMeans'
            self.best_clustering_labels = initial_labels
            self._log("✅ 初回クラスタリングで十分な品質が得られました")
        
        # ステップ7: 次元削減（v0.8改: PCAは団子状態になるため、t-SNE/UMAPを直接使用）
        self._log("【ラウンド1】次元削減（t-SNE/UMAP）")
        
        # ステップ8: 代替次元削減手法の実行（PCAをスキップ）
        self._log("【実行】t-SNE/UMAP次元削減")
        self._try_alternative_dimensionality_reduction()
        
        # ステップ9: 最適次元削減手法の選択
        self._log("【選択】最適次元削減手法の決定")
        self._select_best_dimensionality_reduction()
        
        # ステップ12: 結果のサマリー
        self._print_summary()
        
        return self._prepare_final_results()
    
    def _standardize_data(self):
        """データを標準化"""
        print("\n📊 特徴量を標準化中...")
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        print("✓ 標準化完了")
    
    def _initial_clustering(self):
        """初回クラスタリング（KMeans）"""
        from clustering import find_optimal_clusters, perform_clustering
        
        best_k, best_score, _ = find_optimal_clusters(self.X_scaled)
        kmeans, labels = perform_clustering(self.X_scaled, best_k)
        
        self.clustering_results['KMeans'] = {
            'model': kmeans,
            'labels': labels,
            'n_clusters': best_k
        }
        
        return labels
    
    def _try_alternative_clustering(self):
        """代替クラスタリング手法を試行"""
        alt_methods = AlternativeClusteringMethods(self.X_scaled)
        
        # 初回のクラスタ数を参考値として使用
        reference_k = self.clustering_results['KMeans']['n_clusters']
        
        # GMM（Gaussian Mixture Model）
        # v0.8改: ベイズ最適化を使用してパラメータ探索
        try:
            gmm_labels = alt_methods.try_gmm(
                use_bayesian=True,
                n_calls=100
            )
            self.clustering_results['GMM'] = alt_methods.results['GMM']
            
            # 評価
            evaluator = ClusterEvaluator(self.X_scaled, gmm_labels)
            self.evaluation_results['GMM'] = evaluator.evaluate_all()
            print(f"   ✅ GMMを代替手法候補に追加しました。")
        except Exception as e:
            print(f"   ⚠️ GMM実行エラー: {e}")
        
        # DBSCAN
        # v0.8改: ベイズ最適化を使用してパラメータ探索を効率化
        try:
            dbscan_labels = alt_methods.try_dbscan(
                use_bayesian=True,
                n_calls=100
            )
            self.clustering_results['DBSCAN'] = alt_methods.results['DBSCAN']
            dbscan_n_clusters = alt_methods.results['DBSCAN']['n_clusters']
            
            # 評価（ノイズポイントを除外）
            mask = dbscan_labels != -1
            if mask.sum() > 1:
                evaluator = ClusterEvaluator(self.X_scaled[mask], dbscan_labels[mask])
                self.evaluation_results['DBSCAN'] = evaluator.evaluate_all()
                print(f"   ✅ DBSCANを代替手法候補に追加しました。")
                    
        except Exception as e:
            print(f"   ⚠️ DBSCAN実行エラー: {e}")
        
        # 🆕 v0.7: HDBSCANとCLASSIXを常に試行（DBSCANの結果に関係なく）
        print(f"\n🔄 密度ベース高度手法を試行します...")
        print(f"   → HDBSCAN目標クラスタ数: {config.HDBSCAN_TARGET_CLUSTERS}")
        print(f"   → CLASSIX目標クラスタ数: {config.HDBSCAN_TARGET_CLUSTERS}")
        
        # HDBSCAN
        # v0.8改: ベイズ最適化を使用してパラメータ探索を効率化
        try:
            hdbscan_labels = alt_methods.try_hdbscan(
                target_clusters=config.HDBSCAN_TARGET_CLUSTERS,
                use_bayesian=True,
                n_calls=100
            )
            if hdbscan_labels is not None and 'HDBSCAN' in alt_methods.results:
                self.clustering_results['HDBSCAN'] = alt_methods.results['HDBSCAN']
                
                # 評価（ノイズポイントを除外）
                mask_hdbscan = hdbscan_labels != -1
                if mask_hdbscan.sum() > 1:
                    evaluator_hdbscan = ClusterEvaluator(self.X_scaled[mask_hdbscan], hdbscan_labels[mask_hdbscan])
                    self.evaluation_results['HDBSCAN'] = evaluator_hdbscan.evaluate_all()
                    print(f"   ✅ HDBSCANを代替手法候補に追加しました。")
        except Exception as e:
            print(f"   ⚠️ HDBSCAN実行エラー: {e}")
        
        # CLASSIX（HDBSCANのノイズ問題を解決する代替手法）
        # ✅ Windows Cython dtype問題を解決（merge_cm_win.pyxを修正してnp.int64に変更）
        # v0.8改: ベイズ最適化を使用してパラメータ探索を効率化
        try:
            classix_labels = alt_methods.try_classix(
                target_clusters=config.CLASSIX_TARGET_CLUSTERS,
                use_bayesian=True,
                n_calls=100
            )
            if classix_labels is not None and 'CLASSIX' in alt_methods.results:
                self.clustering_results['CLASSIX'] = alt_methods.results['CLASSIX']
                
                # 評価（ノイズポイントを除外）
                mask_classix = classix_labels != -1
                if mask_classix.sum() > 1:
                    evaluator_classix = ClusterEvaluator(self.X_scaled[mask_classix], classix_labels[mask_classix])
                    self.evaluation_results['CLASSIX'] = evaluator_classix.evaluate_all()
                    print(f"   ✅ CLASSIXを代替手法候補に追加しました。")
        except Exception as e:
            print(f"   ⚠️ CLASSIX実行エラー: {e}")
    
    def _select_best_clustering(self):
        """最適なクラスタリング手法を選択"""
        # 除外基準: 1) DBSCANのクラスタ数過多、2) HDBSCANのノイズ10%以上
        filtered_results = {}
        for method, result in self.evaluation_results.items():
            if method == 'DBSCAN':
                n_clusters = self.clustering_results[method].get('n_clusters', 0)
                if n_clusters > config.DBSCAN_CLUSTER_THRESHOLD:
                    print(f"\n⚠️  DBSCANのクラスタ数({n_clusters})が閾値({config.DBSCAN_CLUSTER_THRESHOLD})を超えているため、採用候補から除外します。")
                    continue
            elif method == 'HDBSCAN':
                # v0.7改: ノイズ10%以上のHDBSCANを除外
                labels = self.clustering_results[method]['labels']
                n_noise = list(labels).count(-1)
                noise_ratio = n_noise / len(labels)
                if noise_ratio >= config.HDBSCAN_NOISE_THRESHOLD:
                    print(f"\n⚠️  HDBSCANのノイズ比率({noise_ratio*100:.1f}%)が閾値({config.HDBSCAN_NOISE_THRESHOLD*100:.0f}%)以上のため、採用候補から除外します。")
                    continue
            filtered_results[method] = result
        
        # フィルタ後の候補から最適手法を選択
        if not filtered_results:
            print("\n⚠️  すべての手法が除外されました。KMeansを使用します。")
            best_method = 'KMeans'
        else:
            best_method, comparison = compare_methods(filtered_results)
        
        self.best_clustering_method = best_method
        self.best_clustering_labels = self.clustering_results[best_method]['labels']
        
        self._log(f"🎯 選択された手法: {best_method}")
    
    def _initial_dimensionality_reduction(self):
        """初回次元削減（PCA）"""
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(self.X_scaled)
        
        explained_variance = pca.explained_variance_ratio_
        print(f"\n🔍 PCA: 説明された分散 = {explained_variance.sum():.2%}")
        
        self.dimensionality_results['PCA'] = {
            'model': pca,
            'embedding': embedding
        }
        
        return embedding
    
    def _try_alternative_dimensionality_reduction(self):
        """代替次元削減手法を試行"""
        alt_dim = AlternativeDimensionalityReduction(self.X_scaled)
        
        # t-SNE
        try:
            tsne_embedding = alt_dim.try_tsne()
            if tsne_embedding is not None:
                self.dimensionality_results['t-SNE'] = alt_dim.results['t-SNE']
                
                # オーバーラップ評価
                evaluator = DimensionalityReductionEvaluator(
                    tsne_embedding,
                    self.best_clustering_labels
                )
                self.dim_evaluation_results['t-SNE'] = evaluator.evaluate_overlap()
        except Exception as e:
            print(f"   ⚠️ t-SNE実行エラー: {e}")
        
        # UMAP
        try:
            umap_embedding = alt_dim.try_umap(create_3d=True)  # 3次元も作成
            if umap_embedding is not None:
                self.dimensionality_results['UMAP'] = alt_dim.results['UMAP']
                
                # オーバーラップ評価
                evaluator = DimensionalityReductionEvaluator(
                    umap_embedding,
                    self.best_clustering_labels
                )
                self.dim_evaluation_results['UMAP'] = evaluator.evaluate_overlap()
        except Exception as e:
            print(f"   ⚠️ UMAP実行エラー: {e}")
    
    def _select_best_dimensionality_reduction(self):
        """最適な次元削減手法を選択"""
        print("\n" + "="*70)
        print("🏆 次元削減手法の比較")
        print("="*70)
        
        best_method = 'PCA'
        best_score = self.dim_evaluation_results.get('PCA', {}).get('overlap', float('inf'))
        
        print(f"\n手法 | オーバーラップスコア (低↓)")
        print("-" * 70)
        
        for method, scores in self.dim_evaluation_results.items():
            overlap = scores.get('overlap', float('inf'))
            marker = "🥇" if overlap < best_score else "  "
            print(f"{marker} {method:10s} | {overlap:.4f}")
            
            if overlap < best_score:
                best_score = overlap
                best_method = method
        
        self.best_dim_reduction_method = best_method
        self.best_dim_reduction_embedding = self.dimensionality_results[best_method]['embedding']
        
        print(f"\n🎯 選択された手法: {best_method} (オーバーラップ: {best_score:.4f})")
        self._log(f"🎯 次元削減手法: {best_method}")
    
    def _print_summary(self):
        """結果のサマリーを表示"""
        print("\n" + "="*70)
        print("📋 Agenticクラスタリング 最終結果")
        print("="*70)
        
        print(f"\n🎯 最適クラスタリング手法: {self.best_clustering_method}")
        if self.best_clustering_method in self.evaluation_results:
            scores = self.evaluation_results[self.best_clustering_method]
            print(f"   総合スコア: {scores['overall']:.2f}/100")
            print(f"   シルエットスコア: {scores['silhouette']:.4f}")
        
        print(f"\n🎯 最適次元削減手法: {self.best_dim_reduction_method}")
        if self.best_dim_reduction_method in self.dim_evaluation_results:
            scores = self.dim_evaluation_results[self.best_dim_reduction_method]
            print(f"   オーバーラップスコア: {scores['overlap']:.4f}")
        
        print(f"\n📝 改善履歴:")
        for i, log in enumerate(self.improvement_log, 1):
            print(f"   {i}. {log}")
    
    def _log(self, message):
        """改善ログを記録"""
        print(f"\n{'='*70}")
        print(message)
        print('='*70)
        self.improvement_log.append(message)
    
    def _prepare_final_results(self):
        """最終結果を準備"""
        df_result = self.df.copy()
        df_result['cluster'] = self.best_clustering_labels
        
        # クラスタサマリー
        cluster_summary = df_result.groupby('cluster')[self.feature_cols].mean()
        
        # 3次元埋め込みデータを取得（UMAP使用時）
        embedding_3d = None
        if self.best_dim_reduction_method == 'UMAP' and 'UMAP' in self.dimensionality_results:
            umap_result = self.dimensionality_results['UMAP']
            if 'embedding_3d' in umap_result:
                embedding_3d = umap_result['embedding_3d']
        
        result = {
            'df_with_cluster': df_result,
            'cluster_summary': cluster_summary,
            'embedding': self.best_dim_reduction_embedding,
            'labels': self.best_clustering_labels,
            'clustering_method': self.best_clustering_method,
            'dim_reduction_method': self.best_dim_reduction_method,
            'evaluation_scores': self.evaluation_results.get(self.best_clustering_method, {}),
            'improvement_log': self.improvement_log
        }
        
        # 3次元埋め込みがあれば追加
        if embedding_3d is not None:
            result['embedding_3d'] = embedding_3d
        
        return result
