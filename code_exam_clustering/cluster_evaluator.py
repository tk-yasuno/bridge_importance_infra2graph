# -*- coding: utf-8 -*-
"""
クラスター評価モジュール: Agentic Clustering v0.2
クラスタリング結果の品質を多角的に評価
"""

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import pdist, squareform

class ClusterEvaluator:
    """クラスタリング結果を評価するクラス"""
    
    def __init__(self, X_scaled, labels):
        """
        Parameters:
        -----------
        X_scaled : array-like
            標準化された特徴量
        labels : array-like
            クラスタラベル
        """
        self.X_scaled = X_scaled
        self.labels = labels
        self.n_clusters = len(np.unique(labels))
        self.scores = {}
        
    def evaluate_all(self):
        """すべての評価指標を計算"""
        print("\n📊 クラスタリング品質を評価中...")
        
        # シルエットスコア（-1〜1、高いほど良い）
        self.scores['silhouette'] = silhouette_score(self.X_scaled, self.labels)
        
        # Davies-Bouldin指数（0以上、低いほど良い）
        self.scores['davies_bouldin'] = davies_bouldin_score(self.X_scaled, self.labels)
        
        # Calinski-Harabasz指数（高いほど良い）
        self.scores['calinski_harabasz'] = calinski_harabasz_score(self.X_scaled, self.labels)
        
        # クラスタサイズのバランス（0〜1、1に近いほど均等）
        self.scores['balance'] = self._calculate_balance()
        
        # 総合スコア（0〜100）
        self.scores['overall'] = self._calculate_overall_score()
        
        self._print_scores()
        
        return self.scores
    
    def _calculate_balance(self):
        """クラスタサイズのバランスを計算"""
        unique, counts = np.unique(self.labels, return_counts=True)
        max_count = counts.max()
        min_count = counts.min()
        
        if max_count == 0:
            return 0.0
        
        balance = min_count / max_count
        return balance
    
    def _calculate_overall_score(self):
        """総合スコアを計算（0〜100）"""
        # シルエットスコアを0-1に正規化（-1〜1 → 0〜1）
        silhouette_normalized = (self.scores['silhouette'] + 1) / 2
        
        # Davies-Bouldin指数を逆数化して正規化（低いほど良い → 高いほど良い）
        db_normalized = 1 / (1 + self.scores['davies_bouldin'])
        
        # Calinski-Harabasz指数を0-1に正規化（対数スケール）
        ch_normalized = min(1.0, np.log1p(self.scores['calinski_harabasz']) / 10)
        
        # 重み付き平均（シルエット45%、DB 45%、CH 0%、バランス10%）
        overall = (
            silhouette_normalized * 0.45 +
            db_normalized * 0.45 +
            ch_normalized * 0.0 +
            self.scores['balance'] * 0.10
        ) * 100
        
        return overall
    
    def _print_scores(self):
        """評価結果を表示"""
        print(f"\n📈 評価結果:")
        print(f"  ├─ シルエットスコア: {self.scores['silhouette']:.4f} (範囲: -1〜1, 高↑)")
        print(f"  ├─ Davies-Bouldin指数: {self.scores['davies_bouldin']:.4f} (範囲: 0〜, 低↓)")
        print(f"  ├─ Calinski-Harabasz指数: {self.scores['calinski_harabasz']:.2f} (範囲: 0〜, 高↑)")
        print(f"  ├─ クラスタバランス: {self.scores['balance']:.4f} (範囲: 0〜1, 高↑)")
        print(f"  └─ 総合スコア: {self.scores['overall']:.2f}/100")
    
    def needs_improvement(self, threshold=60.0):
        """改善が必要かどうかを判定"""
        needs = self.scores['overall'] < threshold
        
        if needs:
            print(f"\n⚠️  総合スコアが閾値 {threshold} を下回っています。")
            print(f"   → 代替手法による改善を試みます。")
        else:
            print(f"\n✅ 総合スコアが閾値 {threshold} を上回っています。")
            print(f"   → 現在の手法で十分な品質が得られています。")
        
        return needs
    
    def identify_issues(self):
        """具体的な問題点を特定"""
        issues = []
        
        if self.scores['silhouette'] < 0.3:
            issues.append("低シルエットスコア（クラスタ分離が不十分）")
        
        if self.scores['davies_bouldin'] > 1.5:
            issues.append("高Davies-Bouldin指数（クラスタ間の重複が大きい）")
        
        if self.scores['balance'] < 0.3:
            issues.append("クラスタサイズの不均衡")
        
        if len(issues) > 0:
            print(f"\n🔍 検出された問題点:")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
        
        return issues


class DimensionalityReductionEvaluator:
    """次元削減結果を評価するクラス"""
    
    def __init__(self, X_reduced, labels):
        """
        Parameters:
        -----------
        X_reduced : array-like
            次元削減後のデータ（通常は2次元）
        labels : array-like
            クラスタラベル
        """
        self.X_reduced = X_reduced
        self.labels = labels
        self.scores = {}
    
    def evaluate_overlap(self):
        """クラスタのオーバーラップを評価"""
        print("\n🔍 次元削減結果のオーバーラップを評価中...")
        
        # クラスタごとの中心を計算
        unique_labels = np.unique(self.labels)
        centers = []
        
        for label in unique_labels:
            mask = self.labels == label
            center = self.X_reduced[mask].mean(axis=0)
            centers.append(center)
        
        centers = np.array(centers)
        
        # クラスタ中心間の平均距離
        if len(centers) > 1:
            distances = pdist(centers)
            self.scores['mean_center_distance'] = distances.mean()
            self.scores['min_center_distance'] = distances.min()
        else:
            self.scores['mean_center_distance'] = 0
            self.scores['min_center_distance'] = 0
        
        # クラスタ内の平均分散
        variances = []
        for label in unique_labels:
            mask = self.labels == label
            cluster_points = self.X_reduced[mask]
            if len(cluster_points) > 1:
                variance = cluster_points.var(axis=0).mean()
                variances.append(variance)
        
        self.scores['mean_variance'] = np.mean(variances) if variances else 0
        
        # オーバーラップスコア（0〜1、低いほどオーバーラップが少ない）
        if self.scores['mean_variance'] > 0:
            self.scores['overlap'] = self.scores['mean_variance'] / (
                self.scores['mean_center_distance'] + 1e-10
            )
        else:
            self.scores['overlap'] = 0
        
        self._print_scores()
        
        return self.scores
    
    def _print_scores(self):
        """評価結果を表示"""
        print(f"\n📊 次元削減評価:")
        print(f"  ├─ クラスタ中心間の平均距離: {self.scores['mean_center_distance']:.4f}")
        print(f"  ├─ クラスタ中心間の最小距離: {self.scores['min_center_distance']:.4f}")
        print(f"  ├─ クラスタ内平均分散: {self.scores['mean_variance']:.4f}")
        print(f"  └─ オーバーラップスコア: {self.scores['overlap']:.4f} (低↓)")
    
    def has_high_overlap(self, threshold=0.5):
        """オーバーラップが高いかどうかを判定"""
        has_overlap = self.scores['overlap'] > threshold
        
        if has_overlap:
            print(f"\n⚠️  オーバーラップスコアが閾値 {threshold} を上回っています。")
            print(f"   → 代替次元削減手法（t-SNE/UMAP）を試みます。")
        else:
            print(f"\n✅ オーバーラップスコアが閾値 {threshold} を下回っています。")
            print(f"   → PCAで十分な分離が得られています。")
        
        return has_overlap


def compare_methods(results_dict):
    """複数の手法の結果を比較"""
    print("\n" + "="*70)
    print("🏆 手法比較")
    print("="*70)
    
    comparison = []
    
    for method_name, result in results_dict.items():
        if 'overall' in result:
            comparison.append({
                'method': method_name,
                'overall': result['overall'],
                'silhouette': result['silhouette'],
                'davies_bouldin': result['davies_bouldin']
            })
    
    # 総合スコアでソート
    comparison.sort(key=lambda x: x['overall'], reverse=True)
    
    print("\n順位 | 手法 | 総合スコア | シルエット | DB指数")
    print("-" * 70)
    
    for i, result in enumerate(comparison, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{medal} {i}位 | {result['method']:15s} | "
              f"{result['overall']:6.2f} | "
              f"{result['silhouette']:6.3f} | "
              f"{result['davies_bouldin']:6.3f}")
    
    best_method = comparison[0]['method']
    print(f"\n🎯 最適手法: {best_method} (総合スコア: {comparison[0]['overall']:.2f})")
    
    return best_method, comparison
