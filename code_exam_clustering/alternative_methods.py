# -*- coding: utf-8 -*-
"""
代替クラスタリング手法モジュール: Agentic Clustering v0.8
GMM, DBSCANなどの代替アルゴリズムを提供
HDBSCAN, CLASSIXのベイズ最適化の探索も含む
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
import config

class AlternativeClusteringMethods:
    """代替クラスタリング手法を提供するクラス"""
    
    def __init__(self, X_scaled):
        """
        Parameters:
        -----------
        X_scaled : array-like
            標準化された特徴量
        """
        self.X_scaled = X_scaled
        self.results = {}
    
    def try_kmeans(self, n_clusters):
        """KMeansクラスタリング"""
        print(f"\n🔵 KMeans (k={n_clusters}) を実行中...")
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=config.RANDOM_STATE,
            n_init=10,
            max_iter=300
        )
        
        labels = kmeans.fit_predict(self.X_scaled)
        
        # クラスタ分布を表示
        unique, counts = np.unique(labels, return_counts=True)
        print(f"   クラスタ数: {len(unique)}")
        for cluster_id, count in zip(unique, counts):
            print(f"   クラスタ {cluster_id}: {count}件")
        
        self.results['KMeans'] = {
            'model': kmeans,
            'labels': labels,
            'n_clusters': n_clusters
        }
        
        return labels
    
    def try_gmm(self, n_components_range=None, use_bayesian=False, n_calls=100):
        """ガウス混合モデル（GMM）
        
        Parameters:
        -----------
        n_components_range : range, optional
            探索するコンポーネント数の範囲（グリッドサーチの場合）
        use_bayesian : bool, default=False
            Trueの場合、ベイズ最適化を使用
        n_calls : int, default=100
            ベイズ最適化の評価回数
        """
        print(f"\n🟣 GMM (Gaussian Mixture Model) を実行中...")
        
        # ベイズ最適化を試行
        if use_bayesian:
            try:
                from skopt import gp_minimize
                from skopt.space import Integer
                from skopt.utils import use_named_args
                print(f"   🔬 ベイズ最適化でパラメータ探索を実行します（{n_calls}回評価）")
                return self._try_gmm_bayesian(n_calls=n_calls)
            except ImportError:
                print("   ⚠️ scikit-optimizeが利用できません。グリッドサーチにフォールバックします。")
        
        # グリッドサーチ
        if n_components_range is None:
            n_components_range = range(config.MIN_CLUSTERS, config.MAX_CLUSTERS + 1)
        
        best_gmm = None
        best_labels = None
        best_score = -1
        best_n = config.MIN_CLUSTERS
        
        for n in n_components_range:
            gmm = GaussianMixture(
                n_components=n,
                covariance_type='full',
                random_state=config.RANDOM_STATE,
                n_init=10
            )
            
            labels = gmm.fit_predict(self.X_scaled)
            
            # ノイズクラスタ（-1）がある場合は除外して評価
            if len(np.unique(labels)) > 1:
                score = silhouette_score(self.X_scaled, labels)
                print(f"   n_components={n}: シルエットスコア = {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_gmm = gmm
                    best_labels = labels
                    best_n = n
        
        print(f"   ✓ 最適コンポーネント数: {best_n} (スコア: {best_score:.4f})")
        
        # クラスタ分布を表示
        unique, counts = np.unique(best_labels, return_counts=True)
        print(f"   クラスタ数: {len(unique)}")
        for cluster_id, count in zip(unique, counts):
            print(f"   クラスタ {cluster_id}: {count}件")
        
        self.results['GMM'] = {
            'model': best_gmm,
            'labels': best_labels,
            'n_clusters': best_n,
            'score': best_score
        }
        
        return best_labels
    
    def _try_gmm_bayesian(self, n_calls=100):
        """GMMのベイズ最適化
        
        Parameters:
        -----------
        n_calls : int
            ベイズ最適化の評価回数
        """
        from skopt import forest_minimize  # 並列化に適した手法
        from skopt.space import Integer
        from skopt.utils import use_named_args
        from sklearn.metrics import davies_bouldin_score
        
        # 探索空間の定義（n_componentsとcovariance_type）
        # covariance_typeは整数でマッピング: 0=full, 1=tied, 2=diag, 3=spherical
        space = [
            Integer(10, 76, name='n_components'),  # クラスタ数: 10-76
            Integer(0, 3, name='cov_type_idx')     # 共分散タイプ: 0-3
        ]
        
        cov_types = ['full', 'tied', 'diag', 'spherical']
        
        @use_named_args(space)
        def objective(n_components, cov_type_idx):
            """最小化する目的関数（負のスコアを返す）"""
            try:
                cov_type = cov_types[cov_type_idx]
                
                # GMMモデルの構築と訓練
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type=cov_type,
                    random_state=config.RANDOM_STATE,
                    n_init=5,  # ベイズ最適化では各評価を高速化
                    max_iter=100
                )
                
                labels = gmm.fit_predict(self.X_scaled)
                n_clusters = len(np.unique(labels))
                
                # クラスタが1つだけの場合はペナルティ
                if n_clusters < 2:
                    return 0.0  # 最悪のスコア
                
                # シルエットスコア（0-1, 高いほど良い）
                silhouette = silhouette_score(self.X_scaled, labels)
                
                # Davis-Bouldin Index（0以上、低いほど良い → 正規化）
                db_index = davies_bouldin_score(self.X_scaled, labels)
                db_normalized = max(0, 1.0 - db_index / 3.0)  # 3.0で割って0-1に正規化
                
                # クラスタバランススコア（標準偏差ベース）
                unique_labels, counts = np.unique(labels, return_counts=True)
                if len(counts) > 1:
                    std_count = counts.std()
                    mean_count = counts.mean()
                    balance_score = max(0, 1.0 - std_count / (mean_count + 1e-6))
                else:
                    balance_score = 0.0
                
                # 複合スコア (0.35:0.35:0.3 = Silhouette:DB:Balance)
                combined_score = 0.35 * silhouette + 0.35 * db_normalized + 0.3 * balance_score
                
                # クラスタ数ペナルティ（10以上、76以下を強制）
                if n_clusters < 10:
                    combined_score *= 0.1  # 90%ペナルティ
                elif n_clusters < 15:
                    combined_score *= 0.6  # 40%ペナルティ
                elif n_clusters > 76:
                    combined_score *= 0.1  # 90%ペナルティ
                elif n_clusters > 60:
                    combined_score *= 0.7  # 30%ペナルティ
                
                return -combined_score  # 最小化するため負の値を返す
                
            except Exception as e:
                print(f"   ⚠️ GMM評価エラー (n={n_components}, cov={cov_types[cov_type_idx]}): {e}")
                return 0.0  # エラー時は最悪のスコア
        
        # 進捗表示用のコールバック
        iteration = [0]
        def on_step(res):
            iteration[0] += 1
            if iteration[0] % 10 == 0 or iteration[0] <= 5:
                print(f"   評価 {iteration[0]}/{n_calls}: 現在の最良スコア = {-res.fun:.4f}")
        
        # ベイズ最適化実行（forest_minimizeで並列処理有効）
        cores_msg = "全CPUコア" if config.N_JOBS == -1 else f"{config.N_JOBS}コア"
        print(f"   🚀 Random Forestベースで{cores_msg}並列処理を実行...")
        print(f"   📊 初期ランダム探索: {config.N_INITIAL_POINTS}回(並列)、逐次探索: {n_calls - config.N_INITIAL_POINTS}回")
        result = forest_minimize(
            objective,
            space,
            n_calls=n_calls,
            n_initial_points=config.N_INITIAL_POINTS,
            n_jobs=config.N_JOBS,  # 並列処理
            random_state=config.RANDOM_STATE,
            callback=[on_step],
            verbose=False
        )
        
        # 最適パラメータの取得
        best_n_components = result.x[0]
        best_cov_type = cov_types[result.x[1]]
        best_score = -result.fun
        
        print(f"\n   ✅ ベイズ最適化完了")
        print(f"   最適パラメータ: n_components={best_n_components}, covariance_type={best_cov_type}")
        print(f"   最良スコア: {best_score:.4f}")
        
        # 最適パラメータで最終モデルを構築
        best_gmm = GaussianMixture(
            n_components=best_n_components,
            covariance_type=best_cov_type,
            random_state=config.RANDOM_STATE,
            n_init=10
        )
        
        best_labels = best_gmm.fit_predict(self.X_scaled)
        
        # クラスタ分布を表示
        unique, counts = np.unique(best_labels, return_counts=True)
        print(f"   クラスタ数: {len(unique)}")
        for cluster_id, count in zip(unique, counts):
            print(f"   クラスタ {cluster_id}: {count}件")
        
        self.results['GMM'] = {
            'model': best_gmm,
            'labels': best_labels,
            'n_clusters': best_n_components,
            'score': best_score
        }
        
        return best_labels
    
    def try_dbscan(self, eps_range=None, min_samples_range=None, target_clusters=50, use_bayesian=False, n_calls=100):
        """DBSCAN(密度ベースクラスタリング)
        
        Parameters:
        -----------
        eps_range : list, optional
            探索するepsの範囲（グリッドサーチの場合）
        min_samples_range : list, optional
            探索するmin_samplesの範囲（グリッドサーチの場合）
        target_clusters : int, default=50
            目標クラスタ数（参考値）
        use_bayesian : bool, default=False
            Trueの場合、ベイズ最適化を使用
        n_calls : int, default=100
            ベイズ最適化の評価回数
        """
        print(f"\n🟢 DBSCAN (Density-Based Spatial Clustering) を実行中...")
        print(f"   目標クラスタ数: {target_clusters}程度")
        
        # ベイズ最適化を試行
        if use_bayesian:
            try:
                from skopt import gp_minimize
                from skopt.space import Real, Integer
                from skopt.utils import use_named_args
                print(f"   🔬 ベイズ最適化でパラメータ探索を実行します（{n_calls}回評価）")
                return self._try_dbscan_bayesian(target_clusters=target_clusters, n_calls=n_calls)
            except ImportError:
                print("   ⚠️ scikit-optimizeが利用できません。グリッドサーチにフォールバックします。")
        
        # デフォルトパラメータ範囲（クラスタ数50程度に調整）
        if eps_range is None:
            eps_range = [0.8, 1.0, 1.2, 1.4, 1.6]
        
        if min_samples_range is None:
            min_samples_range = [15, 20, 25, 30, 35]
        
        best_dbscan = None
        best_labels = None
        best_score = -1
        best_params = None
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(self.X_scaled)
                
                # ノイズポイント（-1）を除いたクラスタ数
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                # クラスタが2つ以上ある場合のみ評価
                if n_clusters >= 2:
                    # ノイズポイントを除外してシルエットスコアを計算
                    mask = labels != -1
                    if mask.sum() > 0:
                        score = silhouette_score(self.X_scaled[mask], labels[mask])
                        
                        # 目標クラスタ数からの距離を計算（ペナルティ）
                        cluster_penalty = abs(n_clusters - target_clusters) / target_clusters
                        adjusted_score = score * (1 - cluster_penalty * 0.5)  # クラスタ数の差に応じてスコアを調整
                        
                        print(f"   eps={eps}, min_samples={min_samples}: "
                              f"クラスタ数={n_clusters}, ノイズ={n_noise}, "
                              f"スコア={score:.4f}, 調整後={adjusted_score:.4f}")
                        
                        # ノイズが少なく、調整後スコアが高く、クラスタ数が目標に近いものを選択
                        if (adjusted_score > best_score and 
                            n_noise < len(labels) * 0.35 and 
                            20 <= n_clusters <= 100):  # クラスタ数の妥当な範囲（60を中心に）
                            best_score = adjusted_score
                            best_dbscan = dbscan
                            best_labels = labels
                            best_params = {'eps': eps, 'min_samples': min_samples}
        
        if best_labels is not None:
            n_clusters_final = len(set(best_labels)) - (1 if -1 in best_labels else 0)
            n_noise_final = list(best_labels).count(-1)
            
            print(f"   ✓ 最適パラメータ: eps={best_params['eps']}, "
                  f"min_samples={best_params['min_samples']} (調整後スコア: {best_score:.4f})")
            print(f"   ✓ クラスタ数: {n_clusters_final}, ノイズ: {n_noise_final}件")
            
            # クラスタ分布を表示（上位10件のみ）
            unique, counts = np.unique(best_labels, return_counts=True)
            sorted_clusters = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)
            print(f"   主要クラスタ分布（上位10件）:")
            for i, (cluster_id, count) in enumerate(sorted_clusters[:10]):
                label_name = "ノイズ" if cluster_id == -1 else f"クラスタ {cluster_id}"
                print(f"     {label_name}: {count}件")
            
            self.results['DBSCAN'] = {
                'model': best_dbscan,
                'labels': best_labels,
                'n_clusters': len(set(best_labels)) - (1 if -1 in best_labels else 0),
                'score': best_score,
                'params': best_params
            }
        else:
            print(f"   ⚠️ 適切なDBSCANパラメータが見つかりませんでした。")
            # フォールバック: デフォルトパラメータで実行
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            best_labels = dbscan.fit_predict(self.X_scaled)
            
            self.results['DBSCAN'] = {
                'model': dbscan,
                'labels': best_labels,
                'n_clusters': len(set(best_labels)) - (1 if -1 in best_labels else 0),
                'score': -1,
                'params': {'eps': 0.5, 'min_samples': 5}
            }
        
        return best_labels
    
    def _try_dbscan_bayesian(self, target_clusters=50, n_calls=100):
        """DBSCANのベイズ最適化
        
        Parameters:
        -----------
        target_clusters : int
            目標クラスタ数（参考値）
        n_calls : int
            ベイズ最適化の評価回数
        """
        from skopt import forest_minimize  # 並列化に適した手法
        from skopt.space import Real, Integer
        from skopt.utils import use_named_args
        from sklearn.metrics import davies_bouldin_score
        
        # 探索空間の定義（eps: 0.5-2.0, min_samples: 5-50）
        space = [
            Real(0.5, 2.0, name='eps'),
            Integer(5, 50, name='min_samples')
        ]
        
        @use_named_args(space)
        def objective(eps, min_samples):
            """最小化する目的関数（負のスコアを返す）"""
            try:
                # DBSCANモデルの構築と訓練
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(self.X_scaled)
                
                # ノイズポイント（-1）を除いたクラスタ数
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                noise_ratio = n_noise / len(labels)
                
                # クラスタが2つ未満の場合はペナルティ
                if n_clusters < 2:
                    return 0.0  # 最悪のスコア
                
                # ノイズポイントを除外して評価
                mask = labels != -1
                if mask.sum() < 2:
                    return 0.0
                
                # シルエットスコア（0-1, 高いほど良い）
                silhouette = silhouette_score(self.X_scaled[mask], labels[mask])
                
                # Davis-Bouldin Index（0以上、低いほど良い → 正規化）
                db_index = davies_bouldin_score(self.X_scaled[mask], labels[mask])
                db_normalized = max(0, 1.0 - db_index / 3.0)  # 3.0で割って0-1に正規化
                
                # クラスタバランススコア（標準偏差ベース）
                unique_labels, counts = np.unique(labels[mask], return_counts=True)
                if len(counts) > 1:
                    std_count = counts.std()
                    mean_count = counts.mean()
                    balance_score = max(0, 1.0 - std_count / (mean_count + 1e-6))
                else:
                    balance_score = 0.0
                
                # 複合スコア (0.35:0.35:0.3 = Silhouette:DB:Balance)
                combined_score = 0.35 * silhouette + 0.35 * db_normalized + 0.3 * balance_score
                
                # クラスタ数ペナルティ（10以上、76以下を強制）
                if n_clusters < 10:
                    combined_score *= 0.1  # 90%ペナルティ
                elif n_clusters < 15:
                    combined_score *= 0.6  # 40%ペナルティ
                elif n_clusters > 76:
                    combined_score *= 0.1  # 90%ペナルティ
                elif n_clusters > 60:
                    combined_score *= 0.7  # 30%ペナルティ
                
                # ノイズ比率ペナルティ（35%以上は大幅減点）
                if noise_ratio < 0.10:
                    combined_score *= (1.0 - noise_ratio * 0.5)
                elif noise_ratio < 0.35:
                    combined_score *= max(0.1, 1.0 - noise_ratio * 1.5)
                else:
                    combined_score *= 0.05  # 95%ペナルティ
                
                return -combined_score  # 最小化するため負の値を返す
                
            except Exception as e:
                print(f"   ⚠️ DBSCAN評価エラー (eps={eps:.2f}, min_samples={min_samples}): {e}")
                return 0.0  # エラー時は最悪のスコア
        
        # 進捗表示用のコールバック
        iteration = [0]
        def on_step(res):
            iteration[0] += 1
            if iteration[0] % 10 == 0 or iteration[0] <= 5:
                print(f"   評価 {iteration[0]}/{n_calls}: 現在の最良スコア = {-res.fun:.4f}")
        
        # ベイズ最適化実行（forest_minimizeで並列処理有効）
        cores_msg = "全CPUコア" if config.N_JOBS == -1 else f"{config.N_JOBS}コア"
        print(f"   🚀 Random Forestベースで{cores_msg}並列処理を実行...")
        print(f"   📊 初期ランダム探索: {config.N_INITIAL_POINTS}回(並列)、逐次探索: {n_calls - config.N_INITIAL_POINTS}回")
        result = forest_minimize(
            objective,
            space,
            n_calls=n_calls,
            n_initial_points=config.N_INITIAL_POINTS,
            n_jobs=config.N_JOBS,  # 並列処理
            random_state=config.RANDOM_STATE,
            callback=[on_step],
            verbose=False
        )
        
        # 最適パラメータの取得
        best_eps = result.x[0]
        best_min_samples = result.x[1]
        best_score = -result.fun
        
        print(f"\n   ✅ ベイズ最適化完了")
        print(f"   最適パラメータ: eps={best_eps:.3f}, min_samples={best_min_samples}")
        print(f"   最良スコア: {best_score:.4f}")
        
        # 最適パラメータで最終モデルを構築
        best_dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
        best_labels = best_dbscan.fit_predict(self.X_scaled)
        
        # クラスタ情報を表示
        n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
        n_noise = list(best_labels).count(-1)
        noise_ratio = n_noise / len(best_labels)
        
        print(f"   クラスタ数: {n_clusters}, ノイズ: {n_noise}件 ({noise_ratio*100:.1f}%)")
        
        # クラスタ分布を表示（上位10件のみ）
        unique, counts = np.unique(best_labels, return_counts=True)
        sorted_clusters = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)
        print(f"   主要クラスタ分布（上位10件）:")
        for i, (cluster_id, count) in enumerate(sorted_clusters[:10]):
            label_name = "ノイズ" if cluster_id == -1 else f"クラスタ {cluster_id}"
            print(f"     {label_name}: {count}件")
        
        self.results['DBSCAN'] = {
            'model': best_dbscan,
            'labels': best_labels,
            'n_clusters': n_clusters,
            'score': best_score,
            'params': {'eps': best_eps, 'min_samples': best_min_samples}
        }
        
        return best_labels
    
    def try_hdbscan(self, min_cluster_size_range=None, min_samples_range=None, target_clusters=50, use_bayesian=True, n_calls=100):
        """HDBSCAN (Hierarchical Density-Based Spatial Clustering) with Bayesian Optimization
        
        DBSCANのクラスタ数が多すぎる場合の代替手法として使用。
        HDBSCANは階層的な密度ベースクラスタリングで、より適応的なクラスタを生成する。
        
        Args:
            use_bayesian: ベイズ最適化を使用するか（デフォルト: True）
            n_calls: ベイズ最適化の評価回数（デフォルト: 100）
        """
        print(f"\n🟡 HDBSCAN (Hierarchical DBSCAN) を実行中...")
        print(f"   目標クラスタ数: {target_clusters}程度")
        if use_bayesian:
            print(f"   最適化手法: ベイズ最適化（評価回数: {n_calls}回）")
        
        try:
            import hdbscan
        except ImportError:
            print(f"   ⚠️ HDBSCANがインストールされていません。")
            print(f"   'pip install hdbscan' でインストールしてください。")
            return None
        
        # ベイズ最適化を使用する場合
        if use_bayesian:
            try:
                from skopt import gp_minimize
                from skopt.space import Integer
                from skopt.utils import use_named_args
            except ImportError:
                print(f"   ⚠️ scikit-optimizeがインストールされていません。")
                print(f"   'pip install scikit-optimize' でインストールするか、use_bayesian=Falseに設定してください。")
                print(f"   グリッドサーチにフォールバックします...")
                use_bayesian = False
        
        if use_bayesian:
            return self._try_hdbscan_bayesian(target_clusters, n_calls)
        
        # グリッドサーチ（従来の方法）
        # v0.8改: パラメータ探索密度を2倍に増加（v0.6の13個→26個）
        if min_cluster_size_range is None:
            # 探索密度2倍: 3-100を26個のポイントで探索
            min_cluster_size_range = [3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
        
        if min_samples_range is None:
            # 探索密度2倍: 1-25を18個のポイントで探索
            min_samples_range = [1, 2, 3, 5, 8, 10, 15, 20, 25]
        
        best_hdbscan = None
        best_labels = None
        best_score = -1
        best_params = None
        
        for min_cluster_size in min_cluster_size_range:
            for min_samples in min_samples_range:
                try:
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        cluster_selection_method='eom',  # Excess of Mass
                        metric='euclidean'
                    )
                    
                    labels = clusterer.fit_predict(self.X_scaled)
                    
                    # クラスタ数とノイズポイント数
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    
                    # クラスタが2つ以上ある場合のみ評価
                    if n_clusters >= 2:
                        # ノイズポイントを除外してシルエットスコアを計算
                        mask = labels != -1
                        if mask.sum() > 1 and len(set(labels[mask])) > 1:
                            score = silhouette_score(self.X_scaled[mask], labels[mask])
                            
                            # 目標クラスタ数からの距離を計算(ペナルティ)
                            cluster_penalty = abs(n_clusters - target_clusters) / target_clusters
                            # v0.7改: ノイズ10%未満を最優先目標とする
                            noise_ratio = n_noise / len(labels)
                            
                            # ノイズ10%未満なら大幅ボーナス、それ以上は大幅ペナルティ
                            if noise_ratio < 0.10:
                                noise_bonus = 1.0 + (0.10 - noise_ratio) * 10.0  # 10%未満なら最大+100%ボーナス
                            else:
                                noise_bonus = 1.0 - (noise_ratio - 0.10) * 5.0  # 10%超過ごとに-50%ペナルティ
                            
                            adjusted_score = score * (1 - cluster_penalty * 0.2) * max(0.01, noise_bonus)
                            
                            print(f"   min_cluster_size={min_cluster_size}, min_samples={min_samples}: "
                                  f"クラスタ数={n_clusters}, ノイズ={n_noise}({noise_ratio*100:.1f}%), "
                                  f"スコア={score:.4f}, 調整後={adjusted_score:.4f}")
                            
                            # v0.7改: ノイズ閾値を撤廃（調整後スコアで自動的に10%未満が優遇される）
                            # クラスタ数範囲: 10〜150個（より広く許容）
                            if (adjusted_score > best_score and 
                                10 <= n_clusters <= 150):
                                best_score = adjusted_score
                                best_hdbscan = clusterer
                                best_labels = labels
                                best_params = {'min_cluster_size': min_cluster_size, 'min_samples': min_samples}
                
                except Exception as e:
                    print(f"   ⚠️ min_cluster_size={min_cluster_size}, min_samples={min_samples}でエラー: {e}")
                    continue
        
        # ノイズ15%以下の候補がない場合、最良の結果を採用
        if best_labels is None:
            print(f"   ⚠️ ノイズ15%以下の候補が見つかりませんでした。最良の結果を採用します。")
            # 最小ノイズの結果を再探索
            for min_cluster_size in min_cluster_size_range:
                for min_samples in min_samples_range:
                    try:
                        clusterer = hdbscan.HDBSCAN(
                            min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            metric='euclidean'
                        )
                        labels = clusterer.fit_predict(self.X_scaled)
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        n_noise = list(labels).count(-1)
                        
                        if n_clusters >= 2:
                            mask = labels != -1
                            if mask.sum() > 1 and len(set(labels[mask])) > 1:
                                score = silhouette_score(self.X_scaled[mask], labels[mask])
                                cluster_penalty = abs(n_clusters - target_clusters) / target_clusters
                                noise_ratio = n_noise / len(labels)
                                noise_penalty = noise_ratio
                                adjusted_score = score * (1 - cluster_penalty * 0.5) * (1 - noise_penalty * 0.6)
                                
                                if (adjusted_score > best_score and 
                                    n_noise < len(labels) * 0.30 and  # 緩和した基準
                                    20 <= n_clusters <= 80):
                                    best_score = adjusted_score
                                    best_hdbscan = clusterer
                                    best_labels = labels
                                    best_params = {'min_cluster_size': min_cluster_size, 'min_samples': min_samples}
                    except:
                        continue
        
        if best_labels is not None:
            n_clusters_final = len(set(best_labels)) - (1 if -1 in best_labels else 0)
            n_noise_final = list(best_labels).count(-1)
            noise_ratio_final = n_noise_final / len(best_labels)
            
            print(f"   ✓ 最適パラメータ: min_cluster_size={best_params['min_cluster_size']}, "
                  f"min_samples={best_params['min_samples']} (調整後スコア: {best_score:.4f})")
            print(f"   ✓ クラスタ数: {n_clusters_final}, ノイズ: {n_noise_final}件 ({noise_ratio_final*100:.1f}%)")
            
            # クラスタ分布を表示（上位10件のみ）
            unique, counts = np.unique(best_labels, return_counts=True)
            sorted_clusters = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)
            print(f"   主要クラスタ分布（上位10件）:")
            for i, (cluster_id, count) in enumerate(sorted_clusters[:10]):
                label_name = "ノイズ" if cluster_id == -1 else f"クラスタ {cluster_id}"
                print(f"     {label_name}: {count}件")
            
            self.results['HDBSCAN'] = {
                'model': best_hdbscan,
                'labels': best_labels,
                'n_clusters': n_clusters_final,
                'score': best_score,
                'params': best_params
            }
        else:
            print(f"   ⚠️ 適切なHDBSCANパラメータが見つかりませんでした。")
            self.results['HDBSCAN'] = None
        
        return best_labels
    
    def _try_hdbscan_bayesian(self, target_clusters, n_calls):
        """ベイズ最適化を使用したHDBSCANパラメータ探索
        
        Note: target_clustersは互換性のために残しているが、最適化では使用しない
        """
        import hdbscan
        from skopt import forest_minimize  # 並列化に適した手法
        from skopt.space import Integer
        from skopt.utils import use_named_args
        
        # 探索空間の定義（v0.8改: より適切な範囲に絞る）
        space = [
            Integer(10, 100, name='min_cluster_size'),  # 下限を10に引き上げ
            Integer(3, 25, name='min_samples')  # 下限を3に引き上げ
        ]
        
        # 目的関数（最小化するため、調整後スコアの負値を返す）
        @use_named_args(space)
        def objective(min_cluster_size, min_samples):
            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_method='eom',
                    metric='euclidean'
                )
                
                labels = clusterer.fit_predict(self.X_scaled)
                
                # クラスタ数計算
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                noise_ratio = n_noise / len(labels)
                
                # 制約チェック
                if n_clusters < 2:
                    return 10.0  # ペナルティ大
                
                # ノイズポイントを除外してシルエットスコアを計算
                mask = labels != -1
                if mask.sum() <= 1 or len(set(labels[mask])) <= 1:
                    return 10.0
                
                # v0.8改: Silhouette係数とDavis-Bouldin Indexを0.5:0.5で評価
                from sklearn.metrics import davies_bouldin_score
                
                silhouette = silhouette_score(self.X_scaled[mask], labels[mask])
                db_index = davies_bouldin_score(self.X_scaled[mask], labels[mask])
                
                # Silhouette: 高い方が良い (0-1), DB Index: 低い方が良い (0-)
                # DB Indexを0-1に正規化（典型的な範囲0-3を想定）
                db_normalized = max(0, 1.0 - db_index / 3.0)
                
                # クラスタバランススコア（各クラスタの均等性を評価）
                unique_labels, counts = np.unique(labels[mask], return_counts=True)
                if len(counts) > 1:
                    # 標準偏差を使用して不均衡度を計算（0-1に正規化）
                    mean_count = counts.mean()
                    std_count = counts.std()
                    balance_score = max(0, 1.0 - std_count / (mean_count + 1e-6))
                else:
                    balance_score = 0.0
                
                # 複合スコア (0.35:0.35:0.3 = Silhouette:DB:Balance)
                # v0.8改: バランススコアの重みを0.2→0.3に増加
                combined_score = 0.35 * silhouette + 0.35 * db_normalized + 0.3 * balance_score
                
                # クラスタ数ペナルティ（10以上、76以下を強制、15-76が理想範囲）
                # 76 = 自治体総数19 × 4
                if n_clusters < 10:
                    combined_score *= 0.1  # 90%減点（非常に大きなペナルティ）
                elif n_clusters < 15:
                    combined_score *= 0.6  # 40%減点
                elif n_clusters > 76:
                    combined_score *= 0.1  # 90%減点（上限超過も厳しくペナルティ）
                elif n_clusters > 60:
                    combined_score *= 0.7  # 30%減点
                
                # ノイズペナルティ（10%未満は軽微、10%以上は大きなペナルティ）
                if noise_ratio < 0.10:
                    combined_score *= (1.0 - noise_ratio * 0.5)  # 最大5%減点
                else:
                    combined_score *= max(0.01, 1.0 - noise_ratio * 2.0)  # 大幅減点
                
                return -combined_score  # 最小化するため負値
                
            except Exception as e:
                return 10.0  # エラー時は大きなペナルティ
        
        # ベイズ最適化の実行（進捗表示付き）
        print(f"   ベイズ最適化を開始（{n_calls}回の評価）...")
        
        # コールバック関数で進捗表示
        iteration = [0]
        def on_step(res):
            iteration[0] += 1
            if iteration[0] % 10 == 0 or iteration[0] <= 5:
                best_score = -res.fun
                print(f"      評価 {iteration[0]}/{n_calls}: 現在の最良スコア = {best_score:.4f}")
        
        # ベイズ最適化実行（forest_minimizeで並列処理有効）
        cores_msg = "全CPUコア" if config.N_JOBS == -1 else f"{config.N_JOBS}コア"
        print(f"   🚀 Random Forestベースで{cores_msg}並列処理を実行...")
        print(f"   📊 初期ランダム探索: {config.N_INITIAL_POINTS}回(並列)、逐次探索: {n_calls - config.N_INITIAL_POINTS}回")
        result = forest_minimize(
            objective,
            space,
            n_calls=n_calls,
            random_state=config.RANDOM_STATE,
            verbose=False,
            n_initial_points=config.N_INITIAL_POINTS,
            n_jobs=config.N_JOBS,  # 並列処理
            callback=[on_step]
        )
        
        # 最適パラメータで最終モデルを構築
        best_min_cluster_size = result.x[0]
        best_min_samples = result.x[1]
        best_score = -result.fun
        
        print(f"   ✓ 最適パラメータ: min_cluster_size={best_min_cluster_size}, min_samples={best_min_samples}")
        print(f"   ✓ 調整後スコア: {best_score:.4f}")
        
        # 最適パラメータで最終クラスタリング
        best_hdbscan = hdbscan.HDBSCAN(
            min_cluster_size=best_min_cluster_size,
            min_samples=best_min_samples,
            cluster_selection_method='eom',
            metric='euclidean'
        )
        best_labels = best_hdbscan.fit_predict(self.X_scaled)
        
        n_clusters_final = len(set(best_labels)) - (1 if -1 in best_labels else 0)
        n_noise_final = list(best_labels).count(-1)
        noise_ratio_final = n_noise_final / len(best_labels)
        
        print(f"   ✓ クラスタ数: {n_clusters_final}, ノイズ: {n_noise_final}件 ({noise_ratio_final*100:.1f}%)")
        
        # クラスタ分布を表示（上位10件のみ）
        unique, counts = np.unique(best_labels, return_counts=True)
        sorted_clusters = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)
        print(f"   主要クラスタ分布（上位10件）:")
        for i, (cluster_id, count) in enumerate(sorted_clusters[:10]):
            label_name = "ノイズ" if cluster_id == -1 else f"クラスタ {cluster_id}"
            print(f"     {label_name}: {count}件")
        
        self.results['HDBSCAN'] = {
            'model': best_hdbscan,
            'labels': best_labels,
            'n_clusters': n_clusters_final,
            'score': best_score,
            'params': {'min_cluster_size': best_min_cluster_size, 'min_samples': best_min_samples}
        }
        
        return best_labels
    
    def try_classix(self, radius_range=None, min_samples_range=None, target_clusters=28, use_bayesian=True, n_calls=100):
        """CLASSIX (Fast and Explainable Clustering) with Bayesian Optimization
        
        HDBSCANのノイズ問題を解決する代替手法。
        CLASSIXは高速で、ノイズが少なく、説明可能なクラスタリングを実現する。
        
        Args:
            use_bayesian: ベイズ最適化を使用するか（デフォルト: True）
            n_calls: ベイズ最適化の評価回数（デフォルト: 100）
        """
        print(f"\n🟠 CLASSIX (Fast Clustering) を実行中...")
        print(f"   目標クラスタ数: {target_clusters}程度")
        if use_bayesian:
            print(f"   最適化手法: ベイズ最適化（評価回数: {n_calls}回）")
        
        try:
            from classix import CLASSIX
        except ImportError:
            print(f"   ⚠️ CLASSIXがインストールされていません。")
            print(f"   'pip install classixclustering' でインストールしてください。")
            return None
        
        # ベイズ最適化を使用する場合
        if use_bayesian:
            try:
                from skopt import gp_minimize
                from skopt.space import Real, Integer
                from skopt.utils import use_named_args
            except ImportError:
                print(f"   ⚠️ scikit-optimizeがインストールされていません。")
                print(f"   'pip install scikit-optimize' でインストールするか、use_bayesian=Falseに設定してください。")
                print(f"   グリッドサーチにフォールバックします...")
                use_bayesian = False
        
        if use_bayesian:
            return self._try_classix_bayesian(target_clusters, n_calls)
        
        # グリッドサーチ（従来の方法）
        # v0.8改: パラメータ探索密度を8倍に増加（より細かく最適化）
        if radius_range is None:
            # 探索密度8倍: 0.05-1.0を56個のポイントで探索
            radius_range = [round(r, 3) for r in list(np.arange(0.05, 0.31, 0.02)) + list(np.arange(0.31, 0.61, 0.03)) + list(np.arange(0.61, 1.01, 0.05))]
        
        if min_samples_range is None:
            # 探索密度8倍: 1-40を40個のポイントで探索
            min_samples_range = list(range(1, 11)) + list(range(12, 21, 2)) + list(range(22, 41, 3))
        
        best_classix = None
        best_labels = None
        best_score = -1
        best_params = None
        
        for radius in radius_range:
            for min_samples in min_samples_range:
                try:
                    # データ型をfloat64に明示的に変換（dtype mismatch回避）
                    X_scaled_float64 = self.X_scaled.astype(np.float64)
                    
                    classix = CLASSIX(
                        radius=radius,
                        minPts=min_samples,
                        verbose=0,
                        post_alloc=True  # ノイズポイントを近隣クラスタに再配置
                    )
                    
                    classix.fit(X_scaled_float64)
                    labels = classix.labels_
                    
                    # ラベルをint64に変換（dtype互換性確保）
                    labels = np.array(labels, dtype=np.int64)
                    
                    # クラスタ数計算（-1はノイズ）
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    
                    # クラスタが2つ以上ある場合のみ評価
                    if n_clusters >= 2:
                        # ノイズポイントを除外してシルエットスコアを計算
                        mask = labels != -1
                        if mask.sum() > 1 and len(set(labels[mask])) > 1:
                            score = silhouette_score(self.X_scaled[mask], labels[mask])
                            
                            # 目標クラスタ数からの距離を計算(ペナルティ)
                            cluster_penalty = abs(n_clusters - target_clusters) / target_clusters
                            # ノイズ比率も考慮
                            noise_penalty = n_noise / len(labels)
                            adjusted_score = score * (1 - cluster_penalty * 0.5) * (1 - noise_penalty * 0.3)
                            
                            print(f"   radius={radius}, minPts={min_samples}: "
                                  f"クラスタ数={n_clusters}, ノイズ={n_noise}, "
                                  f"スコア={score:.4f}, 調整後={adjusted_score:.4f}")
                            
                            # ノイズが少なく、調整後スコアが高く、クラスタ数が適切なものを選択
                            # v0.7改: クラスタ数範囲を50〜150個に拡大（目標112）
                            # ノイズ比率: 10%以下（厳格化）
                            if (adjusted_score > best_score and 
                                n_noise < len(labels) * 0.10 and 
                                50 <= n_clusters <= 150):
                                best_score = adjusted_score
                                best_classix = classix
                                best_labels = labels
                                best_params = {'radius': radius, 'minPts': min_samples}
                
                except Exception as e:
                    print(f"   ⚠️ radius={radius}, minPts={min_samples}でエラー: {e}")
                    continue
        
        if best_labels is not None:
            n_clusters_final = len(set(best_labels)) - (1 if -1 in best_labels else 0)
            n_noise_final = list(best_labels).count(-1)
            
            print(f"   ✓ 最適パラメータ: radius={best_params['radius']}, "
                  f"minPts={best_params['minPts']} (調整後スコア: {best_score:.4f})")
            print(f"   ✓ クラスタ数: {n_clusters_final}, ノイズ: {n_noise_final}件 ({n_noise_final/len(best_labels)*100:.1f}%)")
            
            # クラスタ分布を表示（上位10件のみ）
            unique, counts = np.unique(best_labels, return_counts=True)
            sorted_clusters = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)
            print(f"   主要クラスタ分布（上位10件）:")
            for i, (cluster_id, count) in enumerate(sorted_clusters[:10]):
                label_name = "ノイズ" if cluster_id == -1 else f"クラスタ {cluster_id}"
                print(f"     {label_name}: {count}件")
            
            self.results['CLASSIX'] = {
                'model': best_classix,
                'labels': best_labels,
                'n_clusters': n_clusters_final,
                'score': best_score,
                'params': best_params
            }
        else:
            print(f"   ⚠️ 適切なCLASSIXパラメータが見つかりませんでした。")
            self.results['CLASSIX'] = None
        
        return best_labels
    
    def _try_classix_bayesian(self, target_clusters, n_calls):
        """ベイズ最適化を使用したCLASSIXパラメータ探索
        
        Note: target_clustersは互換性のために残しているが、最適化では使用しない
        """
        from classix import CLASSIX
        from skopt import forest_minimize  # 並列化に適した手法
        from skopt.space import Real, Integer
        from skopt.utils import use_named_args
        
        # 探索空間の定義（v0.8改: より適切な範囲に絞る）
        space = [
            Real(0.3, 0.5, name='radius'),  # クラスタ数を増やすため範囲を狭める
            Integer(3, 40, name='minPts')  # 下限を3に引き上げ
        ]
        
        # 目的関数（最小化するため、調整後スコアの負値を返す）
        @use_named_args(space)
        def objective(radius, minPts):
            try:
                # データ型をfloat64に明示的に変換（dtype mismatch回避）
                X_scaled_float64 = self.X_scaled.astype(np.float64)
                
                classix = CLASSIX(
                    radius=radius,
                    minPts=minPts,
                    verbose=0,
                    post_alloc=True
                )
                
                classix.fit(X_scaled_float64)
                labels = classix.labels_
                # Windowsでの互換性のためint64に変換
                if isinstance(labels, list):
                    labels = np.array(labels, dtype=np.int64)
                else:
                    labels = labels.astype(np.int64)
                
                # クラスタ数計算
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                # 制約チェック
                if n_clusters < 2:
                    return 10.0  # ペナルティ大
                
                # ノイズポイントを除外してシルエットスコアを計算
                mask = labels != -1
                if mask.sum() <= 1 or len(set(labels[mask])) <= 1:
                    return 10.0
                
                # v0.8改: Silhouette係数とDavis-Bouldin Indexを0.5:0.5で評価
                from sklearn.metrics import davies_bouldin_score
                
                silhouette = silhouette_score(self.X_scaled[mask], labels[mask])
                db_index = davies_bouldin_score(self.X_scaled[mask], labels[mask])
                
                # Silhouette: 高い方が良い (0-1), DB Index: 低い方が良い (0-)
                # DB Indexを0-1に正規化（典型的な範囲0-3を想定）
                db_normalized = max(0, 1.0 - db_index / 3.0)
                
                # クラスタバランススコア（各クラスタの均等性を評価）
                unique_labels, counts = np.unique(labels[mask], return_counts=True)
                if len(counts) > 1:
                    # 標準偏差を使用して不均衡度を計算（0-1に正規化）
                    mean_count = counts.mean()
                    std_count = counts.std()
                    balance_score = max(0, 1.0 - std_count / (mean_count + 1e-6))
                else:
                    balance_score = 0.0
                
                # 複合スコア (0.35:0.35:0.3 = Silhouette:DB:Balance)
                # v0.8改: バランススコアの重みを0.2→0.3に増加
                combined_score = 0.35 * silhouette + 0.35 * db_normalized + 0.3 * balance_score
                
                noise_ratio = n_noise / len(labels)
                
                # クラスタ数ペナルティ（10以上、76以下を強制、50-76が理想範囲）
                # 76 = 自治体総数19 × 4
                if n_clusters < 10:
                    combined_score *= 0.1  # 90%減点（非常に大きなペナルティ）
                elif n_clusters < 50:
                    combined_score *= 0.7  # 30%減点
                elif n_clusters > 76:
                    combined_score *= 0.1  # 90%減点（上限超過も厳しくペナルティ）
                
                # ノイズペナルティ（10%未満は軽微、10%以上は大きなペナルティ）
                if noise_ratio < 0.10:
                    combined_score *= (1.0 - noise_ratio * 0.5)  # 最大5%減点
                else:
                    combined_score *= max(0.01, 1.0 - noise_ratio * 2.0)  # 大幅減点
                
                return -combined_score  # 最小化するため負値
                
            except Exception as e:
                return 10.0  # エラー時は大きなペナルティ
        
        # ベイズ最適化の実行（進捗表示付き）
        print(f"   ベイズ最適化を開始（{n_calls}回の評価）...")
        
        # コールバック関数で進捗表示
        iteration = [0]
        def on_step(res):
            iteration[0] += 1
            if iteration[0] % 10 == 0 or iteration[0] <= 5:
                best_score = -res.fun
                print(f"      評価 {iteration[0]}/{n_calls}: 現在の最良スコア = {best_score:.4f}")
        
        # ベイズ最適化実行（forest_minimizeで並列処理有効）
        cores_msg = "全CPUコア" if config.N_JOBS == -1 else f"{config.N_JOBS}コア"
        print(f"   🚀 Random Forestベースで{cores_msg}並列処理を実行...")
        print(f"   📊 初期ランダム探索: {config.N_INITIAL_POINTS}回(並列)、逐次探索: {n_calls - config.N_INITIAL_POINTS}回")
        result = forest_minimize(
            objective,
            space,
            n_calls=n_calls,
            random_state=config.RANDOM_STATE,
            verbose=False,
            n_initial_points=config.N_INITIAL_POINTS,
            n_jobs=config.N_JOBS,  # 並列処理
            callback=[on_step]
        )
        
        # 最適パラメータで最終モデルを構築
        best_radius = result.x[0]
        best_minPts = result.x[1]
        best_score = -result.fun  # 負値を戻す
        
        print(f"   ✓ 最適パラメータ: radius={best_radius:.3f}, minPts={best_minPts}")
        print(f"   ✓ 調整後スコア: {best_score:.4f}")
        
        # 最適パラメータで最終クラスタリング
        X_scaled_float64 = self.X_scaled.astype(np.float64)
        best_classix = CLASSIX(
            radius=best_radius,
            minPts=best_minPts,
            verbose=0,
            post_alloc=True
        )
        best_classix.fit(X_scaled_float64)
        best_labels = np.array(best_classix.labels_, dtype=np.int64)
        
        n_clusters_final = len(set(best_labels)) - (1 if -1 in best_labels else 0)
        n_noise_final = list(best_labels).count(-1)
        
        print(f"   ✓ クラスタ数: {n_clusters_final}, ノイズ: {n_noise_final}件 ({n_noise_final/len(best_labels)*100:.1f}%)")
        
        # クラスタ分布を表示（上位10件のみ）
        unique, counts = np.unique(best_labels, return_counts=True)
        sorted_clusters = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)
        print(f"   主要クラスタ分布（上位10件）:")
        for i, (cluster_id, count) in enumerate(sorted_clusters[:10]):
            label_name = "ノイズ" if cluster_id == -1 else f"クラスタ {cluster_id}"
            print(f"     {label_name}: {count}件")
        
        self.results['CLASSIX'] = {
            'model': best_classix,
            'labels': best_labels,
            'n_clusters': n_clusters_final,
            'score': best_score,
            'params': {'radius': best_radius, 'minPts': best_minPts}
        }
        
        return best_labels
    
    def get_results(self):
        """すべての結果を返す"""
        return self.results


class AlternativeDimensionalityReduction:
    """代替次元削減手法を提供するクラス"""
    
    def __init__(self, X_scaled):
        """
        Parameters:
        -----------
        X_scaled : array-like
            標準化された特徴量
        """
        self.X_scaled = X_scaled
        self.results = {}
    
    def try_tsne(self, n_components=2, perplexity_range=None):
        """t-SNE（t-distributed Stochastic Neighbor Embedding）"""
        print(f"\n🔴 t-SNE を実行中...")
        
        from sklearn.manifold import TSNE
        
        if perplexity_range is None:
            # データサイズに応じて適切なperplexityを選択
            n_samples = len(self.X_scaled)
            perplexity_range = [10,
                               min(30, n_samples // 4), 
                               min(50, n_samples // 3)]
        
        best_tsne = None
        best_embedding = None
        best_perplexity = perplexity_range[0]
        
        for perplexity in perplexity_range:
            try:
                # scikit-learnのバージョンに応じてパラメータを調整
                tsne_params = {
                    'n_components': n_components,
                    'perplexity': perplexity,
                    'random_state': config.RANDOM_STATE
                }
                
                # バージョン互換性のため、max_iterとn_iterの両方を試す
                try:
                    tsne = TSNE(**tsne_params, n_iter=1000, n_iter_without_progress=300)
                except TypeError:
                    tsne = TSNE(**tsne_params, max_iter=1000, n_iter_without_progress=300)
                
                embedding = tsne.fit_transform(self.X_scaled)
                
                # KL divergenceが利用可能な場合は表示
                if hasattr(tsne, 'kl_divergence_'):
                    print(f"   perplexity={perplexity}: KL divergence = {tsne.kl_divergence_:.4f}")
                    if best_tsne is None or tsne.kl_divergence_ < best_tsne.kl_divergence_:
                        best_tsne = tsne
                        best_embedding = embedding
                        best_perplexity = perplexity
                else:
                    print(f"   perplexity={perplexity}: 完了")
                    if best_tsne is None:
                        best_tsne = tsne
                        best_embedding = embedding
                        best_perplexity = perplexity
            
            except Exception as e:
                print(f"   ⚠️ perplexity={perplexity}でエラー: {e}")
                continue
        
        if best_embedding is not None:
            print(f"   ✓ 最適perplexity: {best_perplexity}")
            
            self.results['t-SNE'] = {
                'model': best_tsne,
                'embedding': best_embedding,
                'perplexity': best_perplexity
            }
        else:
            print(f"   ⚠️ t-SNEの実行に失敗しました。")
            self.results['t-SNE'] = None
        
        return best_embedding
    
    def try_umap(self, n_components=2, n_neighbors_range=None, create_3d=False):
        """UMAP（Uniform Manifold Approximation and Projection）
        
        Parameters:
        -----------
        n_components : int
            次元数（デフォルト: 2）
        n_neighbors_range : list, optional
            探索するn_neighborsの範囲
        create_3d : bool
            3次元埋め込みも作成するか（デフォルト: False）
        """
        print(f"\n🟠 UMAP を実行中...")
        
        try:
            import umap
        except ImportError:
            print(f"   ⚠️ UMAPがインストールされていません。")
            print(f"   'pip install umap-learn' でインストールしてください。")
            self.results['UMAP'] = None
            return None
        
        if n_neighbors_range is None:
            n_samples = len(self.X_scaled)
            n_neighbors_range = [min(15, n_samples // 10),
                                 min(30, n_samples // 5)]
        
        best_umap = None
        best_embedding = None
        best_n_neighbors = n_neighbors_range[0]
        
        for n_neighbors in n_neighbors_range:
            try:
                umap_model = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    random_state=config.RANDOM_STATE,
                    min_dist=0.1
                )
                
                embedding = umap_model.fit_transform(self.X_scaled)
                
                print(f"   n_neighbors={n_neighbors}: 完了")
                
                if best_umap is None:
                    best_umap = umap_model
                    best_embedding = embedding
                    best_n_neighbors = n_neighbors
            
            except Exception as e:
                print(f"   ⚠️ n_neighbors={n_neighbors}でエラー: {e}")
                continue
        
        # 3次元埋め込みの作成
        embedding_3d = None
        if best_embedding is not None and create_3d:
            print(f"\n🟠 UMAP 3次元を実行中...")
            try:
                umap_model_3d = umap.UMAP(
                    n_components=3,
                    n_neighbors=best_n_neighbors,
                    random_state=config.RANDOM_STATE,
                    min_dist=0.1
                )
                embedding_3d = umap_model_3d.fit_transform(self.X_scaled)
                print(f"   ✓ 3次元埋め込み作成完了")
            except Exception as e:
                print(f"   ⚠️ 3次元埋め込みでエラー: {e}")
        
        if best_embedding is not None:
            print(f"   ✓ 最適n_neighbors: {best_n_neighbors}")
            
            self.results['UMAP'] = {
                'model': best_umap,
                'embedding': best_embedding,
                'embedding_3d': embedding_3d,
                'n_neighbors': best_n_neighbors
            }
        else:
            print(f"   ⚠️ UMAPの実行に失敗しました。")
            self.results['UMAP'] = None
        
        return best_embedding
    
    def get_results(self):
        """すべての結果を返す"""
        return self.results
