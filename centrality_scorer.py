"""
媒介中心性計算とスコアリングモジュール
Bridge Importance Scoring MVP

NetworkXを使った媒介中心性の計算と橋梁重要度スコアの生成
"""

import networkx as nx
import pandas as pd
import geopandas as gpd
from typing import Dict, Tuple, Optional
import logging
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BridgeImportanceScorer:
    """橋梁重要度スコアリングクラス"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 設定辞書
        """
        self.config = config
        self.centrality_config = config.get('centrality', {})
        self.scoring_config = config.get('scoring', {})
        
    def compute_betweenness_centrality(
        self,
        G: nx.Graph,
        bridge_nodes: list
    ) -> Dict[str, float]:
        """
        橋梁ノードの媒介中心性を計算
        
        Args:
            G: 異種グラフ
            bridge_nodes: 橋梁ノードのリスト
        
        Returns:
            {bridge_id: betweenness_score}の辞書
        """
        logger.info(f"Computing betweenness centrality for {len(bridge_nodes)} bridges...")
        logger.info(f"Graph size: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # 連結成分の確認
        if not nx.is_connected(G):
            logger.warning("Graph is not connected. Using largest connected component.")
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            logger.info(f"Largest component: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            
            # 橋梁ノードを連結成分内に限定
            bridge_nodes = [n for n in bridge_nodes if n in G.nodes]
            logger.info(f"Bridges in largest component: {len(bridge_nodes)}")
        
        # エッジに重みがない場合はデフォルト重みを追加
        for u, v in G.edges():
            if 'weight' not in G[u][v]:
                G[u][v]['weight'] = 1.0
        
        # 媒介中心性の計算
        normalized = self.centrality_config.get('normalized', True)
        endpoints = self.centrality_config.get('endpoints', False)
        k = self.centrality_config.get('k', None)
        
        try:
            if k is not None and k < G.number_of_nodes():
                logger.info(f"Using approximation with k={k} samples")
                bc = nx.betweenness_centrality(
                    G,
                    k=k,
                    normalized=normalized,
                    endpoints=endpoints,
                    weight='weight'
                )
            else:
                logger.info("Computing exact betweenness centrality...")
                bc = nx.betweenness_centrality(
                    G,
                    normalized=normalized,
                    endpoints=endpoints,
                    weight='weight'
                )
            
            # 橋梁ノードのみ抽出
            bridge_bc = {n: bc.get(n, 0.0) for n in bridge_nodes}
            
            logger.info(f"Betweenness centrality computed. Range: [{min(bridge_bc.values()):.6f}, {max(bridge_bc.values()):.6f}]")
            
            return bridge_bc
            
        except Exception as e:
            logger.error(f"Error computing betweenness centrality: {e}")
            logger.warning("Returning zero centrality for all bridges")
            return {n: 0.0 for n in bridge_nodes}
    
    def compute_alternative_metrics(
        self,
        G: nx.Graph,
        bridge_nodes: list
    ) -> Dict[str, Dict[str, float]]:
        """
        媒介中心性以外の補助的な指標を計算
        
        Args:
            G: 異種グラフ
            bridge_nodes: 橋梁ノードのリスト
        
        Returns:
            {bridge_id: {metric_name: value}}の辞書
        """
        logger.info("Computing alternative metrics...")
        
        metrics = {}
        
        for bridge_id in tqdm(bridge_nodes, desc="Computing metrics"):
            if bridge_id not in G.nodes:
                metrics[bridge_id] = {
                    'degree': 0,
                    'closeness': 0.0,
                    'clustering': 0.0
                }
                continue
            
            # 次数（接続数）
            degree = G.degree(bridge_id)
            
            # 近接中心性（計算コストが高いためスキップ可能）
            try:
                # closeness = nx.closeness_centrality(G, bridge_id)
                closeness = 0.0  # 大規模グラフではスキップ
            except:
                closeness = 0.0
            
            # クラスタリング係数
            try:
                clustering = nx.clustering(G, bridge_id)
            except:
                clustering = 0.0
            
            metrics[bridge_id] = {
                'degree': degree,
                'closeness': closeness,
                'clustering': clustering
            }
        
        return metrics
    
    def compute_feature_counts(
        self,
        G: nx.Graph,
        bridge_nodes: list
    ) -> pd.DataFrame:
        """
        各橋梁に接続された地物（建物、バス停など）の数を計算
        
        Args:
            G: 異種グラフ
            bridge_nodes: 橋梁ノードのリスト
        
        Returns:
            特徴量カウントのDataFrame
        """
        logger.info("Computing feature counts for bridges...")
        
        feature_counts = []
        
        for bridge_id in bridge_nodes:
            if bridge_id not in G.nodes:
                feature_counts.append({
                    'bridge_id': bridge_id,
                    'num_buildings': 0,
                    'num_public_facilities': 0,
                    'num_hospitals': 0,
                    'num_schools': 0,
                    'num_bus_stops': 0,
                    'num_street_connections': 0
                })
                continue
            
            neighbors = list(G.neighbors(bridge_id))
            
            num_buildings = 0
            num_public = 0
            num_hospitals = 0
            num_schools = 0
            num_bus_stops = 0
            num_streets = 0
            
            for neighbor in neighbors:
                node_data = G.nodes[neighbor]
                node_type = node_data.get('node_type', '')
                
                if node_type == 'building':
                    num_buildings += 1
                    category = node_data.get('category', '')
                    if category in ['public', 'emergency']:
                        num_public += 1
                    if category == 'hospital':
                        num_hospitals += 1
                    if category == 'school':
                        num_schools += 1
                elif node_type == 'bus_stop':
                    num_bus_stops += 1
                elif node_type == 'street':
                    num_streets += 1
            
            feature_counts.append({
                'bridge_id': bridge_id,
                'num_buildings': num_buildings,
                'num_public_facilities': num_public,
                'num_hospitals': num_hospitals,
                'num_schools': num_schools,
                'num_bus_stops': num_bus_stops,
                'num_street_connections': num_streets
            })
        
        return pd.DataFrame(feature_counts)
    
    def compute_importance_scores(
        self,
        bridges: gpd.GeoDataFrame,
        betweenness: Dict[str, float],
        feature_counts: pd.DataFrame,
        alternative_metrics: Optional[Dict] = None
    ) -> gpd.GeoDataFrame:
        """
        総合的な重要度スコアを計算
        
        Args:
            bridges: 橋梁データ
            betweenness: 媒介中心性スコア
            feature_counts: 地物カウント
            alternative_metrics: その他の指標
        
        Returns:
            スコア付き橋梁データ
        """
        logger.info("Computing importance scores...")
        
        # 媒介中心性を追加
        bridges = bridges.copy()
        bridges['betweenness'] = bridges['bridge_id'].map(betweenness).fillna(0.0)
        
        # 地物カウントをマージ
        bridges = bridges.merge(
            feature_counts,
            on='bridge_id',
            how='left'
        )
        
        # 0-100スケールの基本スコア（媒介中心性ベース）
        b = bridges['betweenness']
        b_min, b_max = b.min(), b.max()
        
        if b_max > b_min:
            bridges['importance_score_base'] = 100 * (b - b_min) / (b_max - b_min)
        else:
            bridges['importance_score_base'] = 50.0
        
        # 重み付きスコア（設定に基づく）
        weights = self.scoring_config.get('weights', {})
        w_betweenness = weights.get('betweenness', 0.6)
        w_public = weights.get('public_access', 0.2)
        w_traffic = weights.get('traffic_volume', 0.2)
        
        # 公共施設アクセススコア（正規化）
        public_score = bridges['num_public_facilities'] + bridges['num_hospitals'] * 2 + bridges['num_schools'] * 1.5
        public_max = public_score.max()
        if public_max > 0:
            public_score = 100 * public_score / public_max
        else:
            public_score = 0.0
        
        # 交通量代理スコア（次数とバス停）
        traffic_score = bridges['num_street_connections'] + bridges['num_bus_stops'] * 3
        traffic_max = traffic_score.max()
        if traffic_max > 0:
            traffic_score = 100 * traffic_score / traffic_max
        else:
            traffic_score = 0.0
        
        # 重み付き総合スコア
        bridges['public_access_score'] = public_score
        bridges['traffic_proxy_score'] = traffic_score
        
        bridges['importance_score'] = (
            w_betweenness * bridges['importance_score_base'] +
            w_public * public_score +
            w_traffic * traffic_score
        )
        
        # ランク付け
        bridges['importance_rank'] = bridges['importance_score'].rank(
            ascending=False,
            method='dense'
        ).astype(int)
        
        # カテゴリ分類
        bridges['importance_category'] = self._categorize_importance(
            bridges['importance_score']
        )
        
        logger.info(f"Importance scores computed. Range: [{bridges['importance_score'].min():.2f}, {bridges['importance_score'].max():.2f}]")
        
        # スコア分布の統計
        self._log_score_statistics(bridges)
        
        return bridges
    
    def _categorize_importance(self, scores: pd.Series) -> pd.Series:
        """スコアに基づく重要度カテゴリ分類"""
        thresholds = self.config.get('narrative', {}).get('thresholds', {})
        critical = thresholds.get('critical', 90)
        high = thresholds.get('high', 70)
        medium = thresholds.get('medium', 50)
        low = thresholds.get('low', 30)
        
        categories = []
        for score in scores:
            if score >= critical:
                categories.append('critical')
            elif score >= high:
                categories.append('high')
            elif score >= medium:
                categories.append('medium')
            elif score >= low:
                categories.append('low')
            else:
                categories.append('very_low')
        
        return pd.Series(categories, index=scores.index)
    
    def _log_score_statistics(self, bridges: gpd.GeoDataFrame):
        """スコア統計をログ出力"""
        logger.info("\n=== Bridge Importance Score Statistics ===")
        logger.info(f"Total bridges: {len(bridges)}")
        logger.info(f"Score range: [{bridges['importance_score'].min():.2f}, {bridges['importance_score'].max():.2f}]")
        logger.info(f"Mean score: {bridges['importance_score'].mean():.2f}")
        logger.info(f"Median score: {bridges['importance_score'].median():.2f}")
        logger.info(f"Std dev: {bridges['importance_score'].std():.2f}")
        
        logger.info("\nCategory distribution:")
        category_counts = bridges['importance_category'].value_counts()
        for cat, count in category_counts.items():
            logger.info(f"  {cat}: {count} ({100*count/len(bridges):.1f}%)")
        
        logger.info("\nTop 10 bridges:")
        top10 = bridges.nlargest(10, 'importance_score')[
            ['bridge_id', 'importance_score', 'betweenness', 'importance_rank']
        ]
        for idx, row in top10.iterrows():
            logger.info(f"  Rank {row['importance_rank']}: {row['bridge_id']} (Score: {row['importance_score']:.2f}, BC: {row['betweenness']:.6f})")


def score_bridge_importance(
    bridges: gpd.GeoDataFrame,
    G: nx.Graph,
    config: Dict
) -> gpd.GeoDataFrame:
    """
    橋梁重要度スコアリングの便利関数
    
    Args:
        bridges: 橋梁データ
        G: 異種グラフ
        config: 設定辞書
    
    Returns:
        スコア付き橋梁データ
    """
    scorer = BridgeImportanceScorer(config)
    
    # 橋梁ノードのリスト
    bridge_nodes = bridges['bridge_id'].tolist()
    
    # 媒介中心性の計算
    betweenness = scorer.compute_betweenness_centrality(G, bridge_nodes)
    
    # 補助的な指標
    alternative_metrics = scorer.compute_alternative_metrics(G, bridge_nodes)
    
    # 地物カウント
    feature_counts = scorer.compute_feature_counts(G, bridge_nodes)
    
    # 総合スコア
    scored_bridges = scorer.compute_importance_scores(
        bridges,
        betweenness,
        feature_counts,
        alternative_metrics
    )
    
    return scored_bridges
