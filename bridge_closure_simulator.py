"""
Bridge Closure Impact Simulator
Bridge Importance Scoring MVP v1.2

橋梁閉鎖シナリオをシミュレーションし、
交通ネットワークへの影響を定量的に評価する。

評価指標:
- 平均最短経路長の変化
- 到達不能になるノード数
- 使用できなくなるバス停の数
"""

import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import logging
from pathlib import Path
from tqdm import tqdm
import pickle

logger = logging.getLogger(__name__)


class BridgeClosureSimulator:
    """橋梁閉鎖シミュレータクラス"""
    
    def __init__(self, G: nx.Graph):
        """
        Args:
            G: 異種グラフ（橋梁、道路、バス停を含む）
        """
        self.G_original = G.copy()  # オリジナルを保持
        self.baseline_metrics = None
        
        # ノードタイプごとに分類
        self.bridge_nodes = self._extract_nodes_by_type('bridge')
        self.street_nodes = self._extract_nodes_by_type('street')
        self.bus_stop_nodes = self._extract_nodes_by_type('bus_stop')
        
        logger.info(f"Graph loaded: {self.G_original.number_of_nodes()} nodes, {self.G_original.number_of_edges()} edges")
        logger.info(f"  Bridge nodes: {len(self.bridge_nodes)}")
        logger.info(f"  Street nodes: {len(self.street_nodes)}")
        logger.info(f"  Bus stop nodes: {len(self.bus_stop_nodes)}")
    
    def _extract_nodes_by_type(self, node_type: str) -> List:
        """
        指定タイプのノードを抽出
        
        Args:
            node_type: ノードタイプ（'bridge', 'street', 'bus_stop'など）
        
        Returns:
            該当ノードのリスト
        """
        nodes = [
            n for n, data in self.G_original.nodes(data=True)
            if data.get('node_type') == node_type
        ]
        return nodes
    
    def compute_baseline_metrics(self, sample_size: int = 500) -> Dict:
        """
        ベースラインメトリクスを計算（橋梁閉鎖前の基準値）
        
        Args:
            sample_size: 最短経路計算のサンプルサイズ（全ノードペアは計算負荷が高いため）
        
        Returns:
            ベースラインメトリクスの辞書
        """
        logger.info("Computing baseline metrics (before any bridge closure)...")
        
        G = self.G_original
        
        # 連結性チェック
        if not nx.is_connected(G):
            logger.warning("Original graph is not fully connected")
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            logger.info(f"Using largest connected component: {G.number_of_nodes()} nodes")
        
        # 最大連結成分内のノード
        connected_nodes = list(G.nodes())
        
        # 平均最短経路長（サンプリング）
        avg_shortest_path = self._compute_average_shortest_path(
            G, 
            sample_nodes=connected_nodes,
            sample_size=min(sample_size, len(connected_nodes))
        )
        
        # 到達可能なバス停（全バス停が基準）
        accessible_bus_stops = len([
            bs for bs in self.bus_stop_nodes
            if bs in G.nodes()
        ])
        
        # 連結成分数
        num_components = nx.number_connected_components(self.G_original)
        
        baseline = {
            'avg_shortest_path': avg_shortest_path,
            'num_connected_nodes': len(connected_nodes),
            'num_components': num_components,
            'accessible_bus_stops': accessible_bus_stops,
            'total_nodes': self.G_original.number_of_nodes(),
            'total_edges': self.G_original.number_of_edges(),
        }
        
        self.baseline_metrics = baseline
        
        logger.info("Baseline metrics computed:")
        logger.info(f"  Average shortest path: {avg_shortest_path:.2f}")
        logger.info(f"  Connected nodes: {len(connected_nodes)}")
        logger.info(f"  Connected components: {num_components}")
        logger.info(f"  Accessible bus stops: {accessible_bus_stops}")
        
        return baseline
    
    def _compute_average_shortest_path(
        self, 
        G: nx.Graph, 
        sample_nodes: List = None,
        sample_size: int = 500
    ) -> float:
        """
        平均最短経路長を計算（サンプリングベース）
        
        Args:
            G: グラフ
            sample_nodes: サンプリング対象ノード（Noneの場合は全ノード）
            sample_size: サンプルサイズ
        
        Returns:
            平均最短経路長
        """
        if sample_nodes is None:
            sample_nodes = list(G.nodes())
        
        if len(sample_nodes) == 0:
            return 0.0
        
        # サンプリング
        if len(sample_nodes) > sample_size:
            sample = np.random.choice(sample_nodes, size=sample_size, replace=False)
        else:
            sample = sample_nodes
        
        path_lengths = []
        
        for source in sample:
            # 各sourceからの最短経路長を計算
            try:
                lengths = nx.single_source_shortest_path_length(G, source)
                # source自身を除く
                lengths = {t: l for t, l in lengths.items() if t != source}
                if lengths:
                    path_lengths.extend(lengths.values())
            except nx.NetworkXError:
                # 孤立ノードの場合はスキップ
                continue
        
        if len(path_lengths) == 0:
            return 0.0
        
        return np.mean(path_lengths)
    
    def simulate_bridge_closure(
        self, 
        bridge_id: str,
        sample_size: int = 500
    ) -> Dict:
        """
        特定の橋梁を閉鎖してシミュレーション
        
        Args:
            bridge_id: 橋梁ID
            sample_size: 最短経路計算のサンプルサイズ
        
        Returns:
            シミュレーション結果の辞書
        """
        # 橋梁ノードを探す
        bridge_node = None
        for node, data in self.G_original.nodes(data=True):
            if data.get('bridge_id') == bridge_id:
                bridge_node = node
                break
        
        if bridge_node is None:
            logger.warning(f"Bridge {bridge_id} not found in graph")
            return {
                'bridge_id': bridge_id,
                'error': 'Bridge not found in graph',
            }
        
        # グラフをコピーして橋梁を削除
        G_closed = self.G_original.copy()
        
        # 橋梁ノードとその関連エッジを削除
        if bridge_node in G_closed:
            G_closed.remove_node(bridge_node)
        
        # 閉鎖後のメトリクスを計算
        if not nx.is_connected(G_closed):
            # 最大連結成分を取得
            components = list(nx.connected_components(G_closed))
            largest_cc = max(components, key=len)
            G_closed_main = G_closed.subgraph(largest_cc).copy()
        else:
            G_closed_main = G_closed
        
        # 到達可能ノード
        connected_nodes = list(G_closed_main.nodes())
        
        # 平均最短経路長
        avg_shortest_path = self._compute_average_shortest_path(
            G_closed_main,
            sample_nodes=connected_nodes,
            sample_size=min(sample_size, len(connected_nodes))
        )
        
        # 到達可能なバス停
        accessible_bus_stops = len([
            bs for bs in self.bus_stop_nodes
            if bs in G_closed_main.nodes()
        ])
        
        # 連結成分数
        num_components = nx.number_connected_components(G_closed)
        
        # 影響度の計算（ベースラインとの差分）
        if self.baseline_metrics is None:
            logger.warning("Baseline metrics not computed. Run compute_baseline_metrics() first.")
            impact_metrics = {}
        else:
            baseline = self.baseline_metrics
            
            # 変化量（絶対値）
            delta_avg_path = avg_shortest_path - baseline['avg_shortest_path']
            delta_connected_nodes = len(connected_nodes) - baseline['num_connected_nodes']
            delta_bus_stops = accessible_bus_stops - baseline['accessible_bus_stops']
            delta_components = num_components - baseline['num_components']
            
            # 変化率（%）
            pct_path_increase = (delta_avg_path / baseline['avg_shortest_path'] * 100) if baseline['avg_shortest_path'] > 0 else 0
            pct_nodes_lost = (abs(delta_connected_nodes) / baseline['num_connected_nodes'] * 100) if baseline['num_connected_nodes'] > 0 else 0
            pct_bus_stops_lost = (abs(delta_bus_stops) / baseline['accessible_bus_stops'] * 100) if baseline['accessible_bus_stops'] > 0 else 0
            
            impact_metrics = {
                'delta_avg_shortest_path': delta_avg_path,
                'delta_connected_nodes': delta_connected_nodes,
                'delta_accessible_bus_stops': delta_bus_stops,
                'delta_num_components': delta_components,
                'pct_path_increase': pct_path_increase,
                'pct_nodes_lost': pct_nodes_lost,
                'pct_bus_stops_lost': pct_bus_stops_lost,
            }
        
        result = {
            'bridge_id': bridge_id,
            'bridge_node': bridge_node,
            'avg_shortest_path_after': avg_shortest_path,
            'num_connected_nodes_after': len(connected_nodes),
            'num_components_after': num_components,
            'accessible_bus_stops_after': accessible_bus_stops,
            **impact_metrics,
        }
        
        return result
    
    def simulate_multiple_bridges(
        self,
        bridge_ids: List[str],
        sample_size: int = 500,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        複数の橋梁について閉鎖シミュレーションを実行
        
        Args:
            bridge_ids: 橋梁IDのリスト
            sample_size: 最短経路計算のサンプルサイズ
            show_progress: プログレスバーを表示するか
        
        Returns:
            シミュレーション結果のDataFrame
        """
        logger.info(f"Starting closure simulation for {len(bridge_ids)} bridges...")
        
        # ベースラインが未計算なら計算
        if self.baseline_metrics is None:
            self.compute_baseline_metrics(sample_size=sample_size)
        
        results = []
        
        iterator = tqdm(bridge_ids, desc="Simulating bridge closures") if show_progress else bridge_ids
        
        for bridge_id in iterator:
            result = self.simulate_bridge_closure(bridge_id, sample_size=sample_size)
            results.append(result)
        
        df = pd.DataFrame(results)
        
        logger.info(f"Simulation completed for {len(results)} bridges")
        
        return df
    
    def generate_impact_report(
        self,
        results_df: pd.DataFrame,
        output_path: Path
    ):
        """
        影響度レポートを生成（Markdown形式）
        
        Args:
            results_df: シミュレーション結果のDataFrame
            output_path: 出力パス
        """
        logger.info(f"Generating impact report to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Bridge Closure Impact Simulation Report\n\n")
            f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # ベースライン情報
            f.write("## Baseline Metrics (Before Any Closure)\n\n")
            if self.baseline_metrics:
                baseline = self.baseline_metrics
                f.write(f"- **Average Shortest Path Length:** {baseline['avg_shortest_path']:.2f}\n")
                f.write(f"- **Connected Nodes:** {baseline['num_connected_nodes']:,}\n")
                f.write(f"- **Connected Components:** {baseline['num_components']}\n")
                f.write(f"- **Accessible Bus Stops:** {baseline['accessible_bus_stops']}\n")
                f.write(f"- **Total Nodes:** {baseline['total_nodes']:,}\n")
                f.write(f"- **Total Edges:** {baseline['total_edges']:,}\n\n")
            
            # サマリー統計
            f.write("## Simulation Summary\n\n")
            f.write(f"- **Number of Bridges Simulated:** {len(results_df)}\n")
            f.write(f"- **Average Path Length Increase:** {results_df['delta_avg_shortest_path'].mean():.2f}±{results_df['delta_avg_shortest_path'].std():.2f}\n")
            f.write(f"- **Average Nodes Lost:** {abs(results_df['delta_connected_nodes'].mean()):.1f}±{results_df['delta_connected_nodes'].std():.1f}\n")
            f.write(f"- **Average Bus Stops Lost:** {abs(results_df['delta_accessible_bus_stops'].mean()):.1f}±{results_df['delta_accessible_bus_stops'].std():.1f}\n\n")
            
            # トップ10影響度の高い橋梁
            f.write("## Top 10 Bridges by Closure Impact\n\n")
            
            # 経路長増加でソート
            f.write("### By Average Shortest Path Increase\n\n")
            top_path = results_df.nlargest(10, 'delta_avg_shortest_path')
            f.write("| Rank | Bridge ID | Δ Avg Path | % Increase | Nodes Lost | Bus Stops Lost |\n")
            f.write("|------|-----------|------------|------------|------------|----------------|\n")
            for rank, (idx, row) in enumerate(top_path.iterrows(), 1):
                f.write(f"| {rank} | {row['bridge_id']} | +{row['delta_avg_shortest_path']:.2f} | {row['pct_path_increase']:.1f}% | {abs(row['delta_connected_nodes'])} | {abs(row['delta_accessible_bus_stops'])} |\n")
            f.write("\n")
            
            # ノード損失でソート
            f.write("### By Nodes Lost (Unreachable)\n\n")
            top_nodes = results_df.nsmallest(10, 'delta_connected_nodes')
            f.write("| Rank | Bridge ID | Nodes Lost | % Lost | Δ Avg Path | Bus Stops Lost |\n")
            f.write("|------|-----------|------------|--------|------------|----------------|\n")
            for rank, (idx, row) in enumerate(top_nodes.iterrows(), 1):
                f.write(f"| {rank} | {row['bridge_id']} | {abs(row['delta_connected_nodes'])} | {row['pct_nodes_lost']:.2f}% | +{row['delta_avg_shortest_path']:.2f} | {abs(row['delta_accessible_bus_stops'])} |\n")
            f.write("\n")
            
            # バス停損失でソート
            f.write("### By Bus Stops Lost (Inaccessible)\n\n")
            top_bus_stops = results_df.nsmallest(10, 'delta_accessible_bus_stops')
            f.write("| Rank | Bridge ID | Bus Stops Lost | % Lost | Δ Avg Path | Nodes Lost |\n")
            f.write("|------|-----------|----------------|--------|------------|------------|\n")
            for rank, (idx, row) in enumerate(top_bus_stops.iterrows(), 1):
                f.write(f"| {rank} | {row['bridge_id']} | {abs(row['delta_accessible_bus_stops'])} | {row['pct_bus_stops_lost']:.2f}% | +{row['delta_avg_shortest_path']:.2f} | {abs(row['delta_connected_nodes'])} |\n")
            f.write("\n")
            
            # 影響度分布
            f.write("## Impact Distribution\n\n")
            
            # カテゴリ分類（影響度）
            high_impact = results_df[results_df['pct_path_increase'] > 10].shape[0]
            medium_impact = results_df[(results_df['pct_path_increase'] > 5) & (results_df['pct_path_increase'] <= 10)].shape[0]
            low_impact = results_df[results_df['pct_path_increase'] <= 5].shape[0]
            
            f.write(f"- **High Impact** (>10% path increase): {high_impact} bridges ({high_impact/len(results_df)*100:.1f}%)\n")
            f.write(f"- **Medium Impact** (5-10% path increase): {medium_impact} bridges ({medium_impact/len(results_df)*100:.1f}%)\n")
            f.write(f"- **Low Impact** (<5% path increase): {low_impact} bridges ({low_impact/len(results_df)*100:.1f}%)\n\n")
            
            # フルリスト
            f.write("## Full Simulation Results\n\n")
            f.write("| Bridge ID | Δ Avg Path | % Path ↑ | Nodes Lost | % Nodes ↓ | Bus Stops Lost | % Bus Stops ↓ |\n")
            f.write("|-----------|------------|----------|------------|-----------|----------------|---------------|\n")
            
            # 影響度でソート
            results_sorted = results_df.sort_values('delta_avg_shortest_path', ascending=False)
            for idx, row in results_sorted.iterrows():
                f.write(f"| {row['bridge_id']} | +{row['delta_avg_shortest_path']:.2f} | {row['pct_path_increase']:.1f}% | {abs(row['delta_connected_nodes'])} | {row['pct_nodes_lost']:.2f}% | {abs(row['delta_accessible_bus_stops'])} | {row['pct_bus_stops_lost']:.2f}% |\n")
        
        logger.info(f"Impact report saved to {output_path}")
