"""
既存グラフから媒介中心性を再計算するスクリプト
"""
import argparse
import pickle
import yaml
import logging
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from centrality_scorer import BridgeImportanceScorer
from narrative_generator import BridgeNarrativeGenerator
from visualization import visualize_results

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _normalize_series_to_100(values: pd.Series) -> pd.Series:
    """0-100へ正規化（定数列は0）"""
    v_min = values.min()
    v_max = values.max()
    if pd.isna(v_min) or pd.isna(v_max) or np.isclose(v_min, v_max):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - v_min) / (v_max - v_min) * 100.0


def compute_closure_indicators_for_lcc_bridges(
    G: nx.Graph,
    bridges: gpd.GeoDataFrame,
    topological_k_hops: int = 2
) -> pd.DataFrame:
    """
    最大連結成分（LCC）内の橋梁ノードについて、閉鎖時の影響指標を計算する。

    指標:
    - degree: 閉鎖前の次数
    - num_neighbors: 隣接ノード数
    - component_increase: 閉鎖後に増加する連結成分数
    - disconnected_nodes: 閉鎖で主成分から分断されるノード数
    - k_hop_nodes: 橋梁周辺k-hopノード数（局所影響の近似）
    - indirect_damage_score: 間接被害スコア（0-100）
    """
    logger.info("\nSTEP 5: Computing closure indicators for bridges in largest connected component...")

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G_cc = G.subgraph(largest_cc).copy()
        logger.info(
            f"Graph is not connected. Using largest connected component: "
            f"{G_cc.number_of_nodes()} nodes, {G_cc.number_of_edges()} edges"
        )
    else:
        G_cc = G

    bridge_nodes_lcc = [
        n for n, data in G_cc.nodes(data=True)
        if data.get('node_type') == 'bridge' and str(n).startswith('BR_')
    ]
    bridge_nodes_lcc_set = set(bridge_nodes_lcc)
    logger.info(f"Bridge nodes in LCC: {len(bridge_nodes_lcc)}")

    if len(bridge_nodes_lcc) == 0:
        logger.warning("No bridge nodes found in largest connected component.")
        return pd.DataFrame(columns=[
            'bridge_id', 'degree', 'num_neighbors', 'component_increase',
            'disconnected_nodes', 'k_hop_nodes', 'indirect_damage_score'
        ])

    results = []
    n_nodes_lcc = G_cc.number_of_nodes()

    for i, bridge_node in enumerate(bridge_nodes_lcc, 1):
        degree = G_cc.degree(bridge_node)
        neighbors = list(G_cc.neighbors(bridge_node))

        ego = nx.ego_graph(G_cc, bridge_node, radius=topological_k_hops)
        k_hop_nodes = max(0, ego.number_of_nodes() - 1)

        G_closed = G_cc.copy()
        G_closed.remove_node(bridge_node)

        components_after = list(nx.connected_components(G_closed))
        component_increase = max(0, len(components_after) - 1)
        largest_after = max((len(c) for c in components_after), default=0)
        disconnected_nodes = max(0, (n_nodes_lcc - 1) - largest_after)

        results.append({
            'bridge_id': bridge_node,
            'degree': degree,
            'num_neighbors': len(neighbors),
            'component_increase': component_increase,
            'disconnected_nodes': disconnected_nodes,
            'k_hop_nodes': k_hop_nodes,
        })

        if i % 50 == 0 or i == len(bridge_nodes_lcc):
            logger.info(f"  Closure indicator progress: {i}/{len(bridge_nodes_lcc)}")

    closure_df = pd.DataFrame(results)

    degree_norm = _normalize_series_to_100(closure_df['degree'])
    comp_norm = _normalize_series_to_100(closure_df['component_increase'])
    disc_norm = _normalize_series_to_100(closure_df['disconnected_nodes'])
    khop_norm = _normalize_series_to_100(closure_df['k_hop_nodes'])

    closure_df['indirect_damage_score'] = (
        0.35 * disc_norm +
        0.30 * comp_norm +
        0.20 * degree_norm +
        0.15 * khop_norm
    )

    # bridges側に存在しないIDを除く（データ整合性担保）
    bridge_ids_in_data = set(bridges['bridge_id'].astype(str)) if 'bridge_id' in bridges.columns else set()
    if bridge_ids_in_data:
        before = len(closure_df)
        closure_df = closure_df[closure_df['bridge_id'].astype(str).isin(bridge_ids_in_data)].copy()
        logger.info(f"Closure indicators aligned to bridge table: {len(closure_df)} (from {before})")

    return closure_df.sort_values('indirect_damage_score', ascending=False).reset_index(drop=True)


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="Recompute bridge centrality and v1.3 closure indicators")
    parser.add_argument(
        '--skip-closure-indicators',
        action='store_true',
        help='Skip v1.3 closure-indicator computation for bridge closures'
    )
    parser.add_argument(
        '--k-hop',
        type=int,
        default=2,
        help='Topological hop radius for local closure neighborhood feature (default: 2)'
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Bridge Centrality Recomputation (with Sampling)")
    logger.info("=" * 80)
    
    # 設定ファイル読み込み
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(config['data']['output_dir'])
    
    # 1. 既存グラフの読み込み
    logger.info("\nSTEP 1: Loading existing heterogeneous graph...")
    graph_path = output_dir / 'heterogeneous_graph.pkl'
    
    if not graph_path.exists():
        logger.error(f"Graph file not found: {graph_path}")
        logger.error("Please run main.py first to build the graph.")
        return
    
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    
    logger.info(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # 2. 橋梁データの読み込み
    logger.info("\nSTEP 2: Loading bridge data...")
    
    # 元データから読み込み直す（地物カウントなどが含まれている）
    from data_loader import load_all_data
    bridges, rivers, coasts, boundary = load_all_data(config)
    logger.info(f"Loaded {len(bridges)} bridges from original data")
    
    # 3. 媒介中心性の再計算（サンプリング設定適用）
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Recomputing Betweenness Centrality with Sampling")
    logger.info("=" * 60)
    
    scorer = BridgeImportanceScorer(config)
    
    # 橋梁IDリストの取得
    bridge_nodes = bridges['bridge_id'].tolist()
    
    # 媒介中心性計算
    betweenness_dict = scorer.compute_betweenness_centrality(G, bridge_nodes)
    
    # 結果をDataFrameに追加
    bridges['betweenness'] = bridges['bridge_id'].map(betweenness_dict)
    
    logger.info(f"Betweenness statistics:")
    logger.info(f"  Min: {bridges['betweenness'].min():.6f}")
    logger.info(f"  Max: {bridges['betweenness'].max():.6f}")
    logger.info(f"  Mean: {bridges['betweenness'].mean():.6f}")
    
    # 4. 特徴量カウント計算
    logger.info("\nSTEP 4: Computing feature counts...")
    feature_counts = scorer.compute_feature_counts(G, bridge_nodes)
    
    # 5. スコアリング
    logger.info("\nSTEP 5: Computing importance scores...")
    bridges = scorer.compute_importance_scores(bridges, betweenness_dict, feature_counts)
    
    logger.info(f"Score statistics:")
    logger.info(f"  Min: {bridges['importance_score'].min():.2f}")
    logger.info(f"  Max: {bridges['importance_score'].max():.2f}")
    logger.info(f"  Mean: {bridges['importance_score'].mean():.2f}")
    logger.info(f"  Median: {bridges['importance_score'].median():.2f}")
    
    # 6. v1.3 閉鎖指標の計算（LCC橋梁対象）
    closure_df = None
    if args.skip_closure_indicators:
        logger.info("\nSTEP 6: Skipped closure indicators (--skip-closure-indicators)")
    else:
        closure_df = compute_closure_indicators_for_lcc_bridges(
            G,
            bridges,
            topological_k_hops=max(1, args.k_hop)
        )

        if len(closure_df) > 0:
            closure_csv_path = output_dir / 'bridge_closure_indicators.csv'
            closure_df.to_csv(closure_csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"Saved closure indicators: {closure_csv_path}")

            bridges = bridges.merge(closure_df, on='bridge_id', how='left')

            fill_zero_cols = [
                'degree', 'num_neighbors', 'component_increase',
                'disconnected_nodes', 'k_hop_nodes', 'indirect_damage_score'
            ]
            for col in fill_zero_cols:
                if col in bridges.columns:
                    bridges[col] = bridges[col].fillna(0)

            logger.info(
                "Indirect damage score statistics: "
                f"min={bridges['indirect_damage_score'].min():.2f}, "
                f"max={bridges['indirect_damage_score'].max():.2f}, "
                f"mean={bridges['indirect_damage_score'].mean():.2f}"
            )

    # 7. 説明文生成
    logger.info("\nSTEP 7: Generating narratives...")
    from narrative_generator import generate_narratives_for_all
    bridges = generate_narratives_for_all(bridges, config)
    
    # 8. 結果保存
    logger.info("\nSTEP 8: Saving results...")
    
    # CSV保存
    csv_path = output_dir / 'bridge_importance_scores.csv'
    bridges.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"Saved CSV: {csv_path}")
    
    # GeoJSON保存
    geojson_path = output_dir / 'bridge_importance_scores.geojson'
    bridges.to_file(geojson_path, driver='GeoJSON', encoding='utf-8')
    logger.info(f"Saved GeoJSON: {geojson_path}")
    
    # トップ10保存
    top10 = bridges.nlargest(10, 'importance_score')
    top10_path = output_dir / 'top10_critical_bridges.csv'
    top10.to_csv(top10_path, index=False, encoding='utf-8-sig')
    logger.info(f"Saved top 10: {top10_path}")
    
    # Markdownレポート生成
    report_path = output_dir / 'bridge_importance_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 橋梁重要度スコアリング結果\n\n")
        f.write(f"## 概要\n\n")
        f.write(f"- 解析対象橋梁数: {len(bridges)}\n")
        f.write(f"- グラフ統計: {G.number_of_nodes():,} ノード, {G.number_of_edges():,} エッジ\n")
        f.write(f"- サンプリング設定: k={config['centrality'].get('k', 'None (全ノード)')}\n\n")
        
        f.write("## トップ10橋梁\n\n")
        f.write("| 順位 | 橋梁名 | スコア | 媒介中心性 | 所在地 |\n")
        f.write("|------|--------|--------|------------|--------|\n")
        
        for idx, (_, bridge) in enumerate(top10.iterrows(), 1):
            name = bridge.get('bridge_name', bridge.get('施設名', 'N/A'))
            score = bridge['importance_score']
            betweenness = bridge['betweenness']
            location = bridge.get('location', bridge.get('所在地', 'N/A'))
            f.write(f"| {idx} | {name} | {score:.2f} | {betweenness:.6f} | {location} |\n")
    
    logger.info(f"Saved report: {report_path}")
    
    # 9. 可視化
    logger.info("\nSTEP 9: Generating visualizations...")
    try:
        visualize_results(bridges, config)
        logger.info("Visualizations saved successfully")
    except Exception as e:
        logger.error(f"Visualization error: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Centrality recomputation completed successfully!")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()
