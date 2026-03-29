"""
ユーティリティスクリプト
Bridge Importance Scoring MVP

データ検証、グラフ統計、デバッグ用の便利関数
"""

import geopandas as gpd
import networkx as nx
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Tuple
import pickle

logger = logging.getLogger(__name__)


def validate_bridge_data(bridges: gpd.GeoDataFrame) -> Tuple[bool, list]:
    """
    橋梁データの検証
    
    Args:
        bridges: 橋梁GeoDataFrame
    
    Returns:
        (is_valid, error_messages)のタプル
    """
    errors = []
    
    # 必須カラムの確認
    required_cols = ['bridge_id', 'geometry']
    for col in required_cols:
        if col not in bridges.columns:
            errors.append(f"Missing required column: {col}")
    
    # ジオメトリの検証
    if 'geometry' in bridges.columns:
        invalid_geom = bridges[bridges.geometry.is_empty | bridges.geometry.isna()]
        if len(invalid_geom) > 0:
            errors.append(f"Found {len(invalid_geom)} bridges with invalid geometry")
    
    # 重複チェック
    if 'bridge_id' in bridges.columns:
        duplicates = bridges['bridge_id'].duplicated().sum()
        if duplicates > 0:
            errors.append(f"Found {duplicates} duplicate bridge_id values")
    
    # CRSの確認
    if bridges.crs is None:
        errors.append("CRS is not set for bridge data")
    
    is_valid = len(errors) == 0
    
    if is_valid:
        logger.info("Bridge data validation passed")
    else:
        logger.warning(f"Bridge data validation failed with {len(errors)} errors")
        for error in errors:
            logger.warning(f"  - {error}")
    
    return is_valid, errors


def print_graph_statistics(G: nx.Graph):
    """
    グラフの統計情報を表示
    
    Args:
        G: NetworkXグラフ
    """
    logger.info("\n" + "=" * 60)
    logger.info("Graph Statistics")
    logger.info("=" * 60)
    
    logger.info(f"Number of nodes: {G.number_of_nodes()}")
    logger.info(f"Number of edges: {G.number_of_edges()}")
    
    # ノードタイプ別カウント
    node_types = {}
    for node, data in G.nodes(data=True):
        node_type = data.get('node_type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    logger.info("\nNode types:")
    for node_type, count in sorted(node_types.items()):
        logger.info(f"  {node_type}: {count}")
    
    # エッジタイプ別カウント
    edge_types = {}
    for u, v, data in G.edges(data=True):
        edge_type = data.get('edge_type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    logger.info("\nEdge types:")
    for edge_type, count in sorted(edge_types.items()):
        logger.info(f"  {edge_type}: {count}")
    
    # 連結性
    is_connected = nx.is_connected(G)
    num_components = nx.number_connected_components(G)
    logger.info(f"\nIs connected: {is_connected}")
    logger.info(f"Number of connected components: {num_components}")
    
    if not is_connected:
        largest_cc = max(nx.connected_components(G), key=len)
        logger.info(f"Largest connected component size: {len(largest_cc)}")
    
    # 次数統計
    degrees = [G.degree(n) for n in G.nodes()]
    if degrees:
        logger.info(f"\nDegree statistics:")
        logger.info(f"  Mean: {sum(degrees) / len(degrees):.2f}")
        logger.info(f"  Max: {max(degrees)}")
        logger.info(f"  Min: {min(degrees)}")
    
    logger.info("=" * 60 + "\n")


def load_saved_results(output_dir: str) -> Tuple[gpd.GeoDataFrame, nx.Graph, Dict]:
    """
    保存済みの結果を読み込み
    
    Args:
        output_dir: 出力ディレクトリパス
    
    Returns:
        (bridges, graph, metadata)のタプル
    """
    output_path = Path(output_dir)
    
    # 橋梁データ（GeoJSON）
    geojson_path = output_path / 'bridge_importance_scores.geojson'
    if geojson_path.exists():
        bridges = gpd.read_file(geojson_path)
        logger.info(f"Loaded bridges from {geojson_path}")
    else:
        raise FileNotFoundError(f"Bridge data not found at {geojson_path}")
    
    # グラフ（Pickle）
    graph_path = output_path / 'heterogeneous_graph.pkl'
    if graph_path.exists():
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        logger.info(f"Loaded graph from {graph_path}")
    else:
        logger.warning(f"Graph not found at {graph_path}")
        graph = None
    
    # メタデータ（YAML）
    metadata_path = output_path / 'metadata.yaml'
    if metadata_path.exists():
        import yaml
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f)
        logger.info(f"Loaded metadata from {metadata_path}")
    else:
        logger.warning(f"Metadata not found at {metadata_path}")
        metadata = {}
    
    return bridges, graph, metadata


def export_for_gis(
    bridges: gpd.GeoDataFrame,
    output_path: str,
    format: str = 'shapefile'
):
    """
    GISソフト用にデータをエクスポート
    
    Args:
        bridges: 橋梁データ
        output_path: 出力パス
        format: 出力フォーマット（'shapefile', 'geojson', 'gpkg'）
    """
    output_path = Path(output_path)
    
    # WGS84に変換
    bridges_wgs84 = bridges.to_crs('EPSG:4326')
    
    # カラム名の短縮（Shapefileの10文字制限対応）
    if format == 'shapefile':
        bridges_wgs84 = bridges_wgs84.rename(columns={
            'importance_score': 'imp_score',
            'importance_rank': 'imp_rank',
            'importance_category': 'imp_cat',
            'betweenness': 'betw_cent',
            'num_public_facilities': 'num_public',
            'num_hospitals': 'num_hosp',
            'num_schools': 'num_school',
            'num_bus_stops': 'num_bus',
            'dist_to_river': 'dist_river',
            'dist_to_coast': 'dist_coast'
        })
    
    # エクスポート
    if format == 'shapefile':
        bridges_wgs84.to_file(output_path, driver='ESRI Shapefile', encoding='utf-8')
    elif format == 'geojson':
        bridges_wgs84.to_file(output_path, driver='GeoJSON', encoding='utf-8')
    elif format == 'gpkg':
        bridges_wgs84.to_file(output_path, driver='GPKG', encoding='utf-8')
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Exported to {output_path} ({format})")


def compare_centrality_measures(
    G: nx.Graph,
    bridge_nodes: list,
    limit: int = 10
):
    """
    複数の中心性指標を比較
    
    Args:
        G: NetworkXグラフ
        bridge_nodes: 橋梁ノードリスト
        limit: 表示する上位N件
    """
    logger.info("Computing multiple centrality measures...")
    
    # 連結成分のチェック
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        bridge_nodes = [n for n in bridge_nodes if n in G.nodes]
    
    # 各中心性の計算
    betweenness = nx.betweenness_centrality(G, normalized=True)
    closeness = nx.closeness_centrality(G)
    degree_cent = nx.degree_centrality(G)
    
    # 橋梁ノードのみ抽出
    results = []
    for node in bridge_nodes:
        if node in G.nodes:
            results.append({
                'bridge_id': node,
                'betweenness': betweenness.get(node, 0),
                'closeness': closeness.get(node, 0),
                'degree': degree_cent.get(node, 0)
            })
    
    df = pd.DataFrame(results)
    
    # 各指標ごとにランク付け
    df['betw_rank'] = df['betweenness'].rank(ascending=False)
    df['close_rank'] = df['closeness'].rank(ascending=False)
    df['deg_rank'] = df['degree'].rank(ascending=False)
    
    logger.info(f"\nTop {limit} bridges by betweenness centrality:")
    top_betw = df.nlargest(limit, 'betweenness')
    print(top_betw[['bridge_id', 'betweenness', 'betw_rank']].to_string(index=False))
    
    logger.info(f"\nTop {limit} bridges by closeness centrality:")
    top_close = df.nlargest(limit, 'closeness')
    print(top_close[['bridge_id', 'closeness', 'close_rank']].to_string(index=False))
    
    # 相関分析
    logger.info("\nCorrelation between centrality measures:")
    corr = df[['betweenness', 'closeness', 'degree']].corr()
    print(corr)
    
    return df


def quick_analysis(output_dir: str = 'output/bridge_importance'):
    """
    保存済みデータのクイック分析
    
    Args:
        output_dir: 出力ディレクトリパス
    """
    try:
        bridges, graph, metadata = load_saved_results(output_dir)
        
        logger.info("\n" + "=" * 60)
        logger.info("Quick Analysis")
        logger.info("=" * 60)
        
        # 橋梁統計
        logger.info(f"\nTotal bridges: {len(bridges)}")
        logger.info(f"Score range: [{bridges['importance_score'].min():.2f}, {bridges['importance_score'].max():.2f}]")
        
        # カテゴリ分布
        logger.info("\nCategory distribution:")
        category_counts = bridges['importance_category'].value_counts()
        for cat, count in category_counts.items():
            print(f"  {cat}: {count} ({100*count/len(bridges):.1f}%)")
        
        # トップ5
        logger.info("\nTop 5 bridges:")
        top5 = bridges.nlargest(5, 'importance_score')
        for rank, (idx, bridge) in enumerate(top5.iterrows(), 1):
            name = bridge.get('name', bridge['bridge_id'])
            score = bridge['importance_score']
            print(f"  {rank}. {name}: {score:.1f}")
        
        # グラフ統計
        if graph is not None:
            print_graph_statistics(graph)
        
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Quick analysis failed: {e}", exc_info=True)


if __name__ == '__main__':
    # スタンドアロン実行時のクイック分析
    quick_analysis()
