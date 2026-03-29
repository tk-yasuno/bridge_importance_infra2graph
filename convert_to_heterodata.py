"""
HeteroData 変換実行スクリプト
Bridge Importance Scoring MVP v1.1

既存の NetworkX グラフと GeoDataFrame から PyTorch Geometric HeteroData を生成
"""

import torch
import yaml
import logging
import pickle
from pathlib import Path
import geopandas as gpd
import pandas as pd

from hetero_data_converter import HeteroGraphConverter

__version__ = "1.1.0"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """メイン実行関数"""
    logger.info(f"HeteroData Conversion v{__version__}")
    logger.info("=" * 80)
    
    # 設定ファイルの読み込み
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 入力ファイルのパス
    graph_pkl_path = Path("output/bridge_importance/heterogeneous_graph.pkl")
    bridge_geojson_path = Path("output/bridge_importance/bridge_importance_scores.geojson")
    
    # ファイルの存在確認
    if not graph_pkl_path.exists():
        logger.error(f"Graph file not found: {graph_pkl_path}")
        logger.info("Please run main.py first to generate the heterogeneous graph.")
        return
    
    if not bridge_geojson_path.exists():
        logger.error(f"Bridge data file not found: {bridge_geojson_path}")
        logger.info("Please run main.py first to generate bridge importance scores.")
        return
    
    # 1. NetworkX グラフの読み込み
    logger.info(f"Loading NetworkX graph from {graph_pkl_path}...")
    with open(graph_pkl_path, 'rb') as f:
        G = pickle.load(f)
    logger.info(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # 2. Bridge GeoDataFrame の読み込み
    logger.info(f"Loading bridge data from {bridge_geojson_path}...")
    bridge_gdf = gpd.read_file(bridge_geojson_path)
    logger.info(f"Loaded {len(bridge_gdf)} bridges with {len(bridge_gdf.columns)} columns")
    
    # 3. Street/Building/Bus stop データの取得（グラフから抽出）
    # ノードタイプごとに分類
    street_nodes = []
    building_nodes = []
    bus_stop_nodes = []
    
    for node_id, node_data in G.nodes(data=True):
        node_type = node_data.get('node_type', 'unknown')
        geometry = node_data.get('geometry')
        
        if node_type == 'street':
            street_nodes.append({
                'node_id': node_id,
                'x': node_data.get('x', 0),
                'y': node_data.get('y', 0),
                'geometry': geometry
            })
        elif node_type == 'building':
            building_nodes.append({
                'node_id': node_id,
                'category': node_data.get('category', 'other'),
                'geometry': geometry
            })
        elif node_type == 'bus_stop':
            bus_stop_nodes.append({
                'node_id': node_id,
                'geometry': geometry
            })
    
    # DataFrames に変換（空でない場合のみ）
    street_nodes_df = pd.DataFrame(street_nodes) if street_nodes else None
    building_nodes_df = pd.DataFrame(building_nodes) if building_nodes else None
    bus_stop_nodes_df = pd.DataFrame(bus_stop_nodes) if bus_stop_nodes else None
    
    if street_nodes_df is not None:
        logger.info(f"Extracted {len(street_nodes_df)} street nodes")
    if building_nodes_df is not None:
        logger.info(f"Extracted {len(building_nodes_df)} building nodes")
    if bus_stop_nodes_df is not None:
        logger.info(f"Extracted {len(bus_stop_nodes_df)} bus stop nodes")
    
    # 4. HeteroData への変換
    logger.info("Converting to HeteroData...")
    converter = HeteroGraphConverter(config)
    
    data = converter.convert_to_hetero_data(
        G=G,
        bridge_gdf=bridge_gdf,
        street_nodes=street_nodes_df,
        buildings=building_nodes_df,
        bus_stops=bus_stop_nodes_df
    )
    
    # 5. 逆方向エッジ（bridge→street）を追加
    if ('street', 'to', 'bridge') in data.edge_types:
        src, dst = data['street', 'to', 'bridge'].edge_index
        data['bridge', 'to', 'street'].edge_index = torch.stack([dst, src], dim=0)
        if hasattr(data['street', 'to', 'bridge'], 'edge_attr'):
            data['bridge', 'to', 'street'].edge_attr = data['street', 'to', 'bridge'].edge_attr.clone()
        logger.info(f"Added reverse edges (bridge→street): {data['bridge', 'to', 'street'].edge_index.shape[1]}")

    # 6. HeteroData の保存
    output_path = Path("output/bridge_importance/heterogeneous_graph_heterodata.pt")
    converter.save_hetero_data(data, str(output_path))
    
    # 7. サマリーの表示
    logger.info("=" * 80)
    logger.info("HeteroData Summary:")
    logger.info(f"  Node types: {list(data.node_types)}")
    logger.info(f"  Edge types: {list(data.edge_types)}")
    
    for node_type in data.node_types:
        if hasattr(data[node_type], 'x'):
            logger.info(f"  {node_type}:")
            logger.info(f"    - Nodes: {data[node_type].x.shape[0]}")
            logger.info(f"    - Features: {data[node_type].x.shape[1]}")
            if hasattr(data[node_type], 'y'):
                logger.info(f"    - Target: {data[node_type].y.shape}")
    
    for edge_type in data.edge_types:
        if hasattr(data[edge_type], 'edge_index'):
            logger.info(f"  {edge_type}:")
            logger.info(f"    - Edges: {data[edge_type].edge_index.shape[1]}")
    
    logger.info("=" * 80)
    logger.info(f"HeteroData saved to {output_path}")
    logger.info("Conversion completed successfully!")
    logger.info("\nNext steps:")
    logger.info("  1. Review the HeteroData structure")
    logger.info("  2. Run train_hgnn.py to train the HGNN model")
    logger.info("  3. Evaluate the model performance")


if __name__ == "__main__":
    main()
