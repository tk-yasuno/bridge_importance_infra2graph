"""
Bridge Importance Scoring MVP
メインパイプライン

山口市791橋の重要度スコアリング
City2Graph + NetworkX媒介中心性を活用した異種グラフ分析

Version: 1.0.0
"""

__version__ = "1.1.0"

import yaml
import logging
from pathlib import Path
import geopandas as gpd
import pickle
from datetime import datetime
import sys
import argparse

# モジュールのインポート
from data_loader import load_all_data
from graph_builder import HeterogeneousGraphBuilder
from centrality_scorer import score_bridge_importance
from narrative_generator import generate_narratives_for_all, BridgeNarrativeGenerator

# ログ設定
def setup_logging(config):
    """ロギングの設定"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('bridge_importance_scoring.log', encoding='utf-8')
        ]
    )

logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config.yaml'):
    """設定ファイルの読み込み"""
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def save_results(bridges: gpd.GeoDataFrame, config: dict, metadata: dict):
    """結果の保存"""
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving results to {output_dir}")
    
    # 1. CSV出力（スコア付き橋梁リスト）
    csv_path = output_dir / 'bridge_importance_scores.csv'
    bridges_csv = bridges.copy()
    bridges_csv['geometry'] = bridges_csv.geometry.apply(lambda g: f"{g.x},{g.y}")
    bridges_csv.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"Saved CSV: {csv_path}")
    
    # 2. GeoJSON出力（地図化用）
    geojson_path = output_dir / 'bridge_importance_scores.geojson'
    bridges_geojson = bridges.to_crs('EPSG:4326')  # WGS84に変換
    bridges_geojson.to_file(geojson_path, driver='GeoJSON', encoding='utf-8')
    logger.info(f"Saved GeoJSON: {geojson_path}")
    
    # 3. レポート生成
    generator = BridgeNarrativeGenerator(config)
    report = generator.generate_report(bridges)
    report_path = output_dir / 'bridge_importance_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Saved report: {report_path}")
    
    # 4. トップ10詳細（CSV）
    top10_path = output_dir / 'top10_critical_bridges.csv'
    top10 = bridges.nlargest(10, 'importance_score')
    top10_csv = top10[[
        'bridge_id', 'importance_rank', 'importance_score', 'importance_category',
        'betweenness', 'num_public_facilities', 'num_hospitals', 'num_schools',
        'num_bus_stops', 'dist_to_river', 'dist_to_coast', 'narrative'
    ]].copy()
    top10_csv.to_csv(top10_path, index=False, encoding='utf-8-sig')
    logger.info(f"Saved top 10: {top10_path}")
    
    # 5. メタデータ（YAML）
    metadata_path = output_dir / 'metadata.yaml'
    metadata['timestamp'] = datetime.now().isoformat()
    with open(metadata_path, 'w', encoding='utf-8') as f:
        yaml.dump(metadata, f, allow_unicode=True)
    logger.info(f"Saved metadata: {metadata_path}")
    
    logger.info("All results saved successfully")


def main(use_merged_network: bool = False):
    """メイン処理"""
    print("=" * 80)
    print("Bridge Importance Scoring MVP")
    print("山口市1316橋の重要度スコアリング")
    print("=" * 80)
    print(f"Bridge Importance Scoring MVP v{__version__}")
    print("山口市791橋の重要度スコアリング")
    if use_merged_network:
        print("[Grid Mode] Using pre-fetched merged network")
    print("=" * 80)
    print()
    
    # 1. 設定の読み込み
    config = load_config('config.yaml')
    setup_logging(config)
    
    logger.info(f"Starting Bridge Importance Scoring pipeline... (v{__version__})")
    
    try:
        # 2. データの読み込み
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: Data Loading")
        logger.info("=" * 60)
        
        bridges, rivers, coastline, boundary = load_all_data(config)
        logger.info(f"Loaded {len(bridges)} bridges")
        logger.info(f"Boundary area: {boundary.geometry.iloc[0].area / 1e6:.1f} km²")
        
        # 3. 異種グラフの構築
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Heterogeneous Graph Construction")
        logger.info("=" * 60)
        
        graph_builder = HeterogeneousGraphBuilder(config)
        G, graph_metadata = graph_builder.build_heterogeneous_graph(
            bridges, 
            boundary,
            use_merged_network=use_merged_network,
            merged_prefix="yamaguchi_merged"
        )
        
        logger.info(f"Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # グラフの保存（オプション）
        graph_path = Path(config['data']['output_dir']) / 'heterogeneous_graph.pkl'
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        with open(graph_path, 'wb') as f:
            pickle.dump(G, f)
        logger.info(f"Graph saved to {graph_path}")
        
        # 4. 媒介中心性の計算とスコアリング
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Betweenness Centrality & Scoring")
        logger.info("=" * 60)
        
        scored_bridges = score_bridge_importance(bridges, G, config)
        
        # 5. 説明文の生成
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Narrative Generation")
        logger.info("=" * 60)
        
        final_bridges = generate_narratives_for_all(scored_bridges, config)
        
        # 6. 結果の保存
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: Saving Results")
        logger.info("=" * 60)
        
        metadata = {
            'config': config,
            'graph_metadata': graph_metadata,
            'num_bridges_analyzed': len(final_bridges)
        }
        
        save_results(final_bridges, config, metadata)
        
        # 7. サマリーの表示
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        print("\n" + "=" * 80)
        print("処理完了サマリー")
        print("=" * 80)
        print(f"分析橋梁数: {len(final_bridges)}橋")
        print(f"グラフ規模: {G.number_of_nodes()}ノード, {G.number_of_edges()}エッジ")
        print(f"\n重要度カテゴリ分布:")
        category_counts = final_bridges['importance_category'].value_counts()
        for cat, count in category_counts.items():
            print(f"  {cat}: {count}橋 ({100*count/len(final_bridges):.1f}%)")
        
        print(f"\nトップ5橋梁:")
        top5 = final_bridges.nlargest(5, 'importance_score')
        for rank, (idx, bridge) in enumerate(top5.iterrows(), 1):
            bridge_name = bridge.get('name', bridge['bridge_id'])
            score = bridge['importance_score']
            print(f"  {rank}. {bridge_name} (スコア: {score:.1f})")
        
        print(f"\n出力ディレクトリ: {config['data']['output_dir']}")
        print("=" * 80)
        
        return final_bridges, G, metadata
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(
        description="Bridge Importance Scoring MVP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 通常モード（OSMから直接取得、小規模エリア向け）
  python main.py
  
  # グリッドモード（事前取得済みマージネットワーク使用、大規模エリア向け）
  python main.py --use-merged-network
  
Note:
  グリッドモードを使用する場合は、事前に fetch_osm_grid.py を実行して
  マージされたネットワークを生成してください。
        """
    )
    parser.add_argument(
        '--use-merged-network',
        action='store_true',
        help='Use pre-fetched merged road network (from fetch_osm_grid.py)'
    )
    
    args = parser.parse_args()
    main(use_merged_network=args.use_merged_network)
