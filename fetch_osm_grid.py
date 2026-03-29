"""
OSM Grid Fetch 実行スクリプト
Bridge Importance Scoring MVP v1.1

山口市全域のOSMデータを4×4グリッドに分割して取得し、
マージされたネットワークを生成する。

Usage:
    python fetch_osm_grid.py
"""

import yaml
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from osm_grid_fetcher import OSMGridFetcher

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """メイン処理"""
    logger.info("=" * 80)
    logger.info("OSM Grid Fetch v1.1 - Yamaguchi City")
    logger.info("=" * 80)
    
    # 設定ファイルの読み込み
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # OSM Grid Fetcher の初期化
    fetcher = OSMGridFetcher(config)
    
    # ステップ1: 山口市ポリゴンを取得
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Fetching city polygon")
    logger.info("=" * 80)
    
    place_name = "山口市, 山口県, 日本"
    city_poly = fetcher.get_city_polygon(place_name)
    
    # ステップ2: 4×4 グリッドに分割
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Creating 4×4 grid")
    logger.info("=" * 80)
    
    grid_gdf = fetcher.make_grid_over_polygon(city_poly, n_rows=4, n_cols=4)
    
    # グリッドの可視化（オプション）
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        city_poly.plot(ax=ax, edgecolor="red", facecolor="none", linewidth=2, label="City boundary")
        grid_gdf.boundary.plot(ax=ax, color="blue", linewidth=1, label="Grid cells")
        
        # セルIDを表示
        for idx, row in grid_gdf.iterrows():
            centroid = row.geometry.centroid
            ax.text(centroid.x, centroid.y, str(row['cell_id']), 
                   fontsize=10, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        ax.set_title(f"Yamaguchi City 4×4 Grid ({len(grid_gdf)} cells)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend()
        
        # 保存
        output_path = fetcher.output_dir.parent / "grid_visualization.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Grid visualization saved to: {output_path}")
        plt.close()
    except Exception as e:
        logger.warning(f"Grid visualization failed: {e}")
    
    # ステップ3: 各セルのOSMデータを取得
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Fetching OSM data for each cell")
    logger.info("=" * 80)
    logger.info("⚠ This will take 15-30 minutes for all cells with retries")
    logger.info("⚠ Rate limit delays: 5 seconds between cells")
    
    stats = fetcher.run_for_all_cells(
        grid_gdf,
        fetch_roads=True,        # 道路ネットワーク（必須）
        fetch_buildings=False,   # 建物（オプション、時間短縮のためスキップ可）
        fetch_bus_stops=False,   # バス停（オプション、時間短縮のためスキップ可）
        inter_cell_delay=5       # セル間の待機時間（秒）
    )
    
    # ステップ4: 道路ネットワークをマージ
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Merging road networks from all cells")
    logger.info("=" * 80)
    
    try:
        nodes_merged, edges_merged = fetcher.merge_cell_roads()
        
        # マージされたネットワークを保存
        fetcher.save_merged_networks(nodes_merged, edges_merged, output_prefix="yamaguchi_merged")
        
        logger.info("\n" + "=" * 80)
        logger.info("OSM Grid Fetch COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"Statistics:")
        logger.info(f"  Total cells: {stats['total_cells']}")
        logger.info(f"  Roads fetched: {stats['roads_success']}/{stats['total_cells']}")
        logger.info(f"  Merged nodes: {len(nodes_merged)}")
        logger.info(f"  Merged edges: {len(edges_merged)}")
        logger.info("\nNext steps:")
        logger.info("  1. Run main.py to build heterogeneous graph using merged network")
        logger.info("  2. Review output/bridge_importance/yamaguchi_merged_roads_*.gpkg")
        
    except Exception as e:
        logger.error(f"Failed to merge networks: {e}", exc_info=True)
        logger.info("\n⚠ Partial data may be available in output/bridge_importance/osm_cells/")
        logger.info("  You can manually verify cell_*_roads_*.gpkg files")


if __name__ == "__main__":
    main()
