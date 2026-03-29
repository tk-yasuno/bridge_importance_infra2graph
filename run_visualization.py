"""
可視化実行スクリプト
既存の結果データを読み込んで可視化を生成

Version: 1.0.0

使い方:
    python run_visualization.py
"""

__version__ = "1.1.0"

import yaml
import logging
import sys
import argparse
from pathlib import Path
import geopandas as gpd
import pandas as pd

# モジュールのインポート
from visualization import visualize_results, BridgeVisualizer

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description="Bridge visualization runner (legacy / v1.4)"
    )
    parser.add_argument(
        "--mode",
        choices=["legacy", "v1_4"],
        default="legacy",
        help="可視化モード（legacy: importance_score, v1_4: indirect_damage_score）"
    )
    parser.add_argument(
        "--score-column",
        default=None,
        help="可視化対象のスコア列名（指定時はmode設定を上書き）"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="静的マップで表示する上位橋梁数"
    )
    return parser.parse_args()


def _assign_category(score_100: float) -> str:
    """0-100正規化スコアからカテゴリを付与"""
    if score_100 >= 80:
        return "critical"
    if score_100 >= 60:
        return "high"
    if score_100 >= 40:
        return "medium"
    if score_100 >= 20:
        return "low"
    return "very_low"


def prepare_visualization_dataframe(
    bridges: gpd.GeoDataFrame,
    score_column: str
) -> gpd.GeoDataFrame:
    """可視化に必要な共通カラムへ整形"""
    if score_column not in bridges.columns:
        raise KeyError(f"Score column not found: {score_column}")

    out = bridges.copy()
    out['importance_score'] = pd.to_numeric(out[score_column], errors='coerce').fillna(0.0)

    score_min = float(out['importance_score'].min())
    score_max = float(out['importance_score'].max())
    denom = (score_max - score_min) if (score_max - score_min) > 1e-9 else 1.0
    out['importance_score_100'] = ((out['importance_score'] - score_min) / denom) * 100.0
    out['importance_category'] = out['importance_score_100'].apply(_assign_category)

    # 1始まり順位（同値は最小順位）
    out['importance_rank'] = out['importance_score'].rank(method='min', ascending=False).astype(int)

    # ポップアップに使う名称列を補正
    if 'name' not in out.columns:
        if '施設名' in out.columns:
            out['name'] = out['施設名']
        else:
            out['name'] = out['bridge_id']

    return out


def main():
    """可視化メイン処理"""
    args = parse_args()

    print("=" * 80)
    print(f"Bridge Importance Scoring - Visualization v{__version__}")
    print(f"既存結果データの可視化 (mode={args.mode})")
    print("=" * 80)
    print()
    
    # 設定の読み込み
    logger.info("Loading configuration...")
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 結果ファイルのパス
    output_dir = Path(config['data']['output_dir'])
    csv_path = output_dir / 'bridge_importance_scores.csv'
    geojson_path = output_dir / 'bridge_importance_scores.geojson'
    
    # 結果ファイルの存在確認
    if not csv_path.exists():
        logger.error(f"Result file not found: {csv_path}")
        logger.error("Please run 'python main.py' first to generate results.")
        sys.exit(1)
    
    # データの読み込み
    logger.info(f"Loading results from {geojson_path}...")
    try:
        # GeoJSONから読み込み（ジオメトリ含む）
        bridges = gpd.read_file(geojson_path)
        logger.info(f"Loaded {len(bridges)} bridges")
        
        # データ型の確認
        logger.info(f"CRS: {bridges.crs}")
        logger.info(f"Columns: {bridges.columns.tolist()}")
        
    except Exception as e:
        logger.error(f"Error loading GeoJSON: {e}")
        logger.info("Trying to load from CSV...")
        
        # CSVから読み込み（座標から再構築）
        bridges_df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        # geometryカラムから座標を抽出
        if 'geometry' in bridges_df.columns:
            # "x,y"形式の文字列を分割
            coords = bridges_df['geometry'].str.split(',', expand=True)
            bridges_df['lon'] = coords[0].astype(float)
            bridges_df['lat'] = coords[1].astype(float)
            
            # GeoDataFrameに変換
            from shapely.geometry import Point
            geometry = [Point(xy) for xy in zip(bridges_df['lon'], bridges_df['lat'])]
            bridges = gpd.GeoDataFrame(bridges_df, geometry=geometry, crs='EPSG:4326')
            
            logger.info(f"Loaded {len(bridges)} bridges from CSV")
        else:
            logger.error("No geometry information found in CSV")
            sys.exit(1)
    
    # 可視化対象列
    score_column = args.score_column
    if not score_column:
        score_column = 'importance_score' if args.mode == 'legacy' else 'indirect_damage_score'

    logger.info(f"Using score column: {score_column}")
    bridges = prepare_visualization_dataframe(bridges, score_column=score_column)

    # 出力ファイル名のプレフィックス
    prefix = 'v1_4_' if args.mode == 'v1_4' else ''

    # 可視化の実行
    logger.info("\n" + "=" * 60)
    logger.info("Generating Visualizations")
    logger.info("=" * 60)
    
    visualizer = BridgeVisualizer(config)
    
    # 1. スコア分布プロット
    logger.info("\n1. Creating score distribution plot...")
    visualizer.plot_score_distribution(
        bridges,
        save_path=output_dir / f'{prefix}score_distribution.png'
    )
    print(f"✓ Saved: {output_dir / f'{prefix}score_distribution.png'}")
    
    # 2. トップ20橋梁マップ（静的）
    logger.info("\n2. Creating top 20 bridges map...")
    visualizer.plot_top_bridges_map(
        bridges,
        top_n=args.top_n,
        save_path=output_dir / f'{prefix}top{args.top_n}_bridges_map.png'
    )
    print(f"✓ Saved: {output_dir / f'{prefix}top{args.top_n}_bridges_map.png'}")
    
    # 3. 対話的地図（HTML）
    logger.info("\n3. Creating interactive map...")
    visualizer.create_interactive_map(
        bridges,
        save_path=output_dir / f'{prefix}interactive_bridge_map.html'
    )
    print(f"✓ Saved: {output_dir / f'{prefix}interactive_bridge_map.html'}")
    
    # 完了
    print("\n" + "=" * 80)
    print("可視化完了！")
    print("=" * 80)
    print(f"\n生成ファイル:")
    print(f"  - {output_dir / f'{prefix}score_distribution.png'}")
    print(f"  - {output_dir / f'{prefix}top{args.top_n}_bridges_map.png'}")
    print(f"  - {output_dir / f'{prefix}interactive_bridge_map.html'}")
    print(f"\nブラウザで開く:")
    print(f"  file:///{output_dir.absolute() / f'{prefix}interactive_bridge_map.html'}")
    print()


if __name__ == '__main__':
    main()
