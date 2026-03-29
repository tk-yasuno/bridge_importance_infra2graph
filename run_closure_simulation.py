"""
Bridge Closure Simulation 実行スクリプト
Bridge Importance Scoring MVP v1.2

重要度Low以上（132橋）を対象に、
橋梁閉鎖シナリオのシミュレーションを実行する。

Usage:
    python run_closure_simulation.py [--sample-size 500]
"""

import yaml
import logging
import argparse
from pathlib import Path
import geopandas as gpd
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from bridge_closure_simulator import BridgeClosureSimulator

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_bridge_importance_scores(config: dict) -> gpd.GeoDataFrame:
    """
    橋梁重要度スコアを読み込み
    
    Args:
        config: 設定辞書
    
    Returns:
        橋梁データのGeoDataFrame
    """
    output_dir = Path(config['data']['output_dir'])
    geojson_path = output_dir / 'bridge_importance_scores.geojson'
    
    if not geojson_path.exists():
        raise FileNotFoundError(
            f"Bridge importance scores not found: {geojson_path}\n"
            "Please run main.py first to generate importance scores."
        )
    
    logger.info(f"Loading bridge importance scores from {geojson_path}")
    bridges = gpd.read_file(geojson_path)
    
    logger.info(f"Loaded {len(bridges)} bridges")
    
    return bridges


def filter_bridges_by_importance(
    bridges: gpd.GeoDataFrame,
    exclude_very_low: bool = True
) -> gpd.GeoDataFrame:
    """
    重要度カテゴリでフィルタリング
    
    Args:
        bridges: 橋梁データ
        exclude_very_low: Very Lowカテゴリを除外するか
    
    Returns:
        フィルタリングされた橋梁データ
    """
    if 'importance_category' not in bridges.columns:
        logger.warning("importance_category column not found. Using all bridges.")
        return bridges
    
    if exclude_very_low:
        # Very Low を除外
        filtered = bridges[bridges['importance_category'] != 'very_low'].copy()
        logger.info(f"Filtered bridges: {len(filtered)} (excluding Very Low)")
        
        # カテゴリ分布
        category_counts = filtered['importance_category'].value_counts()
        logger.info("Target bridge categories:")
        for cat, count in category_counts.items():
            logger.info(f"  {cat}: {count} bridges")
    else:
        filtered = bridges.copy()
        logger.info(f"Using all {len(filtered)} bridges")
    
    return filtered


def load_heterogeneous_graph(config: dict):
    """
    異種グラフを読み込み
    
    Args:
        config: 設定辞書
    
    Returns:
        NetworkX グラフ
    """
    output_dir = Path(config['data']['output_dir'])
    graph_path = output_dir / 'heterogeneous_graph.pkl'
    
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Heterogeneous graph not found: {graph_path}\n"
            "Please run main.py first to build the graph."
        )
    
    logger.info(f"Loading heterogeneous graph from {graph_path}")
    
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    
    logger.info(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G


def visualize_impact_results(
    results_df: pd.DataFrame,
    output_dir: Path
):
    """
    シミュレーション結果を可視化
    
    Args:
        results_df: シミュレーション結果
        output_dir: 出力ディレクトリ
    """
    logger.info("Generating visualizations...")
    
    # 4つのサブプロット
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 経路長増加の分布
    ax1 = axes[0, 0]
    results_df['delta_avg_shortest_path'].hist(bins=30, ax=ax1, edgecolor='black', alpha=0.7)
    ax1.axvline(results_df['delta_avg_shortest_path'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax1.set_xlabel('Δ Average Shortest Path Length')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Path Length Increase\n(After Bridge Closure)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. ノード損失の分布
    ax2 = axes[0, 1]
    results_df['delta_connected_nodes'].apply(abs).hist(bins=30, ax=ax2, edgecolor='black', alpha=0.7, color='orange')
    ax2.axvline(abs(results_df['delta_connected_nodes']).mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax2.set_xlabel('Number of Nodes Lost (Unreachable)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Node Loss\n(After Bridge Closure)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. バス停損失の分布
    ax3 = axes[1, 0]
    results_df['delta_accessible_bus_stops'].apply(abs).hist(bins=30, ax=ax3, edgecolor='black', alpha=0.7, color='green')
    ax3.axvline(abs(results_df['delta_accessible_bus_stops']).mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax3.set_xlabel('Number of Bus Stops Lost (Inaccessible)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Bus Stop Loss\n(After Bridge Closure)')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. 経路長増加率の分布
    ax4 = axes[1, 1]
    results_df['pct_path_increase'].hist(bins=30, ax=ax4, edgecolor='black', alpha=0.7, color='purple')
    ax4.axvline(results_df['pct_path_increase'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax4.set_xlabel('Path Length Increase (%)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Path Length Increase (%)\n(After Bridge Closure)')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    viz_path = output_dir / 'closure_impact_distribution.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    logger.info(f"Visualization saved to {viz_path}")
    plt.close()
    
    # トップ10影響度の高い橋梁（棒グラフ）
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 経路長増加トップ10
    ax1 = axes[0]
    top10_path = results_df.nlargest(10, 'delta_avg_shortest_path')
    ax1.barh(range(10), top10_path['delta_avg_shortest_path'].values, color='steelblue')
    ax1.set_yticks(range(10))
    ax1.set_yticklabels(top10_path['bridge_id'].values, fontsize=9)
    ax1.set_xlabel('Δ Average Shortest Path Length')
    ax1.set_title('Top 10 Bridges by Path Length Increase')
    ax1.invert_yaxis()
    ax1.grid(alpha=0.3, axis='x')
    
    # ノード損失トップ10
    ax2 = axes[1]
    top10_nodes = results_df.nsmallest(10, 'delta_connected_nodes')
    ax2.barh(range(10), top10_nodes['delta_connected_nodes'].apply(abs).values, color='coral')
    ax2.set_yticks(range(10))
    ax2.set_yticklabels(top10_nodes['bridge_id'].values, fontsize=9)
    ax2.set_xlabel('Number of Nodes Lost')
    ax2.set_title('Top 10 Bridges by Node Loss')
    ax2.invert_yaxis()
    ax2.grid(alpha=0.3, axis='x')
    
    # バス停損失トップ10
    ax3 = axes[2]
    top10_bus = results_df.nsmallest(10, 'delta_accessible_bus_stops')
    ax3.barh(range(10), top10_bus['delta_accessible_bus_stops'].apply(abs).values, color='seagreen')
    ax3.set_yticks(range(10))
    ax3.set_yticklabels(top10_bus['bridge_id'].values, fontsize=9)
    ax3.set_xlabel('Number of Bus Stops Lost')
    ax3.set_title('Top 10 Bridges by Bus Stop Loss')
    ax3.invert_yaxis()
    ax3.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # 保存
    top10_path = output_dir / 'closure_impact_top10.png'
    plt.savefig(top10_path, dpi=150, bbox_inches='tight')
    logger.info(f"Top 10 visualization saved to {top10_path}")
    plt.close()


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="Bridge Closure Impact Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # デフォルト設定で実行（サンプルサイズ500）
  python run_closure_simulation.py
  
  # サンプルサイズを増やして精度向上（計算時間増加）
  python run_closure_simulation.py --sample-size 1000
  
  # Very Lowカテゴリを含めて全橋梁を対象
  python run_closure_simulation.py --include-very-low

Note:
  本スクリプトは、main.pyで生成された以下のファイルが必要です:
  - output/bridge_importance/bridge_importance_scores.geojson
  - output/bridge_importance/heterogeneous_graph.pkl
        """
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=500,
        help='Sample size for shortest path computation (default: 500)'
    )
    parser.add_argument(
        '--include-very-low',
        action='store_true',
        help='Include Very Low importance bridges (default: exclude)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Bridge Closure Impact Simulation v1.2")
    logger.info("=" * 80)
    logger.info(f"Sample size: {args.sample_size}")
    logger.info(f"Include Very Low: {args.include_very_low}")
    logger.info("")
    
    # 設定ファイルの読み込み
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(config['data']['output_dir']) / 'closure_simulation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # ステップ1: データ読み込み
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Loading Data")
        logger.info("=" * 80)
        
        bridges = load_bridge_importance_scores(config)
        G = load_heterogeneous_graph(config)
        
        # ステップ2: 対象橋梁のフィルタリング
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Filtering Target Bridges")
        logger.info("=" * 80)
        
        target_bridges = filter_bridges_by_importance(
            bridges,
            exclude_very_low=not args.include_very_low
        )
        
        # bridge_id のリストを抽出
        bridge_ids = target_bridges['bridge_id'].tolist()
        
        logger.info(f"Target bridges for simulation: {len(bridge_ids)}")
        
        # ステップ3: シミュレーターの初期化
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Initializing Simulator")
        logger.info("=" * 80)
        
        simulator = BridgeClosureSimulator(G)
        
        # ベースラインメトリクスの計算
        baseline = simulator.compute_baseline_metrics(sample_size=args.sample_size)
        
        # ステップ4: 閉鎖シミュレーションの実行
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Running Closure Simulations")
        logger.info("=" * 80)
        logger.info(f"⚠ This will take approximately {len(bridge_ids) * 2 // 60} minutes for {len(bridge_ids)} bridges")
        
        results_df = simulator.simulate_multiple_bridges(
            bridge_ids,
            sample_size=args.sample_size,
            show_progress=True
        )
        
        # ステップ5: 結果の保存
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Saving Results")
        logger.info("=" * 80)
        
        # CSV保存
        csv_path = output_dir / 'closure_simulation_results.csv'
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"Results saved to {csv_path}")
        
        # レポート生成
        report_path = output_dir / 'closure_impact_report.md'
        simulator.generate_impact_report(results_df, report_path)
        
        # 可視化
        visualize_impact_results(results_df, output_dir)
        
        # ステップ6: サマリー表示
        logger.info("\n" + "=" * 80)
        logger.info("SIMULATION COMPLETED")
        logger.info("=" * 80)
        
        print("\n" + "=" * 80)
        print("Bridge Closure Impact Simulation Summary")
        print("=" * 80)
        print(f"Bridges Simulated: {len(results_df)}")
        print(f"\nBaseline Metrics:")
        print(f"  Average Shortest Path: {baseline['avg_shortest_path']:.2f}")
        print(f"  Connected Nodes: {baseline['num_connected_nodes']:,}")
        print(f"  Accessible Bus Stops: {baseline['accessible_bus_stops']}")
        
        print(f"\nAverage Impact per Bridge Closure:")
        print(f"  Path Length Increase: +{results_df['delta_avg_shortest_path'].mean():.2f} ({results_df['pct_path_increase'].mean():.2f}%)")
        print(f"  Nodes Lost: {abs(results_df['delta_connected_nodes'].mean()):.1f} ({results_df['pct_nodes_lost'].mean():.2f}%)")
        print(f"  Bus Stops Lost: {abs(results_df['delta_accessible_bus_stops'].mean()):.1f} ({results_df['pct_bus_stops_lost'].mean():.2f}%)")
        
        print(f"\nTop 5 Most Impactful Bridges:")
        top5 = results_df.nlargest(5, 'delta_avg_shortest_path')
        for rank, (idx, row) in enumerate(top5.iterrows(), 1):
            print(f"  {rank}. {row['bridge_id']}: +{row['delta_avg_shortest_path']:.2f} path length ({row['pct_path_increase']:.1f}% increase)")
        
        print(f"\nOutput Directory: {output_dir}")
        print(f"  - closure_simulation_results.csv")
        print(f"  - closure_impact_report.md")
        print(f"  - closure_impact_distribution.png")
        print(f"  - closure_impact_top10.png")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
