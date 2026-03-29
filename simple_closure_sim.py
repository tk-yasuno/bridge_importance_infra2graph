"""
Simple Bridge Closure Simulation
Very Low除外後の41橋について、閉鎖シナリオのシミュレーション

軽量版：グラフの最大接続コンポーネント内のみで処理
"""

import pandas as pd
import pickle
import logging
import sys
from pathlib import Path
import networkx as nx
from datetime import datetime
import numpy as np

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('simple_closure_sim.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def load_data():
    """グラフとスコアをロード"""
    logger.info("Loading data...")
    
    with open('output/bridge_importance/heterogeneous_graph.pkl', 'rb') as f:
        G = pickle.load(f)
    
    scores_df = pd.read_csv('output/bridge_importance/bridge_importance_scores.csv')
    
    logger.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    logger.info(f"Scores: {len(scores_df)} bridges")
    
    return G, scores_df


def prepare_graph(G):
    """最大接続コンポーネントを抽出"""
    if not nx.is_connected(G):
        logger.warning("Graph not connected. Using largest component...")
        largest_cc = max(nx.connected_components(G), key=len)
        G_cc = G.subgraph(largest_cc).copy()
        logger.info(f"Largest component: {G_cc.number_of_nodes()} nodes, {G_cc.number_of_edges()} edges")
        return G_cc
    return G


def filter_target_bridges(scores_df):
    """Very Low除外後の41橋を抽出"""
    target = scores_df[scores_df['importance_category'] != 'very_low'].copy()
    target = target.sort_values('importance_score', ascending=False).reset_index(drop=True)
    
    logger.info(f"\n【Target Bridges: {len(target)}】")
    for cat in ['high', 'medium', 'low']:
        count = len(target[target['importance_category'] == cat])
        logger.info(f"  {cat}: {count} bridges")
    
    return target


def calculate_closure_impact(G, bridge_id, scores_df):
    """
    橋梁閉鎖時の影響を計算
    
    指標：
    - 接続ノード数（度数）
    - 接続された建物数
    - 接続されたバス停数
    - 切断後のコンポーネント増加
    """
    try:
        # ノード名取得
        bridge_node = f"BR_{bridge_id.split('_')[1]}" if not bridge_id.startswith('BR_') else bridge_id
        
        if bridge_node not in G.nodes():
            logger.warning(f"Bridge {bridge_node} not in graph")
            return None
        
        # 閉鎖前の度数
        degree = G.degree(bridge_node)
        neighbors = list(G.neighbors(bridge_node))
        
        # 隣接ノードのタイプ分類
        buildings = sum(1 for n in neighbors if str(n).startswith('BE_'))
        bus_stops = sum(1 for n in neighbors if str(n).startswith('BS_'))
        
        # コンポーネント数の変化
        G_closed = G.copy()
        G_closed.remove_node(bridge_node)
        
        components_before = nx.number_connected_components(G)
        components_after = nx.number_connected_components(G_closed)
        
        return {
            'bridge_id': bridge_id,
            'degree': degree,
            'num_neighbors': len(neighbors),
            'buildings': buildings,
            'bus_stops': bus_stops,
            'component_increase': components_after - components_before,
        }
    
    except Exception as e:
        logger.error(f"Error for {bridge_id}: {e}")
        return None


def run_simulation(G, target_bridges):
    """シミュレーション実行"""
    results = []
    
    logger.info(f"\n【Running Simulation: {len(target_bridges)} bridges】")
    
    for idx, row in target_bridges.iterrows():
        bridge_id = row['bridge_id']
        
        impact = calculate_closure_impact(G, bridge_id, target_bridges)
        
        if impact:
            impact['facility_name'] = row['施設名']
            impact['importance_score'] = row['importance_score']
            impact['importance_rank'] = row['importance_rank']
            impact['importance_category'] = row['importance_category']
            results.append(impact)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"  Completed: {idx + 1}/{len(target_bridges)}")
    
    return pd.DataFrame(results)


def save_results(results_df):
    """結果を保存"""
    output_dir = Path('output/closure_simulation_simple')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV保存
    csv_path = output_dir / 'closure_results.csv'
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"\nSaved results: {csv_path}")
    
    # トップ10（度数別）
    top_degree = results_df.nlargest(10, 'degree')
    top_path = output_dir / 'top10_by_degree.csv'
    top_degree.to_csv(top_path, index=False, encoding='utf-8-sig')
    
    # Markdownレポート
    report_path = output_dir / 'closure_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Bridge Closure Impact Analysis\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Target Bridges**: {len(results_df)} (Low + Medium + High)\n")
        f.write(f"- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d')}\n\n")
        
        f.write("## Category Distribution\n\n")
        for cat in ['high', 'medium', 'low']:
            cat_data = results_df[results_df['importance_category'] == cat]
            if len(cat_data) > 0:
                f.write(f"### {cat.upper()}\n\n")
                f.write(f"- **Count**: {len(cat_data)}\n")
                f.write(f"- **Avg Degree**: {cat_data['degree'].mean():.1f}\n")
                f.write(f"- **Avg Neighbors**: {cat_data['num_neighbors'].mean():.1f}\n")
                f.write(f"- **Total Buildings**: {cat_data['buildings'].sum():.0f}\n")
                f.write(f"- **Total Bus Stops**: {cat_data['bus_stops'].sum():.0f}\n\n")
        
        f.write("## Top 10 by Network Degree\n\n")
        top10 = results_df.nlargest(10, 'degree')
        for rank, (idx, row) in enumerate(top10.iterrows(), 1):
            f.write(f"### {rank}. {row['bridge_id']}: {row['facility_name']}\n\n")
            f.write(f"- **Category**: {row['importance_category'].upper()}\n")
            f.write(f"- **Score**: {row['importance_score']:.1f}\n")
            f.write(f"- **Degree**: {row['degree']:.0f}\n")
            f.write(f"- **Neighbors**: {row['num_neighbors']:.0f}\n")
            f.write(f"- **Buildings**: {row['buildings']:.0f}\n")
            f.write(f"- **Bus Stops**: {row['bus_stops']:.0f}\n\n")
    
    logger.info(f"Saved report: {report_path}")
    
    return output_dir


def main():
    logger.info("="*80)
    logger.info("Bridge Closure Impact Simulation (Simple)")
    logger.info("="*80)
    
    # データロード
    G, scores_df = load_data()
    
    # グラフ準備
    G = prepare_graph(G)
    
    # 対象橋梁抽出
    target_bridges = filter_target_bridges(scores_df)
    
    # シミュレーション実行
    results_df = run_simulation(G, target_bridges)
    
    # 統計情報
    logger.info(f"\n【Simulation Results】")
    logger.info(f"Average degree: {results_df['degree'].mean():.1f}")
    logger.info(f"Max degree: {results_df['degree'].max():.0f}")
    logger.info(f"Total buildings affected: {results_df['buildings'].sum():.0f}")
    logger.info(f"Total bus stops affected: {results_df['bus_stops'].sum():.0f}")
    
    # 結果保存
    output_dir = save_results(results_df)
    
    logger.info("\n" + "="*80)
    logger.info("Simulation completed successfully!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
