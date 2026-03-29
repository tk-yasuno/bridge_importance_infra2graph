"""
Bridge Closure Impact Visualization
ネットワークへの影響度指標の可視化
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

# 出力ディレクトリ
OUTPUT_DIR = Path('output/closure_simulation_simple')

def load_data():
    """結果データの読み込み"""
    df = pd.read_csv(OUTPUT_DIR / 'closure_results.csv')
    print(f"Loaded {len(df)} bridges")
    return df

def plot_degree_distribution(df):
    """度数分布のヒストグラム"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # カテゴリ別に色分け
    colors = {'high': '#d62728', 'medium': '#ff7f0e', 'low': '#2ca02c'}
    
    for category in ['high', 'medium', 'low']:
        data = df[df['importance_category'] == category]['degree']
        ax.hist(data, bins=20, alpha=0.6, label=category.upper(), 
                color=colors[category], edgecolor='black')
    
    ax.set_xlabel('ネットワーク度数 (Degree)', fontsize=12)
    ax.set_ylabel('橋梁数', fontsize=12)
    ax.set_title('橋梁のネットワーク度数分布', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'degree_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_category_comparison(df):
    """カテゴリ別の平均度数比較"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 平均度数
    category_stats = df.groupby('importance_category').agg({
        'degree': 'mean',
        'component_increase': 'mean',
        'bridge_id': 'count'
    }).reindex(['high', 'medium', 'low'])
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    
    # グラフ1: 平均度数
    bars1 = ax1.bar(category_stats.index.str.upper(), category_stats['degree'], 
                     color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('平均ネットワーク度数', fontsize=12)
    ax1.set_title('カテゴリ別平均ネットワーク度数', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 値を表示
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # グラフ2: 平均コンポーネント増加
    bars2 = ax2.bar(category_stats.index.str.upper(), category_stats['component_increase'], 
                     color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('平均コンポーネント増加数', fontsize=12)
    ax2.set_title('カテゴリ別平均コンポーネント増加数', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 値を表示
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'category_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_score_vs_degree(df):
    """重要度スコアと度数の散布図"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {'high': '#d62728', 'medium': '#ff7f0e', 'low': '#2ca02c'}
    
    for category in ['high', 'medium', 'low']:
        data = df[df['importance_category'] == category]
        ax.scatter(data['importance_score'], data['degree'], 
                  label=category.upper(), color=colors[category],
                  s=100, alpha=0.7, edgecolors='black', linewidth=1)
    
    # 注目橋梁をアノテーション
    # 最高スコア橋梁
    highest_score = df.loc[df['importance_score'].idxmax()]
    ax.annotate(f"{highest_score['facility_name']}\n(最高スコア)", 
                xy=(highest_score['importance_score'], highest_score['degree']),
                xytext=(10, 20), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 最高度数橋梁
    highest_degree = df.loc[df['degree'].idxmax()]
    ax.annotate(f"{highest_degree['facility_name']}\n(最高度数)", 
                xy=(highest_degree['importance_score'], highest_degree['degree']),
                xytext=(10, -30), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', fc='cyan', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax.set_xlabel('重要度スコア', fontsize=12)
    ax.set_ylabel('ネットワーク度数', fontsize=12)
    ax.set_title('重要度スコア vs ネットワーク度数の関係', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'score_vs_degree.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_top_bridges(df, top_n=10):
    """トップN橋梁の度数比較"""
    top_bridges = df.nlargest(top_n, 'degree')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {'high': '#d62728', 'medium': '#ff7f0e', 'low': '#2ca02c'}
    bar_colors = [colors[cat] for cat in top_bridges['importance_category']]
    
    y_pos = np.arange(len(top_bridges))
    bars = ax.barh(y_pos, top_bridges['degree'], color=bar_colors, 
                    edgecolor='black', linewidth=1.5)
    
    # 橋梁名とカテゴリを表示
    labels = [f"{row['facility_name']}\n({row['importance_category'].upper()}, スコア: {row['importance_score']:.1f})" 
              for _, row in top_bridges.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    
    ax.set_xlabel('ネットワーク度数', fontsize=12)
    ax.set_title(f'ネットワーク度数トップ{top_n}橋梁', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 値を表示
    for i, (bar, degree) in enumerate(zip(bars, top_bridges['degree'])):
        ax.text(degree + 50, bar.get_y() + bar.get_height()/2., 
                f'{degree:,}',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'top_bridges_degree.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_component_impact(df):
    """コンポーネント増加の影響分析"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # グラフ1: コンポーネント増加のヒストグラム
    ax1.hist(df['component_increase'], bins=30, color='steelblue', 
             edgecolor='black', alpha=0.7)
    ax1.set_xlabel('コンポーネント増加数', fontsize=12)
    ax1.set_ylabel('橋梁数', fontsize=12)
    ax1.set_title('橋梁閉鎖時のコンポーネント増加分布', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(df['component_increase'].mean(), color='red', 
                linestyle='--', linewidth=2, label=f'平均: {df["component_increase"].mean():.1f}')
    ax1.legend()
    
    # グラフ2: 度数とコンポーネント増加の関係
    colors = {'high': '#d62728', 'medium': '#ff7f0e', 'low': '#2ca02c'}
    
    for category in ['high', 'medium', 'low']:
        data = df[df['importance_category'] == category]
        ax2.scatter(data['degree'], data['component_increase'], 
                   label=category.upper(), color=colors[category],
                   s=80, alpha=0.6, edgecolors='black', linewidth=1)
    
    ax2.set_xlabel('ネットワーク度数', fontsize=12)
    ax2.set_ylabel('コンポーネント増加数', fontsize=12)
    ax2.set_title('度数とネットワーク分断の関係', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'component_impact.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_summary_dashboard(df):
    """総合ダッシュボード"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    colors_cat = {'high': '#d62728', 'medium': '#ff7f0e', 'low': '#2ca02c'}
    
    # 1. カテゴリ分布（円グラフ）
    ax1 = fig.add_subplot(gs[0, 0])
    category_counts = df['importance_category'].value_counts().reindex(['high', 'medium', 'low'])
    ax1.pie(category_counts, labels=[f'{cat.upper()}\n({count}橋)' for cat, count in category_counts.items()],
            colors=['#d62728', '#ff7f0e', '#2ca02c'], autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax1.set_title('カテゴリ分布', fontsize=12, fontweight='bold')
    
    # 2. 度数の統計（ボックスプロット）
    ax2 = fig.add_subplot(gs[0, 1:])
    category_order = ['high', 'medium', 'low']
    df_plot = df.copy()
    df_plot['category_label'] = df_plot['importance_category'].str.upper()
    
    bp = ax2.boxplot([df[df['importance_category']==cat]['degree'].values 
                       for cat in category_order],
                      labels=[cat.upper() for cat in category_order],
                      patch_artist=True, showmeans=True)
    
    for patch, color in zip(bp['boxes'], ['#d62728', '#ff7f0e', '#2ca02c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('ネットワーク度数', fontsize=11)
    ax2.set_title('カテゴリ別度数分布', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. スコアと度数の散布図
    ax3 = fig.add_subplot(gs[1, :])
    for category in ['high', 'medium', 'low']:
        data = df[df['importance_category'] == category]
        ax3.scatter(data['importance_score'], data['degree'], 
                   label=category.upper(), color=colors_cat[category],
                   s=100, alpha=0.7, edgecolors='black', linewidth=1)
    
    ax3.set_xlabel('重要度スコア', fontsize=11)
    ax3.set_ylabel('ネットワーク度数', fontsize=11)
    ax3.set_title('重要度スコア vs ネットワーク度数', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. トップ10橋梁
    ax4 = fig.add_subplot(gs[2, :])
    top_10 = df.nlargest(10, 'degree')
    bar_colors = [colors_cat[cat] for cat in top_10['importance_category']]
    
    y_pos = np.arange(len(top_10))
    ax4.barh(y_pos, top_10['degree'], color=bar_colors, 
             edgecolor='black', linewidth=1.5, alpha=0.8)
    
    labels = [f"{row['facility_name']} ({row['importance_category'].upper()})" 
              for _, row in top_10.iterrows()]
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(labels, fontsize=9)
    ax4.set_xlabel('ネットワーク度数', fontsize=11)
    ax4.set_title('度数トップ10橋梁', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 全体タイトル
    fig.suptitle('橋梁閉鎖影響度分析ダッシュボード', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    output_path = OUTPUT_DIR / 'summary_dashboard.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    print("=" * 80)
    print("Bridge Closure Impact Visualization")
    print("=" * 80)
    
    # データ読み込み
    df = load_data()
    
    # 各種可視化
    print("\nGenerating visualizations...")
    plot_degree_distribution(df)
    plot_category_comparison(df)
    plot_score_vs_degree(df)
    plot_top_bridges(df, top_n=10)
    plot_component_impact(df)
    plot_summary_dashboard(df)
    
    print("\n" + "=" * 80)
    print("All visualizations completed!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == '__main__':
    main()
