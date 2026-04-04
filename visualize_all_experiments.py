"""
全実験（exp2-13）の統一的な可視化
SAGE最適化実験（exp2-10）+ Attention実験（exp11-13）の比較
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 実験結果データ
experiments = [
    # SAGE最適化実験（exp2-10）
    {"id": "exp2", "name": "SAGE mean\n(baseline)", "r2": 0.8551, "mae": 1.29, "gpu_mb": 900, "time_s": 13.84, "category": "SAGE最適化"},
    {"id": "exp3", "name": "SAGE max\n(BEST)", "r2": 0.8939, "mae": 1.30, "gpu_mb": 900, "time_s": 26.66, "category": "SAGE最適化"},
    {"id": "exp4", "name": "SAGE sum", "r2": 0.8468, "mae": 1.59, "gpu_mb": 900, "time_s": 21.08, "category": "SAGE最適化"},
    {"id": "exp5", "name": "4 layers", "r2": 0.8229, "mae": 1.44, "gpu_mb": 900, "time_s": 17.57, "category": "SAGE最適化"},
    {"id": "exp6", "name": "5 layers", "r2": 0.8476, "mae": 1.39, "gpu_mb": 900, "time_s": 20.79, "category": "SAGE最適化"},
    {"id": "exp7", "name": "dropout=0.2", "r2": 0.8491, "mae": 1.31, "gpu_mb": 900, "time_s": 13.46, "category": "SAGE最適化"},
    {"id": "exp8", "name": "dropout=0.4", "r2": 0.8537, "mae": 1.35, "gpu_mb": 882, "time_s": 12.19, "category": "SAGE最適化"},
    {"id": "exp9", "name": "hidden=256", "r2": 0.8645, "mae": 1.18, "gpu_mb": 2401, "time_s": 23.44, "category": "SAGE最適化"},
    {"id": "exp10", "name": "hidden=512", "r2": 0.8886, "mae": 1.10, "gpu_mb": 4810, "time_s": 45.58, "category": "SAGE最適化"},
    
    # Attention実験（exp11-13）
    {"id": "exp11", "name": "Simple\nAttention", "r2": 0.8517, "mae": 1.32, "gpu_mb": 2201, "time_s": 24.42, "category": "Attention追加"},
    {"id": "exp12", "name": "GATv2Style\nAttention", "r2": 0.8338, "mae": 1.38, "gpu_mb": 3205, "time_s": 23.87, "category": "Attention追加"},
    {"id": "exp13", "name": "Metapath\nAttention", "r2": 0.8741, "mae": 1.22, "gpu_mb": 3863, "time_s": 29.31, "category": "Attention追加"},
]

df = pd.DataFrame(experiments)

# カテゴリ別の色設定
category_colors = {
    "SAGE最適化": "steelblue",
    "Attention追加": "coral"
}

# 図の作成
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Bridge Importance HGNN All Experiments Comparison (exp2-13)", fontsize=16, fontweight='bold')

# (1) R²スコア比較
ax1 = axes[0, 0]
colors = [category_colors[cat] for cat in df['category']]
bars = ax1.bar(range(len(df)), df['r2'], color=colors, alpha=0.7, edgecolor='black')

# exp3を強調
best_idx = df['r2'].idxmax()
bars[best_idx].set_color('gold')
bars[best_idx].set_edgecolor('darkred')
bars[best_idx].set_linewidth(3)

ax1.axhline(y=0.8939, color='darkred', linestyle='--', linewidth=2, label='exp3 (BEST): R²=0.8939')
ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax1.set_title('(1) R² Score Comparison', fontsize=13, fontweight='bold')
ax1.set_xticks(range(len(df)))
ax1.set_xticklabels(df['name'], rotation=45, ha='right', fontsize=9)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0.8, 0.91)

# 値ラベル
for i, (idx, row) in enumerate(df.iterrows()):
    ax1.text(i, row['r2'] + 0.002, f"{row['r2']:.4f}", 
             ha='center', va='bottom', fontsize=8, fontweight='bold')

# (2) MAE比較
ax2 = axes[0, 1]
bars2 = ax2.bar(range(len(df)), df['mae'], color=colors, alpha=0.7, edgecolor='black')
bars2[best_idx].set_color('gold')
bars2[best_idx].set_edgecolor('darkred')
bars2[best_idx].set_linewidth(3)

ax2.axhline(y=1.30, color='darkred', linestyle='--', linewidth=2, label='exp3 (BEST): MAE=1.30')
ax2.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
ax2.set_title('(2) MAE Comparison (lower is better)', fontsize=13, fontweight='bold')
ax2.set_xticks(range(len(df)))
ax2.set_xticklabels(df['name'], rotation=45, ha='right', fontsize=9)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

for i, (idx, row) in enumerate(df.iterrows()):
    ax2.text(i, row['mae'] + 0.02, f"{row['mae']:.2f}", 
             ha='center', va='bottom', fontsize=8, fontweight='bold')

# (3) GPU Memory使用量
ax3 = axes[1, 0]
bars3 = ax3.bar(range(len(df)), df['gpu_mb'], color=colors, alpha=0.7, edgecolor='black')
bars3[best_idx].set_color('gold')
bars3[best_idx].set_edgecolor('darkred')
bars3[best_idx].set_linewidth(3)

ax3.axhline(y=900, color='darkred', linestyle='--', linewidth=2, label='exp3 (BEST): 900MB')
ax3.set_ylabel('GPU Memory (MB)', fontsize=12, fontweight='bold')
ax3.set_title('(3) GPU Memory Usage (lower is better)', fontsize=13, fontweight='bold')
ax3.set_xticks(range(len(df)))
ax3.set_xticklabels(df['name'], rotation=45, ha='right', fontsize=9)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

for i, (idx, row) in enumerate(df.iterrows()):
    ax3.text(i, row['gpu_mb'] + 100, f"{int(row['gpu_mb'])}MB", 
             ha='center', va='bottom', fontsize=8, fontweight='bold')

# (4) Training Time
ax4 = axes[1, 1]
bars4 = ax4.bar(range(len(df)), df['time_s'], color=colors, alpha=0.7, edgecolor='black')
bars4[best_idx].set_color('gold')
bars4[best_idx].set_edgecolor('darkred')
bars4[best_idx].set_linewidth(3)

ax4.axhline(y=26.66, color='darkred', linestyle='--', linewidth=2, label='exp3 (BEST): 26.66s')
ax4.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
ax4.set_title('(4) Training Time (lower is better)', fontsize=13, fontweight='bold')
ax4.set_xticks(range(len(df)))
ax4.set_xticklabels(df['name'], rotation=45, ha='right', fontsize=9)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

for i, (idx, row) in enumerate(df.iterrows()):
    ax4.text(i, row['time_s'] + 1, f"{row['time_s']:.1f}s", 
             ha='center', va='bottom', fontsize=8, fontweight='bold')

# カテゴリ凡例を追加
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='steelblue', edgecolor='black', label='SAGE Optimization Experiments (exp2-10)'),
    Patch(facecolor='coral', edgecolor='black', label='Attention Addition Experiments (exp11-13)'),
    Patch(facecolor='gold', edgecolor='darkred', linewidth=3, label='BEST: exp3 (SAGE max)')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11, 
           bbox_to_anchor=(0.5, -0.02), frameon=True, shadow=True)

# 上下の図の間隔を広げる（1cm ≈ hspace=0.4）
plt.subplots_adjust(hspace=0.4)
plt.tight_layout(rect=[0, 0.02, 1, 0.97])

# 保存
output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "all_experiments_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ 可視化を保存: {output_path}")

plt.show()
