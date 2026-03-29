"""
可視化モジュール
Bridge Importance Scoring MVP

橋梁の重要度スコアを地図上に可視化
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Optional
import folium
from folium import plugins
import pandas as pd

logger = logging.getLogger(__name__)

plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']  # 日本語フォント


class BridgeVisualizer:
    """橋梁可視化クラス"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 設定辞書
        """
        self.config = config
        self.output_dir = Path(config['data']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_score_distribution(
        self, 
        bridges: gpd.GeoDataFrame,
        save_path: Optional[str] = None
    ):
        """
        スコア分布のヒストグラムを作成
        
        Args:
            bridges: スコア付き橋梁データ
            save_path: 保存先パス（Noneの場合は表示のみ）
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 重要度スコアの分布
        ax1 = axes[0, 0]
        bridges['importance_score'].hist(bins=50, ax=ax1, edgecolor='black')
        ax1.set_xlabel('重要度スコア', fontsize=12)
        ax1.set_ylabel('橋梁数', fontsize=12)
        ax1.set_title('重要度スコアの分布', fontsize=14, fontweight='bold')
        ax1.axvline(bridges['importance_score'].mean(), color='red', 
                   linestyle='--', label=f'平均: {bridges["importance_score"].mean():.1f}')
        ax1.legend()
        
        # 2. 媒介中心性の分布
        ax2 = axes[0, 1]
        bridges['betweenness'].hist(bins=50, ax=ax2, edgecolor='black', log=True)
        ax2.set_xlabel('媒介中心性', fontsize=12)
        ax2.set_ylabel('橋梁数（対数）', fontsize=12)
        ax2.set_title('媒介中心性の分布', fontsize=14, fontweight='bold')
        
        # 3. カテゴリ別分布
        ax3 = axes[1, 0]
        category_counts = bridges['importance_category'].value_counts()
        category_order = ['critical', 'high', 'medium', 'low', 'very_low']
        category_counts = category_counts.reindex(category_order, fill_value=0)
        category_counts.plot(kind='bar', ax=ax3, color='steelblue', edgecolor='black')
        ax3.set_xlabel('重要度カテゴリ', fontsize=12)
        ax3.set_ylabel('橋梁数', fontsize=12)
        ax3.set_title('カテゴリ別橋梁数', fontsize=14, fontweight='bold')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
        
        # 4. スコアと媒介中心性の相関
        ax4 = axes[1, 1]
        ax4.scatter(bridges['betweenness'], bridges['importance_score'], 
                   alpha=0.5, s=30)
        ax4.set_xlabel('媒介中心性', fontsize=12)
        ax4.set_ylabel('重要度スコア', fontsize=12)
        ax4.set_title('媒介中心性 vs 重要度スコア', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Score distribution plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_interactive_map(
        self,
        bridges: gpd.GeoDataFrame,
        save_path: Optional[str] = None
    ) -> folium.Map:
        """
        Foliumを使った対話的地図を作成
        
        Args:
            bridges: スコア付き橋梁データ（WGS84）
            save_path: 保存先HTMLパス
        
        Returns:
            Foliumマップオブジェクト
        """
        logger.info("Creating interactive map...")
        
        # WGS84に変換（まだの場合）
        if bridges.crs != 'EPSG:4326':
            bridges = bridges.to_crs('EPSG:4326')
        
        # 地図の中心座標
        center_lat = bridges.geometry.y.mean()
        center_lon = bridges.geometry.x.mean()
        
        # ベースマップ
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles='OpenStreetMap'
        )
        
        # カラーマップ（重要度スコアベース）
        min_score = bridges['importance_score'].min()
        max_score = bridges['importance_score'].max()
        
        def get_color(score):
            """スコアに基づく色を取得"""
            normalized = (score - min_score) / (max_score - min_score + 1e-9)
            if normalized >= 0.8:
                return 'red'
            elif normalized >= 0.6:
                return 'orange'
            elif normalized >= 0.4:
                return 'yellow'
            elif normalized >= 0.2:
                return 'lightgreen'
            else:
                return 'green'
        
        # マーカーの追加
        for idx, bridge in bridges.iterrows():
            score = bridge['importance_score']
            rank = bridge.get('importance_rank', 0)
            name = bridge.get('name', bridge['bridge_id'])
            narrative = bridge.get('narrative', '説明なし')
            
            # ポップアップ内容
            popup_html = f"""
            <div style="width:300px">
                <h4>{name}</h4>
                <hr>
                <b>ランク:</b> {rank}位<br>
                <b>スコア:</b> {score:.1f}<br>
                <b>媒介中心性:</b> {bridge['betweenness']:.6f}<br>
                <b>カテゴリ:</b> {bridge['importance_category']}<br>
                <br>
                <b>公共施設:</b> {int(bridge.get('num_public_facilities', 0))}箇所<br>
                <b>病院:</b> {int(bridge.get('num_hospitals', 0))}箇所<br>
                <b>学校:</b> {int(bridge.get('num_schools', 0))}箇所<br>
                <b>バス停:</b> {int(bridge.get('num_bus_stops', 0))}箇所<br>
                <br>
                <b>河川距離:</b> {bridge.get('dist_to_river', 0):.0f}m<br>
                <b>海岸距離:</b> {bridge.get('dist_to_coast', 0):.0f}m<br>
                <hr>
                <p style="font-size:11px">{narrative}</p>
            </div>
            """
            
            # マーカーサイズ（スコアに比例）
            radius = 3 + (score / max_score) * 7
            
            folium.CircleMarker(
                location=[bridge.geometry.y, bridge.geometry.x],
                radius=radius,
                popup=folium.Popup(popup_html, max_width=350),
                tooltip=f"{name} (スコア: {score:.1f})",
                color=get_color(score),
                fill=True,
                fillColor=get_color(score),
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
        
        # レジェンドの追加
        legend_html = '''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 180px; height: 180px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p style="margin:0; font-weight:bold;">橋梁重要度</p>
        <hr style="margin:5px 0;">
        <p style="margin:5px 0;"><span style="color:red;">●</span> 最重要 (80-100)</p>
        <p style="margin:5px 0;"><span style="color:orange;">●</span> 高 (60-80)</p>
        <p style="margin:5px 0;"><span style="color:yellow;">●</span> 中 (40-60)</p>
        <p style="margin:5px 0;"><span style="color:lightgreen;">●</span> 低 (20-40)</p>
        <p style="margin:5px 0;"><span style="color:green;">●</span> 最低 (0-20)</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # フルスクリーンボタン
        plugins.Fullscreen().add_to(m)
        
        # 保存
        if save_path:
            m.save(save_path)
            logger.info(f"Interactive map saved to {save_path}")
        
        return m
    
    def plot_top_bridges_map(
        self,
        bridges: gpd.GeoDataFrame,
        top_n: int = 20,
        save_path: Optional[str] = None
    ):
        """
        トップN橋梁の地図プロット（静的）
        
        Args:
            bridges: スコア付き橋梁データ
            top_n: 表示する橋梁数
            save_path: 保存先パス
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # 全橋梁（薄いグレー）
        bridges.plot(ax=ax, color='lightgray', markersize=10, alpha=0.3, label='全橋梁')
        
        # トップN橋梁
        top_bridges = bridges.nlargest(top_n, 'importance_score')
        
        # スコアに基づく色分け
        top_bridges.plot(
            ax=ax,
            column='importance_score',
            cmap='YlOrRd',
            markersize=100,
            alpha=0.8,
            legend=True,
            legend_kwds={'label': '重要度スコア', 'orientation': 'horizontal'}
        )
        
        # 橋梁名のラベル（トップ10のみ）
        for idx, bridge in top_bridges.head(10).iterrows():
            name = bridge.get('name', bridge['bridge_id'])
            ax.annotate(
                name,
                xy=(bridge.geometry.x, bridge.geometry.y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
            )
        
        ax.set_xlabel('経度', fontsize=12)
        ax.set_ylabel('緯度', fontsize=12)
        ax.set_title(f'トップ{top_n}重要橋梁の位置', fontsize=16, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Top bridges map saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def visualize_results(bridges: gpd.GeoDataFrame, config: Dict):
    """
    結果を可視化する便利関数
    
    Args:
        bridges: スコア付き橋梁データ
        config: 設定辞書
    """
    visualizer = BridgeVisualizer(config)
    output_dir = Path(config['data']['output_dir'])
    
    logger.info("Generating visualizations...")
    
    # 1. スコア分布プロット
    visualizer.plot_score_distribution(
        bridges,
        save_path=output_dir / 'score_distribution.png'
    )
    
    # 2. トップ橋梁マップ（静的）
    visualizer.plot_top_bridges_map(
        bridges,
        top_n=20,
        save_path=output_dir / 'top20_bridges_map.png'
    )
    
    # 3. 対話的地図（HTML）
    visualizer.create_interactive_map(
        bridges,
        save_path=output_dir / 'interactive_bridge_map.html'
    )
    
    logger.info("All visualizations completed")
