"""
説明文生成モジュール
Bridge Importance Scoring MVP

橋梁の重要度スコアに基づいて人間可読な説明文を生成
"""

import geopandas as gpd
import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class BridgeNarrativeGenerator:
    """橋梁の説明文生成クラス"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 設定辞書
        """
        self.config = config
        self.narrative_config = config.get('narrative', {})
        self.thresholds = self.narrative_config.get('thresholds', {})
        self.risk_config = self.narrative_config.get('risk', {})
        
    def generate_narrative(self, bridge: pd.Series) -> str:
        """
        単一の橋梁に対する説明文を生成
        
        Args:
            bridge: 橋梁データの行（Series）
        
        Returns:
            説明文（日本語）
        """
        parts = []
        
        # 1. 重要度の総合評価
        importance_desc = self._describe_importance_level(bridge)
        parts.append(importance_desc)
        
        # 2. ネットワーク中心性の説明
        centrality_desc = self._describe_centrality(bridge)
        if centrality_desc:
            parts.append(centrality_desc)
        
        # 3. 社会的影響の説明
        social_desc = self._describe_social_impact(bridge)
        if social_desc:
            parts.append(social_desc)
        
        # 4. 公共交通の説明
        transit_desc = self._describe_transit_access(bridge)
        if transit_desc:
            parts.append(transit_desc)
        
        # 5. リスクの説明
        risk_desc = self._describe_risks(bridge)
        if risk_desc:
            parts.append(risk_desc)
        
        # 最終的な説明文
        narrative = " ".join(parts)
        
        return narrative
    
    def _describe_importance_level(self, bridge: pd.Series) -> str:
        """重要度レベルの説明"""
        score = bridge.get('importance_score', 0)
        rank = bridge.get('importance_rank', 0)
        category = bridge.get('importance_category', 'unknown')
        
        if category == 'critical' or score >= self.thresholds.get('critical', 90):
            return f"【最重要橋梁】市内で最上位クラス（ランク{rank}位、スコア{score:.1f}）のボトルネック橋梁です。"
        elif category == 'high' or score >= self.thresholds.get('high', 70):
            return f"【高重要度】高い重要度を持つ橋梁です（ランク{rank}位、スコア{score:.1f}）。"
        elif category == 'medium' or score >= self.thresholds.get('medium', 50):
            return f"【中程度の重要度】中程度の重要度を持つ橋梁です（ランク{rank}位、スコア{score:.1f}）。"
        elif category == 'low' or score >= self.thresholds.get('low', 30):
            return f"【低重要度】比較的重要度が低い橋梁です（ランク{rank}位、スコア{score:.1f}）。"
        else:
            return f"【最低重要度】重要度が最も低いクラスの橋梁です（ランク{rank}位、スコア{score:.1f}）。"
    
    def _describe_centrality(self, bridge: pd.Series) -> str:
        """媒介中心性の説明"""
        betweenness = bridge.get('betweenness', 0)
        
        # 媒介中心性が高い場合
        if betweenness > 0.01:  # 正規化済みの場合の閾値
            return "この橋梁は、多数の交通経路が通過する重要な結節点であり、通行不能になると広範囲に影響が及びます。"
        elif betweenness > 0.001:
            return "ネットワーク上で一定の交通経路が通過しています。"
        else:
            return ""
    
    def _describe_social_impact(self, bridge: pd.Series) -> str:
        """社会的影響の説明"""
        num_public = bridge.get('num_public_facilities', 0)
        num_hospitals = bridge.get('num_hospitals', 0)
        num_schools = bridge.get('num_schools', 0)
        num_buildings = bridge.get('num_buildings', 0)
        
        parts = []
        
        # 公共施設の集中
        public_threshold = self.narrative_config.get('public_facility_threshold', 3)
        if num_public >= public_threshold:
            parts.append(f"周辺に{int(num_public)}箇所の公共施設があり、社会的影響が大きいと考えられます。")
        
        # 病院アクセス
        if num_hospitals > 0:
            parts.append(f"周辺に病院が{int(num_hospitals)}箇所あり、緊急医療アクセスルートとしての役割があります。")
        
        # 学校アクセス
        if num_schools >= 2:
            parts.append(f"周辺に学校が{int(num_schools)}箇所あり、通学路としての利用が想定されます。")
        
        # 建物密度
        if num_buildings >= 10:
            parts.append(f"周辺に{int(num_buildings)}棟の建物があり、住宅密集地へのアクセス路として重要です。")
        
        return " ".join(parts)
    
    def _describe_transit_access(self, bridge: pd.Series) -> str:
        """公共交通アクセスの説明"""
        num_bus_stops = bridge.get('num_bus_stops', 0)
        
        if num_bus_stops >= 3:
            return f"周辺に{int(num_bus_stops)}箇所のバス停があり、公共交通の要衝として機能しています。"
        elif num_bus_stops >= 1:
            return f"周辺にバス停が{int(num_bus_stops)}箇所あります。"
        else:
            return ""
    
    def _describe_risks(self, bridge: pd.Series) -> str:
        """リスク要因の説明"""
        dist_to_river = bridge.get('dist_to_river', float('inf'))
        dist_to_coast = bridge.get('dist_to_coast', float('inf'))
        
        parts = []
        
        # 河川リスク
        if dist_to_river < 50:  # 50m以内
            parts.append("河川を跨いでおり、洪水時の通行不能が広域の分断を引き起こす可能性があります。")
        elif dist_to_river < 100:
            parts.append("河川に近接しており、洪水リスクに留意が必要です。")
        
        # 塩害リスク
        salt_damage_distance = self.risk_config.get('salt_damage_distance', 3000)
        if dist_to_coast < salt_damage_distance:
            if dist_to_coast < 1000:
                parts.append(f"海岸線から{int(dist_to_coast)}mと非常に近く、飛来塩分による腐食劣化リスクが極めて高い状況です。")
            elif dist_to_coast < 2000:
                parts.append(f"海岸線から{int(dist_to_coast)}mの距離にあり、塩害による劣化リスクが高いと考えられます。")
            else:
                parts.append(f"海岸線から{int(dist_to_coast/1000):.1f}km圏内にあり、塩害リスクに留意が必要です。")
        
        return " ".join(parts)
    
    def generate_summary_statistics(self, bridges: gpd.GeoDataFrame) -> Dict:
        """
        全橋梁の統計サマリーを生成
        
        Args:
            bridges: スコア付き橋梁データ
        
        Returns:
            統計サマリー辞書
        """
        summary = {
            'total_bridges': len(bridges),
            'score_statistics': {
                'mean': float(bridges['importance_score'].mean()),
                'median': float(bridges['importance_score'].median()),
                'std': float(bridges['importance_score'].std()),
                'min': float(bridges['importance_score'].min()),
                'max': float(bridges['importance_score'].max())
            },
            'category_distribution': bridges['importance_category'].value_counts().to_dict(),
            'top_10_bridges': []
        }
        
        # トップ10橋梁
        top10 = bridges.nlargest(10, 'importance_score')
        for idx, bridge in top10.iterrows():
            summary['top_10_bridges'].append({
                'bridge_id': bridge['bridge_id'],
                'name': bridge.get('name', bridge['bridge_id']),
                'rank': int(bridge['importance_rank']),
                'score': float(bridge['importance_score']),
                'category': bridge['importance_category']
            })
        
        return summary
    
    def generate_report(self, bridges: gpd.GeoDataFrame) -> str:
        """
        全体レポートを生成
        
        Args:
            bridges: スコア付き橋梁データ
        
        Returns:
            マークダウン形式のレポート
        """
        report_lines = [
            "# 橋梁重要度評価レポート",
            "",
            f"## 概要",
            f"- 評価対象橋梁数: {len(bridges)}橋",
            f"- 平均重要度スコア: {bridges['importance_score'].mean():.2f}",
            f"- スコア範囲: {bridges['importance_score'].min():.2f} 〜 {bridges['importance_score'].max():.2f}",
            "",
            "## カテゴリ別分布",
            ""
        ]
        
        # カテゴリ別統計
        category_counts = bridges['importance_category'].value_counts()
        for cat in ['critical', 'high', 'medium', 'low', 'very_low']:
            count = category_counts.get(cat, 0)
            pct = 100 * count / len(bridges) if len(bridges) > 0 else 0
            report_lines.append(f"- **{cat.capitalize()}**: {count}橋 ({pct:.1f}%)")
        
        report_lines.extend([
            "",
            "## トップ10重要橋梁",
            ""
        ])
        
        # トップ10
        top10 = bridges.nlargest(10, 'importance_score')
        for rank, (idx, bridge) in enumerate(top10.iterrows(), 1):
            bridge_name = bridge.get('name', bridge['bridge_id'])
            score = bridge['importance_score']
            narrative = self.generate_narrative(bridge)
            
            report_lines.extend([
                f"### {rank}位: {bridge_name} (スコア: {score:.1f})",
                "",
                narrative,
                ""
            ])
        
        report_lines.extend([
            "## リスク要因分析",
            ""
        ])
        
        # リスク統計
        num_river_risk = (bridges['dist_to_river'] < 50).sum()
        num_coast_risk = (bridges['dist_to_coast'] < 3000).sum()
        
        report_lines.extend([
            f"- 河川近接橋梁（50m以内）: {num_river_risk}橋",
            f"- 塩害リスク橋梁（海岸3km以内）: {num_coast_risk}橋",
            ""
        ])
        
        return "\n".join(report_lines)


def generate_narratives_for_all(
    bridges: gpd.GeoDataFrame,
    config: Dict
) -> gpd.GeoDataFrame:
    """
    全橋梁の説明文を生成する便利関数
    
    Args:
        bridges: スコア付き橋梁データ
        config: 設定辞書
    
    Returns:
        説明文付き橋梁データ
    """
    generator = BridgeNarrativeGenerator(config)
    
    logger.info("Generating narratives for all bridges...")
    
    bridges = bridges.copy()
    bridges['narrative'] = bridges.apply(generator.generate_narrative, axis=1)
    
    logger.info("Narratives generated successfully")
    
    return bridges
