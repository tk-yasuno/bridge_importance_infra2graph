"""
データローダーモジュール
Bridge Importance Scoring MVP

橋梁リスト、河川、海岸線などのデータを読み込む
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
from shapely.geometry import Point

logger = logging.getLogger(__name__)


class BridgeDataLoader:
    """橋梁および関連データのローダークラス"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 設定辞書（YAMLから読み込み）
        """
        self.config = config
        self.data_paths = config['data']
        self.crs = config['crs']
        
    def load_bridge_data(self) -> gpd.GeoDataFrame:
        """
        橋梁リストを読み込み、GeoDataFrameに変換
        
        Returns:
            橋梁データのGeoDataFrame
        """
        logger.info(f"Loading bridge data from {self.data_paths['bridge_list']}")
        
        # Excelファイルの読み込み
        df = pd.read_excel(self.data_paths['bridge_list'])
        
        logger.info(f"Loaded {len(df)} bridges")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # 市町村でフィルタリング（設定がある場合）
        filter_config = self.config.get('data', {}).get('filter', {})
        target_city = filter_config.get('city')
        city_column = filter_config.get('city_column', '市町村')
        
        if target_city and city_column in df.columns:
            original_count = len(df)
            df = df[df[city_column].str.contains(target_city, na=False)]
            logger.info(f"Filtered by city '{target_city}': {len(df)} bridges (from {original_count})")
        elif target_city:
            logger.warning(f"City column '{city_column}' not found. Available columns: {df.columns.tolist()}")
        
        # 座標カラムの検出（一般的なカラム名のバリエーションに対応）
        lon_col = self._find_column(df, ['経度', 'longitude', 'lon', 'x', 'X', 'Longitude'])
        lat_col = self._find_column(df, ['緯度', 'latitude', 'lat', 'y', 'Y', 'Latitude'])
        
        if lon_col is None or lat_col is None:
            # 座標が見つからない場合、最初の2つの数値カラムを使用
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) >= 2:
                lon_col, lat_col = numeric_cols[0], numeric_cols[1]
                logger.warning(f"Using {lon_col} and {lat_col} as coordinates")
            else:
                raise ValueError("Cannot find coordinate columns in bridge data")
        
        # ジオメトリの作成
        geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
        
        # GeoDataFrameの作成
        gdf = gpd.GeoDataFrame(
            df, 
            geometry=geometry,
            crs=self.crs['geographic']
        )
        
        # bridge_idの作成（存在しない場合）
        if 'bridge_id' not in gdf.columns:
            gdf['bridge_id'] = [f"BR_{i:04d}" for i in range(len(gdf))]
        
        # ノードタイプの追加
        gdf['node_type'] = 'bridge'
        
        # 距離計算用にメートル系座標に変換
        gdf_projected = gdf.to_crs(self.crs['projected'])
        
        logger.info(f"Bridge data loaded: {len(gdf_projected)} bridges")
        
        return gdf_projected
    
    def load_river_data(self) -> Optional[gpd.GeoDataFrame]:
        """
        河川データを読み込む
        
        Returns:
            河川データのGeoDataFrame（存在しない場合はNone）
        """
        river_path = Path(self.data_paths['river_data'])
        
        if not river_path.exists():
            logger.warning(f"River data path does not exist: {river_path}")
            return None
        
        try:
            # Streamシェープファイルを探す
            stream_files = list(river_path.glob("*_Stream.shp"))
            
            if not stream_files:
                logger.warning("No river stream shapefile found")
                return None
            
            logger.info(f"Loading river data from {stream_files[0]}")
            gdf = gpd.read_file(stream_files[0])
            
            # CRSの統一
            if gdf.crs is None:
                logger.warning("River data has no CRS, assuming EPSG:4326")
                gdf = gdf.set_crs(self.crs['geographic'])
            
            gdf = gdf.to_crs(self.crs['projected'])
            
            logger.info(f"Loaded {len(gdf)} river features")
            
            return gdf
            
        except Exception as e:
            logger.error(f"Error loading river data: {e}")
            return None
    
    def load_coastline_data(self) -> Optional[gpd.GeoDataFrame]:
        """
        海岸線データを読み込む
        
        Returns:
            海岸線データのGeoDataFrame（存在しない場合はNone）
        """
        coast_path = Path(self.data_paths['coastline_data'])
        
        if not coast_path.exists():
            logger.warning(f"Coastline data path does not exist: {coast_path}")
            return None
        
        try:
            # Coastlineシェープファイルを探す
            coast_files = list(coast_path.glob("*_Coastline.shp"))
            
            if not coast_files:
                logger.warning("No coastline shapefile found")
                return None
            
            logger.info(f"Loading coastline data from {coast_files[0]}")
            gdf = gpd.read_file(coast_files[0])
            
            # CRSの統一
            if gdf.crs is None:
                logger.warning("Coastline data has no CRS, assuming EPSG:4326")
                gdf = gdf.set_crs(self.crs['geographic'])
            
            gdf = gdf.to_crs(self.crs['projected'])
            
            logger.info(f"Loaded {len(gdf)} coastline features")
            
            return gdf
            
        except Exception as e:
            logger.error(f"Error loading coastline data: {e}")
            return None
    
    def get_boundary_from_bridges(
        self,
        bridges: gpd.GeoDataFrame,
        buffer_km: float = 1.0
    ) -> gpd.GeoDataFrame:
        """
        橋梁データから解析範囲を生成
        
        Args:
            bridges: 橋梁データ
            buffer_km: バッファ距離（km）
        
        Returns:
            解析範囲のGeoDataFrame
        """
        # 全橋梁を含む凸包を作成
        boundary = bridges.unary_union.convex_hull
        
        # バッファを追加（kmをmに変換）
        boundary = boundary.buffer(buffer_km * 1000)
        
        # GeoDataFrameに変換
        boundary_gdf = gpd.GeoDataFrame(
            {'geometry': [boundary]},
            crs=self.crs['projected']
        )
        
        logger.info(f"Generated boundary with {buffer_km}km buffer")
        
        return boundary_gdf
    
    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
        """
        データフレームから候補カラム名を探す
        
        Args:
            df: データフレーム
            candidates: 候補カラム名のリスト
        
        Returns:
            見つかったカラム名（見つからない場合はNone）
        """
        for col in df.columns:
            if col in candidates:
                return col
            # 部分一致も試す
            for candidate in candidates:
                if candidate.lower() in col.lower():
                    return col
        return None
    
    def compute_river_proximity(
        self,
        bridges: gpd.GeoDataFrame,
        rivers: Optional[gpd.GeoDataFrame]
    ) -> pd.Series:
        """
        橋梁から河川までの距離を計算
        
        Args:
            bridges: 橋梁データ
            rivers: 河川データ
        
        Returns:
            各橋梁の最近傍河川までの距離（m）
        """
        if rivers is None or len(rivers) == 0:
            logger.warning("No river data available")
            return pd.Series([float('inf')] * len(bridges), index=bridges.index)
        
        logger.info("Computing river proximity...")
        
        # 河川のunion geometry
        river_union = rivers.unary_union
        
        # 各橋梁から河川までの距離
        distances = bridges.geometry.distance(river_union)
        
        logger.info(f"River proximity computed: min={distances.min():.1f}m, max={distances.max():.1f}m")
        
        return distances
    
    def compute_coast_proximity(
        self,
        bridges: gpd.GeoDataFrame,
        coastline: Optional[gpd.GeoDataFrame]
    ) -> pd.Series:
        """
        橋梁から海岸線までの距離を計算
        
        Args:
            bridges: 橋梁データ
            coastline: 海岸線データ
        
        Returns:
            各橋梁の海岸線までの距離（m）
        """
        if coastline is None or len(coastline) == 0:
            logger.warning("No coastline data available")
            return pd.Series([float('inf')] * len(bridges), index=bridges.index)
        
        logger.info("Computing coastline proximity...")
        
        # 海岸線のunion geometry
        coast_union = coastline.unary_union
        
        # 各橋梁から海岸線までの距離
        distances = bridges.geometry.distance(coast_union)
        
        logger.info(f"Coast proximity computed: min={distances.min():.1f}m, max={distances.max():.1f}m")
        
        return distances


def load_all_data(config: Dict) -> Tuple[gpd.GeoDataFrame, ...]:
    """
    全データを読み込む便利関数
    
    Args:
        config: 設定辞書
    
    Returns:
        (bridges, rivers, coastline, boundary)のタプル
    """
    loader = BridgeDataLoader(config)
    
    # 橋梁データ
    bridges = loader.load_bridge_data()
    
    # 河川・海岸線データ
    rivers = loader.load_river_data()
    coastline = loader.load_coastline_data()
    
    # 解析範囲（configからバッファサイズを取得）
    buffer_km = config.get('data', {}).get('boundary', {}).get('buffer_km', 1.0)
    boundary = loader.get_boundary_from_bridges(bridges, buffer_km=buffer_km)
    
    # 近接性の事前計算
    bridges['dist_to_river'] = loader.compute_river_proximity(bridges, rivers)
    bridges['dist_to_coast'] = loader.compute_coast_proximity(bridges, coastline)
    
    return bridges, rivers, coastline, boundary
