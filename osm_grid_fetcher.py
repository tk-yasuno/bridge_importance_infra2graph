"""
OSM グリッド分割取得モジュール
Bridge Importance Scoring MVP v1.1

大規模エリア（山口市 ~1000km²）のOSMデータを、
レート制限回避のため4×4グリッドに分割して取得し、
密な結合グラフを構築する。
"""

import os
import time
import geopandas as gpd
import osmnx as ox
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from shapely.geometry import box, Point, LineString
import pandas as pd
import networkx as nx

logger = logging.getLogger(__name__)


class OSMGridFetcher:
    """OSM データのグリッド分割取得クラス"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 設定辞書
        """
        self.config = config
        self.crs = config['crs']
        self.output_dir = Path(config['data']['output_dir']) / "osm_cells"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # OSMnx設定
        ox.settings.use_cache = True
        ox.settings.log_console = True
        ox.settings.timeout = 180  # 3分タイムアウト
        
        logger.info(f"OSM Grid Fetcher initialized: output={self.output_dir}")
    
    def get_city_polygon(self, place_name: str) -> gpd.GeoSeries:
        """
        市町村名からポリゴンを取得
        
        Args:
            place_name: 地名（例: "山口市, 山口県, 日本"）
        
        Returns:
            市域ポリゴン（GeoSeries）
        """
        logger.info(f"Fetching city polygon for: {place_name}")
        
        try:
            gdf = ox.geocode_to_gdf(place_name)
            # マルチポリゴンの場合は unary_union で結合
            poly = gdf.unary_union
            gs = gpd.GeoSeries([poly], crs=gdf.crs)
            
            area_km2 = gs.iloc[0].area / 1e6
            logger.info(f"City polygon obtained: area={area_km2:.1f} km²")
            
            return gs
        except Exception as e:
            logger.error(f"Failed to fetch city polygon: {e}")
            raise
    
    def make_grid_over_polygon(
        self, 
        poly_gs: gpd.GeoSeries, 
        n_rows: int = 4, 
        n_cols: int = 4
    ) -> gpd.GeoDataFrame:
        """
        ポリゴンの外接矩形を n_rows × n_cols に分割し、
        ポリゴンと交差する部分のみを抽出
        
        Args:
            poly_gs: 市域ポリゴン（GeoSeries）
            n_rows: グリッドの行数（デフォルト: 4）
            n_cols: グリッドの列数（デフォルト: 4）
        
        Returns:
            グリッドセルの GeoDataFrame（cell_id, geometry）
        """
        logger.info(f"Creating {n_rows}×{n_cols} grid over polygon...")
        
        poly = poly_gs.iloc[0]
        minx, miny, maxx, maxy = poly.bounds
        
        dx = (maxx - minx) / n_cols
        dy = (maxy - miny) / n_rows
        
        cells = []
        cell_ids = []
        cell_info = []
        
        for i in range(n_rows):
            for j in range(n_cols):
                x1 = minx + j * dx
                x2 = minx + (j + 1) * dx
                y1 = miny + i * dy
                y2 = miny + (i + 1) * dy
                
                cell = box(x1, y1, x2, y2)
                # 市域ポリゴンと交差する部分だけ残す
                inter = cell.intersection(poly)
                
                if not inter.is_empty:
                    cell_id = i * n_cols + j
                    cells.append(inter)
                    cell_ids.append(cell_id)
                    
                    # セル情報（デバッグ用）
                    area_km2 = inter.area / 1e6 if hasattr(inter, 'area') else 0
                    cell_info.append({
                        'cell_id': cell_id,
                        'row': i,
                        'col': j,
                        'area_km2': area_km2,
                    })
        
        grid_gdf = gpd.GeoDataFrame(
            cell_info,
            geometry=cells,
            crs=poly_gs.crs,
        )
        
        logger.info(f"Grid created: {len(grid_gdf)} cells (total area: {grid_gdf['area_km2'].sum():.1f} km²)")
        
        return grid_gdf
    
    def get_bbox_from_geom(self, geom) -> Tuple[float, float, float, float]:
        """
        ジオメトリからバウンディングボックスを取得（OSMnx形式）
        
        Args:
            geom: Shapely geometry
        
        Returns:
            (south, north, west, east) タプル
        """
        minx, miny, maxx, maxy = geom.bounds
        return miny, maxy, minx, maxx  # south, north, west, east
    
    def fetch_roads_for_cell(
        self, 
        geom, 
        cell_id: int,
        retry_count: int = 3,
        retry_delay: int = 10
    ) -> bool:
        """
        セルごとに道路ネットワークを取得
        
        Args:
            geom: セルのジオメトリ
            cell_id: セルID
            retry_count: リトライ回数
            retry_delay: リトライ待機時間（秒）
        
        Returns:
            成功したら True
        """
        south, north, west, east = self.get_bbox_from_geom(geom)
        
        for attempt in range(retry_count):
            try:
                logger.info(f"[Cell {cell_id:02d}] Fetching roads... (attempt {attempt+1}/{retry_count})")
                
                G = ox.graph_from_bbox(
                    north=north,
                    south=south,
                    east=east,
                    west=west,
                    network_type="drive",
                    simplify=True,
                    retain_all=False,
                )
                
                # NetworkX → GeoDataFrame に変換
                nodes, edges = ox.graph_to_gdfs(G)
                
                # 保存
                nodes_path = self.output_dir / f"cell_{cell_id:02d}_roads_nodes.gpkg"
                edges_path = self.output_dir / f"cell_{cell_id:02d}_roads_edges.gpkg"
                
                nodes.to_file(nodes_path, driver="GPKG")
                edges.to_file(edges_path, driver="GPKG")
                
                logger.info(f"[Cell {cell_id:02d}] Roads saved: {len(nodes)} nodes, {len(edges)} edges")
                return True
                
            except Exception as e:
                logger.warning(f"[Cell {cell_id:02d}] Roads fetch failed (attempt {attempt+1}): {e}")
                if attempt < retry_count - 1:
                    logger.info(f"[Cell {cell_id:02d}] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"[Cell {cell_id:02d}] Roads fetch failed after {retry_count} attempts")
                    return False
    
    def fetch_buildings_for_cell(
        self, 
        geom, 
        cell_id: int,
        retry_count: int = 3,
        retry_delay: int = 10
    ) -> bool:
        """
        セルごとに建物を取得
        
        Args:
            geom: セルのジオメトリ
            cell_id: セルID
            retry_count: リトライ回数
            retry_delay: リトライ待機時間（秒）
        
        Returns:
            成功したら True
        """
        south, north, west, east = self.get_bbox_from_geom(geom)
        
        for attempt in range(retry_count):
            try:
                logger.info(f"[Cell {cell_id:02d}] Fetching buildings... (attempt {attempt+1}/{retry_count})")
                
                buildings = ox.features_from_bbox(
                    north=north,
                    south=south,
                    east=east,
                    west=west,
                    tags={"building": True},
                )
                
                # 保存
                buildings_path = self.output_dir / f"cell_{cell_id:02d}_buildings.gpkg"
                buildings.to_file(buildings_path, driver="GPKG")
                
                logger.info(f"[Cell {cell_id:02d}] Buildings saved: {len(buildings)} features")
                return True
                
            except Exception as e:
                logger.warning(f"[Cell {cell_id:02d}] Buildings fetch failed (attempt {attempt+1}): {e}")
                if attempt < retry_count - 1:
                    logger.info(f"[Cell {cell_id:02d}] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"[Cell {cell_id:02d}] Buildings fetch failed after {retry_count} attempts")
                    return False
    
    def fetch_bus_stops_for_cell(
        self, 
        geom, 
        cell_id: int,
        retry_count: int = 3,
        retry_delay: int = 10
    ) -> bool:
        """
        セルごとにバス停を取得
        
        Args:
            geom: セルのジオメトリ
            cell_id: セルID
            retry_count: リトライ回数
            retry_delay: リトライ待機時間（秒）
        
        Returns:
            成功したら True
        """
        south, north, west, east = self.get_bbox_from_geom(geom)
        
        for attempt in range(retry_count):
            try:
                logger.info(f"[Cell {cell_id:02d}] Fetching bus stops... (attempt {attempt+1}/{retry_count})")
                
                tags = {"highway": "bus_stop"}
                bus_stops = ox.features_from_bbox(
                    north=north,
                    south=south,
                    east=east,
                    west=west,
                    tags=tags,
                )
                
                # 保存
                bus_stops_path = self.output_dir / f"cell_{cell_id:02d}_bus_stops.gpkg"
                bus_stops.to_file(bus_stops_path, driver="GPKG")
                
                logger.info(f"[Cell {cell_id:02d}] Bus stops saved: {len(bus_stops)} features")
                return True
                
            except Exception as e:
                logger.warning(f"[Cell {cell_id:02d}] Bus stops fetch failed (attempt {attempt+1}): {e}")
                if attempt < retry_count - 1:
                    logger.info(f"[Cell {cell_id:02d}] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"[Cell {cell_id:02d}] Bus stops fetch failed after {retry_count} attempts")
                    return False
    
    def run_for_all_cells(
        self, 
        grid_gdf: gpd.GeoDataFrame,
        fetch_roads: bool = True,
        fetch_buildings: bool = False,
        fetch_bus_stops: bool = False,
        inter_cell_delay: int = 5
    ) -> Dict[str, int]:
        """
        全セルに対してOSMデータを取得
        
        Args:
            grid_gdf: グリッドセルの GeoDataFrame
            fetch_roads: 道路ネットワークを取得するか
            fetch_buildings: 建物を取得するか
            fetch_bus_stops: バス停を取得するか
            inter_cell_delay: セル間の待機時間（秒、レート制限対策）
        
        Returns:
            統計情報の辞書
        """
        logger.info(f"Starting OSM data fetch for {len(grid_gdf)} cells...")
        logger.info(f"Options: roads={fetch_roads}, buildings={fetch_buildings}, bus_stops={fetch_bus_stops}")
        
        stats = {
            'total_cells': len(grid_gdf),
            'roads_success': 0,
            'buildings_success': 0,
            'bus_stops_success': 0,
        }
        
        for idx, row in grid_gdf.iterrows():
            cell_id = int(row["cell_id"])
            geom = row.geometry
            
            logger.info(f"=== Processing Cell {cell_id:02d} ({idx+1}/{len(grid_gdf)}) ===")
            
            if fetch_roads:
                if self.fetch_roads_for_cell(geom, cell_id):
                    stats['roads_success'] += 1
            
            if fetch_buildings:
                if self.fetch_buildings_for_cell(geom, cell_id):
                    stats['buildings_success'] += 1
            
            if fetch_bus_stops:
                if self.fetch_bus_stops_for_cell(geom, cell_id):
                    stats['bus_stops_success'] += 1
            
            # セル間で待機（レート制限対策）
            if idx < len(grid_gdf) - 1:  # 最後のセル以外
                logger.info(f"Waiting {inter_cell_delay} seconds before next cell...")
                time.sleep(inter_cell_delay)
        
        logger.info(f"OSM data fetch completed:")
        logger.info(f"  Roads: {stats['roads_success']}/{stats['total_cells']}")
        logger.info(f"  Buildings: {stats['buildings_success']}/{stats['total_cells']}")
        logger.info(f"  Bus stops: {stats['bus_stops_success']}/{stats['total_cells']}")
        
        return stats
    
    def merge_cell_roads(self) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        全セルの道路ネットワークをマージし、ノードIDを一意化
        
        Returns:
            (nodes_merged, edges_merged) のタプル
        """
        logger.info("Merging road networks from all cells...")
        
        nodes_list = []
        edges_list = []
        
        # 全セルのファイルを探索
        cell_files = sorted(self.output_dir.glob("cell_*_roads_nodes.gpkg"))
        
        if not cell_files:
            raise FileNotFoundError(f"No road network files found in {self.output_dir}")
        
        logger.info(f"Found {len(cell_files)} cell files to merge")
        
        # ノードIDのオフセット管理
        node_id_offset = 0
        node_id_mapping = {}  # {(cell_id, original_osmid) -> new_osmid}
        
        for cell_file in cell_files:
            # セルIDを抽出
            cell_id = int(cell_file.stem.split('_')[1])
            
            # ノードとエッジを読み込み
            nodes_path = self.output_dir / f"cell_{cell_id:02d}_roads_nodes.gpkg"
            edges_path = self.output_dir / f"cell_{cell_id:02d}_roads_edges.gpkg"
            
            if not nodes_path.exists() or not edges_path.exists():
                logger.warning(f"[Cell {cell_id:02d}] Missing files, skipping")
                continue
            
            nodes = gpd.read_file(nodes_path)
            edges = gpd.read_file(edges_path)
            
            logger.info(f"[Cell {cell_id:02d}] Loaded {len(nodes)} nodes, {len(edges)} edges")
            
            # ノードIDをリマップ（重複回避）
            original_osmids = nodes.index.tolist()
            
            for orig_id in original_osmids:
                new_id = node_id_offset
                node_id_mapping[(cell_id, orig_id)] = new_id
                node_id_offset += 1
            
            # ノードのインデックスを更新
            nodes['original_osmid'] = nodes.index
            nodes['cell_id'] = cell_id
            nodes.index = [node_id_mapping[(cell_id, oid)] for oid in nodes['original_osmid']]
            
            # エッジのu, vをリマップ
            edges['u_new'] = edges['u'].apply(lambda x: node_id_mapping.get((cell_id, x), x))
            edges['v_new'] = edges['v'].apply(lambda x: node_id_mapping.get((cell_id, x), x))
            edges['u'] = edges['u_new']
            edges['v'] = edges['v_new']
            edges.drop(columns=['u_new', 'v_new'], inplace=True)
            edges['cell_id'] = cell_id
            
            nodes_list.append(nodes)
            edges_list.append(edges)
        
        # 全セルをマージ
        nodes_merged = pd.concat(nodes_list, ignore_index=False)
        edges_merged = pd.concat(edges_list, ignore_index=True)
        
        # 重複ノードの削除（同一座標のノードを統合）
        logger.info("Removing duplicate nodes (same coordinates)...")
        original_count = len(nodes_merged)
        
        # 座標による重複判定（1m以内を同一とみなす）
        nodes_merged['x_round'] = nodes_merged.geometry.x.round(5)  # 約1mの精度
        nodes_merged['y_round'] = nodes_merged.geometry.y.round(5)
        
        # 最初の出現を保持
        nodes_merged = nodes_merged[~nodes_merged.duplicated(subset=['x_round', 'y_round'], keep='first')]
        nodes_merged.drop(columns=['x_round', 'y_round'], inplace=True)
        
        logger.info(f"Merged nodes: {original_count} → {len(nodes_merged)} (removed {original_count - len(nodes_merged)} duplicates)")
        logger.info(f"Merged edges: {len(edges_merged)}")
        
        return nodes_merged, edges_merged
    
    def save_merged_networks(
        self,
        nodes: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame,
        output_prefix: str = "merged"
    ):
        """
        マージされたネットワークを保存
        
        Args:
            nodes: ノード GeoDataFrame
            edges: エッジ GeoDataFrame
            output_prefix: 出力ファイル名のプレフィックス
        """
        nodes_path = self.output_dir.parent / f"{output_prefix}_roads_nodes.gpkg"
        edges_path = self.output_dir.parent / f"{output_prefix}_roads_edges.gpkg"
        
        nodes.to_file(nodes_path, driver="GPKG")
        edges.to_file(edges_path, driver="GPKG")
        
        logger.info(f"Merged network saved:")
        logger.info(f"  Nodes: {nodes_path}")
        logger.info(f"  Edges: {edges_path}")
