"""
異種グラフ構築モジュール
Bridge Importance Scoring MVP

City2Graphを使って橋梁、道路、建物、バス停の異種グラフを構築
"""

import geopandas as gpd
import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from shapely.geometry import Point, LineString
import numpy as np

logger = logging.getLogger(__name__)

try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    logger.warning("OSMnx not available. OSM data fetching will be limited.")
    OSMNX_AVAILABLE = False


class HeterogeneousGraphBuilder:
    """異種グラフ構築クラス"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 設定辞書
        """
        self.config = config
        self.crs = config['crs']
        self.graph_config = config['graph']
        self.proximity = config['graph']['proximity']
        
        # OSMnx設定（タイムアウト設定）
        if OSMNX_AVAILABLE:
            ox.settings.timeout = 300  # 5分タイムアウト
            ox.settings.use_cache = True
            logger.info("OSMnx settings configured: timeout=300s, use_cache=True")
        
    def fetch_osm_streets(
        self, 
        boundary: gpd.GeoDataFrame
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        OSMから道路ネットワークを取得
        
        Args:
            boundary: 解析範囲
        
        Returns:
            (nodes_gdf, edges_gdf)のタプル
        """
        if not OSMNX_AVAILABLE:
            raise ImportError("OSMnx is required for fetching OSM data")
        
        # 境界の面積をチェック
        area_km2 = boundary.geometry.iloc[0].area / 1e6
        logger.info(f"OSM data fetch area: {area_km2:.1f} km²")
        
        if area_km2 > 5000:
            logger.warning(f"⚠ Very large area ({area_km2:.1f} km²). OSM fetch may be slow or fail.")
            logger.warning("Consider reducing buffer size or filtering data to target city only.")
        
        logger.info("Fetching street network from OSM...")
        
        # 境界をWGS84に変換
        boundary_wgs84 = boundary.to_crs(self.crs['geographic'])
        polygon = boundary_wgs84.geometry.iloc[0]
        
        # OSMから道路ネットワークを取得
        try:
            G = ox.graph_from_polygon(
                polygon,
                network_type=self.graph_config['street']['network_type'],
                simplify=self.graph_config['street']['simplify']
            )
            
            # GeoDataFrameに変換
            nodes, edges = ox.graph_to_gdfs(G)
            
            # メートル系座標に変換
            nodes = nodes.to_crs(self.crs['projected'])
            edges = edges.to_crs(self.crs['projected'])
            
            # ノードタイプの追加
            nodes['node_type'] = 'street'
            edges['edge_type'] = 'street_to_street'
            
            # ノードIDをリセット
            nodes = nodes.reset_index()
            edges = edges.reset_index()
            
            logger.info(f"Fetched {len(nodes)} street nodes and {len(edges)} edges")
            
            return nodes, edges
            
        except Exception as e:
            logger.error(f"Error fetching OSM streets: {e}")
            logger.info("Creating minimal street network from bridge locations...")
            return self._create_minimal_street_network(boundary)
    
    def load_merged_streets(
        self, 
        merged_prefix: str = "yamaguchi_merged"
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        グリッド分割取得でマージされた道路ネットワークを読み込む
        
        Args:
            merged_prefix: マージされたファイルのプレフィックス
        
        Returns:
            (nodes_gdf, edges_gdf)のタプル
        """
        logger.info("Loading merged street network from grid fetch...")
        
        from pathlib import Path
        
        output_dir = Path(self.config['data']['output_dir'])
        nodes_path = output_dir / f"{merged_prefix}_roads_nodes.gpkg"
        edges_path = output_dir / f"{merged_prefix}_roads_edges.gpkg"
        
        if not nodes_path.exists() or not edges_path.exists():
            raise FileNotFoundError(
                f"Merged road network not found:\n"
                f"  Expected: {nodes_path}\n"
                f"  Expected: {edges_path}\n"
                f"  Please run fetch_osm_grid.py first to generate merged network."
            )
        
        # GeoPackageから読み込み
        nodes = gpd.read_file(nodes_path)
        edges = gpd.read_file(edges_path)
        
        logger.info(f"Loaded merged network: {len(nodes)} nodes, {len(edges)} edges")
        
        # メートル系座標に変換（まだの場合）
        if nodes.crs != self.crs['projected']:
            nodes = nodes.to_crs(self.crs['projected'])
            edges = edges.to_crs(self.crs['projected'])
        
        # ノードタイプの追加（まだない場合）
        if 'node_type' not in nodes.columns:
            nodes['node_type'] = 'street'
        if 'edge_type' not in edges.columns:
            edges['edge_type'] = 'street_to_street'
        
        # インデックスのリセット（必要に応じて）
        if nodes.index.name == 'osmid':
            nodes = nodes.reset_index()
        if 'u' not in edges.columns and edges.index.names == ['u', 'v', 'key']:
            edges = edges.reset_index()
        
        logger.info(f"Processed merged network: {len(nodes)} nodes, {len(edges)} edges")
        
        return nodes, edges
    
    def fetch_osm_buildings(
        self,
        boundary: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        OSMから建物を取得
        
        Args:
            boundary: 解析範囲
        
        Returns:
            建物のGeoDataFrame
        """
        if not OSMNX_AVAILABLE:
            logger.warning("OSMnx not available, skipping building data")
            return gpd.GeoDataFrame()
        
        logger.info("Fetching buildings from OSM...")
        
        boundary_wgs84 = boundary.to_crs(self.crs['geographic'])
        polygon = boundary_wgs84.geometry.iloc[0]
        
        try:
            # 建物タグで取得
            tags = {'building': True}
            buildings = ox.features_from_polygon(polygon, tags=tags)
            
            # ポイントまたはポリゴンのみ保持
            buildings = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon', 'Point'])]
            
            # メートル系座標に変換
            buildings = buildings.to_crs(self.crs['projected'])
            
            # 重心を追加（ポリゴンの場合）
            buildings['centroid'] = buildings.geometry.centroid
            
            # ノードタイプの追加
            buildings['node_type'] = 'building'
            
            # カテゴリ分類
            buildings['category'] = self._classify_buildings(buildings)
            
            logger.info(f"Fetched {len(buildings)} buildings")
            
            return buildings.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Error fetching OSM buildings: {e}")
            return gpd.GeoDataFrame()
    
    def fetch_osm_pois(
        self,
        boundary: gpd.GeoDataFrame,
        poi_type: str = 'bus_stop'
    ) -> gpd.GeoDataFrame:
        """
        OSMからPOI（バス停など）を取得
        
        Args:
            boundary: 解析範囲
            poi_type: POIタイプ（'bus_stop', 'hospital'など）
        
        Returns:
            POIのGeoDataFrame
        """
        if not OSMNX_AVAILABLE:
            logger.warning("OSMnx not available, skipping POI data")
            return gpd.GeoDataFrame()
        
        logger.info(f"Fetching {poi_type} from OSM...")
        
        boundary_wgs84 = boundary.to_crs(self.crs['geographic'])
        polygon = boundary_wgs84.geometry.iloc[0]
        
        # POIタグの取得
        tags = self.config.get('poi_tags', {}).get(poi_type, {})
        
        if not tags:
            logger.warning(f"No tags configured for {poi_type}")
            return gpd.GeoDataFrame()
        
        try:
            pois = ox.features_from_polygon(polygon, tags=tags)
            
            # ポイントのみ保持（またはポリゴンの重心を使用）
            if len(pois) > 0:
                pois = pois.to_crs(self.crs['projected'])
                
                # ポイントでない場合は重心を使用
                mask = ~pois.geometry.type.isin(['Point', 'MultiPoint'])
                if mask.any():
                    pois.loc[mask, 'geometry'] = pois.loc[mask, 'geometry'].centroid
                
                pois['node_type'] = poi_type
                pois['poi_type'] = poi_type
                
                logger.info(f"Fetched {len(pois)} {poi_type} POIs")
                
                return pois.reset_index(drop=True)
            else:
                logger.warning(f"No {poi_type} POIs found")
                return gpd.GeoDataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching OSM POIs: {e}")
            return gpd.GeoDataFrame()
    
    def build_proximity_graph(
        self,
        sources: gpd.GeoDataFrame,
        targets: gpd.GeoDataFrame,
        max_distance: float,
        edge_type: str
    ) -> nx.Graph:
        """
        近接性に基づくグラフを構築
        
        Args:
            sources: ソースノード（例：橋梁）
            targets: ターゲットノード（例：建物）
            max_distance: 最大距離（m）
            edge_type: エッジタイプ名
        
        Returns:
            NetworkXグラフ
        """
        logger.info(f"Building {edge_type} proximity graph (max_distance={max_distance}m)...")
        
        G = nx.Graph()
        edge_count = 0
        
        # 各ソースノードについて最近傍のターゲットを探す
        for idx, source in sources.iterrows():
            source_geom = source.geometry
            source_id = source.get('bridge_id', f"src_{idx}")
            
            # ターゲットまでの距離を計算
            distances = targets.geometry.distance(source_geom)
            
            # 閾値以内のターゲットを抽出
            within_range = distances[distances <= max_distance]
            
            for target_idx, dist in within_range.items():
                target = targets.loc[target_idx]
                target_id = target.get('osmid', f"tgt_{target_idx}")
                
                G.add_edge(
                    source_id,
                    target_id,
                    edge_type=edge_type,
                    distance=dist,
                    weight=1.0 / (dist + 1.0)  # 距離の逆数を重みに
                )
                edge_count += 1
        
        logger.info(f"Created {edge_count} {edge_type} edges")
        
        return G
    
    def snap_bridges_to_streets(
        self,
        bridges: gpd.GeoDataFrame,
        street_edges: gpd.GeoDataFrame,
        max_distance: float = 30.0
    ) -> Tuple[nx.Graph, pd.DataFrame]:
        """
        橋梁を道路ネットワークにスナップ
        
        Args:
            bridges: 橋梁データ
            street_edges: 道路エッジ
            max_distance: 最大スナップ距離（m）
        
        Returns:
            (グラフ, スナップ情報DataFrame)のタプル
        """
        logger.info(f"Snapping bridges to street network (max_distance={max_distance}m)...")
        
        G = nx.Graph()
        snap_info = []
        
        for idx, bridge in bridges.iterrows():
            bridge_geom = bridge.geometry
            bridge_id = bridge.get('bridge_id', f"BR_{idx}")
            
            # 最近傍の道路エッジを探す
            distances = street_edges.geometry.distance(bridge_geom)
            min_dist_idx = distances.idxmin()
            min_dist = distances[min_dist_idx]
            
            if min_dist <= max_distance:
                nearest_edge = street_edges.loc[min_dist_idx]
                
                # エッジの始点・終点ノードを取得
                u = nearest_edge.get('u', f"node_u_{min_dist_idx}")
                v = nearest_edge.get('v', f"node_v_{min_dist_idx}")
                
                # 橋梁と道路ネットワークを接続
                G.add_edge(bridge_id, u, edge_type='bridge_to_street', distance=min_dist)
                G.add_edge(bridge_id, v, edge_type='bridge_to_street', distance=min_dist)
                
                snap_info.append({
                    'bridge_id': bridge_id,
                    'nearest_edge': min_dist_idx,
                    'distance': min_dist,
                    'snapped': True
                })
            else:
                snap_info.append({
                    'bridge_id': bridge_id,
                    'nearest_edge': None,
                    'distance': min_dist,
                    'snapped': False
                })
                logger.warning(f"Bridge {bridge_id} could not be snapped (distance={min_dist:.1f}m)")
        
        snap_df = pd.DataFrame(snap_info)
        logger.info(f"Snapped {snap_df['snapped'].sum()}/{len(snap_df)} bridges to street network")
        
        return G, snap_df
    
    def build_heterogeneous_graph(
        self,
        bridges: gpd.GeoDataFrame,
        boundary: gpd.GeoDataFrame,
        use_merged_network: bool = False,
        merged_prefix: str = "yamaguchi_merged"
    ) -> Tuple[nx.Graph, Dict]:
        """
        異種グラフを構築（統合）
        
        Args:
            bridges: 橋梁データ
            boundary: 解析範囲
            use_merged_network: グリッド分割取得でマージされたネットワークを使用するか
            merged_prefix: マージされたファイルのプレフィックス
        
        Returns:
            (統合グラフ, メタデータ辞書)のタプル
        """
        logger.info("Building heterogeneous graph...")
        
        # 1. OSMから道路ネットワークを取得
        if use_merged_network:
            logger.info("Using pre-fetched merged road network from grid fetch")
            street_nodes, street_edges = self.load_merged_streets(merged_prefix)
        else:
            logger.info("Fetching road network from OSM (single request)")
            street_nodes, street_edges = self.fetch_osm_streets(boundary)
        
        # 2. 道路ネットワークグラフを構築
        street_graph = self._edges_to_graph(street_edges)
        
        # 3. 橋梁を道路にスナップ
        bridge_street_graph, snap_info = self.snap_bridges_to_streets(
            bridges, 
            street_edges, 
            max_distance=self.proximity['bridge_to_street']
        )
        
        # 4. 建物データ（設定に応じて取得）
        fetch_buildings = self.graph_config.get('fetch', {}).get('buildings', True)
        if fetch_buildings:
            buildings = self.fetch_osm_buildings(boundary)
        else:
            logger.info("Skipping building data fetch (config: fetch.buildings=false)")
            buildings = gpd.GeoDataFrame()
        
        # 5. バス停データ（設定に応じて取得）
        fetch_bus_stops = self.graph_config.get('fetch', {}).get('bus_stops', True)
        if fetch_bus_stops:
            bus_stops = self.fetch_osm_pois(boundary, poi_type='bus_stop')
        else:
            logger.info("Skipping bus stop data fetch (config: fetch.bus_stops=false)")
            bus_stops = gpd.GeoDataFrame()
        
        # 6. 近接グラフの構築
        graphs_to_merge = [street_graph, bridge_street_graph]
        
        if len(buildings) > 0:
            bridge_building_graph = self.build_proximity_graph(
                bridges,
                buildings,
                max_distance=self.proximity['bridge_to_building'],
                edge_type='bridge_to_building'
            )
            graphs_to_merge.append(bridge_building_graph)
        
        if len(bus_stops) > 0:
            bridge_bus_graph = self.build_proximity_graph(
                bridges,
                bus_stops,
                max_distance=self.proximity['bridge_to_bus_stop'],
                edge_type='bridge_to_bus'
            )
            graphs_to_merge.append(bridge_bus_graph)
        
        # 7. 全グラフを統合
        logger.info("Merging all graphs...")
        G = nx.compose_all(graphs_to_merge)
        
        # 8. ノード属性の追加
        self._add_node_attributes(G, bridges, street_nodes, buildings, bus_stops)
        
        # メタデータ
        metadata = {
            'num_bridges': len(bridges),
            'num_street_nodes': len(street_nodes),
            'num_street_edges': len(street_edges),
            'num_buildings': len(buildings),
            'num_bus_stops': len(bus_stops),
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'snap_info': snap_info
        }
        
        logger.info(f"Heterogeneous graph built: {metadata['total_nodes']} nodes, {metadata['total_edges']} edges")
        
        return G, metadata
    
    def _classify_buildings(self, buildings: gpd.GeoDataFrame) -> pd.Series:
        """建物をカテゴリ分類"""
        categories = []
        
        for idx, building in buildings.iterrows():
            # amenityタグから分類
            amenity = building.get('amenity', '')
            building_type = building.get('building', '')
            
            if amenity in ['hospital', 'clinic']:
                cat = 'hospital'
            elif amenity in ['school', 'university', 'kindergarten']:
                cat = 'school'
            elif amenity in ['townhall', 'library', 'community_centre']:
                cat = 'public'
            elif amenity in ['fire_station', 'police']:
                cat = 'emergency'
            elif building_type in ['house', 'residential', 'apartments']:
                cat = 'residential'
            else:
                cat = 'other'
            
            categories.append(cat)
        
        return pd.Series(categories, index=buildings.index)
    
    def _edges_to_graph(self, edges: gpd.GeoDataFrame) -> nx.Graph:
        """エッジGeoDataFrameからNetworkXグラフを構築"""
        G = nx.Graph()
        
        for idx, edge in edges.iterrows():
            u = edge.get('u', idx)
            v = edge.get('v', idx + 0.5)
            length = edge.get('length', edge.geometry.length if hasattr(edge.geometry, 'length') else 1.0)
            
            G.add_edge(u, v, edge_type='street_to_street', length=length, weight=length)
        
        return G
    
    def _add_node_attributes(
        self, 
        G: nx.Graph, 
        bridges: gpd.GeoDataFrame,
        street_nodes: gpd.GeoDataFrame,
        buildings: gpd.GeoDataFrame,
        bus_stops: gpd.GeoDataFrame
    ):
        """ノード属性を追加"""
        # 橋梁ノード
        for idx, bridge in bridges.iterrows():
            bridge_id = bridge.get('bridge_id', f"BR_{idx}")
            if bridge_id in G.nodes:
                G.nodes[bridge_id]['node_type'] = 'bridge'
                G.nodes[bridge_id]['geometry'] = bridge.geometry
        
        # 道路ノード
        for idx, node in street_nodes.iterrows():
            node_id = node.get('osmid', idx)
            if node_id in G.nodes:
                G.nodes[node_id]['node_type'] = 'street'
                G.nodes[node_id]['geometry'] = node.geometry
        
        # 建物ノード
        for idx, building in buildings.iterrows():
            building_id = building.get('osmid', f"bld_{idx}")
            if building_id in G.nodes:
                G.nodes[building_id]['node_type'] = 'building'
                G.nodes[building_id]['category'] = building.get('category', 'other')
        
        # バス停ノード
        for idx, bus_stop in bus_stops.iterrows():
            bus_id = bus_stop.get('osmid', f"bus_{idx}")
            if bus_id in G.nodes:
                G.nodes[bus_id]['node_type'] = 'bus_stop'
    
    def _create_minimal_street_network(
        self, 
        boundary: gpd.GeoDataFrame
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        OSM取得失敗時の最小限の道路ネットワーク生成
        """
        logger.warning("Creating minimal street network (fallback)")
        
        # 境界の重心を中心にグリッドを生成
        centroid = boundary.geometry.iloc[0].centroid
        
        # 簡単なグリッドノード
        nodes_data = []
        for i in range(5):
            for j in range(5):
                x = centroid.x + (i - 2) * 1000
                y = centroid.y + (j - 2) * 1000
                nodes_data.append({
                    'osmid': f"grid_{i}_{j}",
                    'geometry': Point(x, y),
                    'node_type': 'street'
                })
        
        nodes = gpd.GeoDataFrame(nodes_data, crs=self.crs['projected'])
        
        # エッジ（グリッド接続）
        edges_data = []
        for i in range(5):
            for j in range(5):
                node_id = f"grid_{i}_{j}"
                # 右隣
                if i < 4:
                    next_id = f"grid_{i+1}_{j}"
                    edges_data.append({
                        'u': node_id,
                        'v': next_id,
                        'geometry': LineString([
                            nodes[nodes['osmid'] == node_id].geometry.iloc[0],
                            nodes[nodes['osmid'] == next_id].geometry.iloc[0]
                        ]),
                        'edge_type': 'street_to_street',
                        'length': 1000
                    })
                # 下隣
                if j < 4:
                    next_id = f"grid_{i}_{j+1}"
                    edges_data.append({
                        'u': node_id,
                        'v': next_id,
                        'geometry': LineString([
                            nodes[nodes['osmid'] == node_id].geometry.iloc[0],
                            nodes[nodes['osmid'] == next_id].geometry.iloc[0]
                        ]),
                        'edge_type': 'street_to_street',
                        'length': 1000
                    })
        
        edges = gpd.GeoDataFrame(edges_data, crs=self.crs['projected'])
        
        return nodes, edges
