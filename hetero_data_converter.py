"""
HeteroData 変換モジュール
Bridge Importance Scoring MVP v1.1

City2Graph で作成した NetworkX 異種グラフを
PyTorch Geometric の HeteroData 形式に変換する
"""

import torch
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from typing import Dict, List, Optional, Tuple
import logging
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class HeteroGraphConverter:
    """異種グラフの HeteroData 変換クラス"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 設定辞書
        """
        self.config = config
        self.node_type_mapping = {}  # {node_id -> (node_type, type_specific_id)}
        self.reverse_mapping = {}  # {(node_type, type_specific_id) -> node_id}
        self._bridge_gdf = None
        self._street_nodes = None
        
    def convert_to_hetero_data(
        self,
        G: nx.Graph,
        bridge_gdf: gpd.GeoDataFrame,
        street_nodes: Optional[gpd.GeoDataFrame] = None,
        buildings: Optional[gpd.GeoDataFrame] = None,
        bus_stops: Optional[gpd.GeoDataFrame] = None
    ) -> HeteroData:
        """
        NetworkX グラフと GeoDataFrame から HeteroData を構築
        
        Args:
            G: NetworkX 異種グラフ
            bridge_gdf: 橋梁 GeoDataFrame（特徴量を含む）
            street_nodes: 道路ノード GeoDataFrame
            buildings: 建物 GeoDataFrame
            bus_stops: バス停 GeoDataFrame
        
        Returns:
            PyTorch Geometric HeteroData
        """
        logger.info("Converting NetworkX graph to PyTorch Geometric HeteroData...")

        # 後段のエッジ再定義（kNN）で使用
        self._bridge_gdf = bridge_gdf
        self._street_nodes = street_nodes
        
        data = HeteroData()
        
        # 1. ノードタイプごとの ID マッピングを作成
        self._create_node_mappings(G, bridge_gdf, street_nodes, buildings, bus_stops)
        
        # 2. ノード特徴量を抽出
        self._extract_node_features(data, bridge_gdf, street_nodes, buildings, bus_stops)
        
        # 3. エッジインデックスを抽出
        self._extract_edge_indices(data, G)

        # 3.1 v1.4 Exp-2: bridge-streetエッジをkNNで再定義（任意）
        self._apply_knn_bridge_edges(data)
        
        # 4. メタデータを追加
        data.num_nodes_dict = {
            node_type: len(indices) 
            for node_type, indices in self.reverse_mapping.items()
        }
        
        logger.info(f"HeteroData created:")
        logger.info(f"  Node types: {list(data.node_types)}")
        logger.info(f"  Edge types: {list(data.edge_types)}")
        for node_type in data.node_types:
            if hasattr(data[node_type], 'x'):
                logger.info(f"  {node_type}: {data[node_type].x.shape[0]} nodes, {data[node_type].x.shape[1]} features")
        
        return data

    def _apply_knn_bridge_edges(self, data: HeteroData):
        """v1.4実験: street->bridgeエッジをkNNで再定義"""
        hgnn_cfg = self.config.get('hgnn', {})
        edge_mode = hgnn_cfg.get('bridge_edge_mode', 'graph')
        knn_k = int(hgnn_cfg.get('bridge_edge_knn_k', 3))
        use_edge_attr = bool(hgnn_cfg.get('use_edge_attr', False))
        dist_scale = float(hgnn_cfg.get('edge_distance_scale_m', 100.0))

        if edge_mode != 'knn':
            return

        if 'bridge' not in self.reverse_mapping or 'street' not in self.reverse_mapping:
            logger.warning("kNN edge mode requested, but bridge/street mappings are unavailable.")
            return
        if self._bridge_gdf is None or self._street_nodes is None or len(self._street_nodes) == 0:
            logger.warning("kNN edge mode requested, but bridge_gdf/street_nodes are unavailable.")
            return

        bridge_xy_lookup = {}
        if 'bridge_id' in self._bridge_gdf.columns and 'geometry' in self._bridge_gdf.columns:
            for _, row in self._bridge_gdf.iterrows():
                geom = row.get('geometry')
                bid = row.get('bridge_id')
                if bid is None or geom is None:
                    continue
                if hasattr(geom, 'x') and hasattr(geom, 'y'):
                    bridge_xy_lookup[bid] = (float(geom.x), float(geom.y))

        street_xy_lookup = {}
        if 'node_id' in self._street_nodes.columns:
            for _, row in self._street_nodes.iterrows():
                nid = row.get('node_id')
                geom = row.get('geometry') if 'geometry' in self._street_nodes.columns else None
                if geom is not None and hasattr(geom, 'x') and hasattr(geom, 'y'):
                    street_xy_lookup[nid] = (float(geom.x), float(geom.y))
                elif 'x' in self._street_nodes.columns and 'y' in self._street_nodes.columns:
                    x = row.get('x', None)
                    y = row.get('y', None)
                    if pd.notna(x) and pd.notna(y):
                        street_xy_lookup[nid] = (float(x), float(y))

        bridge_type_ids = sorted(self.reverse_mapping['bridge'].keys())
        street_type_ids = sorted(self.reverse_mapping['street'].keys())

        bridge_xy = []
        bridge_id_list = []
        for b_tid in bridge_type_ids:
            b_node_id = self.reverse_mapping['bridge'][b_tid]
            xy = bridge_xy_lookup.get(b_node_id)
            if xy is not None:
                bridge_xy.append(xy)
                bridge_id_list.append(b_tid)

        street_xy = []
        street_id_list = []
        for s_tid in street_type_ids:
            s_node_id = self.reverse_mapping['street'][s_tid]
            xy = street_xy_lookup.get(s_node_id)
            if xy is not None:
                street_xy.append(xy)
                street_id_list.append(s_tid)

        if len(bridge_xy) == 0 or len(street_xy) == 0:
            logger.warning("kNN edge mode requested, but valid bridge/street coordinates were not found.")
            return

        bridge_xy = np.asarray(bridge_xy, dtype=float)
        street_xy = np.asarray(street_xy, dtype=float)
        k_eff = max(1, min(knn_k, len(street_xy)))

        nn = NearestNeighbors(n_neighbors=k_eff)
        nn.fit(street_xy)
        distances, neighbor_indices = nn.kneighbors(bridge_xy)

        edges = []
        edge_attrs = []
        for i, b_tid in enumerate(bridge_id_list):
            for n_idx, j in enumerate(neighbor_indices[i]):
                s_tid = street_id_list[int(j)]
                edges.append([s_tid, b_tid])  # (street -> bridge)
                if use_edge_attr:
                    d = float(distances[i][n_idx])
                    w = np.exp(-d / max(dist_scale, 1e-6))
                    edge_attrs.append([float(w)])

        if len(edges) == 0:
            logger.warning("kNN edge mode produced no edges; keeping original graph edges.")
            return

        data['street', 'to', 'bridge'].edge_index = torch.tensor(edges, dtype=torch.long).t()
        if use_edge_attr:
            data['street', 'to', 'bridge'].edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        logger.info(f"  Replaced ('street','to','bridge') edges with kNN edges: {len(edges)} (k={k_eff})")
    
    def _create_node_mappings(
        self,
        G: nx.Graph,
        bridge_gdf: gpd.GeoDataFrame,
        street_nodes: Optional[gpd.GeoDataFrame],
        buildings: Optional[gpd.GeoDataFrame],
        bus_stops: Optional[gpd.GeoDataFrame]
    ):
        """ノード ID のマッピングを作成"""
        logger.info("Creating node ID mappings...")
        
        # ノードタイプごとのカウンタ
        type_counters = {'bridge': 0, 'street': 0, 'building': 0, 'bus_stop': 0}
        
        # NetworkX グラフのすべてのノードを走査
        for node_id, node_data in G.nodes(data=True):
            node_type = node_data.get('node_type', 'unknown')
            
            if node_type in type_counters:
                type_specific_id = type_counters[node_type]
                self.node_type_mapping[node_id] = (node_type, type_specific_id)
                
                if node_type not in self.reverse_mapping:
                    self.reverse_mapping[node_type] = {}
                self.reverse_mapping[node_type][type_specific_id] = node_id
                
                type_counters[node_type] += 1
        
        logger.info(f"Node type counts: {type_counters}")
    
    def _extract_node_features(
        self,
        data: HeteroData,
        bridge_gdf: gpd.GeoDataFrame,
        street_nodes: Optional[gpd.GeoDataFrame],
        buildings: Optional[gpd.GeoDataFrame],
        bus_stops: Optional[gpd.GeoDataFrame]
    ):
        """ノードタイプごとに特徴量を抽出"""
        logger.info("Extracting node features...")
        
        # Bridge ノード特徴量
        if 'bridge' in self.reverse_mapping:
            data['bridge'].x = self._extract_bridge_features(
                bridge_gdf,
                street_nodes=street_nodes,
                buildings=buildings,
                bus_stops=bus_stops
            )

            # ターゲット変数（v1.3: configから選択）
            hgnn_config = self.config.get('hgnn', {})
            requested_target = hgnn_config.get('target_column', 'betweenness')
            fallback_targets = [requested_target, 'indirect_damage_score', 'betweenness', 'importance_score']
            target_column = next((c for c in fallback_targets if c in bridge_gdf.columns), None)

            if target_column is None:
                raise ValueError(
                    "No valid target column found in bridge_gdf. "
                    f"Requested '{requested_target}', checked {fallback_targets}."
                )

            target_values = pd.to_numeric(bridge_gdf[target_column], errors='coerce').fillna(0.0).values
            data['bridge'].y = torch.tensor(target_values, dtype=torch.float).unsqueeze(1)
            data['bridge'].target_name = target_column
            logger.info(f"  Bridge target ({target_column}) shape: {data['bridge'].y.shape}")
        
        # Street ノード特徴量
        if 'street' in self.reverse_mapping and street_nodes is not None:
            data['street'].x = self._extract_street_features(street_nodes)
        
        # Building ノード特徴量
        if 'building' in self.reverse_mapping and buildings is not None:
            data['building'].x = self._extract_building_features(buildings)
        
        # Bus stop ノード特徴量
        if 'bus_stop' in self.reverse_mapping and bus_stops is not None:
            data['bus_stop'].x = self._extract_bus_stop_features(bus_stops)
    
    def _extract_bridge_features(
        self,
        bridge_gdf: gpd.GeoDataFrame,
        street_nodes: Optional[gpd.GeoDataFrame],
        buildings: Optional[gpd.GeoDataFrame],
        bus_stops: Optional[gpd.GeoDataFrame]
    ) -> torch.Tensor:
        """
        橋梁ノードの特徴量を抽出
        
        特徴量:
        - 健全度区分 (1-4, one-hot encoded)
        - 橋齢（年）
        - 橋長（m）
        - 幅員（m）
        - 環境リスク: 河川からの距離、海岸からの距離
        - 既存の重要度指標: betweenness, 周辺施設数
        - Binary flags: 離島架橋、長大橋、特殊橋、重要物流道路、緊急輸送道路
        """
        logger.info("  Extracting bridge features...")
        
        features = []
        feature_names = []
        
        # 1. 健全度区分（one-hot encoded）
        condition_cols = ['健全度Ⅰ', '健全度Ⅱ', '健全度Ⅲ', '健全度Ⅳ']
        if all(col in bridge_gdf.columns for col in condition_cols):
            condition_onehot = np.zeros((len(bridge_gdf), 4))
            for idx, row in bridge_gdf.iterrows():
                for i, col in enumerate(condition_cols):
                    if row[col] == '○':
                        condition_onehot[idx, i] = 1
                        break
            features.append(condition_onehot)
            feature_names.extend(['健全度Ⅰ', '健全度Ⅱ', '健全度Ⅲ', '健全度Ⅳ'])
        else:
            logger.warning("  Condition columns not found, using default")
            features.append(np.zeros((len(bridge_gdf), 4)))
            feature_names.extend(['健全度Ⅰ', '健全度Ⅱ', '健全度Ⅲ', '健全度Ⅳ'])
        
        # 2. 橋齢（2026 - 架設年）
        if '架設年（西暦）' in bridge_gdf.columns:
            construction_year = pd.to_numeric(bridge_gdf['架設年（西暦）'], errors='coerce').fillna(2000)
            age = 2026 - construction_year.values
            age = np.clip(age, 0, 200).reshape(-1, 1)  # 異常値を制限
            features.append(age)
            feature_names.append('橋齢')
        else:
            features.append(np.zeros((len(bridge_gdf), 1)))
            feature_names.append('橋齢')
        
        # 3. 橋長（m）
        if '橋長（m）' in bridge_gdf.columns:
            length = pd.to_numeric(bridge_gdf['橋長（m）'], errors='coerce').fillna(0).values.reshape(-1, 1)
            features.append(length)
            feature_names.append('橋長')
        else:
            features.append(np.zeros((len(bridge_gdf), 1)))
            feature_names.append('橋長')
        
        # 4. 幅員（m）
        if '幅員（m）' in bridge_gdf.columns:
            width = pd.to_numeric(bridge_gdf['幅員（m）'], errors='coerce').fillna(0).values.reshape(-1, 1)
            features.append(width)
            feature_names.append('幅員')
        else:
            features.append(np.zeros((len(bridge_gdf), 1)))
            feature_names.append('幅員')
        
        # 5. 環境リスク: 河川・海岸からの距離
        if 'dist_to_river' in bridge_gdf.columns:
            dist_river = bridge_gdf['dist_to_river'].fillna(1000).values.reshape(-1, 1)
            features.append(np.log1p(dist_river))  # log スケール
            feature_names.append('log_dist_river')
        else:
            features.append(np.zeros((len(bridge_gdf), 1)))
            feature_names.append('log_dist_river')
        
        if 'dist_to_coast' in bridge_gdf.columns:
            dist_coast = bridge_gdf['dist_to_coast'].fillna(50000).values.reshape(-1, 1)
            features.append(np.log1p(dist_coast))  # log スケール
            feature_names.append('log_dist_coast')
        else:
            features.append(np.zeros((len(bridge_gdf), 1)))
            feature_names.append('log_dist_coast')
        
        # 6. 既存の重要度指標 - betweennessは予測対象なので入力から除外
        # if 'betweenness' in bridge_gdf.columns:
        #     betweenness = bridge_gdf['betweenness'].fillna(0).values.reshape(-1, 1)
        #     features.append(betweenness)
        #     feature_names.append('betweenness')
        # else:
        #     features.append(np.zeros((len(bridge_gdf), 1)))
        #     feature_names.append('betweenness')
        
        # 7. 周辺施設数
        facility_cols = ['num_buildings', 'num_public_facilities', 'num_hospitals', 'num_schools']
        for col in facility_cols:
            if col in bridge_gdf.columns:
                count = bridge_gdf[col].fillna(0).values.reshape(-1, 1)
                features.append(count)
                feature_names.append(col)
            else:
                features.append(np.zeros((len(bridge_gdf), 1)))
                feature_names.append(col)
        
        # 8. Binary flags
        binary_flags = ['離島架橋', '長大橋', '特殊橋', '重要物流道路', '緊急輸送道路', '跨線橋', '跨道橋']
        for flag in binary_flags:
            if flag in bridge_gdf.columns:
                # '○' を 1、それ以外を 0 に変換
                values = (bridge_gdf[flag] == '○').astype(int).values.reshape(-1, 1)
                features.append(values)
                feature_names.append(flag)
            else:
                features.append(np.zeros((len(bridge_gdf), 1)))
                feature_names.append(flag)

        # 9. v1.3: k-NN による周辺ノード数特徴量
        logger.info("  Computing v1.3 k-NN neighborhood count features...")
        hgnn_config = self.config.get('hgnn', {})
        knn_k = int(hgnn_config.get('knn_k', 64))
        knn_max_distance = float(hgnn_config.get('knn_max_distance_m', 1000.0))

        bridge_coords = self._extract_xy_from_geometry(bridge_gdf)

        street_coords = self._extract_xy_from_geometry(street_nodes)
        building_coords = self._extract_xy_from_geometry(buildings)
        bus_stop_coords = self._extract_xy_from_geometry(bus_stops)

        knn_bus_count = self._compute_knn_count_feature(bridge_coords, bus_stop_coords, knn_k, knn_max_distance)
        knn_building_count = self._compute_knn_count_feature(bridge_coords, building_coords, knn_k, knn_max_distance)
        knn_street_count = self._compute_knn_count_feature(bridge_coords, street_coords, knn_k, knn_max_distance)

        features.append(knn_bus_count)
        feature_names.append('knn_bus_stop_count')

        features.append(knn_building_count)
        feature_names.append('knn_building_count')

        features.append(knn_street_count)
        feature_names.append('knn_street_node_count')
        
        # 特徴量を連結
        X = np.hstack(features)
        logger.info(f"  Bridge features (before processing): {X.shape[0]} nodes, {X.shape[1]} features")
        
        # 分散ゼロの特徴を除去（オプション）
        hgnn_config = self.config.get('hgnn', {})
        if hgnn_config.get('remove_zero_variance', True):
            # 各特徴の標準偏差を計算
            stds = X.std(axis=0)
            valid_features = stds > 1e-6  # 分散がほぼゼロの特徴を除外
            removed_count = (~valid_features).sum()
            
            if removed_count > 0:
                logger.info(f"  Removing {removed_count} features with zero variance")
                removed_names = [feature_names[i] for i, valid in enumerate(valid_features) if not valid]
                logger.info(f"  Removed features: {removed_names}")
                X = X[:, valid_features]
                feature_names = [name for i, name in enumerate(feature_names) if valid_features[i]]
        
        # 特徴量を正規化（オプション）
        if hgnn_config.get('normalize_features', True):
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            logger.info(f"  Features normalized (mean=0, std=1)")
        
        logger.info(f"  Bridge features (after processing): {X.shape[0]} nodes, {X.shape[1]} features")
        logger.info(f"  Final feature names: {feature_names}")
        
        return torch.tensor(X, dtype=torch.float32)

    def _extract_xy_from_geometry(self, df: Optional[pd.DataFrame]) -> Optional[np.ndarray]:
        """geometry列からXY座標を抽出（失敗時はNone）"""
        if df is None or len(df) == 0:
            return None

        if 'geometry' not in df.columns:
            return None

        try:
            coords = []
            for geom in df['geometry']:
                if geom is None:
                    continue
                if hasattr(geom, 'geom_type') and geom.geom_type in ('Polygon', 'MultiPolygon'):
                    c = geom.centroid
                    coords.append([c.x, c.y])
                elif hasattr(geom, 'x') and hasattr(geom, 'y'):
                    coords.append([geom.x, geom.y])
            if len(coords) == 0:
                return None
            return np.array(coords, dtype=float)
        except Exception as e:
            logger.warning(f"  Geometry to XY conversion failed: {e}")
            return None

    def _compute_knn_count_feature(
        self,
        source_xy: Optional[np.ndarray],
        target_xy: Optional[np.ndarray],
        k: int,
        max_distance: float
    ) -> np.ndarray:
        """
        source各点に対して target の k近傍を探索し、max_distance以内の点数を返す。
        """
        if source_xy is None:
            return np.zeros((0, 1), dtype=float)

        n_source = source_xy.shape[0]
        if target_xy is None or len(target_xy) == 0:
            return np.zeros((n_source, 1), dtype=float)

        k_eff = max(1, min(int(k), len(target_xy)))
        nn = NearestNeighbors(n_neighbors=k_eff)
        nn.fit(target_xy)

        distances, _ = nn.kneighbors(source_xy)
        counts = (distances <= float(max_distance)).sum(axis=1).astype(float).reshape(-1, 1)
        return counts
    
    def _extract_street_features(self, street_nodes: gpd.GeoDataFrame) -> torch.Tensor:
        """
        道路ノードの特徴量を抽出
        
        特徴量:
        - 道路種別（one-hot encoded）
        - 制限速度（取得可能な場合）
        - 車線数（取得可能な場合）
        - ノードの次数（接続数）
        """
        logger.info("  Extracting street features...")
        
        features = []
        
        # 基本特徴: 座標の正規化（簡易版）
        if 'x' in street_nodes.columns and 'y' in street_nodes.columns:
            x = street_nodes['x'].fillna(0).values.reshape(-1, 1)
            y = street_nodes['y'].fillna(0).values.reshape(-1, 1)
            # 正規化
            x = (x - x.mean()) / (x.std() + 1e-8)
            y = (y - y.mean()) / (y.std() + 1e-8)
            features.extend([x, y])
        else:
            # デフォルト：ゼロベクトル
            features.append(np.zeros((len(street_nodes), 2)))
        
        # その他の特徴は将来の実装で追加可能
        # 例: highway タイプ、maxspeed、lanes など
        
        X = np.hstack(features) if len(features) > 1 else features[0]
        logger.info(f"  Street features: {X.shape[0]} nodes, {X.shape[1]} features")
        
        return torch.tensor(X, dtype=torch.float32)
    
    def _extract_building_features(self, buildings: gpd.GeoDataFrame) -> torch.Tensor:
        """
        建物ノードの特徴量を抽出
        
        特徴量:
        - 建物種別（one-hot encoded）
        - 延床面積（取得可能な場合）
        """
        logger.info("  Extracting building features...")
        
        features = []
        
        # 建物種別（簡易版：category カラムがあれば使用）
        if 'category' in buildings.columns:
            categories = buildings['category'].fillna('other')
            unique_cats = ['residential', 'hospital', 'school', 'public', 'other']
            cat_onehot = np.zeros((len(buildings), len(unique_cats)))
            for idx, cat in enumerate(categories):
                if cat in unique_cats:
                    cat_onehot[idx, unique_cats.index(cat)] = 1
                else:
                    cat_onehot[idx, -1] = 1  # 'other'
            features.append(cat_onehot)
        else:
            # デフォルト: すべて 'other'
            cat_onehot = np.zeros((len(buildings), 5))
            cat_onehot[:, -1] = 1
            features.append(cat_onehot)
        
        X = np.hstack(features) if len(features) > 1 else features[0]
        logger.info(f"  Building features: {X.shape[0]} nodes, {X.shape[1]} features")
        
        return torch.tensor(X, dtype=torch.float32)
    
    def _extract_bus_stop_features(self, bus_stops: gpd.GeoDataFrame) -> torch.Tensor:
        """
        バス停ノードの特徴量を抽出
        
        特徴量:
        - 路線数（取得可能な場合）
        - 近接人口（将来実装）
        """
        logger.info("  Extracting bus stop features...")
        
        # 簡易版：座標のみ
        features = np.zeros((len(bus_stops), 2))
        
        logger.info(f"  Bus stop features: {features.shape[0]} nodes, {features.shape[1]} features")
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _extract_edge_indices(self, data: HeteroData, G: nx.Graph):
        """エッジインデックスを抽出"""
        logger.info("Extracting edge indices...")
        hgnn_cfg = self.config.get('hgnn', {})
        use_edge_attr = bool(hgnn_cfg.get('use_edge_attr', False))
        
        # エッジタイプごとに集約
        edge_dict = {}
        edge_attr_dict = {}
        
        for u, v, edge_data in G.edges(data=True):
            # ノードタイプを取得
            if u not in self.node_type_mapping or v not in self.node_type_mapping:
                continue
            
            u_type, u_id = self.node_type_mapping[u]
            v_type, v_id = self.node_type_mapping[v]
            
            # エッジタイプの作成
            edge_type = (u_type, 'to', v_type)
            
            if edge_type not in edge_dict:
                edge_dict[edge_type] = []
                if use_edge_attr:
                    edge_attr_dict[edge_type] = []
            
            edge_dict[edge_type].append([u_id, v_id])
            if use_edge_attr:
                w = edge_data.get('weight', 1.0)
                try:
                    w = float(w)
                except Exception:
                    w = 1.0
                edge_attr_dict[edge_type].append([w])
        
        # HeteroData にエッジインデックスを追加
        for edge_type, edges in edge_dict.items():
            if len(edges) > 0:
                edge_index = torch.tensor(edges, dtype=torch.long).t()
                data[edge_type].edge_index = edge_index
                logger.info(f"  {edge_type}: {edge_index.shape[1]} edges")
                if use_edge_attr:
                    edge_attr = torch.tensor(edge_attr_dict[edge_type], dtype=torch.float32)
                    data[edge_type].edge_attr = edge_attr
        
        return data
    
    def save_hetero_data(self, data: HeteroData, path: str):
        """HeteroData を保存"""
        torch.save(data, path)
        logger.info(f"HeteroData saved to {path}")
    
    def load_hetero_data(self, path: str) -> HeteroData:
        """HeteroData を読み込み"""
        data = torch.load(path)
        logger.info(f"HeteroData loaded from {path}")
        return data
