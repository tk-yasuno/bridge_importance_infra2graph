"""
HGNN モデル定義
Bridge Importance Scoring MVP v1.1

HeteroConv + GATConv/SAGEConv を使った異種グラフニューラルネットワーク
橋梁ノードの重要度スコアを予測（ノード回帰タスク）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv, Linear
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BridgeImportanceHGNN(nn.Module):
    """
    橋梁重要度予測のための Heterogeneous GNN
    
    アーキテクチャ:
    - Input: HeteroData (bridge, street, building, bus_stop)
    - Hidden: HeteroConv layers with GATConv or SAGEConv
    - Output: Bridge ノードの重要度スコア (回帰)
    """
    
    def __init__(
        self,
        node_types: list,
        edge_types: list,
        in_channels_dict: Dict[str, int],
        hidden_channels: int = 64,
        out_channels: int = 1,
        num_layers: int = 2,
        conv_type: str = 'GAT',
        dropout: float = 0.2,
        heads: int = 4,
        use_edge_attr: bool = False
    ):
        """
        Args:
            node_types: ノードタイプのリスト ['bridge', 'street', ...]
            edge_types: エッジタイプのリスト [('bridge', 'to', 'street'), ...]
            in_channels_dict: ノードタイプごとの入力特徴次元 {'bridge': 25, 'street': 2, ...}
            hidden_channels: 隠れ層の次元数
            out_channels: 出力次元数（デフォルト: 1, 回帰タスク）
            num_layers: HeteroConv 層の数
            conv_type: 'GAT' or 'SAGE'
            dropout: ドロップアウト率
            heads: GAT の attention heads 数
        """
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.conv_type = conv_type
        self.dropout = dropout
        self.heads = heads if conv_type == 'GAT' else 1
        self.use_edge_attr = use_edge_attr and (conv_type == 'GAT')
        
        # ノードタイプごとの入力線形変換（特徴次元を統一）
        self.node_encoders = nn.ModuleDict()
        for node_type in node_types:
            in_channels = in_channels_dict.get(node_type, 1)
            self.node_encoders[node_type] = Linear(in_channels, hidden_channels)
        
        # HeteroConv 層
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                src_type, _, dst_type = edge_type
                
                if conv_type == 'GAT':
                    # GAT: Multi-head attention
                    if i == 0:
                        conv_dict[edge_type] = GATConv(
                            hidden_channels, 
                            hidden_channels // heads,
                            heads=heads,
                            dropout=dropout,
                            add_self_loops=False,
                            edge_dim=1 if self.use_edge_attr else None
                        )
                    else:
                        conv_dict[edge_type] = GATConv(
                            hidden_channels,
                            hidden_channels // heads,
                            heads=heads,
                            dropout=dropout,
                            add_self_loops=False,
                            edge_dim=1 if self.use_edge_attr else None
                        )
                elif conv_type == 'SAGE':
                    # GraphSAGE
                    conv_dict[edge_type] = SAGEConv(
                        hidden_channels,
                        hidden_channels,
                        aggr='mean'
                    )
                else:
                    raise ValueError(f"Unknown conv_type: {conv_type}")
            
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
        
        # 出力層（bridge ノードのみ）
        self.output_layer = nn.Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_channels // 2, out_channels)
        )
        
        logger.info(f"BridgeImportanceHGNN initialized:")
        logger.info(f"  Node types: {node_types}")
        logger.info(f"  Edge types: {len(edge_types)} types")
        logger.info(f"  Hidden channels: {hidden_channels}")
        logger.info(f"  Num layers: {num_layers}")
        logger.info(f"  Conv type: {conv_type}")
        logger.info(f"  Dropout: {dropout}")
        if conv_type == 'GAT':
            logger.info(f"  GAT heads: {heads}")
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """
        Forward pass
        
        Args:
            x_dict: ノードタイプごとの特徴量辞書 {'bridge': Tensor, 'street': Tensor, ...}
            edge_index_dict: エッジタイプごとのエッジインデックス {('bridge', 'to', 'street'): Tensor, ...}
        
        Returns:
            bridge ノードの予測スコア (Tensor, shape: [num_bridges, 1])
        """
        # 1. ノードエンコーディング（各ノードタイプの特徴を hidden_channels に変換）
        x_dict_encoded = {}
        for node_type, x in x_dict.items():
            if node_type in self.node_encoders:
                x_dict_encoded[node_type] = F.relu(self.node_encoders[node_type](x))
            else:
                # 未知のノードタイプはスキップ
                logger.warning(f"Node type '{node_type}' not in encoders, skipping")
        
        # 2. HeteroConv 層を通過
        for i, conv in enumerate(self.convs):
            if self.use_edge_attr and edge_attr_dict is not None:
                x_dict_encoded = conv(x_dict_encoded, edge_index_dict, edge_attr_dict=edge_attr_dict)
            else:
                x_dict_encoded = conv(x_dict_encoded, edge_index_dict)
            
            # 活性化関数とドロップアウト（最後の層以外）
            if i < self.num_layers - 1:
                x_dict_encoded = {
                    key: F.relu(x) for key, x in x_dict_encoded.items()
                }
                x_dict_encoded = {
                    key: F.dropout(x, p=self.dropout, training=self.training)
                    for key, x in x_dict_encoded.items()
                }
        
        # 3. Bridge ノードの出力を取得
        if 'bridge' not in x_dict_encoded:
            raise ValueError("'bridge' node type not found in x_dict_encoded")
        
        bridge_embeddings = x_dict_encoded['bridge']
        
        # 4. 出力層（回帰）
        out = self.output_layer(bridge_embeddings)
        
        return out
    
    def get_node_embeddings(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """
        ノード埋め込みを取得（評価・可視化用）
        
        Returns:
            ノードタイプごとの埋め込み辞書
        """
        # Forward pass without output layer
        x_dict_encoded = {}
        for node_type, x in x_dict.items():
            if node_type in self.node_encoders:
                x_dict_encoded[node_type] = F.relu(self.node_encoders[node_type](x))
        
        for conv in self.convs:
            if self.use_edge_attr and edge_attr_dict is not None:
                x_dict_encoded = conv(x_dict_encoded, edge_index_dict, edge_attr_dict=edge_attr_dict)
            else:
                x_dict_encoded = conv(x_dict_encoded, edge_index_dict)
            x_dict_encoded = {
                key: F.relu(x) for key, x in x_dict_encoded.items()
            }
        
        return x_dict_encoded


class BridgeImportanceHGNN_Simple(nn.Module):
    """
    簡易版 HGNN（小規模データ用）
    1層のみの HeteroConv
    """
    
    def __init__(
        self,
        node_types: list,
        edge_types: list,
        in_channels_dict: Dict[str, int],
        hidden_channels: int = 32,
        out_channels: int = 1,
        conv_type: str = 'SAGE'
    ):
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        
        # ノードエンコーダ
        self.node_encoders = nn.ModuleDict()
        for node_type in node_types:
            in_channels = in_channels_dict.get(node_type, 1)
            self.node_encoders[node_type] = Linear(in_channels, hidden_channels)
        
        # 1層の HeteroConv
        conv_dict = {}
        for edge_type in edge_types:
            if conv_type == 'SAGE':
                conv_dict[edge_type] = SAGEConv(hidden_channels, hidden_channels)
            elif conv_type == 'GAT':
                conv_dict[edge_type] = GATConv(hidden_channels, hidden_channels, heads=1)
        
        self.conv = HeteroConv(conv_dict, aggr='sum')
        
        # 出力層
        self.output = Linear(hidden_channels, out_channels)
        
        logger.info(f"BridgeImportanceHGNN_Simple initialized:")
        logger.info(f"  Node types: {node_types}")
        logger.info(f"  Hidden channels: {hidden_channels}")
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # エンコーディング
        x_dict_encoded = {
            node_type: F.relu(self.node_encoders[node_type](x))
            for node_type, x in x_dict.items()
            if node_type in self.node_encoders
        }
        
        # HeteroConv
        x_dict_encoded = self.conv(x_dict_encoded, edge_index_dict)
        x_dict_encoded = {key: F.relu(x) for key, x in x_dict_encoded.items()}
        
        # Bridge ノードの出力
        bridge_out = self.output(x_dict_encoded['bridge'])
        
        return bridge_out


def create_model(
    data,
    model_type: str = 'standard',
    hidden_channels: int = 64,
    num_layers: int = 2,
    conv_type: str = 'GAT',
    dropout: float = 0.2,
    heads: int = 4,
    use_edge_attr: bool = False
):
    """
    モデルを作成するヘルパー関数
    
    Args:
        data: HeteroData
        model_type: 'standard' or 'simple'
        hidden_channels: 隠れ層の次元数
        num_layers: 層の数
        conv_type: 'GAT' or 'SAGE'
        dropout: ドロップアウト率
        heads: GAT の attention heads 数
    
    Returns:
        HGNN モデル
    """
    # ノードタイプと入力次元を抽出
    node_types = list(data.node_types)
    edge_types = list(data.edge_types)
    
    in_channels_dict = {}
    for node_type in node_types:
        if hasattr(data[node_type], 'x'):
            in_channels_dict[node_type] = data[node_type].x.shape[1]
        else:
            in_channels_dict[node_type] = 1
    
    if model_type == 'simple':
        model = BridgeImportanceHGNN_Simple(
            node_types=node_types,
            edge_types=edge_types,
            in_channels_dict=in_channels_dict,
            hidden_channels=hidden_channels,
            conv_type=conv_type
        )
    else:
        model = BridgeImportanceHGNN(
            node_types=node_types,
            edge_types=edge_types,
            in_channels_dict=in_channels_dict,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            conv_type=conv_type,
            dropout=dropout,
            heads=heads,
            use_edge_attr=use_edge_attr
        )
    
    return model
