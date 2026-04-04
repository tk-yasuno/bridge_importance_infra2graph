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
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Custom SAGE + Attention Layers (v1.6 exp11-13)
# ============================================================================

class SimpleAttentionSAGEConv(MessagePassing):
    """
    SAGEConv + Simple Attention (exp11)
    
    ノード特徴ベースの単純なattention重み学習。
    message段階で近傍特徴に重み付けし、max aggregationを適用。
    """
    
    def __init__(self, in_channels, out_channels, aggr='max'):
        super().__init__(aggr=aggr)
        self.lin_l = Linear(in_channels, out_channels)
        self.lin_r = Linear(in_channels, out_channels, bias=False)
        # Attention layer: node features → scalar weight
        self.attention = Linear(in_channels, 1)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        self.attention.reset_parameters()
    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [N, in_channels] or tuple of (src, dst) features
            edge_index: Edge indices [2, E]
        
        Returns:
            out: Updated node features [N, out_channels]
        """
        # Handle heterogeneous edge types (x can be a tuple)
        if isinstance(x, tuple):
            x_src, x_dst = x
        else:
            x_src = x_dst = x
        
        # Compute attention weights from source features
        attn = torch.sigmoid(self.attention(x_src))  # [N_src, 1]
        
        # Propagate messages with attention
        out = self.propagate(edge_index, x=(x_src, x_dst), attn=attn, size=(x_src.size(0), x_dst.size(0)))
        
        # Combine aggregated neighbors + root node
        out = self.lin_l(out) + self.lin_r(x_dst)
        return out
    
    def message(self, x_j, attn_j):
        """
        Apply attention weights before aggregation.
        
        Args:
            x_j: Neighbor features [E, in_channels]
            attn_j: Attention weights for neighbors [E, 1]
        
        Returns:
            Weighted neighbor features [E, in_channels]
        """
        return attn_j * x_j


class GATv2StyleSAGEConv(MessagePassing):
    """
    SAGEConv + GATv2-style Attention (exp12)
    
    ノードペア特徴（src + dst）を使用したattention計算。
    LeakyReLU + softmax正規化でより洗練されたattention。
    """
    
    def __init__(self, in_channels, out_channels, aggr='max', negative_slope=0.2):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        
        self.lin_l = Linear(in_channels, out_channels)
        self.lin_r = Linear(in_channels, out_channels, bias=False)
        
        # GATv2-style attention (additive)
        self.att_src = Linear(in_channels, 1, bias=False)
        self.att_dst = Linear(in_channels, 1, bias=False)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        self.att_src.reset_parameters()
        self.att_dst.reset_parameters()
    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [N, in_channels] or tuple of (src, dst) features
            edge_index: Edge indices [2, E]
        
        Returns:
            out: Updated node features [N, out_channels]
        """
        # Handle heterogeneous edge types (x can be a tuple)
        if isinstance(x, tuple):
            x_src, x_dst = x
        else:
            x_src = x_dst = x
        
        # Propagate with GATv2-style attention
        out = self.propagate(edge_index, x=(x_src, x_dst), size=(x_src.size(0), x_dst.size(0)))
        
        # Combine aggregated neighbors + root node
        out = self.lin_l(out) + self.lin_r(x_dst)
        return out
    
    def message(self, x_i, x_j, index, ptr, size_i):
        """
        Compute GATv2-style attention and apply to neighbor features.
        
        Args:
            x_i: Target node features [E, in_channels]
            x_j: Source (neighbor) node features [E, in_channels]
            index: Target node indices for each edge [E]
            ptr: Index pointer (for softmax)
            size_i: Number of target nodes
        
        Returns:
            Attention-weighted neighbor features [E, in_channels]
        """
        # Additive attention: a(x_i, x_j) = LeakyReLU(W_src^T x_i + W_dst^T x_j)
        alpha = self.att_src(x_i) + self.att_dst(x_j)  # [E, 1]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Softmax normalization per target node
        alpha = softmax(alpha, index, ptr, size_i)  # [E, 1]
        
        # Apply attention weights
        return alpha * x_j


class MetapathAwareSAGEConv(MessagePassing):
    """
    SAGEConv + Metapath-Aware Attention (exp13)
    
    メタパス特徴（5次元: 2hop_neighbors, street_degree_avg等）を
    明示的に活用したattention計算。Query-Key構造で実装。
    """
    
    def __init__(self, in_channels, out_channels, metapath_dim=5, aggr='max', att_hidden=16):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.metapath_dim = metapath_dim
        self.att_hidden = att_hidden
        
        self.lin_l = Linear(in_channels, out_channels)
        self.lin_r = Linear(in_channels, out_channels, bias=False)
        
        # Attention with metapath features (Query-Key-Value structure)
        # Query: target node features + metapath features
        self.att_query = Linear(in_channels + metapath_dim, att_hidden)
        # Key: source (neighbor) node features
        self.att_key = Linear(in_channels, att_hidden)
        # Value projection to scalar attention
        self.att_value = Linear(att_hidden, 1)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        self.att_query.reset_parameters()
        self.att_key.reset_parameters()
        self.att_value.reset_parameters()
    
    def forward(self, x, edge_index, metapath_features=None):
        """
        Args:
            x: Node features [N, in_channels] or tuple of (src, dst) features
            edge_index: Edge indices [2, E]
            metapath_features: Metapath features [N, metapath_dim] (optional)
        
        Returns:
            out: Updated node features [N, out_channels]
        """
        # Handle heterogeneous edge types (x can be a tuple)
        if isinstance(x, tuple):
            x_src, x_dst = x
        else:
            x_src = x_dst = x
        
        # Augment node features with metapath features
        if metapath_features is not None:
            # metapath_features should match source nodes
            if isinstance(metapath_features, tuple):
                metapath_src, metapath_dst = metapath_features
            else:
                metapath_src = metapath_dst = metapath_features
            
            x_aug_src = torch.cat([x_src, metapath_src], dim=-1)  # [N_src, in_channels+metapath_dim]
            x_aug_dst = torch.cat([x_dst, metapath_dst], dim=-1)  # [N_dst, in_channels+metapath_dim]
        else:
            # If no metapath features, pad with zeros
            x_aug_src = torch.cat([x_src, torch.zeros(x_src.size(0), self.metapath_dim, device=x_src.device)], dim=-1)
            x_aug_dst = torch.cat([x_dst, torch.zeros(x_dst.size(0), self.metapath_dim, device=x_dst.device)], dim=-1)
        
        # Propagate with metapath-aware attention
        out = self.propagate(edge_index, x=(x_src, x_dst), x_aug=(x_aug_src, x_aug_dst), size=(x_src.size(0), x_dst.size(0)))
        
        # Combine aggregated neighbors + root node
        out = self.lin_l(out) + self.lin_r(x_dst)
        return out
    
    def message(self, x_j, x_aug_i, x_aug_j, index, ptr, size_i):
        """
        Compute metapath-aware attention.
        
        Args:
            x_j: Source node features [E, in_channels]
            x_aug_i: Target node augmented features [E, in_channels+metapath_dim]
            x_aug_j: Source node augmented features [E, in_channels+metapath_dim]
            index: Target node indices [E]
            ptr: Index pointer
            size_i: Number of target nodes
        
        Returns:
            Attention-weighted neighbor features [E, in_channels]
        """
        # Query from target (with metapath), Key from source
        query = self.att_query(x_aug_i)  # [E, att_hidden]
        key = self.att_key(x_j)          # [E, att_hidden]
        
        # Dot product attention
        alpha = torch.sum(query * key, dim=-1, keepdim=True)  # [E, 1]
        alpha = F.relu(alpha)  # [E, 1]
        
        # Softmax normalization per target node
        alpha = softmax(alpha, index, ptr, size_i)  # [E, 1]
        
        # Apply attention weights
        return alpha * x_j


# ============================================================================
# Main HGNN Model
# ============================================================================

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
        use_edge_attr: bool = False,
        sage_aggr: str = 'mean',
        attention_type: str = 'none'
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
            sage_aggr: SAGE aggregation method ('mean', 'max', 'sum')
            attention_type: SAGE+Attention type ('none', 'simple', 'gatv2', 'metapath')
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
        self.sage_aggr = sage_aggr
        self.attention_type = attention_type
        
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
                    # GraphSAGE with optional Attention
                    if attention_type == 'simple':
                        # exp11: SimpleAttentionSAGE
                        conv_dict[edge_type] = SimpleAttentionSAGEConv(
                            hidden_channels,
                            hidden_channels,
                            aggr=self.sage_aggr
                        )
                    elif attention_type == 'gatv2':
                        # exp12: GATv2StyleSAGE
                        conv_dict[edge_type] = GATv2StyleSAGEConv(
                            hidden_channels,
                            hidden_channels,
                            aggr=self.sage_aggr
                        )
                    elif attention_type == 'metapath':
                        # exp13: MetapathAwareSAGE
                        conv_dict[edge_type] = MetapathAwareSAGEConv(
                            hidden_channels,
                            hidden_channels,
                            metapath_dim=5,  # 5 metapath features
                            aggr=self.sage_aggr
                        )
                    else:
                        # Default: standard SAGEConv (no attention)
                        conv_dict[edge_type] = SAGEConv(
                            hidden_channels,
                            hidden_channels,
                            aggr=self.sage_aggr
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
        elif conv_type == 'SAGE':
            logger.info(f"  SAGE aggregation: {sage_aggr}")
            logger.info(f"  Attention type: {attention_type}")
    
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
    use_edge_attr: bool = False,
    sage_aggr: str = 'mean',
    attention_type: str = 'none'
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
        sage_aggr: SAGE aggregation method ('mean', 'max', 'sum')
        attention_type: SAGE+Attention type ('none', 'simple', 'gatv2', 'metapath')
    
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
            use_edge_attr=use_edge_attr,
            sage_aggr=sage_aggr,
            attention_type=attention_type
        )
    
    return model
