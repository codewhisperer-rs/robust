import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SignedConv


class SGCNEncoder(nn.Module):

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # 第一层
        self.conv1 = SignedConv(
            in_channels=in_channels,
            out_channels=hidden_channels // 2,
            first_aggr=True
        )
        
        # 第二层
        self.conv2 = SignedConv(
            in_channels=hidden_channels // 2,
            out_channels=hidden_channels // 2,
            first_aggr=False
        )
        
    def forward(self, x, edge_index, edge_weight):
        pos_mask = edge_weight > 0
        neg_mask = edge_weight < 0
        
        pos_edge_index = edge_index[:, pos_mask]
        neg_edge_index = edge_index[:, neg_mask]
        
        # 第一层卷积
        x = F.relu(self.conv1(x, pos_edge_index, neg_edge_index))
        
        # 第二层卷积
        x = F.relu(self.conv2(x, pos_edge_index, neg_edge_index))
        
        return x