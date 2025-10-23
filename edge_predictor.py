import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear


class EdgePredictor(nn.Module):

    def __init__(self, embed_dim, num_classes=2):
        super().__init__()
        self.predictor = nn.Sequential(
            Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            Linear(embed_dim, num_classes)
        )
    
    def forward(self, node_embed, edge_index):
        """
        Args:
            node_embed: 节点嵌入 [num_nodes, embed_dim]
            edge_index: 需要预测的边 [2, num_edges]
        Returns:
            edge_logits: 边分类logits [num_edges, num_classes]
        """
        row, col = edge_index
        edge_feat = torch.cat([node_embed[row], node_embed[col]], dim=1)
        return self.predictor(edge_feat)