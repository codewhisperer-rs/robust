import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear


def gumbel_sigmoid(logits, temp=1.0, hard=False):
    """
    Gumbel-Sigmoid采样
    
    Args:
        logits: 未归一化的logits
        temp: 温度参数
        hard: 是否使用硬采样
    Returns:
        采样概率
    """
    eps = 1e-10
    u1 = torch.rand_like(logits).clamp(eps, 1 - eps)
    u2 = torch.rand_like(logits).clamp(eps, 1 - eps)
    
    g1 = -torch.log(-torch.log(u1))
    g2 = -torch.log(-torch.log(u2))
    
    y_soft = torch.sigmoid((logits + g1 - g2) / temp)
    
    if hard:
        y_hard = (y_soft > 0.5).float()
        return y_hard - y_soft.detach() + y_soft
    
    return y_soft


class learnable_edge_pruning(nn.Module):
    """
    使用Gumbel-Sigmoid学习边权重
    """
    def __init__(self, embed_dim, hidden_dim=64):
        super().__init__()
        self.edge_scorer = nn.Sequential(
            Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            Linear(hidden_dim, 1)
        )
        self.temp = 1.0
        
    def forward(self, node_embed, edge_index, hard=False, return_reg=False):
        """
        Args:
            node_embed: 节点嵌入 [num_nodes, embed_dim]
            edge_index: 边索引 [2, num_edges]
            hard: 是否使用硬采样
            return_reg: 是否返回L1正则化项
        Returns:
            edge_probs: 边保留概率 [num_edges]
            l1_reg: L1正则化项（如果return_reg=True）
        """
        row, col = edge_index
        edge_feat = torch.cat([node_embed[row], node_embed[col]], dim=1)
        edge_logits = self.edge_scorer(edge_feat).squeeze(-1)
        
        # Gumbel-Sigmoid采样
        edge_probs = gumbel_sigmoid(edge_logits, self.temp, hard)
        
        if return_reg:
            # 计算L1正则化项
            l1_reg = torch.mean(torch.abs(edge_probs))
            return edge_probs, l1_reg
        
        return edge_probs
    
    def set_temp(self, temp):
        """设置温度（用于温度退火）"""
        self.temp = temp