import torch
import torch.nn as nn
from torch.nn import Linear
from torch.distributions.gumbel import Gumbel
from torch import Tensor


def gumbel_sigmoid(logits: Tensor, temp: float = 1.0, hard: bool = False, threshold: float = 0.5) -> Tensor:
    if temp <= 0:
        raise ValueError("Temperature must be positive.")

    noise = Gumbel(0, 1).sample(logits.shape).to(logits.device) - \
            Gumbel(0, 1).sample(logits.shape).to(logits.device)
    y_soft = torch.sigmoid((logits + noise) / temp)

    if hard:
        y_hard = (y_soft > threshold).float()
        return y_hard - y_soft.detach() + y_soft

    return y_soft


class learnable_edge_pruning(nn.Module):
    """Gumbel-Sigmoid选择边"""

    def __init__(self, embed_dim, hidden_dim=64, edge_threshold=1e-4):
        super().__init__()
        self.edge_scorer = nn.Sequential(
            Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            Linear(hidden_dim, 1)
        )
        self.temp = 1.0
        self.edge_threshold = edge_threshold

    def forward(self, node_embed, edge_index, hard=False, return_reg=False):
        row, col = edge_index
        edge_feat = torch.cat([node_embed[row], node_embed[col]], dim=1)
        edge_logits = self.edge_scorer(edge_feat).squeeze(-1)

        edge_probs = gumbel_sigmoid(
            edge_logits,
            temp=self.temp,
            hard=hard,
            threshold=self.edge_threshold,
        )

        if return_reg:
            l1_reg = torch.mean(torch.abs(edge_probs))
            return edge_probs, l1_reg

        return edge_probs

    def set_temp(self, temp):
        self.temp = temp
