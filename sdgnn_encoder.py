import math
from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class SDRLayerDynamic(nn.Module):
    """SDR layer that mirrors the SDGNN building block but accepts dynamic edge lists."""

    def __init__(self, in_dim: int, out_dim: int, num_relations: int = 4, heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.num_relations = num_relations
        self.heads = heads

        aggs = []
        for _ in range(num_relations):
            aggs.append(
                GATConv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    heads=heads,
                    dropout=dropout,
                    add_self_loops=True,
                    bias=True
                )
            )
        self.aggs = nn.ModuleList(aggs)

        gat_out_dim = out_dim * heads
        self.mlp_layer = nn.Sequential(
            nn.Linear(in_dim + gat_out_dim * num_relations, out_dim),
            nn.Tanh(),
            nn.Linear(out_dim, out_dim)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for agg in self.aggs:
            agg.reset_parameters()
        for module in self.mlp_layer:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5.0))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, edge_lists: List[torch.Tensor]) -> torch.Tensor:
        assert len(edge_lists) == self.num_relations, (
            f"Expected {self.num_relations} relation edge lists, got {len(edge_lists)}"
        )
        neigh_feats = []
        for edges, agg in zip(edge_lists, self.aggs):
            if edges.numel() == 0:
                out_dim = agg.out_channels * (agg.heads if agg.concat else 1)
                neigh_feats.append(x.new_zeros(x.size(0), out_dim))
                continue
            neigh_feats.append(agg(x, edges))

        combined = torch.cat([x] + neigh_feats, dim=1)
        return self.mlp_layer(combined)


class SDGNNEncoder(nn.Module):
    """
    SDGNN-style encoder that produces node embeddings for signed directed graphs.
    It mirrors the behaviour of the original SDGNN forward pass while exposing a
    PyG-friendly interface (x, edge_index, edge_weight) so it can be swapped into
    the existing training pipeline.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        heads: int = 2,
        use_residual: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.dropout = nn.Dropout(dropout)
        self.heads = heads

        layers = []
        norms = []
        for layer_idx in range(num_layers):
            layer_in = in_channels if layer_idx == 0 else hidden_channels
            layers.append(
                SDRLayerDynamic(
                    in_dim=layer_in,
                    out_dim=hidden_channels,
                    num_relations=4,
                    heads=heads,
                    dropout=dropout
                )
            )
            norms.append(nn.LayerNorm(hidden_channels))

        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        edge_lists = self.build_edge_lists(edge_index, edge_weight, x.size(0), x.device)

        h = x
        for layer, norm in zip(self.layers, self.norms):
            residual = h
            h = layer(h, edge_lists)
            h = self.dropout(h)
            if self.use_residual and h.shape == residual.shape:
                h = h + residual
            h = norm(h)
        return h

    @staticmethod
    def build_edge_lists(edge_index: torch.Tensor,
                         edge_weight: torch.Tensor,
                         num_nodes: int,
                         device: torch.device) -> List[torch.Tensor]:
        if edge_index.size(1) == 0:
            empty = torch.empty((2, 0), dtype=torch.long, device=device)
            return [empty, empty.clone(), empty.clone(), empty.clone()]

        edge_index_cpu = edge_index.detach().cpu()
        edge_weight_cpu = edge_weight.detach().cpu()

        pos_mask = edge_weight_cpu > 0
        neg_mask = edge_weight_cpu < 0

        pos_src = edge_index_cpu[0, pos_mask].tolist()
        pos_dst = edge_index_cpu[1, pos_mask].tolist()
        neg_src = edge_index_cpu[0, neg_mask].tolist()
        neg_dst = edge_index_cpu[1, neg_mask].tolist()

        pos_out = defaultdict(set)
        pos_in = defaultdict(set)
        neg_out = defaultdict(set)
        neg_in = defaultdict(set)

        for u, v in zip(pos_src, pos_dst):
            pos_out[u].add(v)
            pos_in[v].add(u)
        for u, v in zip(neg_src, neg_dst):
            neg_out[u].add(v)
            neg_in[v].add(u)

        # Ensure dictionaries cover isolated nodes to keep device tensors consistent.
        for node_id in range(num_nodes):
            pos_out.setdefault(node_id, set())
            pos_in.setdefault(node_id, set())
            neg_out.setdefault(node_id, set())
            neg_in.setdefault(node_id, set())

        return [
            SDGNNEncoder.dict_to_edge_index(pos_out, device),
            SDGNNEncoder.dict_to_edge_index(pos_in, device),
            SDGNNEncoder.dict_to_edge_index(neg_out, device),
            SDGNNEncoder.dict_to_edge_index(neg_in, device),
        ]

    @staticmethod
    def dict_to_edge_index(adj_dict: defaultdict, device: torch.device) -> torch.Tensor:
        rows = []
        cols = []
        for src, targets in adj_dict.items():
            for dst in targets:
                rows.append(src)
                cols.append(dst)

        if not rows:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        return edge_index.to(device=device, non_blocking=True)
