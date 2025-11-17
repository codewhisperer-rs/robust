import torch
import torch.nn as nn
from sdgnn_encoder import SDGNNEncoder
from sgcn_encoder import SGCNEncoder
from learnable_edge_pruning import learnable_edge_pruning
from edge_predictor import EdgePredictor
from structural_balance_expansion import StructuralBalanceExpander


class ThreeStageSGNN(nn.Module):
    """
    Stage 1: 使用原始图编码初始节点嵌入 (h1)
    Stage 2: 结构平衡扩边生成 完整候选图(原始边 + 新增边)
    Stage 3: Gumbel-Sigmoid 基于h1共同训练，筛选完整候选图
    Stage 4: 在精炼图上再次编码 (h2)，融合 (h1, h2) 并预测
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_classes=2,
        fusion='concat',
        encoder_type='sdgnn',
        encoder_kwargs=None,
        balance_config=None,
        prune_threshold=0.5,
    ):
        super().__init__()

        encoder_type = encoder_type.lower()
        if encoder_type == 'sdgnn':
            encoder_cls = SDGNNEncoder
            self.encoder_kwargs = encoder_kwargs or {}
        elif encoder_type == 'sgcn':
            encoder_cls = SGCNEncoder
            self.encoder_kwargs = {}
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.encoder_type = encoder_type

        balance_args = dict(balance_config) if balance_config else {}
        balance_enabled = balance_args.pop('enabled', True)
        if balance_enabled:
            # Stage 2: 结构平衡扩边器，仅提供候选边
            self.balance_expander = StructuralBalanceExpander(**balance_args)
        else:
            self.balance_expander = None

        # Stage 1: 原图编码器
        if self.encoder_type == 'sdgnn':
            self.encoder1 = encoder_cls(in_channels, hidden_channels, **self.encoder_kwargs)
        else:
            self.encoder1 = encoder_cls(in_channels, hidden_channels)

        # Stage 3: Gumbel-Sigmoid候选边筛选器
        self.edge_learner = learnable_edge_pruning(hidden_channels, edge_threshold=prune_threshold)

        # Stage 4: 合并后图上的第二次编码
        if self.encoder_type == 'sdgnn':
            self.encoder2 = encoder_cls(hidden_channels, hidden_channels, **self.encoder_kwargs)
        else:
            self.encoder2 = encoder_cls(hidden_channels, hidden_channels)

        # 嵌入融合
        self.fusion = fusion
        if fusion == 'concat':
            # 拼接：[h1, h2]
            final_dim = 2 * hidden_channels
        elif fusion == 'add':
            # 残差连接：h1 + h2
            final_dim = hidden_channels
        elif fusion == 'attention':
            # 注意力融合
            self.attn = nn.Linear(2 * hidden_channels, 1)
            final_dim = hidden_channels
        elif fusion == 'gate':
            # 门控融合
            self.gate = nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels),
                nn.Sigmoid()
            )
            final_dim = hidden_channels
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

        # 边预测器
        self.edge_predictor = EdgePredictor(final_dim, num_classes)
        self.edge_threshold = prune_threshold

    def forward(self, x, edge_index, edge_weight, pred_edge_index, hard=False):
        """
        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            edge_weight: 边符号权重 [num_edges]
            pred_edge_index: 需要预测的边 [2, num_pred_edges]
            hard: 是否使用硬采样
        Returns:
            edge_logits: 边预测logits [num_pred_edges, num_classes]
        """
        num_nodes = x.size(0)

        # Stage 1: 使用原始图得到初始嵌入
        edge_weight = edge_weight.to(x.dtype)
        h1 = self.encoder1(x, edge_index, edge_weight)

        # Stage 2: 生成完整候选图（原始边 + 新增边）
        if self.balance_expander is not None:
            expanded_edge_index, expanded_edge_weight, balance_stats = self.balance_expander(
                edge_index, edge_weight, num_nodes
            )
            balance_stats = dict(balance_stats)
        else:
            expanded_edge_index, expanded_edge_weight = edge_index, edge_weight
            balance_stats = {
                'num_new_edges': 0,
                'new_edge_index': edge_index.new_empty((2, 0)),
                'new_edge_weight': edge_weight.new_empty(0),
            }

        expanded_edge_weight = expanded_edge_weight.to(x.dtype)

        # Stage 3: 用h1共同训练筛选器
        edge_output = self.edge_learner(h1, expanded_edge_index, hard, return_reg=True)
        if isinstance(edge_output, tuple):
            edge_probs, l1_reg = edge_output
        else:
            edge_probs = edge_output
            l1_reg = None
        refined_weight = expanded_edge_weight * edge_probs
        mask = refined_weight.abs() > self.edge_threshold
        refined_edge_index = expanded_edge_index[:, mask]
        refined_weight = refined_weight[mask]

        candidate_count = expanded_edge_index.size(1)
        retained_count = refined_edge_index.size(1)
        balance_stats['num_candidate_edges'] = candidate_count
        balance_stats['num_retained_edges'] = retained_count

        h2 = self.encoder2(h1, refined_edge_index, refined_weight)

        # Stage 4b: 融合h1和h2
        h_fused = self.fuse_embeddings(h1, h2)

        # Stage 4c: 边符号预测
        edge_logits = self.edge_predictor(h_fused, pred_edge_index)

        return {
            'logits': edge_logits,
            'l1_reg': l1_reg,
            'h1': h1,
            'h2': h2,
            'h_fused': h_fused,
            'edge_probs': edge_probs,
            'refined_edge_index': refined_edge_index,
            'refined_weight': refined_weight,
            'expanded_edge_index': expanded_edge_index,
            'expanded_edge_weight': expanded_edge_weight,
            'balance_stats': balance_stats,
        }

    def fuse_embeddings(self, h1, h2):
        """
        融合初始嵌入h1和refined嵌入h2
        """
        if self.fusion == 'concat':
            return torch.cat([h1, h2], dim=1)
        elif self.fusion == 'add':
            return h1 + h2
        elif self.fusion == 'attention':
            combined = torch.cat([h1, h2], dim=1)
            alpha = torch.sigmoid(self.attn(combined))  # [num_nodes, 1]
            return alpha * h1 + (1 - alpha) * h2
        elif self.fusion == 'gate':
            combined = torch.cat([h1, h2], dim=1)
            gate = self.gate(combined)  # [num_nodes, hidden_channels]
            return gate * h1 + (1 - gate) * h2
        else:
            raise ValueError(f"Unknown fusion: {self.fusion}")

    def get_edge_probs(self, x, edge_index, edge_weight):
        """获取完整候选图在Gumbel筛选前的概率"""
        with torch.no_grad():
            edge_weight = edge_weight.to(x.dtype)
            h1 = self.encoder1(x, edge_index, edge_weight)

            if self.balance_expander is not None:
                expanded_edge_index, _, _ = self.balance_expander(edge_index, edge_weight, x.size(0))
            else:
                expanded_edge_index = edge_index

            if expanded_edge_index.size(1) == 0:
                return h1.new_empty(0)

            edge_output = self.edge_learner(h1.detach(), expanded_edge_index, hard=False, return_reg=True)
            if isinstance(edge_output, tuple):
                edge_probs, _ = edge_output
            else:
                edge_probs = edge_output
        return edge_probs

    def set_temp(self, temp):
        """设置Gumbel温度"""
        self.edge_learner.set_temp(temp)
