import torch
import torch.nn as nn
from sdgnn_encoder import SDGNNEncoder
from sgcn_encoder import SGCNEncoder
from learnable_edge_pruning import learnable_edge_pruning
from edge_predictor import EdgePredictor
from structural_balance_expansion import StructuralBalanceExpander


class ThreeStageSGNN(nn.Module):
    """
    Stage 1: 结构平衡扩边，强化图结构
    Stage 2: 编码器（SDGNN/SGCN）学习初始节点嵌入
    Stage 3: Gumbel-Sigmoid学习边重要性，重加权边
    Stage 4: 再次启用编码器，拼接/融合嵌入，预测边符号
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
        prune_threshold=1e-3,
    ):
        """
        Args:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            num_classes: 分类类别数
            fusion: 嵌入融合方式 ('concat', 'add', 'attention', 'gate')
            encoder_type: 编码器类型 ('sdgnn' 或 'sgcn')
            encoder_kwargs: 额外的编码器参数（针对SDGNN）
        """
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
            self.balance_expander = StructuralBalanceExpander(**balance_args)
        else:
            self.balance_expander = None

        # Stage 2: 初始节点嵌入
        if self.encoder_type == 'sdgnn':
            self.encoder1 = encoder_cls(in_channels, hidden_channels, **self.encoder_kwargs)
        else:
            self.encoder1 = encoder_cls(in_channels, hidden_channels)

        # Stage 3: Gumbel-Sigmoid
        self.edge_learner = learnable_edge_pruning(hidden_channels, edge_threshold=prune_threshold)
        
        # Stage 4: Refined嵌入
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
        # Stage 1: 结构平衡扩边，得到强化图
        if self.balance_expander is not None:
            expanded_edge_index, expanded_edge_weight, balance_stats = self.balance_expander(
                edge_index, edge_weight, num_nodes
            )
        else:
            expanded_edge_index, expanded_edge_weight, balance_stats = (
                edge_index,
                edge_weight,
                {'num_new_edges': 0},
            )

        expanded_edge_weight = expanded_edge_weight.to(x.dtype)

        # Stage 2: 基于扩边后的图计算初始嵌入
        h1 = self.encoder1(x, expanded_edge_index, expanded_edge_weight)

        # Stage 3: Gumbel-Sigmoid
        edge_output = self.edge_learner(h1, expanded_edge_index, hard, return_reg=True)
        if isinstance(edge_output, tuple):
            edge_probs, l1_reg = edge_output
        else:
            edge_probs = edge_output
            l1_reg = None
        
        # 重加权边：原始符号 × 学习到的概率
        refined_weight = expanded_edge_weight * edge_probs
        
        # 过滤低权重边
        mask = refined_weight.abs() > self.edge_threshold
        refined_edge_index = expanded_edge_index[:, mask]
        refined_weight = refined_weight[mask]
        
        # Stage 4a: 再次GNN
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
        
        Args:
            h1: 初始节点嵌入 [num_nodes, hidden_channels]
            h2: refined节点嵌入 [num_nodes, hidden_channels]
        Returns:
            h_fused: 融合后的嵌入
        """
        if self.fusion == 'concat':
            # 简单拼接：保留所有信息
            return torch.cat([h1, h2], dim=1)
        
        elif self.fusion == 'add':
            # 残差连接：h1 + h2
            return h1 + h2
        
        elif self.fusion == 'attention':
            # 注意力加权融合
            combined = torch.cat([h1, h2], dim=1)
            alpha = torch.sigmoid(self.attn(combined))  # [num_nodes, 1]
            return alpha * h1 + (1 - alpha) * h2
        
        elif self.fusion == 'gate':
            # 门控融合
            combined = torch.cat([h1, h2], dim=1)
            gate = self.gate(combined)  # [num_nodes, hidden_channels]
            return gate * h1 + (1 - gate) * h2
        
        else:
            raise ValueError(f"Unknown fusion: {self.fusion}")
    
    def get_edge_probs(self, x, edge_index, edge_weight):
        """获取学习到的边概率"""
        with torch.no_grad():
            if self.balance_expander is not None:
                edge_index_aug, edge_weight_aug, _ = self.balance_expander(edge_index, edge_weight, x.size(0))
            else:
                edge_index_aug, edge_weight_aug = edge_index, edge_weight

            h1 = self.encoder1(x, edge_index_aug, edge_weight_aug.to(x.dtype))
            edge_probs = self.edge_learner(h1, edge_index_aug, hard=False)
        return edge_probs
    
    def set_temp(self, temp):
        """设置Gumbel温度"""
        self.edge_learner.set_temp(temp)
