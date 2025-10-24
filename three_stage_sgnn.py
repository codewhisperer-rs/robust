import torch
import torch.nn as nn
from sgcn_encoder import SGCNEncoder
from learnable_edge_pruning import learnable_edge_pruning
from edge_predictor import EdgePredictor


class ThreeStageSGNN(nn.Module):
    """
    Stage 1: SGCN学习初始节点嵌入
    Stage 2: Gumbel-Sigmoid学习边重要性，重加权边
    Stage 3: 再次SGCN，拼接/融合嵌入，预测边符号
    """
    def __init__(self, in_channels, hidden_channels, num_classes=2, fusion='concat'):
        """
        Args:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            num_classes: 分类类别数
            fusion: 嵌入融合方式 ('concat', 'add', 'attention', 'gate')
        """
        super().__init__()
        
        # Stage 1: 初始节点嵌入
        self.encoder1 = SGCNEncoder(in_channels, hidden_channels)

        # Stage 2: Gumbel-Sigmoid
        # 修复：edge_learner应该接收SGCNEncoder的实际输出维度hidden_channels
        self.edge_learner = learnable_edge_pruning(hidden_channels)
        
        # Stage 3: Refined嵌入
        # 修复：encoder2应该接收SGCNEncoder的实际输出维度hidden_channels
        self.encoder2 = SGCNEncoder(hidden_channels, hidden_channels)
        
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
        
        self.edge_threshold = 1e-3
        
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
        # Stage 1: 初始节点嵌入
        h1 = self.encoder1(x, edge_index, edge_weight)

        # Stage 2: Gumbel-Sigmoid
        edge_output = self.edge_learner(h1, edge_index, hard, return_reg=True)
        if isinstance(edge_output, tuple):
            edge_probs, l1_reg = edge_output
        else:
            edge_probs = edge_output
            l1_reg = None
        
        # 重加权边：原始符号 × 学习到的概率
        refined_weight = edge_weight * edge_probs
        
        # 过滤低权重边
        mask = refined_weight.abs() > self.edge_threshold
        refined_edge_index = edge_index[:, mask]
        refined_weight = refined_weight[mask]
        
        # Stage 3a: 再次GNN
        h2 = self.encoder2(h1, refined_edge_index, refined_weight)
        
        # Stage 3b: 融合h1和h2
        h_fused = self._fuse_embeddings(h1, h2)
        
        # Stage 3c: 边符号预测
        edge_logits = self.edge_predictor(h_fused, pred_edge_index)
        
        return (edge_logits, l1_reg) if l1_reg is not None else edge_logits
    
    def _fuse_embeddings(self, h1, h2):
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
            h1 = self.encoder1(x, edge_index, edge_weight)
            edge_probs = self.edge_learner(h1, edge_index, hard=False)
        return edge_probs
    
    def set_temp(self, temp):
        """设置Gumbel温度"""
        self.edge_learner.set_temp(temp)