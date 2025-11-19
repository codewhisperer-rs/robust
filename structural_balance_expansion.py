from collections import defaultdict
import torch
from torch import Tensor
from torch_sparse import SparseTensor  
from typing import Dict, Any


class StructuralBalanceExpander:
    """
    基于结构平衡理论扩张有向符号图。
    支持三种模式：
        - 'dense' : 传统矩阵乘法，适合 < 20k 节点
        - 'sparse': 稀疏矩阵乘法实现
    """

    def __init__(
        self,
        top_k_per_node: int | None = 20,
        score_threshold: float = 1.0,
        weighting: str = 'sign',       # 'sign' / 'score' / 'tanh'
        symmetrize: bool = False,
        allow_self_loops: bool = False,
        mode: str = 'sparse',           # 默认推荐 sparse！
    ) -> None:
        if weighting not in {'sign', 'score', 'tanh'}:
            raise ValueError(f"Unsupported weighting strategy: {weighting}")
        if mode not in {'dense', 'local', 'sparse'}:
            raise ValueError(f"Unsupported mode: {mode}")

        self.top_k_per_node = None if top_k_per_node is None or top_k_per_node <= 0 else int(top_k_per_node)
        self.score_threshold = float(score_threshold)
        self.weighting = weighting
        self.symmetrize = symmetrize
        self.allow_self_loops = allow_self_loops
        self.mode = mode

    @torch.no_grad()
    def __call__(self, edge_index: Tensor, edge_weight: Tensor, num_nodes: int):
        return self.expand(edge_index, edge_weight, num_nodes)

    @torch.no_grad()
    def expand(self, edge_index: Tensor, edge_weight: Tensor, num_nodes: int):
        if edge_index.numel() == 0:
            device = edge_index.device
            empty_idx = torch.empty((2, 0), dtype=torch.long, device=device)
            empty_w = torch.empty((0,), dtype=edge_weight.dtype, device=device)
            return edge_index, edge_weight, {
                'num_new_edges': 0,
                'new_edge_index': empty_idx,
                'new_edge_weight': empty_w,
            }

        if self.mode == 'dense':
            return self.dense_expand(edge_index, edge_weight, num_nodes)
        elif self.mode == 'local':
            return self.local_expand(edge_index, edge_weight, num_nodes)
        else:  # 'sparse'
            return self.sparse_expand(edge_index, edge_weight, num_nodes)

    def dense_expand(self, edge_index: Tensor, edge_weight: Tensor, num_nodes: int):
        device = edge_index.device
        dtype = torch.float32
        edge_weight = edge_weight.to(dtype=dtype)

        pos_mask = edge_weight > 0
        neg_mask = edge_weight < 0

        pos_adj = torch.zeros((num_nodes, num_nodes), dtype=dtype, device=device)
        neg_adj = torch.zeros((num_nodes, num_nodes), dtype=dtype, device=device)

        if pos_mask.any():
            src_pos = edge_index[0, pos_mask]
            dst_pos = edge_index[1, pos_mask]
            pos_adj[src_pos, dst_pos] = 1.0

        if neg_mask.any():
            src_neg = edge_index[0, neg_mask]
            dst_neg = edge_index[1, neg_mask]
            neg_adj[src_neg, dst_neg] = 1.0

        balanced_support = pos_adj @ pos_adj + neg_adj @ neg_adj
        unbalanced_support = pos_adj @ neg_adj + neg_adj @ pos_adj
        score = balanced_support - unbalanced_support

        if not self.allow_self_loops:
            score.fill_diagonal_(0.0)

        existing_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
        existing_mask[edge_index[0], edge_index[1]] = True
        score = score.masked_fill(existing_mask, 0.0)

        score_abs = score.abs()
        candidate_mask = score_abs >= self.score_threshold if self.score_threshold > 0 else score_abs > 0

        if self.top_k_per_node is not None:
            k = min(self.top_k_per_node, num_nodes)
            topk_vals, topk_idx = torch.topk(score_abs, k=k, dim=1)
            selector = torch.zeros_like(candidate_mask)
            positive_topk = topk_vals > 0
            if positive_topk.any():
                selector.scatter_(1, topk_idx, positive_topk)
            candidate_mask &= selector

        candidate_mask &= ~existing_mask
        if not candidate_mask.any():
            empty_idx = torch.empty((2, 0), dtype=torch.long, device=device)
            empty_w = torch.empty((0,), dtype=dtype, device=device)
            return edge_index, edge_weight, {'num_new_edges': 0, 'new_edge_index': empty_idx, 'new_edge_weight': empty_w}

        rows, cols = candidate_mask.nonzero(as_tuple=True)
        candidate_scores = score[rows, cols]
        weights = self.score_to_weight(candidate_scores)

        if self.symmetrize:
            rows = torch.cat([rows, cols], dim=0)
            cols = torch.cat([cols, rows[:candidate_scores.size(0)]], dim=0)
            weights = torch.cat([weights, weights[:candidate_scores.size(0)]], dim=0)

        if not self.allow_self_loops:
            non_diag = rows != cols
            rows, cols, weights = rows[non_diag], cols[non_diag], weights[non_diag]

        if rows.numel() == 0:
            empty_idx = torch.empty((2, 0), dtype=torch.long, device=device)
            empty_w = torch.empty((0,), dtype=dtype, device=device)
            return edge_index, edge_weight, {'num_new_edges': 0, 'new_edge_index': empty_idx, 'new_edge_weight': empty_w}

        new_edge_index = torch.stack([rows, cols], dim=0)
        new_edge_weight = weights

        # 去重
        if new_edge_index.size(1) > 1:
            linear_idx = new_edge_index[0] * num_nodes + new_edge_index[1]
            linear_idx, perm = torch.sort(linear_idx)
            new_edge_index = new_edge_index[:, perm]
            new_edge_weight = new_edge_weight[perm]
            dedup_mask = torch.ones_like(linear_idx, dtype=torch.bool)
            dedup_mask[1:] = linear_idx[1:] != linear_idx[:-1]
            new_edge_index = new_edge_index[:, dedup_mask]
            new_edge_weight = new_edge_weight[dedup_mask]

        expanded_edge_index = torch.cat([edge_index, new_edge_index], dim=1)
        expanded_edge_weight = torch.cat([edge_weight, new_edge_weight], dim=0)

        return expanded_edge_index, expanded_edge_weight, {
            'num_new_edges': new_edge_index.size(1),
            'new_edge_index': new_edge_index,
            'new_edge_weight': new_edge_weight,
        }

    @torch.no_grad()
    def sparse_expand(self, edge_index: Tensor, edge_weight: Tensor, num_nodes: int):
        device = edge_index.device

        pos_mask = edge_weight > 0
        neg_mask = ~pos_mask

        pos_edge = edge_index[:, pos_mask]
        neg_edge = edge_index[:, neg_mask]

        # 极端情况：只有正边或只有负边
        if pos_edge.size(1) == 0 or neg_edge.size(1) == 0:
            empty_idx = torch.empty((2, 0), dtype=torch.long, device=device)
            empty_w = torch.empty((0,), dtype=edge_weight.dtype, device=device)
            return edge_index, edge_weight, {
                'num_new_edges': 0,
                'new_edge_index': empty_idx,
                'new_edge_weight': empty_w,
            }

        pos_adj = SparseTensor.from_edge_index(pos_edge, sparse_sizes=(num_nodes, num_nodes)).to(device)
        neg_adj = SparseTensor.from_edge_index(neg_edge, sparse_sizes=(num_nodes, num_nodes)).to(device)

        # 四种二跳路径计数
        pp = pos_adj @ pos_adj   # ++ → +
        nn = neg_adj @ neg_adj   # -- → +
        pn = pos_adj @ neg_adj   # +- → -
        np = neg_adj @ pos_adj   # -+ → -

        score_map = defaultdict(float)

        # 平衡三角形贡献 +1
        r, c, _ = pp.coo()
        for ri, ci in zip(r.tolist(), c.tolist()):
            score_map[(ri, ci)] += 1.0
        r, c, _ = nn.coo()
        for ri, ci in zip(r.tolist(), c.tolist()):
            score_map[(ri, ci)] += 1.0

        # 非平衡三角形贡献 -1
        r, c, _ = pn.coo()
        for ri, ci in zip(r.tolist(), c.tolist()):
            score_map[(ri, ci)] -= 1.0
        r, c, _ = np.coo()
        for ri, ci in zip(r.tolist(), c.tolist()):
            score_map[(ri, ci)] -= 1.0

        if not score_map:
            empty_idx = torch.empty((2, 0), dtype=torch.long, device=device)
            empty_w = torch.empty((0,), dtype=edge_weight.dtype, device=device)
            return edge_index, edge_weight, {
                'num_new_edges': 0,
                'new_edge_index': empty_idx,
                'new_edge_weight': empty_w,
            }

        rows, cols, scores = [], [], []
        for (ri, ci), sv in score_map.items():
            if sv == 0:
                continue
            rows.append(ri)
            cols.append(ci)
            scores.append(sv)

        if not rows:
            empty_idx = torch.empty((2, 0), dtype=torch.long, device=device)
            empty_w = torch.empty((0,), dtype=edge_weight.dtype, device=device)
            return edge_index, edge_weight, {
                'num_new_edges': 0,
                'new_edge_index': empty_idx,
                'new_edge_weight': empty_w,
            }

        row = torch.tensor(rows, dtype=torch.long, device=device)
        col = torch.tensor(cols, dtype=torch.long, device=device)
        score = torch.tensor(scores, dtype=torch.float32, device=device)

        # 过滤自环
        if not self.allow_self_loops:
            mask = row != col
            row, col, score = row[mask], col[mask], score[mask]

        # 过滤已有边
        linear_existing = edge_index[0] * num_nodes + edge_index[1]
        linear_cand = row * num_nodes + col
        is_existing = torch.isin(linear_cand, linear_existing)
        mask = ~is_existing
        if self.score_threshold > 0:
            mask &= score.abs() >= self.score_threshold
        else:
            mask &= score != 0

        row, col, score = row[mask], col[mask], score[mask]

        if row.numel() == 0:
            empty_idx = torch.empty((2, 0), dtype=torch.long, device=device)
            empty_w = torch.empty((0,), dtype=edge_weight.dtype, device=device)
            return edge_index, edge_weight, {
                'num_new_edges': 0,
                'new_edge_index': empty_idx,
                'new_edge_weight': empty_w,
            }

        # 每节点 top-k
        if self.top_k_per_node is not None and self.top_k_per_node > 0:
            k = self.top_k_per_node
            src_groups = defaultdict(list)  # src -> [(abs_score, score, dst)]
            for r, c, s in zip(row.tolist(), col.tolist(), score.tolist()):
                src_groups[r].append((abs(s), s, c))

            new_rows, new_cols, new_scores = [], [], []
            for r, cand in src_groups.items():
                cand.sort(reverse=True)
                for _, s, c in cand[:k]:
                    new_rows.append(r)
                    new_cols.append(c)
                    new_scores.append(s)

            row = torch.tensor(new_rows, dtype=torch.long, device=device)
            col = torch.tensor(new_cols, dtype=torch.long, device=device)
            score = torch.tensor(new_scores, dtype=torch.float32, device=device)
        else:
            row = row.long()
            col = col.long()

        # 对称化
        if self.symmetrize:
            orig_size = row.size(0)
            row = torch.cat([row, col])
            col = torch.cat([col, row[:orig_size]])
            score = torch.cat([score, score[:orig_size]])

        # 最终去重（防止 symmetrize 后重复）
        linear = row * num_nodes + col
        linear, perm = torch.sort(linear)
        row, col, score = row[perm], col[perm], score[perm]
        keep = torch.cat([torch.tensor([True], device=device), linear[1:] != linear[:-1]])
        row, col, score = row[keep], col[keep], score[keep]

        new_edge_weight = self.score_to_weight(score)
        new_edge_index = torch.stack([row, col])

        expanded_edge_index = torch.cat([edge_index, new_edge_index], dim=1)
        expanded_edge_weight = torch.cat([edge_weight, new_edge_weight])

        return expanded_edge_index, expanded_edge_weight, {
            'num_new_edges': new_edge_index.size(1),
            'new_edge_index': new_edge_index,
            'new_edge_weight': new_edge_weight,
        }

    def score_to_weight(self, scores: Tensor) -> Tensor:
        if self.weighting == 'sign':
            return torch.sign(scores).to(scores.dtype)
        elif self.weighting == 'score':
            return scores
        elif self.weighting == 'tanh':
            return torch.tanh(scores)
        else:
            raise ValueError(f"Unknown weighting: {self.weighting}")
