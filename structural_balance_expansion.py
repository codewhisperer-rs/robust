from collections import defaultdict
import torch
from torch import Tensor


class StructuralBalanceExpander:
    """
    基于结构平衡理论扩张有向符号图。提供两种实现：
    - dense：使用邻接矩阵乘法统计三角关系，精度高但显存消耗大；
    - local：仅遍历二跳邻居的三角关系，适合 Slashdot 等大规模图。
    """

    def __init__(
        self,
        top_k_per_node: int | None = 20,
        score_threshold: float = 1.0,
        weighting: str = 'sign',
        symmetrize: bool = False,
        allow_self_loops: bool = False,
        mode: str = 'dense',
    ) -> None:
        """
        Args:
            top_k_per_node: 每个源节点保留的扩张边上限，<=0 表示不过滤
            score_threshold: 候选边得分的绝对值下限
            weighting: 新边权重的计算方式：sign / score / tanh
            symmetrize: 是否对扩张边做对称化（无向化）
            allow_self_loops: 是否保留自环
            mode: 'dense' | 'local'
        """
        if weighting not in {'sign', 'score', 'tanh'}:
            raise ValueError(f"Unsupported weighting strategy: {weighting}")
        if mode not in {'dense', 'local'}:
            raise ValueError(f"Unsupported expansion mode: {mode}")

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
            return edge_index, edge_weight, {
                'num_new_edges': 0,
                'new_edge_index': torch.empty((2, 0), dtype=torch.long, device=device),
                'new_edge_weight': torch.empty((0,), dtype=edge_weight.dtype, device=device),
            }

        if self.mode == 'dense':
            return self.dense_expand(edge_index, edge_weight, num_nodes)
        return self.local_expand(edge_index, edge_weight, num_nodes)

   
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
            return edge_index, edge_weight, {
                'num_new_edges': 0,
                'new_edge_index': torch.empty((2, 0), dtype=torch.long, device=device),
                'new_edge_weight': torch.empty((0,), dtype=dtype, device=device),
            }

        rows, cols = candidate_mask.nonzero(as_tuple=True)
        candidate_scores = score[rows, cols]
        weights = self.score_to_weight(candidate_scores)

        if self.symmetrize:
            rows = torch.cat([rows, cols], dim=0)
            cols = torch.cat([cols, rows[:candidate_scores.size(0)]], dim=0)
            weights = torch.cat([weights, weights[:candidate_scores.size(0)]], dim=0)

        if not self.allow_self_loops:
            non_diag = rows != cols
            rows = rows[non_diag]
            cols = cols[non_diag]
            weights = weights[non_diag]

        if rows.numel() == 0:
            return edge_index, edge_weight, {
                'num_new_edges': 0,
                'new_edge_index': torch.empty((2, 0), dtype=torch.long, device=device),
                'new_edge_weight': torch.empty((0,), dtype=dtype, device=device),
            }

        new_edge_index = torch.stack([rows, cols], dim=0)
        new_edge_weight = weights

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

    def local_expand(self, edge_index: Tensor, edge_weight: Tensor, num_nodes: int):
        device = edge_index.device
        edge_index_cpu = edge_index.detach().cpu()
        edge_weight_cpu = edge_weight.detach().cpu()

        pos_out = [set() for _ in range(num_nodes)]
        neg_out = [set() for _ in range(num_nodes)]
        existing_edges = set()

        for idx in range(edge_index_cpu.size(1)):
            src = int(edge_index_cpu[0, idx])
            dst = int(edge_index_cpu[1, idx])
            existing_edges.add((src, dst))
            if edge_weight_cpu[idx] > 0:
                pos_out[src].add(dst)
            elif edge_weight_cpu[idx] < 0:
                neg_out[src].add(dst)

        score_map = defaultdict(int)

        for src in range(num_nodes):
            pos_neighbors = pos_out[src]
            neg_neighbors = neg_out[src]

            for mid in pos_neighbors:
                pos_mid = pos_out[mid]
                neg_mid = neg_out[mid]

                for dst in pos_mid:
                    if not self.allow_self_loops and dst == src:
                        continue
                    if (src, dst) in existing_edges:
                        continue
                    score_map[(src, dst)] += 1

                for dst in neg_mid:
                    if not self.allow_self_loops and dst == src:
                        continue
                    if (src, dst) in existing_edges:
                        continue
                    score_map[(src, dst)] -= 1

            for mid in neg_neighbors:
                pos_mid = pos_out[mid]
                neg_mid = neg_out[mid]

                for dst in neg_mid:
                    if not self.allow_self_loops and dst == src:
                        continue
                    if (src, dst) in existing_edges:
                        continue
                    score_map[(src, dst)] += 1

                for dst in pos_mid:
                    if not self.allow_self_loops and dst == src:
                        continue
                    if (src, dst) in existing_edges:
                        continue
                    score_map[(src, dst)] -= 1

        per_src = defaultdict(list)
        threshold = self.score_threshold
        for (src, dst), score in score_map.items():
            if score == 0:
                continue
            if threshold > 0 and abs(score) < threshold:
                continue
            per_src[src].append((score, dst))

        rows = []
        cols = []
        scores = []

        for src, pairs in per_src.items():
            if not pairs:
                continue
            if self.top_k_per_node is not None and len(pairs) > self.top_k_per_node:
                pairs.sort(key=lambda x: abs(x[0]), reverse=True)
                pairs = pairs[:self.top_k_per_node]
            for score, dst in pairs:
                if not self.allow_self_loops and src == dst:
                    continue
                if (src, dst) in existing_edges:
                    continue
                rows.append(src)
                cols.append(dst)
                scores.append(score)

        if not rows:
            return edge_index, edge_weight, {
                'num_new_edges': 0,
                'new_edge_index': torch.empty((2, 0), dtype=torch.long, device=device),
                'new_edge_weight': torch.empty((0,), dtype=edge_weight.dtype, device=device),
            }

        if self.symmetrize:
            extra_rows = cols.copy()
            extra_cols = rows.copy()
            extra_scores = scores.copy()
            rows.extend(extra_rows)
            cols.extend(extra_cols)
            scores.extend(extra_scores)

        candidate_pairs = set()
        final_rows = []
        final_cols = []
        final_scores = []

        for r, c, s in zip(rows, cols, scores):
            key = (r, c)
            if key in existing_edges or key in candidate_pairs:
                continue
            if not self.allow_self_loops and r == c:
                continue
            candidate_pairs.add(key)
            final_rows.append(r)
            final_cols.append(c)
            final_scores.append(s)

        if not final_rows:
            return edge_index, edge_weight, {
                'num_new_edges': 0,
                'new_edge_index': torch.empty((2, 0), dtype=torch.long, device=device),
                'new_edge_weight': torch.empty((0,), dtype=edge_weight.dtype, device=device),
            }

        new_edge_index = torch.tensor([final_rows, final_cols], dtype=torch.long, device=device)
        score_tensor = torch.tensor(final_scores, dtype=torch.float32, device=device)
        new_edge_weight = self.score_to_weight(score_tensor)

        expanded_edge_index = torch.cat([edge_index, new_edge_index], dim=1)
        expanded_edge_weight = torch.cat([edge_weight, new_edge_weight], dim=0)

        return expanded_edge_index, expanded_edge_weight, {
            'num_new_edges': new_edge_index.size(1),
            'new_edge_index': new_edge_index,
            'new_edge_weight': new_edge_weight,
        }

    def score_to_weight(self, scores: Tensor) -> Tensor:
        if self.weighting == 'sign':
            return torch.where(scores >= 0, torch.ones_like(scores), -torch.ones_like(scores))
        if self.weighting == 'score':
            return scores
        if self.weighting == 'tanh':
            return torch.tanh(scores)