import pickle
import torch

# 读取数据
with open('/home/houyikang/data/processed/bitcoin_alpha/random_masking/noise_0.1/seed_0_num_splits_20/test_0.05_val_0.05/mask_0.75/signed_datasets.pkl', 'rb') as f:
    data = pickle.load(f)

# 获取第一个数据集
dataset = data.get(0)
print(dataset.keys())

if dataset:
    # 获取图的边
    graph_edge_index = dataset.get('graph', None)  # 这是整个图的边
    print("整个图的边索引形状：", graph_edge_index.shape if graph_edge_index is not None else "无数据")
    print("整个图的边数量（有向）：", graph_edge_index.size(1) if graph_edge_index is not None else "无数据")

    # 检查是否为无向图（通过检查边索引矩阵是否表示对称关系）
    if graph_edge_index is not None:
        # 创建邻接矩阵来检查对称性
        num_nodes = graph_edge_index.max().item() + 1
        adj_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
        
        for i in range(graph_edge_index.size(1)):
            src, dst = graph_edge_index[0, i], graph_edge_index[1, i]
            adj_matrix[src, dst] = True

        # 检查邻接矩阵是否对称
        is_symmetric = torch.allclose(adj_matrix.to(torch.int), adj_matrix.t().to(torch.int))  # 转换为int进行比较
        print("图是有向的还是无向的：", "无向图" if is_symmetric else "有向图")
        
        if is_symmetric:
            # 如果是无向图，实际唯一边数是总边数的一半（忽略自环）
            unique_edges_tensor = torch.unique(graph_edge_index, dim=1)
            non_self_loops = unique_edges_tensor[0] != unique_edges_tensor[1]
            unique_non_self_loop_edges = unique_edges_tensor[:, non_self_loops]
            unique_undirected_edges_count = unique_non_self_loop_edges.size(1) // 2 if unique_non_self_loop_edges.size(1) > 0 else 0
            self_loops_count = (~non_self_loops).sum().item()
            total_unique_undirected_edges = unique_undirected_edges_count + self_loops_count
            print("整个图的唯一边数量（无向，包含自环）：", total_unique_undirected_edges)

    # 获取训练集数据
    train_edges = dataset.get('train', {}).get('edges', None)
    train_labels = dataset.get('train', {}).get('label', None)

    print("训练集边：", train_edges[-5:-1] if train_edges is not None else "无数据")
    print("训练集标签：", train_labels[-5:-1] if train_labels is not None else "无数据")
    print("训练集边数量：", train_edges.size(0) if train_edges is not None else "无数据")

    # 获取验证集数据
    val_edges = dataset.get('val', {}).get('edges', None)
    val_labels = dataset.get('val', {}).get('label', None)

    print("验证集边：", val_edges[-5:-1] if val_edges is not None else "无数据")
    print("验证集标签：", val_labels[-5:-1] if val_labels is not None else "无数据")
    print("验证集边数量：", val_edges.size(0) if val_edges is not None else "无数据")

    # 获取测试集数据
    test_edges = dataset.get('test', {}).get('edges', None)
    test_labels = dataset.get('test', {}).get('label', None)

    print("测试集边：", test_edges[:5] if test_edges is not None else "无数据")
    print("测试集标签：", test_labels[:5] if test_labels is not None else "无数据")
    print("测试集边数量：", test_edges.size(0) if test_edges is not None else "无数据")

    # 计算总边数 (训练+验证+测试)
    total_split_edges = 0
    for split_name in ['train', 'val', 'test']:
        split_data = dataset.get(split_name, {})
        edges = split_data.get('edges', None)
        if edges is not None:
            count = edges.size(0)
            total_split_edges += count
            print(f"{split_name}集边数量: {count}")
    print(f"训练/验证/测试集总边数: {total_split_edges}")

    # 查看unknowns和knowns的数据
    unknowns = dataset.get('unknowns', None)
    knowns = dataset.get('knowns', None)

    print("未知节点：", unknowns[:5] if unknowns is not None else "无数据")
    print("已知节点：", knowns[:5] if knowns is not None else "无数据")

else:
    print("未找到键为 0 的数据划分")
