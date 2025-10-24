import torch
import pickle
from torch_geometric.data import Data


def remove_edges(edge_index, edge_weight, edges_to_drop):
    if edge_index.numel() == 0 or edges_to_drop.numel() == 0:
        return edge_index, edge_weight

    # 确保edges_to_drop是正确的形状 [2, num_edges_to_drop]
    if edges_to_drop.dim() != 2 or edges_to_drop.size(0) != 2:
        edges_to_drop = edges_to_drop.t().contiguous()
    
    # 同样确保edge_index是正确的形状 [2, num_edges]
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        edge_index = edge_index.t().contiguous()

    # 验证节点ID非负
    if (edge_index < 0).any() or (edges_to_drop < 0).any():
        raise ValueError("Node indices must be non-negative")

    num_nodes = max(edge_index.max().item(), edges_to_drop.max().item()) + 1
    
    # 将边转换为唯一的整数标识
    drop_edge_ids1 = edges_to_drop[0] * num_nodes + edges_to_drop[1]
    drop_edge_ids2 = edges_to_drop[1] * num_nodes + edges_to_drop[0]
    drop_edge_ids = torch.cat([drop_edge_ids1, drop_edge_ids2], dim=0)
    drop_set = torch.unique(drop_edge_ids)
    
    # 为现有边创建唯一标识
    edge_ids1 = edge_index[0] * num_nodes + edge_index[1]
    edge_ids2 = edge_index[1] * num_nodes + edge_index[0]
    
    # 检查每条边是否应该被保留
    keep_mask1 = ~torch.isin(edge_ids1, drop_set)
    keep_mask2 = ~torch.isin(edge_ids2, drop_set)
    keep_mask = keep_mask1 & keep_mask2
    
    return edge_index[:, keep_mask], edge_weight[keep_mask]


def load_pkl_data(file_path, split_id=0, device='cpu'):
    """
    加载 pickle 文件中的单个数据划分
    
    参数:
        file_path
        split_id: 数据划分的ID (默认为0，范围通常是0-19)
        device
    
    返回:
        train_data, val_data, test_data: PyG Data 对象
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # 数据格式: data[split_id] 包含 'graph', 'train', 'val', 'test', 'features', 'weights'
    # 选择指定的数据划分
    split_data = data[split_id]

    node_features = split_data['features'].to(device)
    
    graph_edge_index = split_data['graph'].to(device)
    edge_weights = split_data['weights'].to(device)
    

    train_edges = split_data['train']['edges'].to(device).t().contiguous()
    train_labels = split_data['train']['label'].to(device)

    val_edges = split_data['val']['edges'].to(device).t().contiguous()
    val_labels = split_data['val']['label'].to(device)

    test_edges = split_data['test']['edges'].to(device).t().contiguous()
    test_labels = split_data['test']['label'].to(device)

    # 去掉验证和测试边，避免训练阶段泄漏
    graph_edge_index, edge_weights = remove_edges(graph_edge_index, edge_weights, val_edges)
    graph_edge_index, edge_weights = remove_edges(graph_edge_index, edge_weights, test_edges)

    train_data = Data(
        x=node_features, 
        edge_index=graph_edge_index,  # 用于GNN消息传递
        edge_weight=edge_weights,      # 边的符号权重
        pred_edge_index=train_edges,  # 需要预测的边
        y=train_labels                 # 需要预测的边的标签
    )
    
    val_data = Data(
        x=node_features,
        edge_index=graph_edge_index,
        edge_weight=edge_weights,
        pred_edge_index=val_edges,
        y=val_labels
    )
    
    test_data = Data(
        x=node_features,
        edge_index=graph_edge_index,
        edge_weight=edge_weights,
        pred_edge_index=test_edges,
        y=test_labels
    )

    return train_data, val_data, test_data

def load_all_splits(file_path, device='cpu'):
    """
    加载所有数据划分
    
    参数:
        file_path
        device
    
    返回:
        all_splits: 包含所有划分的 (train_data, val_data, test_data) 元组
        num_splits: 划分数量
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    num_splits = len(data)
    all_splits = []
    
    for split_id in range(num_splits):
        split_data = data[split_id]

        node_features = split_data['features'].to(device)
        
        graph_edge_index = split_data['graph'].to(device)
        edge_weights = split_data['weights'].to(device)
        
        train_edges = split_data['train']['edges'].to(device).t().contiguous()
        train_labels = split_data['train']['label'].to(device)

        val_edges = split_data['val']['edges'].to(device).t().contiguous()
        val_labels = split_data['val']['label'].to(device)

        test_edges = split_data['test']['edges'].to(device).t().contiguous()
        test_labels = split_data['test']['label'].to(device)

        graph_edge_index, edge_weights = remove_edges(graph_edge_index, edge_weights, val_edges)
        graph_edge_index, edge_weights = remove_edges(graph_edge_index, edge_weights, test_edges)

        # 创建 PyG 的 Data 对象
        train_data = Data(
            x=node_features,
            edge_index=graph_edge_index,
            edge_weight=edge_weights,
            pred_edge_index=train_edges,
            y=train_labels
        )
        
        val_data = Data(
            x=node_features,
            edge_index=graph_edge_index,
            edge_weight=edge_weights,
            pred_edge_index=val_edges,
            y=val_labels
        )
        
        test_data = Data(
            x=node_features,
            edge_index=graph_edge_index,
            edge_weight=edge_weights,
            pred_edge_index=test_edges,
            y=test_labels
        )
        
        all_splits.append((train_data, val_data, test_data))
    
    return all_splits, num_splits

def load_data(file_path, split_id=None, device='cpu'):
    """
    加载数据
    
    参数:
        file_path
        split_id: 数据划分的ID
        device
    """
    if split_id is None:
        return load_all_splits(file_path, device)
    else:
        return load_pkl_data(file_path, split_id, device)
