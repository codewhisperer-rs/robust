import torch
import pickle
from torch_geometric.data import Data


def labels_to_edge_weight(labels: torch.Tensor) -> torch.Tensor:
    signed = labels.detach().clone().to(torch.float32)
    unique_vals = torch.unique(signed)
    if torch.all((unique_vals == 0) | (unique_vals == 1)):
        signed.mul_(2).sub_(1)
        return signed
    if torch.all((unique_vals == -1) | (unique_vals == 1)):
        return signed

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
    
    

    train_edges = split_data['train']['edges'].to(device).t().contiguous()
    train_labels = split_data['train']['label'].to(device)

    val_edges = split_data['val']['edges'].to(device).t().contiguous()
    val_labels = split_data['val']['label'].to(device)

    test_edges = split_data['test']['edges'].to(device).t().contiguous()
    test_labels = split_data['test']['label'].to(device)


    train_edge_weight = labels_to_edge_weight(train_labels)

    train_data = Data(
        x=node_features,
        edge_index=train_edges,  # 用于GNN消息传递
        edge_weight=train_edge_weight,  # 边的符号权重 {-1, +1}
        pred_edge_index=train_edges,  # 需要预测的边
        y=train_labels                 # 需要预测的边的标签
    )
    
    val_data = Data(
        x=node_features,
        edge_index=train_edges,  # 用于GNN消息传递
        edge_weight=train_edge_weight,
        pred_edge_index=val_edges,
        y=val_labels
    )
    
    test_data = Data(
        x=node_features,
        edge_index=train_edges,  # 用于GNN消息传递
        edge_weight=train_edge_weight,
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
    
        
        train_edges = split_data['train']['edges'].to(device).t().contiguous()
        train_labels = split_data['train']['label'].to(device)

        val_edges = split_data['val']['edges'].to(device).t().contiguous()
        val_labels = split_data['val']['label'].to(device)

        test_edges = split_data['test']['edges'].to(device).t().contiguous()
        test_labels = split_data['test']['label'].to(device)

        train_edge_weight = labels_to_edge_weight(train_labels)

        # 创建 PyG 的 Data 对象
        train_data = Data(
            x=node_features,
            edge_index=train_edges,  # 用于GNN消息传递
            edge_weight=train_edge_weight,      # 边的符号权重
            pred_edge_index=train_edges,  # 需要预测的边
            y=train_labels                 # 需要预测的边的标签
        )

        val_data = Data(
            x=node_features,
            edge_index=train_edges,  # 用于GNN消息传递
            edge_weight=train_edge_weight,
            pred_edge_index=val_edges,
            y=val_labels
        )
        
        test_data = Data(
            x=node_features,
            edge_index=train_edges,  # 用于GNN消息传递
            edge_weight=train_edge_weight,
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
