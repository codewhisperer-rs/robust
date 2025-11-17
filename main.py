import argparse
import os
import random
from decimal import Decimal
from pathlib import Path

import numpy as np
import torch

from train import train_all_splits


def format_ratio(value: float) -> str:
    out = format(Decimal(str(value)).normalize(), 'f')
    if '.' in out:
        out = out.rstrip('0').rstrip('.')
    return out or '0'


def build_data_path(data_root: str, dataset: str, noise_ratio: float, unlabeled_ratio: float) -> str:
    base = Path(data_root) / dataset / 'random_masking'
    if noise_ratio > 0:
        base /= f'noise_{format_ratio(noise_ratio)}'
    base /= 'seed_0_num_splits_20'
    base /= 'test_0.05_val_0.05'
    base /= 'mask_0.75'
    if unlabeled_ratio > 0:
        base /= f'unlabeled_{format_ratio(unlabeled_ratio)}'
    return str(base / 'signed_datasets.pkl')


def set_random_seed(seed: int) -> None:
    """Make training repeatable by seeding all relevant RNGs."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    # 数据集参数
    parser.add_argument('--data_root', type=str, default='/home/houyikang/data/processed')
    parser.add_argument('--dataset_name', type=str, default='wiki')
    parser.add_argument('--noise_ratio', type=float, default=0)
    parser.add_argument('--unlabeled_ratio', type=float, default=0)
    parser.add_argument('--data_path', type=str, default=None,help='提供该路径，将覆盖自动构建的路径')

    # 模型参数
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--temp_start', type=float, default=3.0)
    parser.add_argument('--temp_end', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--fusion', type=str, default='concat', choices=['concat', 'add', 'attention', 'gate'])
    parser.add_argument('--l1_reg_weight', type=float, default=0.005)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--grad_clip_value', type=float, default=1.0)
    parser.add_argument('--model', type=str, default='sdgnn', choices=['sdgnn', 'sgcn'])
    parser.add_argument('--sign_product_weight', type=float, default=0.1,help='Sign Product Entropy 辅助损失权重')
    parser.add_argument('--sign_direction_weight', type=float, default=0.1, help='Sign Direction 辅助损失权重')
    parser.add_argument('--sdgnn_layers', type=int, default=2)
    parser.add_argument('--sdgnn_heads', type=int, default=1)
    parser.add_argument('--sdgnn_dropout', type=float, default=0.2)
    parser.add_argument('--sdgnn_no_residual', action='store_true',help='禁用 SDGNN 残差连接')

    # 结构平衡扩边参数
    parser.add_argument('--balance_topk', type=int, default=4,help='结构平衡扩边时每个节点最多保留的新增边数')
    parser.add_argument('--balance_threshold', type=float, default=0.8,help='扩边得分阈值，低于该阈值的候选边会被忽略')
    parser.add_argument('--balance_weighting', type=str, default='sign',choices=['sign', 'score', 'tanh'],help='扩边后新边的权重计算方式')
    parser.add_argument('--balance_symmetrize', action='store_true',help='启用无向图扩边')
    parser.add_argument('--balance_mode', type=str, default='dense', choices=['dense', 'local'],help='扩边实现：dense 使用矩阵乘法；local 采用局部三角统计（适合大图）')
    parser.add_argument('--disable_balance_expansion', action='store_true',help='禁用结构平衡扩边')

    parser.add_argument('--prune_threshold', type=float, default=0.6,help='Gumbel 筛边阶段的权重阈值')
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()

    set_random_seed(args.seed)

    data_path = args.data_path or build_data_path(
        data_root=args.data_root,
        dataset=args.dataset_name,
        noise_ratio=args.noise_ratio,
        unlabeled_ratio=args.unlabeled_ratio,
    )
    
    if args.model == 'sdgnn':
        encoder_kwargs = {
            'num_layers': args.sdgnn_layers,
            'heads': args.sdgnn_heads,
            'dropout': args.sdgnn_dropout,
            'use_residual': not args.sdgnn_no_residual,
        }
    else:
        encoder_kwargs = {}
    
    config = {
        'hidden_channels': args.hidden,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': 1e-3,
        'patience': args.patience,
        'print_every': 1,
        'use_temp_anneal': True,
        'temp_start': args.temp_start,
        'temp_end': args.temp_end,
        'fusion': args.fusion,
        'l1_reg_weight': args.l1_reg_weight,
        'label_smoothing': args.label_smoothing,
        'grad_clip_value': args.grad_clip_value,
        'encoder_type': args.model,
        'sign_product_weight': args.sign_product_weight,
        'sign_direction_weight': args.sign_direction_weight,
        'encoder_kwargs': encoder_kwargs,
        'balance_expander': {
            'enabled': not args.disable_balance_expansion,
            'top_k_per_node': args.balance_topk,
            'score_threshold': args.balance_threshold,
            'weighting': args.balance_weighting,
            'symmetrize': args.balance_symmetrize,
            'mode': args.balance_mode,
        },
        'prune_threshold': args.prune_threshold,
        'seed': args.seed,
    }
    
    train_all_splits(
        data_path,
        config,
        dataset_name=args.dataset_name,
        noise_ratio=args.noise_ratio
    )


if __name__ == '__main__':
    main()
