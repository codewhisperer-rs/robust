import argparse
from train import train_all_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/houyikang/data/processed/slashdot/random_masking/noise_0.2/seed_0_num_splits_20/test_0.05_val_0.05/mask_0.75/unlabeled_0.5/signed_datasets.pkl')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--temp_start', type=float, default=3.0)
    parser.add_argument('--temp_end', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--fusion', type=str, default='concat', choices=['concat', 'add', 'attention', 'gate'])
    parser.add_argument('--l1_reg_weight', type=float, default=0.0)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--grad_clip_value', type=float, default=1.0)
    parser.add_argument('--model', type=str, default='sdgnn', choices=['sdgnn', 'sgcn'],
                        help='选择编码器类型：sdgnn 或 sgcn')
    parser.add_argument('--sign_product_weight', type=float, default=0.0,
                        help='Sign Product Entropy 辅助损失权重')
    parser.add_argument('--sign_direction_weight', type=float, default=0.0,
                        help='Sign Direction 辅助损失权重')
    parser.add_argument('--sdgnn_layers', type=int, default=2,
                        help='SDGNN 编码器层数')
    parser.add_argument('--sdgnn_heads', type=int, default=1,
                        help='SDGNN GAT 注意力头数')
    parser.add_argument('--sdgnn_dropout', type=float, default=0.1,
                        help='SDGNN 层间 dropout')
    parser.add_argument('--sdgnn_no_residual', action='store_true',
                        help='禁用 SDGNN 残差连接')
    
    args = parser.parse_args()
    
    config = {
        'hidden_channels': args.hidden,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': 1e-3,
        'patience': args.patience,
        'print_every': 25,
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
        'encoder_kwargs': {
            'num_layers': args.sdgnn_layers,
            'heads': args.sdgnn_heads,
            'dropout': args.sdgnn_dropout,
            'use_residual': not args.sdgnn_no_residual,
        },
    }
    
    train_all_splits(args.data_path, config)


if __name__ == '__main__':
    main()
