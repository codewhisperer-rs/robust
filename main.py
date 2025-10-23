import argparse
from train import train_all_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--temp_start', type=float, default=3.0)
    parser.add_argument('--temp_end', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--fusion', type=str, default='concat', 
                       choices=['concat', 'add', 'attention', 'gate'])
    
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
    }
    
    train_all_splits(args.data_path, config)


if __name__ == '__main__':
    main()