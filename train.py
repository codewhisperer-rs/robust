import torch
import datetime
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from three_stage_sgnn import ThreeStageSGNN


def compute_metrics(logits, labels):
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    
    # 计算预测概率
    probs = F.softmax(logits, dim=1).cpu().numpy()
    
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs[:, 1])
    return f1, acc, auc


def train_single_split(train_data, val_data, test_data, config, device='cpu'):

    model = ThreeStageSGNN(
        in_channels=train_data.x.size(1),
        hidden_channels=config['hidden_channels'],
        num_classes=2,
        fusion=config.get('fusion', 'concat') 
    ).to(device)
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    
    class_counts = torch.bincount(train_data.y)
    class_weights = torch.sqrt(len(train_data.y) / class_counts.float())
    class_weights = class_weights.to(device)
    
    label_smoothing = config.get('label_smoothing', 0.0)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    
    print(f"    类别: 0={class_counts[0]}, 1={class_counts[1]}")
    print(f"    权重: 0={class_weights[0]:.3f}, 1={class_weights[1]:.3f}")
    
    best_val_f1 = 0
    best_results = {}
    patience_counter = 0
    
    # 获取L1正则化权重参数
    l1_reg_weight = config.get('l1_reg_weight', 0.0)
    
    for epoch in range(config['epochs']):
        if config['use_temp_anneal']:
            progress = epoch / config['epochs']
            temp = config['temp_end'] + (config['temp_start'] - config['temp_end']) * \
                   0.5 * (1 + np.cos(np.pi * progress))
            model.set_temp(max(temp, config['temp_end']))

        model.train()
        optimizer.zero_grad()
        
        output = model(
            train_data.x,
            train_data.edge_index,
            train_data.edge_weight,
            train_data.pred_edge_index,
            hard=False
        )
        
        # 处理模型返回值
        logits = output[0] if isinstance(output, tuple) else output
        l1_reg = output[1] if isinstance(output, tuple) else None
            
        loss = loss_fn(logits, train_data.y)
        
        # L1正则化项
        if l1_reg_weight > 0 and l1_reg is not None:
            loss = loss + l1_reg_weight * l1_reg
        
        loss.backward()
        
        # 梯度裁剪
        grad_clip_value = config.get('grad_clip_value', 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(
                val_data.x, 
                val_data.edge_index, 
                val_data.edge_weight,
                val_data.pred_edge_index, 
                hard=True
            )
            val_logits = val_output[0]
            val_f1, val_acc, val_auc = compute_metrics(val_logits, val_data.y)
            
            test_output = model(
                test_data.x, 
                test_data.edge_index, 
                test_data.edge_weight,
                test_data.pred_edge_index, 
                hard=True
            )
            test_logits = test_output[0]
            test_f1, test_acc, test_auc = compute_metrics(test_logits, test_data.y)
        
        if (epoch + 1) % config['print_every'] == 0:
            print(f"    Epoch {epoch+1:4d} - Loss: {loss.item():.4f} | "
                  f"Val: F1={val_f1:.4f} ACC={val_acc:.4f} AUC={val_auc:.4f} | "
                  f"Test: F1={test_f1:.4f} ACC={test_acc:.4f} AUC={test_auc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_results = {
                'val_f1': val_f1, 'val_acc': val_acc, 'val_auc': val_auc,
                'test_f1': test_f1, 'test_acc': test_acc, 'test_auc': test_auc,
                'epoch': epoch + 1
            }
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= config['patience']:
            print(f"    早停于epoch {epoch+1}")
            break
    
    print(f"    最佳: Epoch {best_results['epoch']} - "
          f"Val F1={best_results['val_f1']:.4f} ACC={best_results['val_acc']:.4f} AUC={best_results['val_auc']:.4f} | "
          f"Test F1={best_results['test_f1']:.4f} ACC={best_results['test_acc']:.4f} AUC={best_results['test_auc']:.4f}")
    
    return best_results


def train_all_splits(data_path, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    from data_loader import load_data
    all_splits, num_splits = load_data(data_path, split_id=None, device=device)
    
    print(f"配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    results = []
    for split_id, (train_data, val_data, test_data) in enumerate(all_splits):
        print(f"\n{'='*60}")
        print(f"划分 {split_id+1}/{num_splits}")
        print(f"{'='*60}")
        
        try:
            res = train_single_split(train_data, val_data, test_data, config, device)
            results.append(res)
        except Exception as e:
            print(f"划分{split_id+1}失败: {e}")
            continue
    
    if results:
        avg_val_f1 = np.mean([r['val_f1'] for r in results])
        std_val_f1 = np.std([r['val_f1'] for r in results])
        avg_val_acc = np.mean([r['val_acc'] for r in results])
        std_val_acc = np.std([r['val_acc'] for r in results])
        avg_test_f1 = np.mean([r['test_f1'] for r in results])
        std_test_f1 = np.std([r['test_f1'] for r in results])
        avg_test_acc = np.mean([r['test_acc'] for r in results])
        std_test_acc = np.std([r['test_acc'] for r in results])
        avg_val_auc = np.mean([r['val_auc'] for r in results])
        std_val_auc = np.std([r['val_auc'] for r in results])
        avg_test_auc = np.mean([r['test_auc'] for r in results])
        std_test_auc = np.std([r['test_auc'] for r in results])
        
        print(f"\n{'='*60}")
        print(f"平均结果 (成功: {len(results)}/{num_splits}):")
        print(f"{'='*60}")
        print(f"验证 F1:  {avg_val_f1:.4f} ± {std_val_f1:.4f}")
        print(f"验证 ACC: {avg_val_acc:.4f} ± {std_val_acc:.4f}")
        print(f"验证 AUC: {avg_val_auc:.4f} ± {std_val_auc:.4f}")
        print(f"测试 F1:  {avg_test_f1:.4f} ± {std_test_f1:.4f}")
        print(f"测试 ACC: {avg_test_acc:.4f} ± {std_test_acc:.4f}")
        print(f"测试 AUC: {avg_test_auc:.4f} ± {std_test_auc:.4f}")
        
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f'results_{timestamp}.txt'
        
        with open(log_filename, 'w') as f:
            f.write(f"训练时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"配置: {config}\n\n")
            f.write("="*60 + "\n")
            f.write("各划分详细结果:\n")
            f.write("="*60 + "\n")
            for i, r in enumerate(results):
                f.write(f"划分{i+1}:\n")
                f.write(f"  验证集 - F1: {r['val_f1']:.4f}, ACC: {r['val_acc']:.4f}\n")
                f.write(f"  测试集 - F1: {r['test_f1']:.4f}, ACC: {r['test_acc']:.4f}\n")
                f.write(f"  最佳Epoch: {r['epoch']}\n\n")
            
            f.write("="*60 + "\n")
            f.write("平均结果:\n")
            f.write("="*60 + "\n")
            f.write(f"验证集:\n")
            f.write(f"  F1:  {avg_val_f1:.4f} ± {std_val_f1:.4f}\n")
            f.write(f"  ACC: {avg_val_acc:.4f} ± {std_val_acc:.4f}\n")
            f.write(f"  AUC: {avg_val_auc:.4f} ± {std_val_auc:.4f}\n")
            f.write(f"测试集:\n")
            f.write(f"  F1:  {avg_test_f1:.4f} ± {std_test_f1:.4f}\n")
            f.write(f"  ACC: {avg_test_acc:.4f} ± {std_test_acc:.4f}\n")
            f.write(f"  AUC: {avg_test_auc:.4f} ± {std_test_auc:.4f}\n")
        
        print(f"\n结果已保存至: {log_filename}")


if __name__ == '__main__':
    config = {
        'hidden_channels': 64,
        'epochs': 1000,
        'lr': 0.001,
        'weight_decay': 1e-3,
        'patience': 100,
        'print_every': 25,
        'use_temp_anneal': True,
        'temp_start': 3.0,
        'temp_end': 0.5,
        'fusion': 'concat',  # 'concat', 'add', 'attention', 'gate'
    }

    data_path = '/home/houyikang/data/processed/slashdot/random_masking/seed_0_num_splits_20/test_0.05_val_0.05/mask_0.75/unlabeled_0.5/signed_datasets.pkl'

    train_all_splits(data_path, config)