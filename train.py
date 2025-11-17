import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from decimal import Decimal
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from three_stage_sgnn import ThreeStageSGNN
from losses import Sign_Product_Entropy_Loss, Sign_Direction_Loss


def format_ratio(value: float) -> str:
    out = format(Decimal(str(value)).normalize(), 'f')
    if '.' in out:
        out = out.rstrip('0').rstrip('.')
    return out or '0'


def infer_dataset_and_noise(data_path: str):
    path = Path(data_path)
    parts = path.parts
    dataset = None
    noise = '0'

    if 'processed' in parts:
        idx = parts.index('processed')
        if idx + 1 < len(parts):
            dataset = parts[idx + 1]
    if dataset is None and len(parts) >= 2:
        dataset = parts[-2]
    if dataset is None:
        dataset = path.stem

    for part in parts:
        if part.startswith('noise_'):
            noise = part.split('noise_', 1)[1] or '0'
            break

    return dataset, noise


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
        fusion=config.get('fusion', 'concat'),
        encoder_type=config.get('encoder_type', 'sdgnn'),
        encoder_kwargs=config.get('encoder_kwargs', None),
        balance_config=config.get('balance_expander', None),
        prune_threshold=config.get('prune_threshold', 1e-3),
    ).to(device)
    
    # class_counts = torch.bincount(train_data.y)
    # class_weights = len(train_data.y) / (2 * class_counts.float())
    class_counts = torch.bincount(train_data.y)
    class_weights = torch.sqrt(len(train_data.y) / class_counts.float())
    class_weights = class_weights.to(device)
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Multi-task signed loss modules
    sign_product_weight = config.get('sign_product_weight', 0.0)
    sign_direction_weight = config.get('sign_direction_weight', 0.0)
    sign_product_loss = Sign_Product_Entropy_Loss().to(device) if sign_product_weight > 0 else None
    sign_direction_loss = Sign_Direction_Loss(emb_dim=config['hidden_channels']).to(device) \
        if sign_direction_weight > 0 else None

    optim_params = list(model.parameters())
    if sign_product_loss is not None:
        optim_params.extend(list(sign_product_loss.parameters()))
    if sign_direction_loss is not None:
        optim_params.extend(list(sign_direction_loss.parameters()))

    optimizer = optim.Adam(
        optim_params,
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    pos_mask_train = train_data.edge_weight > 0
    neg_mask_train = train_data.edge_weight < 0
    pos_edge_index_train = train_data.edge_index[:, pos_mask_train]
    neg_edge_index_train = train_data.edge_index[:, neg_mask_train]
    
    print(f"    类别: 0={class_counts[0]}, 1={class_counts[1]}")
    print(f"    权重: 0={class_weights[0]:.3f}, 1={class_weights[1]:.3f}")
    
    best_val_score = float('-inf')
    best_results = {}
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        if config['use_temp_anneal']:
            progress = epoch / config['epochs']
            temp = config['temp_end'] + (config['temp_start'] - config['temp_end']) * \
                   0.5 * (1 + np.cos(np.pi * progress))
            model.set_temp(max(temp, config['temp_end']))

        model.train()
        if sign_product_loss is not None:
            sign_product_loss.train()
        if sign_direction_loss is not None:
            sign_direction_loss.train()
        optimizer.zero_grad()
        
        output = model(
            train_data.x,
            train_data.edge_index,
            train_data.edge_weight,
            train_data.pred_edge_index,
            hard=False
        )

        balance_stats = output.get('balance_stats', {}) or {}
        num_candidates = balance_stats.get('num_candidate_edges', balance_stats.get('num_new_edges', 0))
        num_retained = balance_stats.get('num_retained_edges', 0)
        
        logits = output['logits']
        l1_reg = output.get('l1_reg', None)
        edge_probs = output.get('edge_probs', None)
        emb_refined = output.get('h2', None)
            
        loss = loss_fn(logits, train_data.y)

        # 额外的符号监督损失
        if emb_refined is not None:
            if sign_product_loss is not None and \
               pos_edge_index_train.numel() > 0 and neg_edge_index_train.numel() > 0:
                loss = loss + sign_product_weight * \
                    sign_product_loss(emb_refined, pos_edge_index_train, neg_edge_index_train)
            if sign_direction_loss is not None and \
               pos_edge_index_train.numel() > 0 and neg_edge_index_train.numel() > 0:
                loss = loss + sign_direction_weight * \
                    sign_direction_loss(emb_refined, pos_edge_index_train, neg_edge_index_train)

        # Gumbel 概率的 L1 稀疏约束
        l1_reg_weight = config.get('l1_reg_weight', 0.0)
        if l1_reg_weight > 0:
            if l1_reg is None and edge_probs is not None and edge_probs.numel() > 0:
                l1_reg = edge_probs.abs().mean()
            if l1_reg is not None:
                loss = loss + l1_reg_weight * l1_reg
        
        loss.backward()
        
        # 梯度裁剪
        grad_clip_value = config.get('grad_clip_value', 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        optimizer.step()

        model.eval()
        if sign_product_loss is not None:
            sign_product_loss.eval()
        if sign_direction_loss is not None:
            sign_direction_loss.eval()
        with torch.no_grad():
            val_output = model(
                val_data.x, 
                val_data.edge_index, 
                val_data.edge_weight,
                val_data.pred_edge_index, 
                hard=True
            )
            val_logits = val_output['logits']
            val_f1, val_acc, val_auc = compute_metrics(val_logits, val_data.y)
            
            test_output = model(
                test_data.x, 
                test_data.edge_index, 
                test_data.edge_weight,
                test_data.pred_edge_index, 
                hard=True
            )
            test_logits = test_output['logits']
            test_f1, test_acc, test_auc = compute_metrics(test_logits, test_data.y)
        
        if epoch == 0 or (epoch + 1) % config['print_every'] == 0:
            print(f"    Epoch {epoch+1:4d} - Loss: {loss.item():.4f} | "
                  f"Val: F1={val_f1:.4f} ACC={val_acc:.4f} AUC={val_auc:.4f} | "
                  f"Test: F1={test_f1:.4f} ACC={test_acc:.4f} AUC={test_auc:.4f} | "
                  f"Expand {num_candidates} -> {num_retained}")

        val_score = val_f1
        if val_score > best_val_score:
            best_val_score = val_score
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


def train_all_splits(data_path, config, dataset_name=None, noise_ratio=None, results_dir='results'):
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

        summary = {
            'dataset': dataset_name or dataset_label,
            'noise_ratio': noise_ratio if noise_ratio is not None else float(noise_label),
            'num_splits': num_splits,
            'num_success': len(results),
            'averages': {
                'val_f1': avg_val_f1,
                'val_f1_std': std_val_f1,
                'val_acc': avg_val_acc,
                'val_acc_std': std_val_acc,
                'val_auc': avg_val_auc,
                'val_auc_std': std_val_auc,
                'test_f1': avg_test_f1,
                'test_f1_std': std_test_f1,
                'test_acc': avg_test_acc,
                'test_acc_std': std_test_acc,
                'test_auc': avg_test_auc,
                'test_auc_std': std_test_auc,
            },
            'per_split': results,
        }
        
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        if dataset_name:
            dataset_label = dataset_name
            inferred_noise = None
        else:
            dataset_label, inferred_noise = infer_dataset_and_noise(data_path)

        if noise_ratio is not None:
            noise_label = format_ratio(noise_ratio)
        else:
            noise_label = inferred_noise or '0'

        log_filename = results_path / f"{dataset_label}-{noise_label}-{timestamp}.txt"
        
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
        
        summary['log_path'] = str(log_filename)
        print(f"\n结果已保存至: {log_filename}")
        return summary
    
    print("\n没有成功的划分，跳过记录。")
    return None
