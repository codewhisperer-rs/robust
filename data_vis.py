import pickle

with open('/home/houyikang/data/processed/bitcoin_alpha/random_masking/seed_0_num_splits_20/test_0.05_val_0.05/mask_0.75/signed_datasets.pkl', 'rb') as f:
    data = pickle.load(f)

dataset = data.get(0)
print(dataset.keys())

if dataset:
    train_edges = dataset.get('train', {}).get('edges', None)
    train_labels = dataset.get('train', {}).get('label', None)

    print("训练集边：", train_edges[:5] if train_edges is not None else "无数据")
    print("训练集标签：", train_labels[:5] if train_labels is not None else "无数据")
    print("训练集边数量：", train_edges.size(0) if train_edges is not None else "无数据")

    val_edges = dataset.get('val', {}).get('edges', None)
    val_labels = dataset.get('val', {}).get('label', None)

    print("验证集边：", val_edges[:5] if val_edges is not None else "无数据")
    print("验证集标签：", val_labels[:5] if val_labels is not None else "无数据")
    print("验证集边数量：", val_edges.size(0) if val_edges is not None else "无数据")

    test_edges = dataset.get('test', {}).get('edges', None)
    test_labels = dataset.get('test', {}).get('label', None)

    print("测试集边：", test_edges[:5] if test_edges is not None else "无数据")
    print("测试集标签：", test_labels[:5] if test_labels is not None else "无数据")
    print("测试集边数量：", test_edges.size(0) if test_edges is not None else "无数据")
else:
    print("未找到键为 0 的数据划分")
