"""
LSTM 训练脚本 —— 使用论文 Krauss et al. (2025) 中的最优参数
============================================================
论文: Incorporating Respiratory Signals for ML-based Multi-Modal Sleep Stage Classification
参数来源: 5-Class / 3-Class Classification - MESA / SHHS Baseline

每次训练自动在 exports_our/<时间戳>/ 下保存 config.json，里面记录了所有参数。
不需要改脚本，通过命令行参数切换配置。

用法:
    # MESA 5 分类 (论文默认, ACT+HRV+RRV, 170 epoch)
    python LSTM_paper_params.py -d MESA_Sleep -c 5stage

    # SHHS1 / SHHS2 (无体动数据，自动使用 HRV+RRV)
    python LSTM_paper_params.py -d SHHS1 -c 5stage
    python LSTM_paper_params.py -d SHHS2 -c 3stage

    # 快速测试 (20人, 3 epoch)
    python LSTM_paper_params.py -d SHHS1 --small --quick

    # 切换模态或超参数
    python LSTM_paper_params.py -c 3stage --hidden 256 --layers 4 --lr 1e-4
    python LSTM_paper_params.py -d MESA_Sleep --modality HRV RRV

    # 查看所有参数
    python LSTM_paper_params.py --help
"""
import argparse
import json
import pickle
import random
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# 20260806: 屏蔽第三方/历史代码的 FutureWarning 噪音 (pandas 位置索引弃用等),
# 训练日志只保留关键信息
warnings.filterwarnings("ignore", category=FutureWarning)

from sleep_analysis.classification.deep_learning.lstm.data_peparation import DataPreparation
from sleep_analysis.classification.deep_learning.lstm.LSTM import LSTM
from sleep_analysis.classification.deep_learning.utils import get_num_input
from sleep_analysis.datasets.helper import get_random_split
from sleep_analysis.datasets.mesadataset import MesaDataset
from sleep_analysis.datasets.shhs_dataset import ShhsDataset
from sleep_analysis.datasets.mixed_dataset import MixedDataset

# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="LSTM Sleep Stage Classification")
# 数据集
parser.add_argument("-d", "--dataset", default="MESA_Sleep",
                    help="数据集: MESA_Sleep / SHHS1 / SHHS2 / MESA_Sleep+SHHS2 等任意 '+' 组合")
parser.add_argument("--small", action="store_true", help="只用 20 个被试验证管线")
parser.add_argument("--split-file", type=str, default=None,
                    help="划分名单 JSON (含 train/val/test 三个被试 ID 数组); "
                         "指定后按名单划分而非随机划分 (保证跨数据集/实验可比)")
# 分类
parser.add_argument("-c", "--classification", default="5stage",
                    choices=["binary", "3stage", "4stage", "5stage"])
parser.add_argument("-m", "--modality", nargs="+", default=None,
                    choices=["ACT", "HRV", "RRV", "EDR"],
                    help="特征模态 (默认: MESA=ACT+HRV+RRV, SHHS=HRV+RRV)")
# 训练
parser.add_argument("--quick", action="store_true", help="快速测试: 只跑 3 个 epoch")
parser.add_argument("--epochs", type=int, default=170)
# 超参数 (默认值来自论文)
parser.add_argument("--seq-len", type=int, default=21)
parser.add_argument("--hidden", type=int, default=556, dest="hidden_size")
parser.add_argument("--layers", type=int, default=6, dest="num_layers")
parser.add_argument("--dropout", type=float, default=0.255)
parser.add_argument("--lr", type=float, default=6.31e-5)
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--focal-gamma", type=float, default=2.0,
                    help="Focal Loss 聚焦参数 (0=普通CE, 越大越关注难样本)")
parser.add_argument("--grad-clip", type=float, default=0.5,
                    help="梯度裁剪阈值 ∥∇∥₂")
parser.add_argument("--seed", type=int, default=42)
# 恢复
parser.add_argument("--load-weights", type=str, default=None,
                    help="从指定 .pt 文件加载权重 (如 exports_our/.../checkpoints/best_model.pt)"
                        " 会自动读取同级目录下的 config.json 恢复超参数")
parser.add_argument("--eval-only", action="store_true",
                    help="仅评估，跳过训练")
parser.add_argument("--causal", action="store_true",
                    help="实时分期模式: 仅在序列左侧padding, 预测每个窗口的最后时刻")

args = parser.parse_args()

# 如果指定了权重，从同次训练的 config.json 中恢复超参数
if args.load_weights:
    weights_dir = Path(args.load_weights).parent.parent  # checkpoints/ → run dir/
    config_file = weights_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            saved_config = json.load(f)
        print(f"[LOAD] Restoring params from {config_file}")
        # 用 config.json 里的值覆盖命令行参数
        args.dataset = saved_config.get("dataset", args.dataset)
        args.classification = saved_config.get("classification", args.classification)
        args.modality = saved_config.get("modality", args.modality)
        args.hidden_size = saved_config.get("hidden_size", args.hidden_size)
        args.num_layers = saved_config.get("num_layers", args.num_layers)
        args.dropout = saved_config.get("dropout", args.dropout)
        args.lr = saved_config.get("learning_rate", args.lr)
        args.seq_len = saved_config.get("seq_len", args.seq_len)
        args.batch_size = saved_config.get("batch_size", args.batch_size)
        args.focal_gamma = saved_config.get("focal_gamma", args.focal_gamma)
        args.grad_clip = saved_config.get("grad_clip", args.grad_clip)
        args.causal = saved_config.get("causal", args.causal)
        args.seed = saved_config.get("seed", args.seed)
    else:
        print(f"[WARNING] {config_file} not found, using current CLI params."
              f" 确保超参数与训练时一致，否则 load_state_dict 会报错!")

# 快速测试覆盖
if args.quick:
    args.epochs = 3
if args.small:
    print("[SMALL mode] Using only 20 subjects")

# 数据集创建
# ---------------------------------------------------------------------------
DATASET_PARTS = args.dataset.split("+")
_DS_REGISTRY = {
    "MESA_Sleep": lambda: MesaDataset(),
    "SHHS1": lambda: ShhsDataset(study="shhs1"),
    "SHHS2": lambda: ShhsDataset(study="shhs2"),
}

# 按数据集默认选 modality
has_mesa = any(not p.startswith("SHHS") for p in DATASET_PARTS)
has_shhs = any(p.startswith("SHHS") for p in DATASET_PARTS)
if args.modality is not None:
    if has_shhs and "ACT" in args.modality:
        print("[WARNING] Some datasets have no actigraphy. Removing ACT from modality.")
        args.modality = [m for m in args.modality if m != "ACT"]
else:
    if has_mesa and has_shhs:
        args.modality = ["HRV", "RRV"]
    elif has_shhs:
        args.modality = ["HRV", "RRV"]
    else:
        args.modality = ["ACT", "HRV", "RRV"]

# 对每个子数据集分别 80/20 划分，再拼成 train/val/test
# SHHS1/SHHS2 共享参与者：先按 nsrrid 联合划分，避免同一人被分到训练集和测试集
_singleton = len(DATASET_PARTS) == 1
_train_sources, _val_sources, _test_sources = {}, {}, {}
_shhs_datasets = [p for p in DATASET_PARTS if p.startswith("SHHS")]
if len(_shhs_datasets) > 1:
    # 收集所有 SHHS 子集的 nsrrid (SHHS 的 subj_id 就是 nsrrid)
    _shhs_pids = set()
    _shhs_ds_map = {}
    for name in _shhs_datasets:
        ds = _DS_REGISTRY[name]()
        if args.small:
            ds = ds[0:20]
        key = name.lower().replace("_", "")
        _shhs_ds_map[key] = ds
        _shhs_pids.update(ds.index["subj_id"].tolist())

    # 按参与者 ID 划分 (80/20 → 80/20)
    _pids_sorted = sorted(_shhs_pids)
    np.random.seed(args.seed)
    np.random.shuffle(_pids_sorted)
    n_test = max(1, int(len(_pids_sorted) * 0.2))
    _test_pids = set(_pids_sorted[:n_test])
    _trainval_list = _pids_sorted[n_test:]   # 保持列表顺序，避免 set 迭代不确定性
    n_val = max(1, int(len(_trainval_list) * 0.2))
    _val_pids_set = set(_trainval_list[:n_val])
    _train_pids = set(_trainval_list[n_val:])

    print(f"[MIXED] SHHS participant-level split: "
          f"train={len(_train_pids)}, val={len(_val_pids_set)}, test={len(_test_pids)}")
    # 防御：确保 train/val/test 无被试重叠
    assert _train_pids.isdisjoint(_val_pids_set), "SHHS train/val overlap detected!"
    assert _train_pids.isdisjoint(_test_pids), "SHHS train/test overlap detected!"
    assert _val_pids_set.isdisjoint(_test_pids), "SHHS val/test overlap detected!"

    for key, ds in _shhs_ds_map.items():
        train_idx = [i for i, sid in enumerate(ds.index["subj_id"]) if sid in _train_pids]
        val_idx = [i for i, sid in enumerate(ds.index["subj_id"]) if sid in _val_pids_set]
        test_idx = [i for i, sid in enumerate(ds.index["subj_id"]) if sid in _test_pids]
        _train_sources[key] = ds[train_idx] if train_idx else ds[0:0]
        _val_sources[key] = ds[val_idx] if val_idx else ds[0:0]
        _test_sources[key] = ds[test_idx] if test_idx else ds[0:0]

# 非 SHHS (MESA) 独立划分
for name in [p for p in DATASET_PARTS if p not in _shhs_datasets or len(_shhs_datasets) <= 1]:
    if name not in _DS_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")
    ds = _DS_REGISTRY[name]()
    if args.small:
        ds = ds[0:20]
    if _singleton:
        dataset = ds
    else:
        key = name.lower().replace("_", "")
        if name.startswith("SHHS"):
            # 单 SHHS 数据集 (len(_shhs_datasets)<=1)：用标准随机划分
            src_train, src_test = get_random_split(ds)
            src_train, src_val = get_random_split(src_train)
        else:
            src_train, src_test = get_random_split(ds)
            src_train, src_val = get_random_split(src_train)
        _train_sources[key] = src_train
        _val_sources[key] = src_val
        _test_sources[key] = src_test

if not _singleton:
    train_set = MixedDataset(_train_sources)
    val_set = MixedDataset(_val_sources)
    test_set = MixedDataset(_test_sources)
    print(f"[MIXED] {', '.join(DATASET_PARTS)}: "
          f"train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    # 最终检查：SHHS1+SHHS2 的 train/val/test 在被试级无重叠
    if len(_shhs_datasets) > 1:
        def _de_prefix(ids):
            """从 shhs1@201206 → 201206 提取原始 nsrrid"""
            return {str(s).split("@", 1)[-1] for s in ids}
        train_ids = _de_prefix(train_set.index["subj_id"])
        val_ids = _de_prefix(val_set.index["subj_id"])
        test_ids = _de_prefix(test_set.index["subj_id"])
        assert train_ids.isdisjoint(val_ids), \
            f"SHHS train/val overlap in final split! {len(train_ids & val_ids)} subjects"
        assert train_ids.isdisjoint(test_ids), \
            f"SHHS train/test overlap in final split! {len(train_ids & test_ids)} subjects"
        assert val_ids.isdisjoint(test_ids), \
            f"SHHS val/test overlap in final split! {len(val_ids & test_ids)} subjects"

# ---------------------------------------------------------------------------
# 输出目录 & 配置保存
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parents[3]
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H%M%S")
OUTPUT_DIR = PROJECT_ROOT / "exports_our" / RUN_TIMESTAMP
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 20260806: 记录本次训练使用的数据来源 — study_data.json 可能随时更改
# (MESA/SHHS 路径切换等), 训练时把快照复制到 run 目录, 保证事后可追溯
with open(PROJECT_ROOT / "study_data.json") as _f:
    _study_cfg = json.load(_f)
shutil.copy(PROJECT_ROOT / "study_data.json", OUTPUT_DIR / "study_data.json")

# 将所有配置保存为 JSON，以后随时查阅
config = {
    "timestamp": RUN_TIMESTAMP,
    "dataset": args.dataset,
    "classification": args.classification,
    "modality": args.modality,
    "small": args.small,
    "quick": args.quick,
    "epochs": args.epochs,
    "seq_len": args.seq_len,
    "hidden_size": args.hidden_size,
    "num_layers": args.num_layers,
    "dropout": args.dropout,
    "learning_rate": args.lr,
    "batch_size": args.batch_size,
    "focal_gamma": args.focal_gamma,
    "grad_clip": args.grad_clip,
    "causal": args.causal,
    "seed": args.seed,
    "load_weights": args.load_weights,
    # 数据来源 (完整快照见同目录 study_data.json)
    "data_paths": {
        "processed_mesa_path_hpc": _study_cfg.get("processed_mesa_path_hpc"),
        "shhs1_processed_path": _study_cfg.get("shhs1_processed_path"),
        "shhs2_processed_path": _study_cfg.get("shhs2_processed_path"),
    },
}
with open(OUTPUT_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"Output directory: {OUTPUT_DIR}")
print(f"Config saved to: {OUTPUT_DIR / 'config.json'}")

# 固定随机种子
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# 1. 加载数据集 & 划分
# ---------------------------------------------------------------------------
print("=" * 60)
print(f"Classification: {args.classification} | Modality: {args.modality}")
print(f"Hyperparams: seq_len={args.seq_len}, hidden={args.hidden_size}, "
      f"layers={args.num_layers}, dropout={args.dropout}, lr={args.lr}, batch={args.batch_size}")
print(f"Epochs: {args.epochs}")
print("=" * 60)

print("\n[1/5] Dataset...")

if args.split_file is not None:
    # 20260806: 按名单划分 (--split-file) — 用固定的 train/val/test ID 列表,
    # 保证跨数据集/跨实验划分一致 (train_test_split 的 shuffle 依赖被试总数,
    # 1121 vs 1120 会把整个划分打乱, 两次实验无法严格对比)
    # 2026-08-07: 支持多数据集格式 (混合训练 MESA+SHHS1+SHHS2):
    #   {"mesasleep": {"train":[...], "val":[...], "test":[...]}, "shhs1": {...}, "shhs2": {...}}
    #   MixedDataset 的 subj_id 带 "src@" 前缀 → 按前缀匹配各自数据集名单;
    #   单数据集训练 (无前缀) → 遍历全部名单匹配 raw ID
    with open(args.split_file) as _f:
        _split = json.load(_f)
    _multi = all(isinstance(v, dict) and "train" in v for v in _split.values())

    if _multi:
        _want = {name: {k: set(v) for k, v in parts.items()} for name, parts in _split.items()}
        if _singleton:
            # 单数据集: subj_id 无前缀, 在所有数据集名单中匹配 raw ID
            _all_parts = {k: set().union(*(_want[n][k] for n in _want)) for k in ("train", "val", "test")}
            _ids = [str(s) for s in dataset.index["subj_id"]]
            train_set = dataset[[i for i, sid in enumerate(_ids) if sid in _all_parts["train"]]]
            val_set = dataset[[i for i, sid in enumerate(_ids) if sid in _all_parts["val"]]]
            test_set = dataset[[i for i, sid in enumerate(_ids) if sid in _all_parts["test"]]]
        else:
            # 混合数据集: 按 src 前缀 (与 MixedDataset 的 subj_id 前缀一致) 匹配各自名单
            _src_name_map = {name.lower().replace("_", ""): name for name in DATASET_PARTS}
            _train_sources, _val_sources, _test_sources = {}, {}, {}
            for _src_key, _parts in _want.items():
                if _src_key not in _src_name_map:
                    print(f"[WARNING] split 名单中的数据集 {_src_key} 不在本次训练数据集内, 跳过")
                    continue
                ds = _DS_REGISTRY[_src_name_map[_src_key]]()
                _ids = [str(s) for s in ds.index["subj_id"]]
                _tr = [i for i, sid in enumerate(_ids) if sid in _parts["train"]]
                _va = [i for i, sid in enumerate(_ids) if sid in _parts["val"]]
                _te = [i for i, sid in enumerate(_ids) if sid in _parts["test"]]
                _train_sources[_src_key] = ds[_tr] if _tr else ds[0:0]
                _val_sources[_src_key] = ds[_va] if _va else ds[0:0]
                _test_sources[_src_key] = ds[_te] if _te else ds[0:0]
            train_set = MixedDataset(_train_sources)
            val_set = MixedDataset(_val_sources)
            test_set = MixedDataset(_test_sources)
            dataset = train_set  # 兼容后续打印 (L395 用 len(dataset), 与原自动划分分支一致)

        # 数据与 split 名单一致性告警 (总集合层面: 全部数据集的 ID, 而非仅 train)
        _all_want = {raw for parts in _want.values() for k in parts for raw in parts[k]}
        if _singleton:
            _all_data = {str(s).split("@")[-1] for s in dataset.index["subj_id"]}
        else:
            _all_data = set()
            for _sources in (_train_sources, _val_sources, _test_sources):
                for _src_ds in _sources.values():
                    _all_data |= {str(s).split("@")[-1] for s in _src_ds.index["subj_id"]}
        _in_split_not_data = sorted(_all_want - _all_data)
        _in_data_not_split = sorted(_all_data - _all_want)
        if _in_split_not_data or _in_data_not_split:
            print("[WARNING] 数据与 split 名单不一致!")
            if _in_split_not_data:
                print(f"  [WARNING] 名单中有但数据缺失 {len(_in_split_not_data)} 个: {_in_split_not_data[:10]}")
            if _in_data_not_split:
                print(f"  [WARNING] 数据中有但名单缺失 {len(_in_data_not_split)} 个: {_in_data_not_split[:10]}")
        for _k in ("train", "val", "test"):
            _n_actual = len(locals()[f"{_k}_set"].index)
            _n_want = sum(len(_want[n][_k]) for n in _want)
            _flag = "  ← 不一致" if _n_actual != _n_want else ""
            print(f"[SPLIT] {_k}: 名单 {_n_want} -> 实际 {_n_actual}{_flag}")
    else:
        # 单数据集格式 (旧): {"train": [...], "val": [...], "test": [...]}
        _want = {k: set(v) for k, v in _split.items()}
        # subj_id 可能带 "source@" 前缀 (MixedDataset), 统一取 "@" 之后
        _ids = [str(s).split("@")[-1] for s in dataset.index["subj_id"]]
        _id_set = set(_ids)
        train_set = dataset[[i for i, sid in enumerate(_ids) if sid in _want["train"]]]
        val_set = dataset[[i for i, sid in enumerate(_ids) if sid in _want["val"]]]
        test_set = dataset[[i for i, sid in enumerate(_ids) if sid in _want["test"]]]

        # 20260806: 数据与 split 名单一致性告警 (双向检查)
        _in_split_not_data = sorted((_want["train"] | _want["val"] | _want["test"]) - _id_set)  # 名单有但数据无
        _in_data_not_split = sorted(_id_set - (_want["train"] | _want["val"] | _want["test"]))  # 数据有但名单无
        if _in_split_not_data or _in_data_not_split:
            print("[WARNING] 数据与 split 名单不一致!")
            if _in_split_not_data:
                print(f"  [WARNING] 名单中有但数据缺失 {len(_in_split_not_data)} 个: {_in_split_not_data}")
            if _in_data_not_split:
                print(f"  [WARNING] 数据中有但名单缺失 {len(_in_data_not_split)} 个: {_in_data_not_split}")
        for _k in ("train", "val", "test"):
            _n_actual = len(locals()[f"{_k}_set"].index)
            _n_want = len(_want[_k])
            _flag = "  ← 不一致" if _n_actual != _n_want else ""
            print(f"[SPLIT] {_k}: 名单 {_n_want} -> 实际 {_n_actual}{_flag}")
elif _singleton:
    train_set, test_set = get_random_split(dataset=dataset)
    train_set, val_set = get_random_split(dataset=train_set)
else:
    # 已在上面拆好，直接打印
    dataset = train_set  # 兼容后续引用

msg = f"  total={len(dataset)} → train={len(train_set)}, val={len(val_set)}, test={len(test_set)}"
if not _singleton:
    # 打印各子集分布
    for split_name, split_ds in [("train", train_set), ("val", val_set), ("test", test_set)]:
        src_counts = split_ds.index["_source"].value_counts().to_dict()
        parts = ", ".join(f"{k}:{v}" for k, v in sorted(src_counts.items()))
        msg += f"\n    {split_name}: {parts}"
print(msg)

# ---------------------------------------------------------------------------
# 2. 构建序列数据
# ---------------------------------------------------------------------------
print("\n[2/5] Preparing sequence data...")
data_loader = DataPreparation(seq_len=args.seq_len, overlap=None, causal=args.causal)
x_train, y_train, x_val, y_val, x_test, y_test, scaler = data_loader.get_final_tensors(
    args.modality, train_set, val_set, test_set, args.classification
)
print(f"  x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"  x_val:   {x_val.shape}, y_val:   {y_val.shape}")
print(f"  x_test:  {len(x_test)} subjects")

# 保存第一层 StandardScaler（训练集拟合），推理时必需
checkpoints_dir = OUTPUT_DIR / "checkpoints"
checkpoints_dir.mkdir(parents=True, exist_ok=True)
scaler_path = checkpoints_dir / "scaler.json"
with open(scaler_path, "w") as f:
    json.dump({
        "n_features": int(scaler.n_features_in_),
        "mean_": scaler.mean_.tolist(),
        "scale_": scaler.scale_.tolist(),
    }, f, indent=2)
print(f"  Scaler saved to: {scaler_path}")

# 混合数据集：为验证集准备子集张量（训练时每 epoch 打印 per-source 指标）
val_sources_dict = {}
if not _singleton:
    for src_name, src_ds in _val_sources.items():
        xs, ys, _ = data_loader.get_data(
            src_ds, modality=args.modality, scaler=scaler,  # 复用训练集 scaler
            classification_type=args.classification, padding=True
        )
        val_sources_dict[src_name] = (xs, ys)
    print(f"  val sources: { {k: v[0].shape[0] for k, v in val_sources_dict.items()} }")

# ---------------------------------------------------------------------------
# 3. 创建模型
# ---------------------------------------------------------------------------
print("\n[3/5] Creating LSTM model...")
num_inputs = get_num_input(args.modality)
print(f"  Input dim: {num_inputs}")
print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

model = LSTM(
    num_epochs=args.epochs,
    input_size=num_inputs,
    hidden_size=args.hidden_size,
    num_layers=args.num_layers,
    learning_rate=args.lr,
    seq_len=args.seq_len,
    dropout=args.dropout,
    batch_size=args.batch_size,
    modality=args.modality,
    dataset_name=args.dataset,
    classification_type=args.classification,
    output_dir=str(OUTPUT_DIR),
    focal_gamma=args.focal_gamma,
    grad_clip=args.grad_clip,
    val_sources=val_sources_dict if not _singleton else None,
)

# 加载已有权重 (如果指定)
if args.load_weights:
    print(f"  Loading weights from: {args.load_weights}")
    model._load_best_model_from_path(args.load_weights)
    # 检查同目录下是否有配套的 scaler.json
    companion_scaler = Path(args.load_weights).parent / "scaler.json"
    if companion_scaler.exists():
        print(f"  Companion scaler found: {companion_scaler}")
    else:
        print(f"  [WARN] No companion scaler found at {companion_scaler}")

# ---------------------------------------------------------------------------
# 4. 训练 (--eval-only 则跳过)
# ---------------------------------------------------------------------------
if not args.eval_only:
    print("\n[4/5] Training...")
    print("-" * 60)
    max_val_mcc = model.train(x_train, y_train, x_val, y_val, retrain=False)
    print(f"\n  Best validation MCC: {max_val_mcc:.4f}")
else:
    print("\n[4/5] Skipping training (--eval-only)")
    max_val_mcc = 0.0

# ---------------------------------------------------------------------------
# 5. 测试 & 保存结果
# ---------------------------------------------------------------------------
print("\n[5/5] Evaluating on test set...")
subject_results, score_mean, pred_dict = model.test(x_test, y_test, retrain=False)

results_dir = OUTPUT_DIR / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# 逐被试指标
result_file = results_dir / "per_subject_metrics.csv"
subject_results.index.name = "metric"
subject_results.to_csv(result_file)
print(f"  Per-subject results saved to: {result_file}")

# 混淆矩阵
from sleep_analysis.classification.ml_algorithms.ml_pipeline_helper import _get_sleep_stage_labels
sleep_stage_labels, conf_matrix = _get_sleep_stage_labels(args.classification)
for subj in subject_results.columns:
    conf_matrix += subject_results[subj]["confusion_matrix"].get_value()

conf_df = pd.DataFrame(conf_matrix, index=sleep_stage_labels, columns=sleep_stage_labels)
print(f"\n  Confusion Matrix (counts, rows=true → cols=pred):")
print(conf_df)

conf_pct = conf_df.div(conf_df.sum(axis=1), axis=0) * 100
print(f"\n  Confusion Matrix (%, rows=true → cols=pred):")
print(conf_pct.round(1))

# 保存
with open(results_dir / "results.json", "w") as f:
    json.dump({
        "per_subject": subject_results.to_dict(),
        "mean": score_mean.to_dict(),
        "confusion_matrix": conf_df.to_dict(),
        "confusion_matrix_pct": conf_pct.round(1).to_dict(),
        "best_val_mcc": float(max_val_mcc),
    }, f, indent=2, default=str)

conf_df.to_csv(results_dir / "confusion_matrix.csv")
conf_pct.round(1).to_csv(results_dir / "confusion_matrix_percent.csv")

with open(results_dir / "predictions.pickle", "wb") as f:
    pickle.dump(pred_dict, f)

# 按子数据集汇总（混合模式）
if not _singleton:
    # 从 prefixed ID 解析来源: "mesasleep@2982" → "mesasleep"
    def _parse_source(col_name):
        i = str(col_name).find("@")
        return str(col_name)[:i] if i > 0 else "unknown"
    src_conf_matrices = {}
    print(f"\n{'=' * 60}")
    print("Per-Source Test Results:")
    print(f"{'=' * 60}")
    for src_name in sorted(set(_parse_source(s) for s in subject_results.columns)):
        src_subjs = [s for s in subject_results.columns if _parse_source(s) == src_name]
        if not src_subjs:
            continue
        numeric_cols = [c for c in subject_results[src_subjs].index
                        if c != "confusion_matrix"]
        src_mean = subject_results[src_subjs].loc[numeric_cols].astype(float).mean(axis=1)
        print(f"\n  [{src_name}]  (n={len(src_subjs)})")
        for metric in ["accuracy", "kappa", "mcc"]:
            if metric in src_mean:
                print(f"    {metric:10s}: {src_mean[metric]:.4f}")
        src_cm = pd.DataFrame(0, index=sleep_stage_labels, columns=sleep_stage_labels)
        for s in src_subjs:
            if "confusion_matrix" in subject_results[s]:
                src_cm += subject_results[s]["confusion_matrix"].get_value()
        src_conf_matrices[src_name] = src_cm
        print(f"    Confusion Matrix (%):")
        pct = src_cm.div(src_cm.sum(axis=1), axis=0) * 100
        print(pct.round(1).to_string())
    for src_name, cm in src_conf_matrices.items():
        cm.to_csv(results_dir / f"confusion_matrix_{src_name}.csv")
        (cm.div(cm.sum(axis=1), axis=0) * 100).round(1).to_csv(
            results_dir / f"confusion_matrix_{src_name}_percent.csv")

# 汇总
print(f"\n{'=' * 60}")
print("Test Set Results (OVERALL mean across subjects):")
print(score_mean.to_string())
print(f"{'=' * 60}")
print(f"All outputs saved to: {OUTPUT_DIR}")
