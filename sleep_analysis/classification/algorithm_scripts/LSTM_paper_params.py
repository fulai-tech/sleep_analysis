"""
LSTM 训练脚本 —— 使用论文 Krauss et al. (2025) 中的最优参数
============================================================
论文: Incorporating Respiratory Signals for ML-based Multi-Modal Sleep Stage Classification
参数来源: 5-Class / 3-Class Classification - MESA / SHHS Baseline

每次训练自动在 exports_our/<时间戳>/ 下保存 config.json，里面记录了所有参数。
不需要改脚本，通过命令行参数切换配置。

用法:
    # MESA 5 分类 (论文默认, ACT+HRV+RRV, 170 epoch)
    python LSTM_paper_params.py -d MESA -c 5stage

    # SHHS1 / SHHS2 (无体动数据，自动使用 HRV+RRV)
    python LSTM_paper_params.py -d SHHS1 -c 5stage
    python LSTM_paper_params.py -d SHHS2 -c 3stage

    # 快速测试 (20人, 3 epoch)
    python LSTM_paper_params.py -d SHHS1 --small --quick

    # 切换模态或超参数
    python LSTM_paper_params.py -c 3stage --hidden 256 --layers 4 --lr 1e-4
    python LSTM_paper_params.py -d MESA --modality HRV RRV

    # 查看所有参数
    python LSTM_paper_params.py --help
"""
import argparse
import json
import pickle
import sys
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
from sleep_analysis.datasets.mixed_dataset import MixedDataset
from sleep_analysis.datasets.registry import get_dataset, get_dataset_class
# import sleep_analysis.datasets 由上面的 registry 导入隐式触发 (__init__ 目录加载各数据集自注册)

# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="LSTM Sleep Stage Classification")
# 数据集
parser.add_argument("-d", "--dataset", default="MESA",
                    help="数据集: MESA / SHHS1 / SHHS2 / MESA+SHHS2 等任意 '+' 组合"
                         " (MESA_Sleep 为 MESA 的旧名别名)")
parser.add_argument("--small", action="store_true", help="只用 20 个被试验证管线")
parser.add_argument("--split-file", type=str, default=None,
                    help="划分名单 JSON (含 train/val/test 三个被试 ID 数组); "
                         "指定后按名单划分而非随机划分 (保证跨数据集/实验可比)。"
                         "不指定时: 若所有请求数据集都声明了 split_file (数据集自描述), "
                         "自动加载各自的划分文件再组合 (20260825)")
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
parser.add_argument("--stateful", action="store_true",
                    help="状态化模式 (2026-08-11): 逐 epoch 扫描 + LSTM 状态携带 + truncated BPTT 训练; "
                         "需要 --causal。消除滑窗重算 (端侧每帧 O(1)), 上下文不受 seq-len 限制")
parser.add_argument("--state-chunk", type=int, default=64,
                    help="truncated BPTT 截断长度 C (stateful 训练); P = batch_size // C 个被试并行")
parser.add_argument("--patience", type=int, default=5,
                    help="早停耐心: 验证 loss 连续 N 个 epoch 不降即停 "
                         "(stateful 训练摆动周期较长时可调大到 10-15)")
parser.add_argument("--inv-freq", action="store_true",
                    help="类权重改为 1/freq (论文公式, 少数类惩罚更重; 默认 1-freq)。"
                         "stateful 的深夜 deep 被时间捷径献祭时用")
parser.add_argument("--internal-norm", action="store_true",
                    help="恢复旧版模型内部 per-batch 归一化 (08-07 前行为, 复现 08-06 基线用)。"
                         "默认关闭 (外部 scaler 作为唯一归一化)")
parser.add_argument("--shuffle-mode", type=str, default="none",
                    choices=["none", "sample", "subject"],
                    help="每 epoch 打乱训练样本 (numpy CPU, seed+epoch 派生, 跨机器可复现)。"
                         "none=固定顺序 (复现历史 run 用); sample=样本级打乱 (拟合快但全量数据"
                         "实测 val 早衰); subject=按被试打乱 (保留批内连续窗口, 推荐)。"
                         "注意: 不用 torch.randperm 的 CUDA 路径 — CUDA RNG 与 GPU 架构相关")
parser.add_argument("--wake-weight", type=float, default=1.0,
                    help="只放大 wake (类0) 的 loss 权重: 权重 = (1-freq)*wake_weight。"
                         "1.0 = 不变; 如 wake:睡眠=3:7 想拉平可试 7/3≈2.33。"
                         "Adam 对 loss 全局缩放近似不变, 一般无需降 lr, 震荡明显再降")

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
        args.stateful = saved_config.get("stateful", args.stateful)
        args.state_chunk = saved_config.get("state_chunk", args.state_chunk)
        args.patience = saved_config.get("patience", args.patience)
        args.inv_freq = saved_config.get("inv_freq", args.inv_freq)
        args.internal_norm = saved_config.get("internal_norm", args.internal_norm)
        # 兼容旧 config: 2026-08-27 前用 "shuffle" 字段 (true=sample), 现统一为 shuffle_mode
        args.shuffle_mode = saved_config.get(
            "shuffle_mode", "sample" if saved_config.get("shuffle") else args.shuffle_mode
        )
        args.wake_weight = saved_config.get("wake_weight", args.wake_weight)
        args.split_file = saved_config.get("split_file", args.split_file)
    else:
        print(f"[WARNING] {config_file} not found, using current CLI params."
              f" 确保超参数与训练时一致，否则 load_state_dict 会报错!")

# 全量数据实测: sample 级打乱在固定 lr 下拟合过快 → val 在 epoch 4-5 达峰后恶化
# (固定序 30+ epoch 持续下降), 早停被真实触发; 与 --internal-norm 无关 (2×2 对照)。
if args.shuffle_mode == "sample":
    print("[WARNING] --shuffle-mode sample: 全量数据实测 val 在 epoch 4-5 达峰后恶化 "
          "(拟合过快, 固定 lr 无衰减)。建议 --shuffle-mode subject 或降 lr。", flush=True)
# ✅2026-08-11: stateful 依赖因果窗口 (状态化 = 只用已见数据); 校验须在 config restore 之后,
# 否则 --load-weights <causal run> --stateful 会被 parse 期校验误杀 (causal 在 restore 后才恢复)
if args.stateful and not args.causal:
    print("[ERROR] --stateful requires --causal (状态化模式依赖因果窗口)", flush=True)
    sys.exit(1)
# ✅2026-08-12: state_chunk 是 truncated BPTT 截断长度, 也是分组除数/range step —
# 0 或负数会在 _stateful_groups/_stateful_build_chunk 里 ZeroDivisionError 或静默空转
if args.stateful and args.state_chunk <= 0:
    print(f"[ERROR] --state-chunk must be a positive integer (got {args.state_chunk})", flush=True)
    sys.exit(1)
if args.patience < 1:
    print(f"[ERROR] --patience must be >= 1 (got {args.patience})", flush=True)
    sys.exit(1)

# 快速测试覆盖
if args.quick:
    args.epochs = 3
if args.small:
    print("[SMALL mode] Using only 20 subjects")

# 数据集创建
# ---------------------------------------------------------------------------
DATASET_PARTS = args.dataset.split("+")

# ✅20260822: 模态默认值改为读数据集类的自描述属性 (modality_defaults /
# has_actigraphy), 不再按前缀猜 — 新增数据集零改动。
# 规则: 各数据集默认模态取并集; 任一数据集无体动 → 剔除 ACT。
_has_actigraphy_all = True
_default_modality = []
for _p in DATASET_PARTS:
    _cls = get_dataset_class(_p)
    _has_actigraphy_all &= _cls.has_actigraphy
    for _m in _cls.modality_defaults:
        if _m not in _default_modality:
            _default_modality.append(_m)
if args.modality is not None:
    if not _has_actigraphy_all and "ACT" in args.modality:
        print("[WARNING] Some datasets have no actigraphy. Removing ACT from modality.")
        args.modality = [m for m in args.modality if m != "ACT"]
else:
    args.modality = [m for m in _default_modality if m != "ACT"] if not _has_actigraphy_all else _default_modality

# 对每个子数据集分别 80/20 划分，再拼成 train/val/test
# ✅20260822: 共享被试身份的数据集组 (person_pool 相同, 如 SHHS1/SHHS2 共享
# nsrrid) 先按人员 ID 联合划分, 避免同一人被分到训练集和测试集 — 由数据集类
# 的 person_pool 钩子驱动, 不再硬编码前缀。
_singleton = len(DATASET_PARTS) == 1
_train_sources, _val_sources, _test_sources = {}, {}, {}
_joint_done = set()

# 按 person_pool 分组: 同池数据集做联合人员级划分
_pools = {}
for _name in DATASET_PARTS:
    _pool = get_dataset_class(_name).person_pool or _name.lower()
    _pools.setdefault(_pool, []).append(_name)

for _pool, _names in _pools.items():
    if len(_names) < 2:
        continue
    # 收集该池所有数据集的被试身份 (如 SHHS 的 subj_id 就是 nsrrid)
    _pool_pids = set()
    _pool_ds_map = {}
    for _name in _names:
        ds = get_dataset(_name)
        if args.small:
            ds = ds[0:20]
        key = _name.lower().replace("_", "")
        _pool_ds_map[key] = ds
        _pool_pids.update(ds.index["subj_id"].tolist())

    # 按参与者 ID 划分 (80/20 → 80/20)
    _pids_sorted = sorted(_pool_pids)
    np.random.seed(args.seed)
    np.random.shuffle(_pids_sorted)
    n_test = max(1, int(len(_pids_sorted) * 0.2))
    _test_pids = set(_pids_sorted[:n_test])
    _trainval_list = _pids_sorted[n_test:]   # 保持列表顺序，避免 set 迭代不确定性
    n_val = max(1, int(len(_trainval_list) * 0.2))
    _val_pids_set = set(_trainval_list[:n_val])
    _train_pids = set(_trainval_list[n_val:])

    print(f"[MIXED] {_pool} participant-level split: "
          f"train={len(_train_pids)}, val={len(_val_pids_set)}, test={len(_test_pids)}")
    # 防御：确保 train/val/test 无被试重叠
    assert _train_pids.isdisjoint(_val_pids_set), f"{_pool} train/val overlap detected!"
    assert _train_pids.isdisjoint(_test_pids), f"{_pool} train/test overlap detected!"
    assert _val_pids_set.isdisjoint(_test_pids), f"{_pool} val/test overlap detected!"

    for key, ds in _pool_ds_map.items():
        train_idx = [i for i, sid in enumerate(ds.index["subj_id"]) if sid in _train_pids]
        val_idx = [i for i, sid in enumerate(ds.index["subj_id"]) if sid in _val_pids_set]
        test_idx = [i for i, sid in enumerate(ds.index["subj_id"]) if sid in _test_pids]
        _train_sources[key] = ds[train_idx] if train_idx else ds[0:0]
        _val_sources[key] = ds[val_idx] if val_idx else ds[0:0]
        _test_sources[key] = ds[test_idx] if test_idx else ds[0:0]
    _joint_done.update(_names)

# 其余数据集 (独立池) 各自随机划分
for name in [p for p in DATASET_PARTS if p not in _joint_done]:
    ds = get_dataset(name)
    if args.small:
        ds = ds[0:20]
    if _singleton:
        dataset = ds
    else:
        key = name.lower().replace("_", "")
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

    # 最终检查：联合划分池 (如 SHHS1+SHHS2) 的 train/val/test 在被试级无重叠
    if _joint_done:
        def _de_prefix(ids):
            """从 shhs1@201206 → 201206 提取原始被试 ID"""
            return {str(s).split("@", 1)[-1] for s in ids}
        train_ids = _de_prefix(train_set.index["subj_id"])
        val_ids = _de_prefix(val_set.index["subj_id"])
        test_ids = _de_prefix(test_set.index["subj_id"])
        assert train_ids.isdisjoint(val_ids), \
            f"train/val overlap in final split! {len(train_ids & val_ids)} subjects"
        assert train_ids.isdisjoint(test_ids), \
            f"train/test overlap in final split! {len(train_ids & test_ids)} subjects"
        assert val_ids.isdisjoint(test_ids), \
            f"val/test overlap in final split! {len(val_ids & test_ids)} subjects"

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
    "stateful": args.stateful,
    "state_chunk": args.state_chunk,
    "patience": args.patience,
    "inv_freq": args.inv_freq,
    "internal_norm": args.internal_norm,
    "shuffle_mode": args.shuffle_mode,
    "wake_weight": args.wake_weight,
    "split_file": args.split_file,
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
            # ✅2026-08-13: 命令行请求的数据集若在名单中缺失, 下面的循环只会重建名单中存在的源,
            # MixedDataset 由残缺 dict 构建 → 该数据集被静默丢弃, 一致性检查也只比对重建的源,
            # 发现不了 → 显式报错
            _missing_keys = {k for k in _src_name_map if k not in _want}
            if _missing_keys:
                print(f"[ERROR] --split-file 缺失数据集名单: {sorted(_missing_keys)} "
                      f"(请求 {sorted(_src_name_map)}, 名单只有 {sorted(_want)})", flush=True)
                sys.exit(1)
            _train_sources, _val_sources, _test_sources = {}, {}, {}
            for _src_key, _parts in _want.items():
                if _src_key not in _src_name_map:
                    print(f"[WARNING] split 名单中的数据集 {_src_key} 不在本次训练数据集内, 跳过")
                    continue
                ds = get_dataset(_src_name_map[_src_key])
                if args.small:
                    # ✅2026-08-13: 与其他分支一致 — split-file 混合模式此前漏掉 --small 切片,
                    # 导致 smoke 跑在全量数据上
                    ds = ds[0:20]
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
elif all(get_dataset(_p).split_file for _p in DATASET_PARTS):
    # ✅20260825: 每个数据集自声明 split 文件 — 按请求的数据集分别加载各自的
    # 划分再组合。同 person_pool 数据集 (如 SHHS1/2) 的划分由 make_splits 联合
    # 生成 (同人两晚不跨集), 组合后这里再做一次人员级跨集断言防泄漏。
    print("[SPLIT] 按数据集自声明 split 文件组合:")
    if _singleton:
        ds = get_dataset(DATASET_PARTS[0])
        if args.small:
            ds = ds[0:20]
        _want = {k: set(v) for k, v in json.load(open(ds.split_file)).items()}
        _ids = [str(s) for s in ds.index["subj_id"]]
        train_set = ds[[i for i, sid in enumerate(_ids) if sid in _want["train"]]]
        val_set = ds[[i for i, sid in enumerate(_ids) if sid in _want["val"]]]
        test_set = ds[[i for i, sid in enumerate(_ids) if sid in _want["test"]]]
        print(f"  [{DATASET_PARTS[0].lower()}] train={len(train_set)} "
              f"val={len(val_set)} test={len(test_set)}")
    else:
        _train_sources, _val_sources, _test_sources = {}, {}, {}
        for _name in DATASET_PARTS:
            ds = get_dataset(_name)
            if args.small:
                ds = ds[0:20]
            _want = {k: set(v) for k, v in json.load(open(ds.split_file)).items()}
            _ids = [str(s) for s in ds.index["subj_id"]]
            _tr = [i for i, sid in enumerate(_ids) if sid in _want["train"]]
            _va = [i for i, sid in enumerate(_ids) if sid in _want["val"]]
            _te = [i for i, sid in enumerate(_ids) if sid in _want["test"]]
            _key = _name.lower().replace("_", "")
            _train_sources[_key] = ds[_tr] if _tr else ds[0:0]
            _val_sources[_key] = ds[_va] if _va else ds[0:0]
            _test_sources[_key] = ds[_te] if _te else ds[0:0]
            _outside = [sid for sid in _ids
                        if sid not in (_want["train"] | _want["val"] | _want["test"])]
            print(f"  [{_key}] train={len(_tr)} val={len(_va)} test={len(_te)}"
                  + (f" (名单外 {len(_outside)} 个)" if _outside else ""))
        train_set = MixedDataset(_train_sources)
        val_set = MixedDataset(_val_sources)
        test_set = MixedDataset(_test_sources)
        dataset = train_set  # 兼容后续打印

        # 同池数据集人员级跨集断言 (如 SHHS1/SHHS2 同人两晚不跨集)
        for _pool, _names in _pools.items():
            if len(_names) < 2:
                continue

            def _de_prefix(ids):
                """从 shhs1@201206 → 201206 提取原始被试 ID"""
                return {str(s).split("@", 1)[-1] for s in ids}

            _tr_ids = _de_prefix(train_set.index["subj_id"])
            _va_ids = _de_prefix(val_set.index["subj_id"])
            _te_ids = _de_prefix(test_set.index["subj_id"])
            assert _tr_ids.isdisjoint(_va_ids), \
                f"{_pool} train/val overlap in per-dataset split! {len(_tr_ids & _va_ids)} subjects"
            assert _tr_ids.isdisjoint(_te_ids), \
                f"{_pool} train/test overlap in per-dataset split! {len(_tr_ids & _te_ids)} subjects"
            assert _va_ids.isdisjoint(_te_ids), \
                f"{_pool} val/test overlap in per-dataset split! {len(_va_ids & _te_ids)} subjects"
            print(f"[SPLIT] {_pool} 人员级跨集检查 ✓")
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

# ✅2026-08-13: 空 train/val 划分是致命的 (stateful: torch.cat([]) 崩; stateless: scaler 无从拟合),
# split 解析只告警不中止 → 在数据加载前显式报错, 给出可读的错误信息
if len(train_set) == 0:
    print("[ERROR] train split is empty — 名单与数据无交集, 无法训练 (检查 --split-file)", flush=True)
    sys.exit(1)
if len(val_set) == 0:
    print("[ERROR] val split is empty — 无法验证/选择最佳模型 (检查 --split-file)", flush=True)
    sys.exit(1)
if len(test_set) == 0:
    print("[ERROR] test split is empty — 无法评估 (检查 --split-file)", flush=True)
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. 构建序列数据
# ---------------------------------------------------------------------------
print("\n[2/5] Preparing sequence data...")
data_loader = DataPreparation(seq_len=args.seq_len, overlap=None, causal=args.causal)

# ✅2026-08-13: --load-weights 时归一化参数 (scaler) 是模型的一部分 — 必须用 checkpoint 的
# 配套 scaler.json, 而不是用当前 train_set 重新拟合。否则权重来自 A run、归一化来自当前
# 数据, --eval-only / 微调结果静默失真 (数据集或划分不同时尤其严重)
_load_scaler = None
if args.load_weights:
    _companion_scaler = Path(args.load_weights).parent / "scaler.json"
    if _companion_scaler.exists():
        with open(_companion_scaler) as _f:
            _sc = json.load(_f)
        _n_sc = int(_sc["n_features"])
        if _n_sc != get_num_input(args.modality):
            print(f"[ERROR] 配套 scaler n_features ({_n_sc}) != 当前 modality 输入维数 "
                  f"({get_num_input(args.modality)}) — 权重与数据不匹配", flush=True)
            sys.exit(1)
        from sklearn.preprocessing import StandardScaler
        _load_scaler = StandardScaler()
        _load_scaler.n_features_in_ = _n_sc
        _load_scaler.mean_ = np.array(_sc["mean_"], dtype=np.float64)
        _load_scaler.scale_ = np.array(_sc["scale_"], dtype=np.float64)
        print(f"  Companion scaler loaded from: {_companion_scaler}")
    else:
        print(f"  [WARN] No companion scaler found at {_companion_scaler} — "
              f"将用当前 train_set 拟合 scaler (与 checkpoint 不配套, 结果不可靠!)")

_train_lens = []
x_train, y_train, x_val, y_val, x_test, y_test, scaler = data_loader.get_final_tensors(
    args.modality, train_set, val_set, test_set, args.classification,
    scaler=_load_scaler, train_subj_lens=_train_lens
)
print(f"  x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"  x_val:   {x_val.shape}, y_val:   {y_val.shape}")
print(f"  x_test:  {len(x_test)} subjects")

# ✅2026-08-11: stateful 模式 — 构建逐被试帧数据 (复用窗口路径拟合的训练集 scaler,
# 保证 stateless↔stateful 缩放一致与 --load-weights 兼容); 训练/测试改传 frames
if args.stateful:
    train_frames = data_loader.get_frame_data(train_set, scaler, args.modality, args.classification)
    val_frames = data_loader.get_frame_data(val_set, scaler, args.modality, args.classification)
    test_frames = data_loader.get_frame_data(test_set, scaler, args.modality, args.classification)
    _n_frames = lambda fs: sum(t[0].shape[0] for t in fs)
    print(f"  [stateful] train frames: {len(train_frames)} subj / {_n_frames(train_frames)} epochs")
    print(f"  [stateful] val frames:   {len(val_frames)} subj / {_n_frames(val_frames)} epochs")
    print(f"  [stateful] test frames:  {len(test_frames)} subj / {_n_frames(test_frames)} epochs")

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
        if args.stateful:
            # stateful: per-source frames list
            val_sources_dict[src_name] = data_loader.get_frame_data(
                src_ds, scaler=scaler, modality=args.modality,
                classification_type=args.classification
            )
        else:
            xs, ys, _ = data_loader.get_data(
                src_ds, modality=args.modality, scaler=scaler,  # 复用训练集 scaler
                classification_type=args.classification, padding=True
            )
            val_sources_dict[src_name] = (xs, ys)
    if args.stateful:
        print(f"  val sources: { {k: sum(t[0].shape[0] for t in v) for k, v in val_sources_dict.items()} }")
    else:
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
    stateful=args.stateful,
    state_chunk=args.state_chunk,
    patience=args.patience,
    inv_freq=args.inv_freq,
    internal_norm=args.internal_norm,
    shuffle_mode=args.shuffle_mode,
    seed=args.seed,
    subject_lens=_train_lens if args.shuffle_mode == "subject" else None,
    wake_weight=args.wake_weight,
)

# 加载已有权重 (如果指定)
if args.load_weights:
    print(f"  Loading weights from: {args.load_weights}")
    model._load_best_model_from_path(args.load_weights)
    # 配套 scaler 已在数据加载前处理 (见 [2/5] 的 companion scaler 逻辑)

# ---------------------------------------------------------------------------
# 4. 训练 (--eval-only 则跳过)
# ---------------------------------------------------------------------------
if not args.eval_only:
    print("\n[4/5] Training...")
    print("-" * 60)
    _train_args = (train_frames, None, val_frames, None) if args.stateful else (x_train, y_train, x_val, y_val)
    max_val_mcc = model.train(*_train_args, retrain=False)
    print(f"\n  Best validation MCC: {max_val_mcc:.4f}")
else:
    print("\n[4/5] Skipping training (--eval-only)")
    max_val_mcc = 0.0

# ---------------------------------------------------------------------------
# 5. 测试 & 保存结果
# ---------------------------------------------------------------------------
print("\n[5/5] Evaluating on test set...")
_test_args = (test_frames, None) if args.stateful else (x_test, y_test)
subject_results, score_mean, pred_dict = model.test(*_test_args, retrain=False)

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
