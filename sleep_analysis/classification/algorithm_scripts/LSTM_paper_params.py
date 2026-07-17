"""
LSTM 训练脚本 —— 使用论文 Krauss et al. (2025) 中的最优参数
============================================================
论文: Incorporating Respiratory Signals for ML-based Multi-Modal Sleep Stage Classification
参数来源: 5-Class Classification - MESA Baseline

每次训练自动在 exports_our/<时间戳>/ 下保存 config.json，里面记录了所有参数。
不需要改脚本，通过命令行参数切换配置。

用法:
    # 论文默认参数 (5分类, ACT+HRV+RRV, 170 epoch)
    PATH="..." PYTHON_KEYRING_BACKEND=... python LSTM_paper_params.py

    # 快速测试 (20人, 3 epoch)
    python LSTM_paper_params.py --small --quick

    # 3分类 + 不同超参数
    python LSTM_paper_params.py -c 3stage --hidden 256 --layers 4 --lr 1e-4

    # 只用 HRV 特征
    python LSTM_paper_params.py --modality HRV

    # 查看所有参数
    python LSTM_paper_params.py --help
"""
import argparse
import json
import pickle
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sleep_analysis.classification.deep_learning.lstm.data_peparation import DataPreparation
from sleep_analysis.classification.deep_learning.lstm.LSTM import LSTM
from sleep_analysis.classification.deep_learning.utils import get_num_input
from sleep_analysis.datasets.helper import get_random_split
from sleep_analysis.datasets.mesadataset import MesaDataset

# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="LSTM Sleep Stage Classification")
# 数据集
parser.add_argument("--small", action="store_true", help="只用 20 个被试验证管线")
# 分类
parser.add_argument("-c", "--classification", default="5stage", choices=["binary", "3stage", "4stage", "5stage"])
parser.add_argument("-m", "--modality", nargs="+", default=["ACT", "HRV", "RRV"],
                    choices=["ACT", "HRV", "RRV", "EDR"])
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
        args.seed = saved_config.get("seed", args.seed)
    else:
        print(f"[WARNING] {config_file} not found, using current CLI params."
              f" 确保超参数与训练时一致，否则 load_state_dict 会报错!")

# 快速测试覆盖
if args.quick:
    args.epochs = 3
if args.small:
    print("[SMALL mode] Using only 20 subjects")

# ---------------------------------------------------------------------------
# 输出目录 & 配置保存
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parents[3]
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H%M%S")
OUTPUT_DIR = PROJECT_ROOT / "exports_our" / RUN_TIMESTAMP
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 将所有配置保存为 JSON，以后随时查阅
config = {
    "timestamp": RUN_TIMESTAMP,
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
    "seed": args.seed,
    "load_weights": args.load_weights,
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

print("\n[1/5] Loading dataset...")
dataset = MesaDataset() if not args.small else MesaDataset()[0:20]

train_set, test_set = get_random_split(dataset=dataset)
train_set, val_set = get_random_split(dataset=train_set)
print(f"  Subjects: {len(dataset)} total → train {len(train_set)}, val {len(val_set)}, test {len(test_set)}")

# ---------------------------------------------------------------------------
# 2. 构建序列数据
# ---------------------------------------------------------------------------
print("\n[2/5] Preparing sequence data...")
data_loader = DataPreparation(seq_len=args.seq_len, overlap=None)
x_train, y_train, x_val, y_val, x_test, y_test = data_loader.get_final_tensors(
    args.modality, train_set, val_set, test_set, args.classification
)
print(f"  x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"  x_val:   {x_val.shape}, y_val:   {y_val.shape}")
print(f"  x_test:  {len(x_test)} subjects")

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
    dataset_name="MESA_Sleep",
    classification_type=args.classification,
    output_dir=str(OUTPUT_DIR),
    focal_gamma=args.focal_gamma,
    grad_clip=args.grad_clip,
)

# 加载已有权重 (如果指定)
if args.load_weights:
    print(f"  Loading weights from: {args.load_weights}")
    model._load_best_model_from_path(args.load_weights)

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

# 汇总
print(f"\n{'=' * 60}")
print("Test Set Results (mean across subjects):")
print(score_mean.to_string())
print(f"{'=' * 60}")
print(f"All outputs saved to: {OUTPUT_DIR}")
