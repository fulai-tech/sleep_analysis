"""
LSTM 超参数搜索 — 使用 Optuna TPE Sampler (与论文一致)
========================================================
在 MESA 数据上搜索最优超参数，支持任意分类类型。

搜索空间（与论文 Table I 一致）:
    seq_len:        {21, 51, 101}  (= 10, 25, 50 min)
    hidden_size:    4–700 (step 4)
    num_layers:     1–10
    dropout:        0.0–0.5
    learning_rate:  1e-6 – 1e-4

用法:
    # 4 分类搜索 (30 trials, 用小数据集加速)
    python LSTM_optuna_search.py -c 4stage --trials 30 --small --quick

    # 正式搜索 (250 trials, 全量数据)
    python LSTM_optuna_search.py -c 4stage --trials 250

    # 3 分类搜索
    python LSTM_optuna_search.py -c 3stage --trials 250
"""
import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.samplers import TPESampler

from sleep_analysis.classification.deep_learning.lstm.data_peparation import DataPreparation
from sleep_analysis.classification.deep_learning.lstm.LSTM import LSTM
from sleep_analysis.classification.deep_learning.utils import get_num_input
from sleep_analysis.datasets.helper import get_random_split
from sleep_analysis.datasets.mesadataset import MesaDataset

# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="LSTM Hyperparameter Search via Optuna TPE")
parser.add_argument("-c", "--classification", default="4stage", choices=["3stage", "4stage", "5stage", "binary"])
parser.add_argument("-m", "--modality", nargs="+", default=["ACT", "HRV", "RRV"])
parser.add_argument("--trials", type=int, default=30, help="Optuna 搜索 trial 数（论文用 250）")
parser.add_argument("--small", action="store_true", help="只用 100 人加速搜索")
parser.add_argument("--quick", action="store_true", help="每个 trial 只训 10 epoch")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--study-name", type=str, default=None,
                    help="Optuna study 名称 (默认自动生成)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# 输出目录
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parents[3]
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H%M%S")
OUTPUT_DIR = PROJECT_ROOT / "exports_our" / f"optuna_{args.classification}_{RUN_TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 固定种子
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("=" * 60)
print(f"Optuna Search: {args.classification} | Modality: {args.modality}")
print(f"Trials: {args.trials} | Small: {args.small} | Quick: {args.quick}")
print(f"Output: {OUTPUT_DIR}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 加载数据（只加载一次，所有 trial 共用）
# ---------------------------------------------------------------------------
print("\n[1/3] Loading dataset ...")
if args.small:
    dataset = MesaDataset()[:100]
else:
    dataset = MesaDataset()
train_set, test_set = get_random_split(dataset=dataset)
train_set, val_set = get_random_split(dataset=train_set)
print(f"  Subjects: {len(dataset)} → train {len(train_set)}, val {len(val_set)}, test {len(test_set)}")

# ---------------------------------------------------------------------------
# 定义 Optuna objective
# ---------------------------------------------------------------------------
def objective(trial):
    """每个 trial 被 Optuna TPE sampler 调用一次"""

    # 采样超参数（搜索空间与论文一致）
    seq_len = trial.suggest_categorical("seq_len", [21, 51, 101])
    hidden_size = trial.suggest_int("hidden_size", 4, 700, step=4)
    num_layers = trial.suggest_int("num_layers", 1, 10)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

    epochs = 10 if args.quick else 50  # 快速搜索时用较少 epoch
    num_inputs = get_num_input(args.modality)

    trial_dir = OUTPUT_DIR / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # 构建数据（seq_len 不同需要重建）
    data_loader = DataPreparation(seq_len=seq_len, overlap=None)
    x_train, y_train, x_val, y_val, _, _, _ = data_loader.get_final_tensors(
        args.modality, train_set, val_set, test_set[:4], args.classification
    )

    # 创建模型
    model = LSTM(
        num_epochs=epochs,
        input_size=num_inputs,
        hidden_size=hidden_size,
        num_layers=num_layers,
        learning_rate=learning_rate,
        seq_len=seq_len,
        dropout=dropout,
        batch_size=batch_size,
        modality=args.modality,
        dataset_name="MESA_Sleep",
        classification_type=args.classification,
        output_dir=str(trial_dir),
    )

    # 训练
    max_val_mcc = model.train(x_train, y_train, x_val, y_val, retrain=False)

    # 记录到 Optuna
    trial.set_user_attr("hidden_size", hidden_size)
    trial.set_user_attr("num_layers", num_layers)
    trial.set_user_attr("dropout", dropout)
    trial.set_user_attr("learning_rate", learning_rate)
    trial.set_user_attr("seq_len", seq_len)
    trial.set_user_attr("batch_size", batch_size)

    return max_val_mcc


# ---------------------------------------------------------------------------
# 运行 Optuna 搜索
# ---------------------------------------------------------------------------
print(f"\n[2/3] Starting Optuna search ({args.trials} trials) ...")
print(f"  Search space:")
print(f"    seq_len:        {{21, 51, 101}}")
print(f"    hidden_size:    4–700 (step 4)")
print(f"    num_layers:     1–10")
print(f"    dropout:        0.0–0.5")
print(f"    learning_rate:  1e-6 – 1e-4 (log-uniform)")
print(f"    batch_size:     {{128, 256, 512}}")
print("-" * 60)

study_name = args.study_name or f"lstm_{args.classification}_{'_'.join(args.modality)}"
study = optuna.create_study(
    study_name=study_name,
    direction="maximize",
    sampler=TPESampler(seed=args.seed),
    storage=f"sqlite:///{OUTPUT_DIR}/optuna_study.db",
    load_if_exists=False,
)

study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

# ---------------------------------------------------------------------------
# 结果汇总
# ---------------------------------------------------------------------------
print(f"\n[3/3] Search complete!")
print("=" * 60)
print(f"Best trial: #{study.best_trial.number}")
print(f"Best validation MCC: {study.best_value:.4f}")
print(f"\nBest hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
print(f"\nAll trial details saved to: {OUTPUT_DIR}/optuna_study.db")

# 保存最优参数为 JSON
best_config = {
    "classification": args.classification,
    "modality": args.modality,
    "best_mcc": float(study.best_value),
    "best_trial": study.best_trial.number,
    "params": study.best_params,
    "search_space": {
        "seq_len": [21, 51, 101],
        "hidden_size": "4–700 step 4",
        "num_layers": "1–10",
        "dropout": "0.0–0.5",
        "learning_rate": "1e-6–1e-4 log-uniform",
        "batch_size": [128, 256, 512],
    },
    "n_trials": args.trials,
    "timestamp": RUN_TIMESTAMP,
}
with open(OUTPUT_DIR / "best_params.json", "w") as f:
    json.dump(best_config, f, indent=2)

# 输出可以直接 --load-weights 的训练命令
print(f"\n{'=' * 60}")
print("用最优参数训练的命令:")
print(f"  python LSTM_paper_params.py -c {args.classification} \\")
print(f"    --seq-len {study.best_params['seq_len']} \\")
print(f"    --hidden {study.best_params['hidden_size']} \\")
print(f"    --layers {study.best_params['num_layers']} \\")
print(f"    --dropout {study.best_params['dropout']} \\")
print(f"    --lr {study.best_params['learning_rate']:.4e} \\")
print(f"    --batch-size {study.best_params['batch_size']}")
print(f"{'=' * 60}")
