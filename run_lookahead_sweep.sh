#!/usr/bin/env bash
# =============================================================================
# 特征平移量 (--lookahead-min) 扫描实验: 5/4/3/2/1/0/-1 分钟
# 无状态 LSTM + 原版 loss (1-freq, 不传 --inv-freq) + 用 --internal-norm (旧版内部归一化, 与 08-06 基线代码路径一致)
# 固定划分 splits/split_mesa_1121_20260806.json, 每个值单独一个日志。
#
# 数据: causal (因果处理)。平移量 k=2*分钟 是"窗口相对预测点"的偏移 —
#   k=5min=10epoch 即原版居中; k=0 即实时。数据不变, 只扫窗口对齐方式。
#
# 用法 (在 sleep_analysis_torch2x 环境下):
#   screen -S lookahead_sweep
#   bash run_lookahead_sweep.sh
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring

SPLIT="splits/split_mesa_1121_20260806.json"

# ---- 数据: causal (因果处理) ----
DATA_SRC="/srv/shared/psgdata/processed_data_causal_20260806"

BACKUP=$(mktemp /tmp/study_data_backup.XXXXXX.json)
cp study_data.json "$BACKUP"
trap 'cp "$BACKUP" study_data.json; rm -f "$BACKUP"' EXIT

python - "$DATA_SRC" <<'PYEOF'
import json, sys
src = sys.argv[1]
cfg = json.load(open("study_data.json"))
n = 0
for k in list(cfg):
    if k.startswith("processed_mesa"):
        cfg[k] = src.rstrip("/") + "/mesa_processed"
        n += 1
json.dump(cfg, open("study_data.json", "w"), indent=2, ensure_ascii=False)
print(f"[DATA] study_data.json: {n} 个 processed_mesa 键 → {src}/mesa_processed")
PYEOF

LOG_BASE="exports_our"

# =============================================================================
# lookahead = 5 min (k=10, 原版居中窗口)
# =============================================================================
LOG_5MIN="${LOG_BASE}/train_lookahead_5min_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "=============================================================================="
echo " [lookahead 5min]  start $(date '+%F %T')  →  log: ${LOG_5MIN}"
echo "=============================================================================="
python -u sleep_analysis/classification/algorithm_scripts/LSTM_paper_params.py \
    -d MESA_Sleep -c 4stage --modality ACT HRV RRV \
    --seq-len 21 --hidden 556 --layers 6 --dropout 0.255 --lr 6.31e-5 --batch-size 512 \
    --split-file "$SPLIT" \
    --lookahead-min 5 --patience 5 --epochs 170 --internal-norm \
    2>&1 | tee "${LOG_5MIN}"
echo " [lookahead 5min]  done  $(date '+%F %T')"

# =============================================================================
# lookahead = 4 min (k=8)
# =============================================================================
LOG_4MIN="${LOG_BASE}/train_lookahead_4min_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "=============================================================================="
echo " [lookahead 4min]  start $(date '+%F %T')  →  log: ${LOG_4MIN}"
echo "=============================================================================="
python -u sleep_analysis/classification/algorithm_scripts/LSTM_paper_params.py \
    -d MESA_Sleep -c 4stage --modality ACT HRV RRV \
    --seq-len 21 --hidden 556 --layers 6 --dropout 0.255 --lr 6.31e-5 --batch-size 512 \
    --split-file "$SPLIT" \
    --lookahead-min 4 --patience 5 --epochs 170 --internal-norm \
    2>&1 | tee "${LOG_4MIN}"
echo " [lookahead 4min]  done  $(date '+%F %T')"

# =============================================================================
# lookahead = 3 min (k=6)
# =============================================================================
LOG_3MIN="${LOG_BASE}/train_lookahead_3min_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "=============================================================================="
echo " [lookahead 3min]  start $(date '+%F %T')  →  log: ${LOG_3MIN}"
echo "=============================================================================="
python -u sleep_analysis/classification/algorithm_scripts/LSTM_paper_params.py \
    -d MESA_Sleep -c 4stage --modality ACT HRV RRV \
    --seq-len 21 --hidden 556 --layers 6 --dropout 0.255 --lr 6.31e-5 --batch-size 512 \
    --split-file "$SPLIT" \
    --lookahead-min 3 --patience 5 --epochs 170 --internal-norm \
    2>&1 | tee "${LOG_3MIN}"
echo " [lookahead 3min]  done  $(date '+%F %T')"

# =============================================================================
# lookahead = 2 min (k=4)
# =============================================================================
LOG_2MIN="${LOG_BASE}/train_lookahead_2min_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "=============================================================================="
echo " [lookahead 2min]  start $(date '+%F %T')  →  log: ${LOG_2MIN}"
echo "=============================================================================="
python -u sleep_analysis/classification/algorithm_scripts/LSTM_paper_params.py \
    -d MESA_Sleep -c 4stage --modality ACT HRV RRV \
    --seq-len 21 --hidden 556 --layers 6 --dropout 0.255 --lr 6.31e-5 --batch-size 512 \
    --split-file "$SPLIT" \
    --lookahead-min 2 --patience 5 --epochs 170 --internal-norm \
    2>&1 | tee "${LOG_2MIN}"
echo " [lookahead 2min]  done  $(date '+%F %T')"

# =============================================================================
# lookahead = 1 min (k=2)
# =============================================================================
LOG_1MIN="${LOG_BASE}/train_lookahead_1min_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "=============================================================================="
echo " [lookahead 1min]  start $(date '+%F %T')  →  log: ${LOG_1MIN}"
echo "=============================================================================="
python -u sleep_analysis/classification/algorithm_scripts/LSTM_paper_params.py \
    -d MESA_Sleep -c 4stage --modality ACT HRV RRV \
    --seq-len 21 --hidden 556 --layers 6 --dropout 0.255 --lr 6.31e-5 --batch-size 512 \
    --split-file "$SPLIT" \
    --lookahead-min 1 --patience 5 --epochs 170 --internal-norm \
    2>&1 | tee "${LOG_1MIN}"
echo " [lookahead 1min]  done  $(date '+%F %T')"

# =============================================================================
# lookahead = 0 min (k=0, 实时)
# =============================================================================
LOG_0MIN="${LOG_BASE}/train_lookahead_0min_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "=============================================================================="
echo " [lookahead 0min]  start $(date '+%F %T')  →  log: ${LOG_0MIN}"
echo "=============================================================================="
python -u sleep_analysis/classification/algorithm_scripts/LSTM_paper_params.py \
    -d MESA_Sleep -c 4stage --modality ACT HRV RRV \
    --seq-len 21 --hidden 556 --layers 6 --dropout 0.255 --lr 6.31e-5 --batch-size 512 \
    --split-file "$SPLIT" \
    --lookahead-min 0 --patience 5 --epochs 170 --internal-norm \
    2>&1 | tee "${LOG_0MIN}"
echo " [lookahead 0min]  done  $(date '+%F %T')"

# =============================================================================
# lookahead = -1 min (k=-2, 预测点晚于数据)
# =============================================================================
LOG_NEG1MIN="${LOG_BASE}/train_lookahead_-1min_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "=============================================================================="
echo " [lookahead -1min]  start $(date '+%F %T')  →  log: ${LOG_NEG1MIN}"
echo "=============================================================================="
python -u sleep_analysis/classification/algorithm_scripts/LSTM_paper_params.py \
    -d MESA_Sleep -c 4stage --modality ACT HRV RRV \
    --seq-len 21 --hidden 556 --layers 6 --dropout 0.255 --lr 6.31e-5 --batch-size 512 \
    --split-file "$SPLIT" \
    --lookahead-min -1 --patience 5 --epochs 170 --internal-norm \
    2>&1 | tee "${LOG_NEG1MIN}"
echo " [lookahead -1min]  done  $(date '+%F %T')"

echo ""
echo "=============================================================================="
echo " 7 个平移量实验全部完成。study_data.json 已恢复。"
echo "=============================================================================="
