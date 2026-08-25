#!/usr/bin/env bash
# =============================================================================
# 不同特征窗长 (seq_len) 三模型实验 + log 域加权投票
#
#  模型 A: --seq-len 11  (窗口 ±5 min,  短时高分辨率)
#  模型 B: --seq-len 21  (窗口 ±10 min, 与参考基线同窗)
#  模型 C: --seq-len 31  (窗口 ±15 min, 长时趋势)
#
# 公共配置 (对齐 2026-08-17_193709 参考基线):
#   MESA 4stage ACT+HRV+RRV, 非 causal, --internal-norm,
#   固定划分 splits/split_mesa_1121_20260806.json, k=1 单点 (--chunk-len 1)
#
# 每个模型独立训练日志 (train_seq*_YYYYMMDD_HHMMSS.log)。
# 训练流程 [5/5] 自动保存每被试 logits (per_subject_predictions/*_logits.csv),
# 三个 run 训练完后 ensemble_checkpoints.py 做 log 域加权投票。
#
# 用法 (挂后台):
#   screen -S seq_ensemble
#   bash run_seq_len_ensemble.sh
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"          # 固定 cwd 到 submodule 根
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring

SPLIT="splits/split_mesa_1121_20260806.json"
PY="sleep_analysis/classification/algorithm_scripts/LSTM_paper_params.py"
ENS="sleep_analysis/classification/algorithm_scripts/ensemble_checkpoints.py"

LOG_SEQ11="exports_our/train_seq11_4stage_$(date +%Y%m%d_%H%M%S).log"
LOG_SEQ21="exports_our/train_seq21_4stage_$(date +%Y%m%d_%H%M%S).log"
LOG_SEQ31="exports_our/train_seq31_4stage_$(date +%Y%m%d_%H%M%S).log"

echo "############################################################"
echo "# 实验: 不同特征窗长三模型集成 (seq_len 11/21/31, k=1)"
echo "# 时间: $(date '+%F %T')"
echo "############################################################"

# =============================================================================
# 1/3 模型 A — seq_len 11 (±5 min)
# =============================================================================
echo "=== [$(date '+%F %T')] seq11_4stage — 日志: ${LOG_SEQ11} ===" | tee "$LOG_SEQ11"
python "$PY" \
    --dataset MESA_Sleep \
    --classification 4stage \
    --modality ACT HRV RRV \
    --split-file "$SPLIT" \
    --epochs 170 \
    --seq-len 11 \
    --hidden 556 \
    --layers 6 \
    --dropout 0.255 \
    --lr 6.31e-05 \
    --batch-size 512 \
    --focal-gamma 2.0 \
    --grad-clip 0.5 \
    --patience 5 \
    --internal-norm \
    --chunk-len 1 \
    2>&1 | tee -a "$LOG_SEQ11"
RUN_SEQ11=$(grep -m1 "All outputs saved to:" "$LOG_SEQ11" | awk '{print $NF}')
if [ -z "$RUN_SEQ11" ] || [ ! -d "$RUN_SEQ11/per_subject_predictions" ]; then
    echo "[ERROR] seq11: run 目录或 logits 未生成, 训练可能失败" >&2
    exit 1
fi
echo "  → run 目录: $RUN_SEQ11" | tee -a "$LOG_SEQ11"

# =============================================================================
# 2/3 模型 B — seq_len 21 (±10 min, 与参考基线同窗)
# =============================================================================
echo "=== [$(date '+%F %T')] seq21_4stage — 日志: ${LOG_SEQ21} ===" | tee "$LOG_SEQ21"
python "$PY" \
    --dataset MESA_Sleep \
    --classification 4stage \
    --modality ACT HRV RRV \
    --split-file "$SPLIT" \
    --epochs 170 \
    --seq-len 21 \
    --hidden 556 \
    --layers 6 \
    --dropout 0.255 \
    --lr 6.31e-05 \
    --batch-size 512 \
    --focal-gamma 2.0 \
    --grad-clip 0.5 \
    --patience 5 \
    --internal-norm \
    --chunk-len 1 \
    2>&1 | tee -a "$LOG_SEQ21"
RUN_SEQ21=$(grep -m1 "All outputs saved to:" "$LOG_SEQ21" | awk '{print $NF}')
if [ -z "$RUN_SEQ21" ] || [ ! -d "$RUN_SEQ21/per_subject_predictions" ]; then
    echo "[ERROR] seq21: run 目录或 logits 未生成, 训练可能失败" >&2
    exit 1
fi
echo "  → run 目录: $RUN_SEQ21" | tee -a "$LOG_SEQ21"

# =============================================================================
# 3/3 模型 C — seq_len 31 (±15 min)
# =============================================================================
echo "=== [$(date '+%F %T')] seq31_4stage — 日志: ${LOG_SEQ31} ===" | tee "$LOG_SEQ31"
python "$PY" \
    --dataset MESA_Sleep \
    --classification 4stage \
    --modality ACT HRV RRV \
    --split-file "$SPLIT" \
    --epochs 170 \
    --seq-len 31 \
    --hidden 556 \
    --layers 6 \
    --dropout 0.255 \
    --lr 6.31e-05 \
    --batch-size 512 \
    --focal-gamma 2.0 \
    --grad-clip 0.5 \
    --patience 5 \
    --internal-norm \
    --chunk-len 1 \
    2>&1 | tee -a "$LOG_SEQ31"
RUN_SEQ31=$(grep -m1 "All outputs saved to:" "$LOG_SEQ31" | awk '{print $NF}')
if [ -z "$RUN_SEQ31" ] || [ ! -d "$RUN_SEQ31/per_subject_predictions" ]; then
    echo "[ERROR] seq31: run 目录或 logits 未生成, 训练可能失败" >&2
    exit 1
fi
echo "  → run 目录: $RUN_SEQ31" | tee -a "$LOG_SEQ31"

# =============================================================================
# 投票 — log 域加权 (默认均匀权重; 可按 val MCC 改 --weights)
# =============================================================================
ENS_LOG="exports_our/ensemble_seq_$(date +%Y%m%d_%H%M%S).log"
echo "=== [$(date '+%F %T')] Ensemble (log 域加权投票, 均匀权重) ===" | tee -a "$ENS_LOG"
python "$ENS" \
    --runs "$RUN_SEQ11" "$RUN_SEQ21" "$RUN_SEQ31" \
    --classification 4stage 2>&1 | tee -a "$ENS_LOG"

echo "============================================================"
echo "完成。投票结果: $ENS_LOG"
echo "============================================================"
