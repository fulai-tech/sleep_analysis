#!/usr/bin/env bash
# MrOS 全量预处理: causal × (visit1, visit2) + 非causal(作者原版) × (visit1, visit2)
# ============================================================================
# 依次执行 4 个独立运行, 各自独立的输出目录 + 日志文件, 每个 8 worker。
# 输出目录约定与现有 MESA/SHHS 一致:
#   causal     → processed_data_causal_20260806/mros{1,2}_processed
#   非 causal  → processed_data_with_leak_20260804/mros{1,2}_processed
# 日志文件名带运行时间戳: mros_{mode}_visit{N}_{YYYYMMDD_HHMMSS}.log
#
# 用法:
#   bash experiments/data_handling/run_preprocess_mros_all.sh                     # 全跑
#   bash experiments/data_handling/run_preprocess_mros_all.sh --visits 2          # 只跑 visit2
#   bash experiments/data_handling/run_preprocess_mros_all.sh --visits "1 2"      # 等价全跑
#   bash experiments/data_handling/run_preprocess_mros_all.sh --workers 16
#   bash experiments/data_handling/run_preprocess_mros_all.sh --n-subjects 50     # 试跑
#
# 断点续跑: 每个输出目录有独立 checkpoint.json, 重跑自动跳过已完成被试;
# 想重做某个运行, 删除该目录的 checkpoint.json 和产出文件即可。
# 注意: causal 与非 causal 的 run_config.json 模式锁互斥, 输出目录不能互换。

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # third_party/sleep_analysis
PYTHON="/home/rdwang/anaconda3/envs/sleep_analysis_torch2x/bin/python"
LOG_DIR="/srv/shared/psgdata/logs"
WORKERS=8
N_SUBJECTS=99999
VISITS="1 2"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)  WORKERS="$2"; shift 2 ;;
        --n-subjects) N_SUBJECTS="$2"; shift 2 ;;
        --visits)   VISITS="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

TS=$(date '+%Y%m%d_%H%M%S')   # 本次运行的时间戳, 用于日志文件名

CAUSAL_V1_LOG="$LOG_DIR/mros_causal_visit1_${TS}.log"
CAUSAL_V2_LOG="$LOG_DIR/mros_causal_visit2_${TS}.log"
NONCAUSAL_V1_LOG="$LOG_DIR/mros_noncausal_visit1_${TS}.log"
NONCAUSAL_V2_LOG="$LOG_DIR/mros_noncausal_visit2_${TS}.log"

CAUSAL_V1_OUT="/srv/shared/psgdata/processed_data_causal_20260806/mros1_processed"
CAUSAL_V2_OUT="/srv/shared/psgdata/processed_data_causal_20260806/mros2_processed"
NONCAUSAL_V1_OUT="/srv/shared/psgdata/processed_data_with_leak_20260804/mros1_processed"
NONCAUSAL_V2_OUT="/srv/shared/psgdata/processed_data_with_leak_20260804/mros2_processed"

RC_C1=0; RC_C2=0; RC_N1=0; RC_N2=0   # 被跳过的运行保持 0

# ============================================================================
# 1/4  causal visit1
# ============================================================================
if [[ " $VISITS " == *" 1 "* ]]; then
echo "" | tee -a "$CAUSAL_V1_LOG"
echo "=== [$(date '+%F %T')] START causal visit1 → $CAUSAL_V1_OUT ===" | tee -a "$CAUSAL_V1_LOG"
cd "$PROJECT_DIR"
SLEEP_CAUSAL=1 "$PYTHON" -u experiments/data_handling/preprocess_mros.py \
    --visit 1 --no-edr --n-workers "$WORKERS" --n-subjects "$N_SUBJECTS" \
    --output-dir "$CAUSAL_V1_OUT" 2>&1 | tee -a "$CAUSAL_V1_LOG"
RC_C1=${PIPESTATUS[0]}
echo "=== [$(date '+%F %T')] END   causal visit1 (exit=${RC_C1}) ===" | tee -a "$CAUSAL_V1_LOG"
fi

# ============================================================================
# 2/4  causal visit2
# ============================================================================
if [[ " $VISITS " == *" 2 "* ]]; then
echo "" | tee -a "$CAUSAL_V2_LOG"
echo "=== [$(date '+%F %T')] START causal visit2 → $CAUSAL_V2_OUT ===" | tee -a "$CAUSAL_V2_LOG"
cd "$PROJECT_DIR"
SLEEP_CAUSAL=1 "$PYTHON" -u experiments/data_handling/preprocess_mros.py \
    --visit 2 --no-edr --n-workers "$WORKERS" --n-subjects "$N_SUBJECTS" \
    --output-dir "$CAUSAL_V2_OUT" 2>&1 | tee -a "$CAUSAL_V2_LOG"
RC_C2=${PIPESTATUS[0]}
echo "=== [$(date '+%F %T')] END   causal visit2 (exit=${RC_C2}) ===" | tee -a "$CAUSAL_V2_LOG"
fi

# ============================================================================
# 3/4  非causal visit1 (不设 SLEEP_CAUSAL = 作者原版逻辑)
# ============================================================================
if [[ " $VISITS " == *" 1 "* ]]; then
echo "" | tee -a "$NONCAUSAL_V1_LOG"
echo "=== [$(date '+%F %T')] START noncausal visit1 → $NONCAUSAL_V1_OUT ===" | tee -a "$NONCAUSAL_V1_LOG"
cd "$PROJECT_DIR"
"$PYTHON" -u experiments/data_handling/preprocess_mros.py \
    --visit 1 --no-edr --n-workers "$WORKERS" --n-subjects "$N_SUBJECTS" \
    --output-dir "$NONCAUSAL_V1_OUT" 2>&1 | tee -a "$NONCAUSAL_V1_LOG"
RC_N1=${PIPESTATUS[0]}
echo "=== [$(date '+%F %T')] END   noncausal visit1 (exit=${RC_N1}) ===" | tee -a "$NONCAUSAL_V1_LOG"
fi

# ============================================================================
# 4/4  非causal visit2
# ============================================================================
if [[ " $VISITS " == *" 2 "* ]]; then
echo "" | tee -a "$NONCAUSAL_V2_LOG"
echo "=== [$(date '+%F %T')] START noncausal visit2 → $NONCAUSAL_V2_OUT ===" | tee -a "$NONCAUSAL_V2_LOG"
cd "$PROJECT_DIR"
"$PYTHON" -u experiments/data_handling/preprocess_mros.py \
    --visit 2 --no-edr --n-workers "$WORKERS" --n-subjects "$N_SUBJECTS" \
    --output-dir "$NONCAUSAL_V2_OUT" 2>&1 | tee -a "$NONCAUSAL_V2_LOG"
RC_N2=${PIPESTATUS[0]}
echo "=== [$(date '+%F %T')] END   noncausal visit2 (exit=${RC_N2}) ===" | tee -a "$NONCAUSAL_V2_LOG"
fi

# ============================================================================
# 汇总
# ============================================================================
echo ""
echo "=========================================="
echo "MrOS 预处理汇总 (worker=${WORKERS}, n-subjects=${N_SUBJECTS}, visits=${VISITS})"
echo "  causal    visit1: exit=${RC_C1}  日志: $CAUSAL_V1_LOG"
echo "  causal    visit2: exit=${RC_C2}  日志: $CAUSAL_V2_LOG"
echo "  noncausal visit1: exit=${RC_N1}  日志: $NONCAUSAL_V1_LOG"
echo "  noncausal visit2: exit=${RC_N2}  日志: $NONCAUSAL_V2_LOG"
echo "=========================================="

[ $RC_C1 -eq 0 ] && [ $RC_C2 -eq 0 ] && [ $RC_N1 -eq 0 ] && [ $RC_N2 -eq 0 ]
