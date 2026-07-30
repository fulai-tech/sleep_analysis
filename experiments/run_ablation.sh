#!/bin/bash
# ============================================================================
# 4 分类消融实验：实时分期 × 特征选择
# ============================================================================
# 基线: --causal=False, modality=ACT+HRV+RRV (作者原文配置)
#
# 实验 1: 实时分期 — 去掉未来信息，预测窗口最后时刻，特征不变
# 实验 2: 无体动特征 — 只用 HRV+RRV，预测位置不变 (作者原文方式)
# 实验 3: 实时分期 + 无体动 — 去掉未来信息 + 只用 HRV+RRV
# ============================================================================

set -e

cd "$(dirname "$0")/.."   # → third_party/sleep_analysis/

# ---- 通用参数 (来自 4 分类基线) ----
BASE_ARGS=(
  -c 4stage
  --seq-len 21 --hidden 556 --layers 6
  --dropout 0.255 --lr 6.31e-5 --batch-size 512
  --focal-gamma 2.0 --grad-clip 0.5
)
PYTHON_CMD="python -u sleep_analysis/classification/algorithm_scripts/LSTM_paper_params.py"

# ---- Experiment 1: 实时分期 (causal) ----
echo "============================================================"
echo "  Experiment 1/3: Causal (real-time) — ACT+HRV+RRV"
echo "============================================================"
$PYTHON_CMD \
  "${BASE_ARGS[@]}" \
  --modality ACT HRV RRV \
  --causal \
  2>&1 | tee "exports_our/ablation_1_causal_$(date +%Y-%m-%d_%H%M%S).log"

# ---- Experiment 2: 无体动，原文预测位置 ----
echo "============================================================"
echo "  Experiment 2/3: No ACT — HRV+RRV only"
echo "============================================================"
$PYTHON_CMD \
  "${BASE_ARGS[@]}" \
  --modality HRV RRV \
  2>&1 | tee "exports_our/ablation_2_no_act_$(date +%Y-%m-%d_%H%M%S).log"

# ---- Experiment 3: 实时分期 + 无体动 ----
echo "============================================================"
echo "  Experiment 3/3: Causal + No ACT — HRV+RRV only"
echo "============================================================"
$PYTHON_CMD \
  "${BASE_ARGS[@]}" \
  --modality HRV RRV \
  --causal \
  2>&1 | tee "exports_our/ablation_3_causal_no_act_$(date +%Y-%m-%d_%H%M%S).log"

echo ""
echo "=============================="
echo "  All 3 experiments finished."
echo "=============================="
