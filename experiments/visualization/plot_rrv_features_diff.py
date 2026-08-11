"""查看 RRV 特征在预处理改造前后的差异（聚焦训练用的 4 个 RRV 特征）。

对比对象:
  - 原版   (无 SLEEP_CAUSAL): 呼吸信号 filtfilt 双向滤波 + decimate(filtfilt)
  - causal (SLEEP_CAUSAL=1):  lfilter 正向滤波 + 因果降采样 (RRV 保留因果)

训练用的 4 个 RRV 特征:
  - 150_RRV_MedianBB / 150_RRV_LF / 270_RRV_MCVBB / 150_RRV_CVBB
  (与 data_peparation.py::_extract_subj_features 和 inference/data_utils.py::_RRV_COLUMNS 一致)

输出: 4 特征 x 3 视图 (时序对比 / 散点对比 / 相对差异直方图)

用法:
    python experiments/visualization/plot_rrv_features_diff.py \
        --orig  <原版 respiration_features_clean CSV> \
        --causal <因果版 CSV> \
        --output <PNG 路径> \
        [--epochs 300]   # 时序图只画前 N 个 epoch, 默认全部
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 训练用的 4 个 RRV 特征（顺序与训练代码一致）
TRAIN_RRV_FEATURES = ["150_RRV_MedianBB", "150_RRV_LF", "270_RRV_MCVBB", "150_RRV_CVBB"]

# 系列配色（dataviz 参考调色板 slot 1/2）
C_ORIG = "#2a78d6"
C_CAUSAL = "#eb6834"
C_DIAG = "#52514e"

DEFAULT_ORIG = (
    "/srv/shared/psgdata/processed_data_with_leak_20260804/mesa_processed/"
    "respiration_features_clean/respiration_features0002.csv"
)
DEFAULT_CAUSAL = (
    "/srv/shared/psgdata/processed_test/manual_mesa/"
    "respiration_features_clean/respiration_features0002.csv"
)
DEFAULT_OUTPUT = "/srv/shared/psgdata/processed_test/rrv_features_diff_0002.png"


def relative_diff(causal: np.ndarray, orig: np.ndarray) -> np.ndarray:
    """逐元素相对差异 |c - o| / |o|, 分母≈0 的元素置 NaN。"""
    denom = np.abs(orig)
    diff = np.abs(causal - orig)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(denom > 1e-9, diff / np.where(denom > 1e-9, denom, np.nan), np.nan)
    return rel


def main():
    parser = argparse.ArgumentParser(description="RRV 特征处理前后差异可视化")
    parser.add_argument("--orig", default=DEFAULT_ORIG, help="原版 respiration_features_clean CSV")
    parser.add_argument("--causal", default=DEFAULT_CAUSAL, help="因果版 respiration_features_clean CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出 PNG 路径")
    parser.add_argument("--epochs", type=int, default=None, help="时序图只画前 N 个 epoch (默认全部)")
    args = parser.parse_args()

    orig_df = pd.read_csv(args.orig)
    causal_df = pd.read_csv(args.causal)

    # 行数对齐（两版 epoch 数可能不同, 截断到短的）
    n = min(len(orig_df), len(causal_df))
    orig_df = orig_df.iloc[:n]
    causal_df = causal_df.iloc[:n]
    print(f"对比: {Path(args.orig).name} vs {Path(args.causal).name}, 共 {n} 个 epoch")

    fig, axes = plt.subplots(
        len(TRAIN_RRV_FEATURES), 3,
        figsize=(16, 3.4 * len(TRAIN_RRV_FEATURES)),
        gridspec_kw={"width_ratios": [3, 1, 1], "wspace": 0.32, "hspace": 0.55},
    )

    epochs = np.arange(n)
    plot_n = args.epochs if args.epochs else n

    for row, feat in enumerate(TRAIN_RRV_FEATURES):
        o = orig_df[feat].values.astype(float)
        c = causal_df[feat].values.astype(float)
        rel = relative_diff(c, o)
        valid = rel[~np.isnan(rel)]

        ax_t, ax_s, ax_h = axes[row]

        # --- 列 1: 时序对比 ---
        ax_t.plot(epochs[:plot_n], o[:plot_n], color=C_ORIG, lw=1.5, label="original (filtfilt)", alpha=0.9)
        ax_t.plot(epochs[:plot_n], c[:plot_n], color=C_CAUSAL, lw=1.5, label="causal (lfilter)", alpha=0.9)
        ax_t.set_ylabel(feat, fontsize=9)
        if row == 0:
            ax_t.legend(loc="upper right", fontsize=8, frameon=False)
        ax_t.grid(axis="y", alpha=0.25, lw=0.5)
        ax_t.tick_params(labelsize=8)
        ax_t.set_xlabel("epoch", fontsize=8) if row == len(TRAIN_RRV_FEATURES) - 1 else None

        # --- 列 2: 散点对比 (x=原版, y=causal, 对角线参考) ---
        lim = [min(o.min(), c.min()), max(o.max(), c.max())]
        ax_s.scatter(o, c, s=10, color=C_CAUSAL, alpha=0.35, edgecolors="none")
        ax_s.plot(lim, lim, color=C_DIAG, lw=1, ls="--", alpha=0.7)
        ax_s.set_xlim(lim)
        ax_s.set_ylim(lim)
        ax_s.set_xlabel("original", fontsize=8)
        ax_s.set_ylabel("causal", fontsize=8)
        ax_s.tick_params(labelsize=7)
        ax_s.set_title(f"median rel. diff {np.nanmedian(valid) * 100:.1f}%", fontsize=9)

        # --- 列 3: 相对差异直方图 ---
        if len(valid):
            ax_h.hist(valid * 100, bins=50, color=C_ORIG, alpha=0.75, edgecolor="none")
            ax_h.axvline(np.nanmedian(valid) * 100, color=C_CAUSAL, lw=1.5, ls="--")
            ax_h.set_xlabel("rel. diff (%)", fontsize=8)
            ax_h.tick_params(labelsize=7)
        else:
            ax_h.text(0.5, 0.5, "no valid", ha="center", va="center", fontsize=8)
        if row == 0:
            ax_s.set_title(f"median rel. diff {np.nanmedian(valid) * 100:.1f}%", fontsize=9)
            ax_h.set_title("distribution", fontsize=9)

    fig.suptitle(
        f"RRV features: original vs causal preprocessing (MESA 0002, {n} epochs)",
        fontsize=13, y=0.995,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"图已保存: {args.output}")

    # 控制台指标表
    print(f"\n{'feature':<20}{'median':>9}{'p90':>9}{'max':>11}")
    print("-" * 50)
    for feat in TRAIN_RRV_FEATURES:
        o = orig_df[feat].values.astype(float)
        c = causal_df[feat].values.astype(float)
        rel = relative_diff(c, o)
        rel = rel[~np.isnan(rel)]
        print(
            f"{feat:<20}"
            f"{np.nanmedian(rel) * 100:>8.1f}%"
            f"{np.nanpercentile(rel, 90) * 100:>8.1f}%"
            f"{np.nanmax(rel) * 100:>10.1f}%"
        )


if __name__ == "__main__":
    main()
