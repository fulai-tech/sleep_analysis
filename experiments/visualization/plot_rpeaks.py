"""
neurokit2 R 峰检测可视化（两级百分位筛选）
==========================================
从 SHHS 原始 EDF 读取 ECG 信号，与预处理产出的 R 峰位置对齐并绘制。

用法:
    # 建索引（首次跑一次）
    python experiments/visualization/plot_rpeaks.py --study shhs1 --build-index

    # 查看各指标的分位表
    python experiments/visualization/plot_rpeaks.py --study shhs1 --list-pct

    # 指定被试，查看该被试内某百分位的数据段
    python experiments/visualization/plot_rpeaks.py --study shhs1 --subj 200377 --seg-pct 50

    # 先按全局指标选被试，再在该被试内按百分位选段
    python experiments/visualization/plot_rpeaks.py --study shhs1 --pct 10 --metric rp_density --seg-pct 10 --seg-metric rp_count

    # 指定被试 + 指定具体时间/epoch
    python experiments/visualization/plot_rpeaks.py --study shhs1 --subj 200001 --epoch 100 --duration 8

被试排序指标 (--metric):
    - rp_density  : 每 epoch 平均 R 峰数
    - hr_coverage : 正常 HR 占比 (30-200 bpm)
    - hr_mean     : 平均心率

段内排序指标 (--seg-metric):
    - rp_count    : 窗口内 R 峰数量 (越少=越稀疏，可能漏检)
    - hr_mean     : 窗口内平均心率
    - hr_std      : 窗口内心率标准差 (越大=越不齐)
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

SHHS_ROOT = Path("/mnt/nas_data/psgdata/SHHS")
PROCESSED_BASE = Path("/srv/shared/psgdata")
OUT_DIR = Path(__file__).parents[2] / "exports_our" / "visualization"

# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_raw_ecg(study, subj):
    """读取 SHHS EDF 中的 ECG 信号，同时返回实际采样率。"""
    import mne
    edf_path = SHHS_ROOT / "polysomnography/edfs" / study / f"{study}-{subj}.edf"
    raw = mne.io.read_raw_edf(edf_path, verbose=False)
    ecg = raw.pick_channels(["ECG"])
    return ecg.get_data()[0, :], raw.times, float(ecg.info["sfreq"])


def load_processed_rpoints(study, subj):
    ecg_file = PROCESSED_BASE / f"{study}_processed/ecg_data_clean/ecg_data_clean{subj}.csv"
    if ecg_file.exists():
        return pd.read_csv(ecg_file)
    rp_file = SHHS_ROOT / "polysomnography/annotations-rpoints" / study / f"{study}-{subj}-rpoint.csv"
    if rp_file.exists():
        return pd.read_csv(rp_file)
    raise FileNotFoundError(f"No R-point data for {subj}")

# ---------------------------------------------------------------------------
# 被试排序索引
# ---------------------------------------------------------------------------

def build_index(study, force=False):
    index_file = OUT_DIR / f"subject_index_{study}.csv"
    if index_file.exists() and not force:
        print(f"Loading cached index: {index_file}")
        return pd.read_csv(index_file)

    processed_dir = PROCESSED_BASE / f"{study}_processed"
    ecg_dir = processed_dir / "ecg_data_clean"
    sleep_dir = processed_dir / "sleep_stages"
    files = sorted(ecg_dir.glob("*.csv"))
    print(f"Building index for {len(files)} subjects from {study}...")

    rows = []
    for i, f in enumerate(files):
        subj = f.name.replace("ecg_data_clean", "").replace(".csv", "")
        try:
            df = pd.read_csv(f)
            n_beats = len(df)
            if "HR" in df.columns:
                hr = df["HR"].values
                hr_ok = ((hr >= 30) & (hr <= 200)).sum()
                hr_coverage = hr_ok / n_beats if n_beats > 0 else 0
                hr_mean = float(np.mean(hr[hr > 0])) if (hr > 0).any() else 0
            else:
                hr_coverage, hr_mean = 1.0, 0
            epochs = int(df["epoch"].nunique()) if "epoch" in df.columns else 0
            rp_density = n_beats / epochs if epochs > 0 else 0
            rows.append({
                "subj": str(subj),
                "n_beats": n_beats, "n_epochs": epochs,
                "rp_density": round(rp_density, 1),
                "hr_coverage": round(hr_coverage, 4),
                "hr_mean": round(hr_mean, 1),
            })
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(files)}...")
        except Exception as e:
            print(f"  {subj}: SKIP ({e})")

    df_idx = pd.DataFrame(rows)
    for m in ["hr_coverage", "rp_density", "n_beats", "n_epochs", "hr_mean"]:
        df_idx[f"pct_{m}"] = (df_idx[m].rank(pct=True) * 100).round(1)
    df_idx.to_csv(index_file, index=False)
    print(f"Index saved: {index_file} ({len(df_idx)} subjects)")
    return df_idx


def find_subject_at_percentile(df_idx, metric, pct):
    pct_col = f"pct_{metric}"
    idx = (df_idx[pct_col] - pct).abs().idxmin()
    row = df_idx.loc[idx].to_dict()
    row["subj"] = str(row["subj"])
    return row


def list_percentiles(df_idx, pcts=None):
    if pcts is None:
        pcts = [5, 10, 25, 50, 75, 90, 95]
    metrics = [c.replace("pct_", "") for c in df_idx.columns if c.startswith("pct_")]
    for m in metrics:
        print(f"\n{'='*65}\n  Metric: {m}\n{'='*65}")
        print(f"{'Pct':>6}  {'Subj':>8}  {'Value':>10}  (range: {df_idx[m].min():.1f} – {df_idx[m].max():.1f})")
        for p in pcts:
            row = find_subject_at_percentile(df_idx, m, p)
            print(f"{p:>5.0f}%  {row['subj']:>8}  {row[m]:>10.2f}")

# ---------------------------------------------------------------------------
# 段内排序 — 核心
# ---------------------------------------------------------------------------

def segment_metrics(df_rp, window_epochs=2):
    """将整条记录按滑动窗口切分，每个窗口计算指标。

    Returns DataFrame 列: epoch_start, epoch_end, start_sec, end_sec,
                         rp_count, hr_mean, hr_std
    """
    if "epoch" not in df_rp.columns:
        return None

    epochs = sorted(df_rp["epoch"].unique())
    if len(epochs) < window_epochs:
        return None

    rows = []
    for i in range(len(epochs) - window_epochs + 1):
        e_start = epochs[i]
        e_end = epochs[i + window_epochs - 1]
        mask = (df_rp["epoch"] >= e_start) & (df_rp["epoch"] <= e_end)
        seg = df_rp[mask]

        rp_count = len(seg)

        hr_mean = 0
        hr_std = 0
        if "HR" in seg.columns:
            hr_vals = seg["HR"].values
            hr_vals = hr_vals[(hr_vals >= 30) & (hr_vals <= 200)]
            if len(hr_vals) > 1:
                hr_mean = float(np.mean(hr_vals))
                hr_std = float(np.std(hr_vals))

        rows.append({
            "epoch_start": int(e_start),
            "epoch_end": int(e_end),
            "start_sec": (e_start - 1) * 30,
            "end_sec": e_end * 30,
            "rp_count": rp_count,
            "hr_mean": round(hr_mean, 1),
            "hr_std": round(hr_std, 1),
        })

    df_seg = pd.DataFrame(rows)
    for m in ["rp_count", "hr_mean", "hr_std"]:
        df_seg[f"pct_{m}"] = (df_seg[m].rank(pct=True) * 100).round(1)
    return df_seg


def find_segment_at_percentile(df_rp, seg_metric, seg_pct, window_epochs=2):
    """在一条记录内，找到指定百分位的窗口。"""
    df_seg = segment_metrics(df_rp, window_epochs)
    if df_seg is None or len(df_seg) == 0:
        return None, None, None

    pct_col = f"pct_{seg_metric}"
    idx = (df_seg[pct_col] - seg_pct).abs().idxmin()
    row = df_seg.loc[idx]

    # 所有窗口在该指标上的分布概况
    low = df_seg[seg_metric].min()
    high = df_seg[seg_metric].max()
    median = df_seg[seg_metric].median()

    desc = (f"{seg_metric}: {row[seg_metric]:.1f}  "
            f"(range across recording: {low:.1f}–{high:.1f}, "
            f"median={median:.1f}, "
            f"pct={row[pct_col]:.0f}%)")
    return float(row["start_sec"]), float(row["end_sec"]), desc

# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

def plot_rpeaks(study, subj, start_sec=None, end_sec=None,
                epoch=None, n_epochs=2, seg_desc=""):
    raw_ecg, t, ecg_rate = load_raw_ecg(study, subj)
    df_rp = load_processed_rpoints(study, subj)

    if start_sec is None:
        if epoch is not None:
            start_sec = (epoch - 1) * 30
            end_sec = start_sec + n_epochs * 30
        else:
            max_start = len(raw_ecg) / ecg_rate - 60
            start_sec = max(0, random.uniform(0, max_start))
            end_sec = start_sec + 60

    start_idx = max(0, int(start_sec * ecg_rate))
    end_idx = min(len(raw_ecg), int(end_sec * ecg_rate))
    ecg_segment = raw_ecg[start_idx:end_idx]
    t_segment = np.arange(start_idx, end_idx) / ecg_rate

    if "seconds" in df_rp.columns:
        rpeaks = (df_rp["seconds"].values * ecg_rate).astype(int)
    elif "RPoint" in df_rp.columns:
        rpeaks = df_rp["RPoint"].values
    else:
        raise ValueError("No position column")

    import neurokit2 as nk
    clean_len = min(len(raw_ecg), 5000000)
    ecg_cleaned = nk.ecg_clean(raw_ecg[:clean_len], sampling_rate=ecg_rate)
    cleaned_segment = ecg_cleaned[start_idx:end_idx]

    rp_mask = (rpeaks >= start_idx) & (rpeaks < end_idx)
    rp_in_seg = rpeaks[rp_mask] - start_idx
    rp_times = rpeaks[rp_mask] / ecg_rate

    if "seconds" in df_rp.columns and "HR" in df_rp.columns:
        hr_mask = (df_rp["seconds"] >= start_sec) & (df_rp["seconds"] < end_sec)
        hr_sec = df_rp["seconds"][hr_mask].values
        hr_vals = df_rp["HR"][hr_mask].values
        has_hr = len(hr_vals) > 0
    else:
        has_hr = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(t_segment, ecg_segment, color="#bdc3c7", linewidth=0.8, alpha=0.6,
             label="Raw ECG")
    ax1.plot(t_segment, cleaned_segment, color="#2c3e50", linewidth=0.8,
             label="Cleaned ECG (neurokit2)")
    if len(rp_in_seg) > 0:
        ax1.scatter(rp_times, cleaned_segment[rp_in_seg], color="#e74c3c",
                    s=40, zorder=5, marker="v", label=f"R-peaks ({len(rp_in_seg)})")

    ax1.set_ylabel("ECG (µV)", fontsize=12)
    title = (f"SHHS R-Peak — {study.upper()} Subject {subj}  "
             f"(epochs {start_sec//30+1:.0f}–{end_sec//30+1:.0f},  "
             f"{end_sec-start_sec:.0f}s, {len(rp_in_seg)} beats)")
    if seg_desc:
        title += f"\n{seg_desc}"
    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    if has_hr:
        hr_clean = np.clip(hr_vals.copy(), 30, 200)
        ax2.plot(hr_sec, hr_clean, "-o", color="#3498db", markersize=3,
                 linewidth=0.8, label="HR (bpm)")
        ax2.axhline(y=np.mean(hr_clean), color="#e74c3c", linestyle="--",
                    linewidth=0.8, label=f"Mean HR = {np.mean(hr_clean):.0f} bpm")
        ax2.set_xlabel("Time (s)", fontsize=12)
        ax2.set_ylabel("Heart Rate (bpm)", fontsize=12)
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.set_visible(False)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"rpeak_{study}_{subj}_epoch{start_sec//30+1:.0f}"
    out_path = OUT_DIR / f"{out_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    dur = max(1, end_sec - start_sec)
    print(f"Segment: {len(rp_in_seg)} beats in {dur:.0f}s "
          f"(avg HR = {len(rp_in_seg)/(dur/60):.0f} bpm)")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="neurokit2 R-peak 可视化")
    parser.add_argument("--study", choices=["shhs1", "shhs2"], default="shhs1")
    parser.add_argument("--subj", type=str, help="被试 ID")

    # 被试级百分位
    parser.add_argument("--pct", type=float, help="被试全局指标百分位 (0-100)")
    parser.add_argument("--metric", type=str, default="rp_density",
                        help="被试排名字段 (rp_density, hr_coverage, hr_mean)")
    parser.add_argument("--list-pct", action="store_true")
    parser.add_argument("--build-index", action="store_true")

    # 段内百分位
    parser.add_argument("--seg-pct", type=float, default=50,
                        help="段内百分位 (0-100, 默认 50=中位)")
    parser.add_argument("--seg-metric", type=str, default="rp_count",
                        help="段内排名字段 (rp_count, hr_mean, hr_std)")
    parser.add_argument("--seg-window", type=int, default=2,
                        help="滑动窗口大小 (epoch 数, 默认 2)")

    # 时间参数
    parser.add_argument("--epoch", type=int, help="起始 epoch")
    parser.add_argument("--start", type=float, help="起始秒数")
    parser.add_argument("--duration", type=float, default=8, help="时长(秒)")

    args = parser.parse_args()

    # --- 确定被试 ---
    subj = args.subj

    if args.build_index:
        build_index(args.study, force=True)
        if args.list_pct:
            df_idx = build_index(args.study)
            list_percentiles(df_idx)
        exit(0) if not args.list_pct else None

    if args.list_pct:
        df_idx = build_index(args.study)
        if df_idx is not None:
            list_percentiles(df_idx)
        if not subj and args.pct is None:
            exit(0)

    if subj is None and args.pct is not None:
        df_idx = build_index(args.study)
        if df_idx is None:
            exit(1)
        row = find_subject_at_percentile(df_idx, args.metric, args.pct)
        subj = row["subj"]
        print(f"Subject: {subj}  ({args.metric}={row[args.metric]}, "
              f"pct ~{row[f'pct_{args.metric}']}%)")

    if subj is None:
        df_idx = build_index(args.study)
        if df_idx is not None:
            row = df_idx.sample(1).iloc[0]
            subj = str(row["subj"])
            print(f"Random subject: {subj}")

    # --- 确定时间范围 ---
    seg_desc = ""

    if args.epoch or args.start:
        start_sec = args.start
        end_sec = start_sec + args.duration if start_sec is not None else None
    elif args.seg_pct is not None:
        # 按段内百分位选窗口
        df_rp = load_processed_rpoints(args.study, subj)
        win_start, win_end, seg_desc = find_segment_at_percentile(
            df_rp, args.seg_metric, args.seg_pct, args.seg_window)
        if win_start is None:
            print("  Could not find segment, falling back to epoch 100")
            win_start, win_end = 99 * 30, 99 * 30 + 60

        # 从窗口中间取出 --duration 秒
        win_center = (win_start + win_end) / 2
        half = args.duration / 2
        start_sec = max(win_start, win_center - half)
        end_sec = min(win_end, start_sec + args.duration)
        print(f"  Segment: epochs {win_start//30+1:.0f}–{win_end//30:.0f} "
              f"(window {win_end-win_start:.0f}s, clipped to {end_sec-start_sec:.0f}s) | {seg_desc}")
    else:
        start_sec, end_sec = None, None

    plot_rpeaks(args.study, subj, start_sec=start_sec, end_sec=end_sec,
                epoch=args.epoch, seg_desc=seg_desc)
