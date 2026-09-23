"""统计训练集的 `internal_norm` 基线（部署时第一晚用它替代批内统计量）。

背景
----
模型 `config.internal_norm=true` 时，`forward` 里有::

    mean_x = x.mean(dim=(0, 1), keepdim=True)     # 对 batch 维 + 21 帧 求平均
    std_x  = x.std(dim=(0, 1), keepdim=True) + 1e-5
    x = (x - mean_x) / std_x

这个统计量**跟着 batch 组成走**：训练是 512 个窗一批、流式部署是 1 个窗一批，
不是同一个变换（实测同一段特征在 batch=1 与 batch=4 下 logits 差 0.128）。

部署策略（2026-09-17 与用户确认）
--------------------------------
用**被试自己的生理基线**替代批内统计量，逐日更新 —— 第一晚用**训练集的平均基线**
兜底。于是这一层归一化从"随批次漂移"变成"随被试收敛"。

本脚本产出那个"训练集平均基线"：

    mean_baseline[j] = 训练集所有窗口的「第 j 列在窗口内的均值」的平均
    std_baseline[j]  = 训练集所有窗口的「第 j 列在窗口内的标准差(ddof=1)」的平均

⚠️ 是 `std` 的**算术平均**（即"norm 参数的均值"），不是 `sqrt(E[方差])`。
   两者都有记录在输出里，部署默认用前者。

与训练管线的一致性
------------------
数据构造逐步对齐 `data_peparation.py::get_final_tensors` + `get_sequence_data(causal=False)`：

1. 特征列顺序 `[ACT..., _has_act, HRV..., _has_hrv, RRV..., _has_rrv]`（missing_mode）
2. 缺模态填 0 且标志置 0
3. **padding 在缩放之前**，`np.pad(..., mode="mean")` —— 注意它是**按列**填该列均值
4. 滑窗 `seq_len=21`，步长 1（`overlap_percent=None`）
5. scaler 在窗口化之后作用于每个 (窗, 帧) 行

**自检**：由 3/5 可知 `E[窗口均值]` 应当精确等于 scaler 的 `mean_`，缩放后为 0。
脚本会把实测值打出来核对 —— 对不上说明数据构造与训练不一致。

用法
----
    python experiments/evaluation/fit_internal_norm_baseline.py \
        --run-dir exports_our/2026-09-11_193146
    # 快速验证（只跑前 N 个被试）
    python experiments/evaluation/fit_internal_norm_baseline.py --run-dir ... --limit 50
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

#: 13 个特征槽位（与训练侧 `pipeline.py` 文档口径一致）
FEATURE_SLOTS = [
    "_acc_mean_1",
    "_hrv_median_nni", "_hrv_ratio_sd2_sd1", "_hrv_median_nni",
    "_hrv_vlf", "_hrv_lf", "_hrv_hf", "_hrv_lf_hf_ratio", "_hrv_total_power",
    "150_RRV_MedianBB", "150_RRV_LF", "270_RRV_MCVBB", "150_RRV_CVBB",
]

#: 模态 → 该模态的槽位（顺序即模型输入顺序）
_MODALITY_SLOTS = {
    "ACT": ["_acc_mean_1"],
    "HRV": ["_hrv_median_nni", "_hrv_ratio_sd2_sd1", "_hrv_median_nni",
            "_hrv_vlf", "_hrv_lf", "_hrv_hf", "_hrv_lf_hf_ratio", "_hrv_total_power"],
    "RRV": ["150_RRV_MedianBB", "150_RRV_LF", "270_RRV_MCVBB", "150_RRV_CVBB"],
}

#: 模态 → 用于判断"该被试有没有这个模态"的列名关键字
_MODALITY_PROBE = {"ACT": "_acc", "HRV": "_hrv", "RRV": "RRV"}


def _project_root() -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(8):
        if (p / "study_data.json").exists():
            return p
        p = p.parent
    raise FileNotFoundError("找不到项目根（study_data.json）")


def _subject_columns(modality: list) -> list:
    """模型输入的 16 列布局（含 `_has_*`）。"""
    cols = []
    for mod in ("ACT", "HRV", "RRV"):
        if mod in modality:
            cols.extend(_MODALITY_SLOTS[mod])
            cols.append(f"_has_{mod.lower()}")
    return cols


def _build_raw_matrix(df: pd.DataFrame, modality: list) -> np.ndarray:
    """由被试的特征表拼出模型输入的**原始**（未缩放）矩阵 `(n_epoch, 16)`。

    缺模态：该模态的特征列填 0、标志列填 0（与 `_extract_subj_features_raw` 一致）。
    """
    n = len(df)
    parts = []
    for mod in ("ACT", "HRV", "RRV"):
        if mod not in modality:
            continue
        probe = _MODALITY_PROBE[mod]
        present = not df.filter(regex=probe).empty
        if present:
            parts.append(df[_MODALITY_SLOTS[mod]].to_numpy(dtype=np.float64))
            parts.append(np.ones((n, 1), dtype=np.float64))
        else:
            parts.append(np.zeros((n, len(_MODALITY_SLOTS[mod])), dtype=np.float64))
            parts.append(np.zeros((n, 1), dtype=np.float64))
    return np.concatenate(parts, axis=1)


def _iter_subjects(ids, processed_dir: Path):
    """产出 `(subj_id, feature_path)`。三个数据集的特征表路径约定相同。"""
    for sid in ids:
        f = processed_dir / "features_full_combined" / f"features_combined{sid}.csv"
        if f.exists():
            yield sid, f


def _subject_windows(raw: np.ndarray, seq_len: int) -> np.ndarray:
    """按训练的方式 pad + 滑窗，返回 `(n_win, seq_len, F)`。"""
    half = seq_len // 2
    # ⚠️ np.pad(mode="mean") 是**按列**填该列均值（不是全局均值），已实测确认
    padded = np.pad(raw, ((half, half), (0, 0)), mode="mean")
    # 步长 1（overlap_percent=None 时 biopsykit 的行为）。
    # ⚠️ sliding_window_view 把窗口轴放在**最后** → 得到 (W, F, seq_len)，
    #    这里转成训练张量的形状 (W, seq_len, F)。
    w = np.lib.stride_tricks.sliding_window_view(padded, seq_len, axis=0)
    return np.moveaxis(w, -1, 1)


class _BatchAccumulator:
    """按**训练时的 batch 结构**累计 norm 参数。

    ⚠️ 这是本脚本的核心，也是最容易搞错的地方。

    训练时 `internal_norm` 的参数是这么算出来的::

        # LSTM.py: batch_loader → x_train.split(batch_size)
        x_batch = x[512 个窗]              # (512, 21, 16)
        mean = x_batch.mean(dim=(0, 1))    # 跨 512 个窗 × 21 帧
        std  = x_batch.std(dim=(0, 1))

    而 `x_train` 是**按被试顺序拼接**的（`data_peparation.py` 里 `for subj in dataset`），
    每个被试约 1032 个窗、`batch_size=512` ⇒ **一个 batch ≈ 半个被试**。

    也就是说：**训练时这一层归一化用的统计量，本质上是「被试自己（半夜间）」的统计量**，
    不是「21 帧窗口内」的、也不是「全训练集」的。

    要复现第一晚的兜底基线，就必须统计**同一个量**：把被试按同样顺序拼起来、
    按 512 切批、每批算一次 (mean, std)、再对批求平均。

    ⚠️ 早先版本错误地统计了「每个 21 帧窗口的 std」得到 0.149~0.910，
       实测与训练真正用的值（0.0~1.156）在 `270_RRV_MCVBB` 上差 2.4 倍、
       `_acc_mean_1` 差 7.8 倍 —— 量纲都对不上。
    """

    def __init__(self, batch_size: int, seq_len: int, n_feat: int):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_feat = n_feat
        self._buf = []                      # 累积的 (seq_len, F) 窗口
        self.sum_mean = np.zeros(n_feat)
        self.sum_std = np.zeros(n_feat)
        self.n_batch = 0
        self.n_win = 0

    def add_subject(self, windows: np.ndarray, scaler_mean, scaler_scale) -> None:
        """喂入一个被试的全部窗口 `(n_win, seq_len, F)`（**原始**特征，未缩放）。"""
        for w in windows:
            self._buf.append(w)
            self.n_win += 1
            if len(self._buf) == self.batch_size:
                self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        b = np.stack(self._buf)             # (B, seq_len, F)
        self.sum_mean += b.mean(axis=(0, 1))
        self.sum_std += b.std(axis=(0, 1), ddof=1)
        self.n_batch += 1
        self._buf = []

    def finish(self) -> None:
        self._flush()                       # 训练时最后一批不足 512 也会算


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个被试（调试）")
    ap.add_argument("--out", default=None, help="输出 JSON（默认写到 run_dir/checkpoints/）")
    args = ap.parse_args()

    root = _project_root()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir

    cfg = json.loads((run_dir / "config.json").read_text())
    modality = cfg["modality"]
    seq_len = int(cfg.get("seq_len", 21))
    input_layout = _subject_columns(modality)

    with open(run_dir / "checkpoints/scaler.json") as f:
        sc = json.load(f)
    scaler_mean = np.asarray(sc["mean_"], dtype=np.float64)
    scaler_scale = np.asarray(sc["scale_"], dtype=np.float64)
    if len(scaler_mean) != len(input_layout):
        raise ValueError(f"scaler 有 {len(scaler_mean)} 列, 布局推出 {len(input_layout)} 列")

    with open(run_dir / "study_data.json") as f:
        sd = json.load(f)

    print("=" * 78)
    print("统计训练集的 internal_norm 基线")
    print("=" * 78)
    print(f"run       : {run_dir}")
    print(f"模态      : {modality}   seq_len={seq_len}")
    print(f"输入布局  : {input_layout}")
    print(f"数据构造  : pad(mode='mean', 按列) → 滑窗 {seq_len} → 按训练 batch 切批"
          f" → 每批算 (均值, 标准差)")
    print(f"统计对象  : E[per-batch 的 norm 参数]（batch_size=512 连续窗 ≈ 半个被试）")

    # 三个数据集的训练名单与路径（按 run 的 dataset 字段挑选）
    jobs = []
    ds_cfg = str(cfg["dataset"]).upper()
    pools = []
    if "MESA" in ds_cfg:
        pools.append(("MESA_Sleep", sd["processed_mesa_path_hpc"], sd["mesa_split_file"]))
    if "SHHS1" in ds_cfg:
        pools.append(("SHHS1", sd["shhs1_processed_path"], sd["shhs1_split_file"]))
    if "SHHS2" in ds_cfg:
        pools.append(("SHHS2", sd["shhs2_processed_path"], sd["shhs2_split_file"]))

    for ds_name, proc_path, split_rel in pools:
        split = json.loads((root / split_rel).read_text())
        ids = split["train"]
        if args.limit:
            ids = ids[:args.limit]
        n_found = len(list(_iter_subjects(ids, Path(proc_path))))
        print(f"  {ds_name:<12s} train {len(ids)} 人, 特征表找到 {n_found} 个")
        jobs.append((ds_name, Path(proc_path), ids))

    # ---- 主循环：按训练时的 batch 结构累计 norm 参数 ----
    batch_size = int(cfg.get("batch_size", 512))
    acc = _BatchAccumulator(batch_size, seq_len, len(input_layout))
    n_subj = 0
    t0 = time.time()

    for ds_name, proc_path, ids in jobs:
        for sid, fpath in _iter_subjects(ids, proc_path):
            try:
                df = pd.read_csv(fpath, index_col=0)
            except Exception as e:
                print(f"  [{ds_name}/{sid}] 读取失败: {e}")
                continue
            if len(df) < seq_len:
                continue
            raw = _build_raw_matrix(df, modality)
            acc.add_subject(_subject_windows(raw, seq_len),
                            scaler_mean, scaler_scale)
            n_subj += 1
            if n_subj % 500 == 0:
                print(f"    ...{n_subj} 人 / {acc.n_win} 窗 / {acc.n_batch} batch"
                      f"  ({(time.time()-t0)/60:.1f} min)")
    acc.finish()

    n_win = acc.n_win
    n_batch = acc.n_batch
    print(f"\n完成: {n_subj} 个被试, {n_win} 个窗口, {n_batch} 个 batch "
          f"(batch_size={batch_size}), 用时 {(time.time()-t0)/60:.1f} min")

    # ---- 缩放域：norm 参数是**缩放后**算的，但 scaler 是逐列线性变换，可解析换算 ----
    mean_raw = acc.sum_mean / n_batch
    std_raw = acc.sum_std / n_batch

    mean_scaled = (mean_raw - scaler_mean) / scaler_scale
    std_scaled = std_raw / scaler_scale

    print(f"\n{'列':>24s} {'E[batch 均值]缩放后':>20s} {'E[batch std]缩放后':>20s}")
    print("-" * 70)
    for j, c in enumerate(input_layout):
        print(f"{c:>24s} {mean_scaled[j]:>20.6f} {std_scaled[j]:>20.6f}")

    # ---- 自检: E[窗口均值] 缩放后应当 ≈ 0（由 scaler 的拟合方式决定）----
    mx = np.abs(mean_scaled).max()
    print(f"\n自检: |E[窗内均值]| 最大 = {mx:.3e}")
    if mx > 1e-2:
        print("  ⚠️ 明显偏离 0 —— 数据构造与训练管线可能不一致（padding/列序/模态判定）")
    else:
        print("  ✓ 与 scaler 的构造一致（scaler 拟合的正是同一批窗口化数据）")

    out_path = Path(args.out) if args.out else run_dir / "checkpoints/internal_norm_baseline.json"
    out_path.write_text(json.dumps({
        "source_run": str(run_dir),
        "dataset": cfg["dataset"],
        "seq_len": seq_len,
        "n_subjects": n_subj,
        "n_windows": n_win,
        "n_batches": n_batch,
        "batch_size": batch_size,
        "columns": input_layout,
        "mean_": mean_scaled.tolist(),
        "std_": std_scaled.tolist(),
        "note": ("E[(batch 均值, batch std)]，batch = 训练时的 batch_size 个连续窗"
                 "（≈ 半个被试）。这才是训练时 internal_norm 实际用的那个量的均值。"),
    }, indent=2, ensure_ascii=False))
    print(f"\n已保存: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
