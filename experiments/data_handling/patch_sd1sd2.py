#!/usr/bin/env python3
"""补表工具：修掉特征表里 `_hrv_ratio_sd2_sd1` 的插值伪影。

**用途**：对**已产出的**预处理数据做原地修补，不重跑预处理。

判据与动作
----------
    `_hrv_sd1 < 1e-6` 的行  →  `_hrv_ratio_sd2_sd1` 置 NaN，**整行删除**
    标签文件同步删除同一批行（保持特征表与标签表行数相等）

`_hrv_sd1 < 1e-6` 同时覆盖两类坏值（实测全量训练集：402 + 1145 行）：
  - `sd1 ≈ 1e-13`（插值段近乎等差）→ `ratio` 爆到 1e13~1e15
  - `sd1 == 0`（严格等差）→ `ratio = inf` → 被 `replace([inf,-inf], 0.0)` 掩盖成 **0**

为什么要删行而不是填 0
----------------------
填 0 会让"无意义"伪装成"变异很小"的正常低值。而 `ratio` 只是**唯一会爆的那个**——
同一批 epoch 的频域槽位（vlf/lf/hf/total_power）实测偏 12~35%，只是不爆，查不出来。

⚠️ 本工具只改能影响训练的两个地方
--------------------------------
- `features_full_combined/features_combined<ID>.csv`（训练直接读）
- 标签文件（MESA 在 `actigraph_data_clean/`，SHHS/MrOS 在 `sleep_stages/`）

**同一被试的其它特征块**（`actigraph_features/` `hrv_features/`
`respiration_features_clean/` `edr_features_clean/`）**不改** —— 它们只在
`merge_features` 重跑时才被用到。若之后重跑 merge，改动会被覆盖，需重跑本工具。

⚠️ CSV 按"不透明列"处理（读时不设 index_col）
--------------------------------------------
MESA 的特征表首列是**无名索引列**，SHHS/MrOS 的**没有**。用 `index_col=0` 读会
把 SHHS 的第一列 `_hrv_mean_nni` 当成索引吃掉、写回时格式也变。所以这里一律
`pd.read_csv(...)`（默认 RangeIndex）+ `to_csv(index=False)`，逐字节保持原格式。

用法
----
    # 先看会改什么（不写盘）
    python experiments/data_handling/patch_sd1sd2.py --root <目录> --dry-run

    # 实际修补
    python experiments/data_handling/patch_sd1sd2.py --root <目录>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

#: `_hrv_sd1` 低于此值即判为无效（单位毫秒；真实 SD1 在 5~50 ms 量级）。
SD1_MIN = 1e-6

#: 参与判定的列名
SD1_COL = "_hrv_sd1"
RATIO_COL = "_hrv_ratio_sd2_sd1"

#: 标签文件相对 `<dataset>_processed/` 的候选路径（按顺序找第一个存在的）
LABEL_TEMPLATES = [
    ("sleep_stages/sleep_stages{id}.csv", "sleep_stages"),
    ("actigraph_data_clean/actigraph_data_clean{id}.csv", "actigraph_data_clean"),
]


def find_label_file(dataset_dir: Path, subj_id: str):
    """找到该被试的标签文件；找不到返回 None。"""
    for tmpl, _ in LABEL_TEMPLATES:
        p = dataset_dir / tmpl.format(id=subj_id)
        if p.exists():
            return p
    return None


def patch_one(feat_path: Path, dataset_dir: Path, dry_run: bool):
    """修补单个被试。返回 (改了几行, 说明)。"""
    subj_id = feat_path.name.replace("features_combined", "").replace(".csv", "")

    # ---- 快速探测：只读判定列，没坏值就整个文件跳过（省掉 99% 的读写）----
    try:
        sd1 = pd.read_csv(feat_path, usecols=[SD1_COL])[SD1_COL].to_numpy(dtype=float)
    except ValueError:
        return 0, f"{subj_id}: 缺 {SD1_COL} 列，跳过"
    bad_mask = np.isfinite(sd1) & (sd1 < SD1_MIN)
    n_bad = int(bad_mask.sum())
    if n_bad == 0:
        return 0, None

    label_path = find_label_file(dataset_dir, subj_id)
    if label_path is None:
        return 0, f"{subj_id}: 有 {n_bad} 行坏值但找不到标签文件，**未修改**"

    if dry_run:
        return n_bad, f"{subj_id}: 将删 {n_bad} 行（特征表+标签表）"

    # ---- 实际修补 ----
    feat = pd.read_csv(feat_path)                 # 不设 index_col，保持原格式
    label = pd.read_csv(label_path)

    if len(feat) != len(label):
        return 0, f"{subj_id}: 特征表 {len(feat)} 行 vs 标签表 {len(label)} 行，**不相等，跳过**"

    # 标志置 NaN 后再删行（保留"这里曾判为无效"的语义痕迹，虽然行会消失）
    feat.loc[bad_mask, RATIO_COL] = np.nan
    keep = ~bad_mask
    feat_out, label_out = feat[keep], label[keep]

    if len(feat_out) != len(label_out):
        return 0, f"{subj_id}: 删行后行数不相等，**未写盘**"

    feat_out.to_csv(feat_path, index=False)
    label_out.to_csv(label_path, index=False)
    return n_bad, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="已产出的数据根目录（如 processed_data_causal_20260806）")
    ap.add_argument("--dry-run", action="store_true", help="只报告，不写盘")
    args = ap.parse_args()

    if not args.root.exists():
        raise SystemExit(f"目录不存在: {args.root}")

    datasets = sorted(d for d in args.root.iterdir()
                      if d.is_dir() and d.name.endswith("_processed"))
    if not datasets:
        raise SystemExit(f"在 {args.root} 下没找到 *_processed 目录")

    print(f"{'[DRY-RUN] ' if args.dry_run else ''}修补 {args.root}")
    print(f"判据: {SD1_COL} < {SD1_MIN:g}\n")

    grand = 0
    for ds in datasets:
        feat_dir = ds / "features_full_combined"
        if not feat_dir.exists():
            print(f"  {ds.name}: 无 features_full_combined，跳过")
            continue
        files = sorted(feat_dir.glob("*.csv"))
        n_here = 0
        notes = []
        for f in files:
            n, note = patch_one(f, ds, args.dry_run)
            n_here += n
            if note:
                notes.append(note)
        grand += n_here
        print(f"  {ds.name}: {len(files)} 个被试, 删 {n_here} 行")
        for note in notes[:10]:
            print(f"      {note}")
        if len(notes) > 10:
            print(f"      ... 另有 {len(notes)-10} 条说明")

    print(f"\n合计删除 {grand} 行")
    if args.dry_run:
        print("（dry-run，未写盘）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
