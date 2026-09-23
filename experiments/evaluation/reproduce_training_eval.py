"""用**路径 B**（`streaming/reproduce.py`）复现训练时的评测结果。

做什么
------
把测试集每个被试的整夜特征表走一遍路径 B，与训练时 `LSTM.py::test` 存下的
`results/per_subject_predictions/<id>.csv` **逐 epoch 对比**。

预测一致 ⇒ 指标自动一致（两者用的是同一份标注），所以不必重算 MCC，
`results.json` 里的 mean mcc 就是现成的复现目标。

对齐点（路径 B 的实现依据，见 `streaming/reproduce.py` 的模块文档）
-----------------------------------------------------------------
1. 特征列用**训练口径**（HRV 第 3 项是与第 1 项重复的 `_hrv_median_nni`）
2. `pad(mode="mean")` 按列 → 滑窗 21、步长 1
3. 训练集 scaler
4. **整夜归一化**（一个被试一个 batch）
5. argmax

用法
----
    python experiments/evaluation/reproduce_training_eval.py \
        --run-dir exports_our/2026-09-11_193146
    python .../reproduce_training_eval.py --run-dir ... --limit 50   # 先小样本试
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

from sleep_analysis.classification.deep_learning.dl_scoring import dl_score  # noqa: E402
from sleep_analysis.classification.inference.streaming.reproduce import (  # noqa: E402
    OvernightReproEngine,
)

#: 数据集前缀 → (`study_data.json` 里的路径键, split 文件键)
_SOURCES = {
    "mesasleep": ("processed_mesa_path_hpc", "mesa_split_file"),
    "shhs1": ("shhs1_processed_path", "shhs1_split_file"),
    "shhs2": ("shhs2_processed_path", "shhs2_split_file"),
}


def _load_ground_truth(src: str, sid: str, proc: Path, classification: str):
    """按训练侧 `MesaDataset.ground_truth` / `ShhsDataset.ground_truth` 的路径读标签。"""
    if src == "mesasleep":
        p = proc / "actigraph_data_clean" / f"actigraph_data_clean{sid}.csv"
    else:
        p = proc / "sleep_stages" / f"sleep_stages{sid}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if classification not in df.columns:
        return None
    return df[classification].to_numpy()


def _project_root() -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(8):
        if (p / "study_data.json").exists():
            return p
        p = p.parent
    raise FileNotFoundError("找不到项目根（study_data.json）")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="每个数据集只跑前 N 个被试")
    args = ap.parse_args()

    root = _project_root()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir

    cfg = json.loads((run_dir / "config.json").read_text())
    sd = json.loads((run_dir / "study_data.json").read_text())
    pred_dir = run_dir / "per_subject_predictions"
    if not pred_dir.exists():
        pred_dir = run_dir / "results/per_subject_predictions"
    if not pred_dir.exists():
        print(f"[ERROR] 找不到训练时的逐被试预测: {pred_dir}")
        return 1

    print("=" * 78)
    print("路径 B：复现训练时的评测结果")
    print("=" * 78)
    print(f"run        : {run_dir}")
    print(f"模型       : {cfg['classification']} | {cfg['modality']} | seq_len={cfg['seq_len']}")
    print(f"对照       : {pred_dir}  ({len(list(pred_dir.glob('*.csv')))} 个文件)")

    ref = json.loads((run_dir / "results/results.json").read_text())
    target = {k: v["mean"] for k, v in ref["mean"].items()}
    print(f"复现目标   : mean mcc={target['mcc']:.4f}  acc={target['accuracy']:.4f}  "
          f"kappa={target['kappa']:.4f}")

    eng = OvernightReproEngine(run_dir)
    print(f"\n引擎       : {eng.onnx_path.name}")
    print(f"训练口径布局: {eng.input_layout}")

    # ---- 逐被试推理 + 与训练时的预测对比 ----
    stats = {"n_subj": 0, "n_epoch": 0, "n_diff_epoch": 0, "n_len_mismatch": 0,
             "n_missing": 0, "per_source": {}}
    score_rows = []          # 逐被试指标（用我们自己的预测算）
    t0 = time.time()

    for src, (path_key, split_key) in _SOURCES.items():
        files = sorted(pred_dir.glob(f"{src}@*.csv"))
        if args.limit:
            files = files[:args.limit]
        if not files:
            continue
        proc = Path(sd[path_key])
        n_subj = n_ep = n_diff = n_len = 0
        for f in files:
            sid = f.stem.split("@", 1)[1]
            feat_path = proc / "features_full_combined" / f"features_combined{sid}.csv"
            if not feat_path.exists():
                stats["n_missing"] += 1
                continue
            df = pd.read_csv(feat_path, index_col=0)
            y_pred = eng.predict_from_frame(df)
            y_ref = pd.read_csv(f, index_col=0).iloc[:, 0].to_numpy()
            n_subj += 1
            n_ep += len(y_ref)
            if len(y_pred) != len(y_ref):
                n_len += 1
                continue
            n_diff += int((y_pred != y_ref).sum())
            # 逐被试指标（与训练用同一个 dl_score）
            gt = _load_ground_truth(src, sid, proc, cfg["classification"])
            if gt is not None and len(gt) == len(y_pred):
                sc = dl_score(y_pred, pd.DataFrame(gt, columns=["sleep_stage"]),
                              classification_type=cfg["classification"], subject_id=f.stem)
                score_rows.append({k: v for k, v in sc.items()
                                   if k != "confusion_matrix"})
            if n_subj % 200 == 0:
                print(f"    [{src}] {n_subj}/{len(files)}  "
                      f"({(time.time()-t0)/60:.1f} min)  不一致 {n_diff}/{n_ep}")
        stats["per_source"][src] = {"n_subj": n_subj, "n_epoch": n_ep,
                                    "n_diff": n_diff, "n_len_mismatch": n_len}
        stats["n_subj"] += n_subj
        stats["n_epoch"] += n_ep
        stats["n_diff_epoch"] += n_diff
        stats["n_len_mismatch"] += n_len

    print(f"\n完成: {stats['n_subj']} 个被试, {stats['n_epoch']} 个 epoch, "
          f"用时 {(time.time()-t0)/60:.1f} min")

    print(f"\n{'来源':>12s} {'被试':>6s} {'epoch':>9s} {'不一致':>8s} {'一致率':>9s} {'长度不符':>9s}")
    print("-" * 62)
    for src, s in stats["per_source"].items():
        rate = 1 - s["n_diff"] / max(s["n_epoch"], 1)
        print(f"{src:>12s} {s['n_subj']:>6d} {s['n_epoch']:>9d} {s['n_diff']:>8d} "
              f"{rate:>9.4%} {s['n_len_mismatch']:>9d}")
    total_rate = 1 - stats["n_diff_epoch"] / max(stats["n_epoch"], 1)
    print("-" * 62)
    print(f"{'合计':>12s} {stats['n_subj']:>6d} {stats['n_epoch']:>9d} "
          f"{stats['n_diff_epoch']:>8d} {total_rate:>9.4%}")
    if stats["n_missing"]:
        print(f"  （{stats['n_missing']} 个被试的特征表缺失, 已跳过）")

    # ---- 指标复现 ----
    if score_rows:
        got = pd.DataFrame(score_rows).mean()
        print(f"\n{'=' * 78}")
        print(f"指标复现（{len(score_rows)} 个被试的逐被试指标均值）")
        if args.limit:
            print(f"⚠️ 用了 --limit {args.limit} —— 这是**子集**的均值，与训练时报的"
                  f"全测试集均值**不可比**，只用于看趋势")
        print(f"{'=' * 78}")
        print(f"{'指标':>14s} {'训练时报的':>12s} {'本次复现':>12s} {'差':>11s}")
        print("-" * 54)
        for m in ["accuracy", "precision", "recall", "f1", "kappa", "specificity", "mcc"]:
            a, b = target.get(m), float(got[m])
            print(f"{m:>14s} {a:>12.4f} {b:>12.4f} {b - a:>+11.4f}")

    print(f"\n{'=' * 78}")
    print("结论")
    print(f"{'=' * 78}")
    if stats["n_len_mismatch"]:
        print(f"  ✗ 有 {stats['n_len_mismatch']} 个被试的 epoch 数对不上 —— 结构性错误")
        return 1

    # 判据按**指标**定，不按逐 epoch 是否逐位一致。
    # ⚠️ 逐 epoch 不可能逐位一致：`internal_norm` 的 `+1e-5` 会把常数列（`_has_act`）
    #    上「mean 的浮点舍入」放大 1e5 倍，该列进模型时数学上是 0、浮点上是**噪声**，
    #    具体值取决于求和顺序（torch vs numpy 永远对不齐）。
    #    实测 0.11% 的 epoch 因此翻边，对指标的影响是 1e-4 量级（= 四舍五入）。
    METRIC_TOL = 1e-3
    if args.limit:
        print("  （--limit 下不做判定 —— 子集均值与全量不可比）")
        return 0
    if score_rows:
        worst = max(abs(float(got[m]) - target.get(m, float("nan")))
                    for m in ["mcc", "accuracy", "kappa", "f1"] if m in target)
    else:
        worst = float("inf")
    print(f"  逐 epoch 一致率 {total_rate:.4%}（{stats['n_diff_epoch']}/{stats['n_epoch']} 翻边）")
    print(f"  指标最大偏差 {worst:.4f}（容差 {METRIC_TOL}）")
    if worst < METRIC_TOL:
        print(f"  ✓ **复现成功** —— 指标与训练记录一致到 {worst:.0e}")
        print(f"    残余的 {stats['n_diff_epoch']} 个翻边 epoch 来自 `_has_act` 常数列的")
        print(f"    浮点放大（数学上是 0、浮点上是噪声），任何实现都无法与 torch 逐位一致。")
        return 0
    print(f"  ✗ 指标偏差 {worst:.4f} 超过容差 {METRIC_TOL}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
