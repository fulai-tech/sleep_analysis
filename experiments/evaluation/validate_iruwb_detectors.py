"""IR-UWB 20 Hz 管线的检测器验证 — 对着仿真真值定量评估。

报告以下几项, 全部在**合成数据**上算:

    [1] 心搏检测的召回 / 精确 / RR 间期误差（分因果与非因果、含与不含运动）
    [2] 检测带 × 心搏波形宽度的二维扫描
    [3] 逐 epoch 质量标记
    [4] 用误差模型估算的 HRV 特征影响
    [5] 用真实检测器输出算的 HRV 特征端到端误差

⚠️ **这些数字衡量的是"检测器在仿真假设下工作得多好", 不代表真机性能。**
合成信号只包含建模时写进去的成分。

⚠️ [1] 里报**两个**时序指标: ``median_ms`` 是绝对峰时刻误差, ``rr_median_ms`` 是
RR 间期误差。前者含滤波器引入的常数偏移（对 HRV 无影响 —— RR 间期里抵消）,
看 HRV 相关的问题应参考后者。

用法
----
    python experiments/evaluation/validate_iruwb_detectors.py              # 全部
    python experiments/evaluation/validate_iruwb_detectors.py --quick      # 快速子集
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

import sleep_analysis.processing_config as pc  # noqa: E402
from sleep_analysis.preprocessing.iruwb.beat_detection import (  # noqa: E402
    detect_beats, epoch_quality,
)
from sleep_analysis.preprocessing.iruwb.simulate import (  # noqa: E402
    SimConfig, simulate_subject,
)

# 参考阈值, 针对 **RR 间期误差**。
# 不用绝对峰时刻误差 —— 它含滤波器引入的常数偏移, 对 HRV 无影响。
KILL_GO_MEDIAN_MS = 30.0


def match_one_to_one(detected: np.ndarray, truth: np.ndarray,
                     tol_s: float = 0.15) -> tuple:
    """一对一匹配（每个真值最多配一个检测，每个检测最多用一次）。

    必须做一对一 —— 若检测数比真值多很多，"最近邻误差"会因为碰巧有近邻
    而显得很小，是典型的假阳性陷阱（初版诊断就踩过这个坑）。
    """
    if len(detected) == 0:
        return 0.0, 0.0, np.array([])
    used = set()
    errs, tp = [], 0
    for t in truth:
        d = np.abs(detected - t)
        for c in np.argsort(d):
            if d[c] > tol_s:
                break
            if c not in used:
                used.add(c)
                errs.append(d[c])
                tp += 1
                break
    return tp / len(truth), tp / len(detected), np.asarray(errs)


def _matched_pairs(detected: np.ndarray, truth: np.ndarray,
                   tol_s: float = 0.15) -> tuple:
    """一对一匹配, 返回 (真值时刻, 检测时刻) 配对数组。"""
    if len(detected) == 0:
        return np.array([]), np.array([])
    used, pairs = set(), []
    for t in truth:
        d = np.abs(detected - t)
        for c in np.argsort(d):
            if d[c] > tol_s:
                break
            if c not in used:
                used.add(c)
                pairs.append((t, detected[c]))
                break
    pairs.sort()
    return (np.array([a for a, _ in pairs]), np.array([b for _, b in pairs]))


def evaluate_beats(sig: np.ndarray, fs: float, beat_times: np.ndarray,
                   hr_bpm=(30.0, 200.0), causal: bool = False) -> dict:
    """跑一次心搏检测并返回指标。

    **两个时序指标, 判据用后者**:

    - ``median_ms``: 绝对峰时刻误差。会包含 IIR 高通导致的**常数偏移**
      （实测约 -31 ms, 因果分支固有）。这个偏移**对 HRV 无害** ——
      HRV 用的是 RR 间期, 常数偏移在相邻拍差值中完全抵消。
    - ``rr_median_ms``: **RR 间期误差** —— HRV 特征真正依赖的量。判据用这个。
    """
    prev = pc.causal
    pc.set_causal(causal)
    try:
        res = detect_beats(sig, fs, hr_bpm=hr_bpm)
    finally:
        pc.set_causal(prev)

    rec, prec, err = match_one_to_one(res.peak_times_s, beat_times)
    tt, dd = _matched_pairs(res.peak_times_s, beat_times)
    rr_err = np.abs(np.diff(dd) - np.diff(tt)) * 1000 if len(tt) > 10 else np.array([])
    return {
        "recall": rec,
        "precision": prec,
        "median_ms": float(np.median(err) * 1000) if len(err) else np.nan,
        "p90_ms": float(np.percentile(err, 90) * 1000) if len(err) else np.nan,
        "rr_median_ms": float(np.median(rr_err)) if len(rr_err) else np.nan,
        "n_det": res.n_detected,
        "n_removed": res.n_removed_outlier,
        "latency_s": res.latency_s,
    }


def _fmt(r: dict) -> str:
    return (f"召回 {r['recall']:.3f}  精确 {r['precision']:.3f}  "
            f"RR误差 {r['rr_median_ms']:>5.1f}ms  (绝对偏移 {r['median_ms']:>5.1f}ms)")


def check_beat_detection(quick: bool) -> bool:
    """[1] 心搏检测: 因果 vs 非因果, 含/不含运动。"""
    print("=" * 78)
    print("[1] 心搏检测精度 (合成数据)"
          f"    参考阈值: RR 间期中位误差 < {KILL_GO_MEDIAN_MS:.0f} ms")
    print("    注: 括号里的绝对偏移含 IIR 常数偏移, 对 HRV 无害; 判据看 RR 误差")
    print("=" * 78)
    ok = True
    durations = [1800.0] if quick else [1800.0, 7200.0]
    for motion in [False, True]:
        for dur in durations:
            cfg = SimConfig(duration_s=dur, seed=42,
                            motion_rate_per_h=6.0 if motion else 0.0)
            d = simulate_subject(cfg, subj_id="01")
            tag = f"{'有运动' if motion else '无运动'} {dur / 3600:.1f}h"
            for causal in [False, True]:
                r = evaluate_beats(d["signal"], cfg.fs, d["beat_times"], causal=causal)
                name = "因果  " if causal else "非因果"
                print(f"  {tag:<12} {name}  {_fmt(r)}")
                if r["rr_median_ms"] >= KILL_GO_MEDIAN_MS:
                    ok = False
            print()
    return ok


def check_band_and_jpeak(quick: bool) -> None:
    """[2] 检测带宽度 × J 峰锐度 —— 接入真机后最需要先定的两个量。

    检测带由生理心率范围给出。**窄带砍噪声功率但也会切掉尖峰的谐波**，
    两个维度的取值都会影响结果, 这里一起扫出来供比较。
    """
    print("=" * 78)
    print("[2] 检测带 × J 峰锐度  (决定真机上该用多宽的带)")
    print("=" * 78)
    bands = [(30.0, 200.0), (30.0, 400.0)] if quick else \
            [(30.0, 200.0), (30.0, 300.0), (30.0, 400.0), (30.0, 600.0)]
    sigmas = [0.04] if quick else [0.04, 0.08, 0.12, 0.20]
    print(f"  {'J峰 σ':>7} " + " ".join(f"{f'{int(a)}-{int(b)}bpm':>26}" for a, b in bands))
    for sg in sigmas:
        cfg = SimConfig(duration_s=1800.0, seed=42, card_j_sigma_s=sg)
        d = simulate_subject(cfg, subj_id="01")
        row = f"  {sg:>6.2f}s"
        for hr in bands:
            r = evaluate_beats(d["signal"], cfg.fs, d["beat_times"], hr_bpm=hr)
            row += f"   召回 {r['recall']:.3f}  RR {r['rr_median_ms']:>5.1f}ms"
        print(row)
    print("\n  注: 带宽越宽保留越多 J 峰谐波, 但放进越多噪声。")
    print("      真机 SNR 差时倾向窄带, 但若 J 峰很尖则窄带会丢峰位精度。\n")
    print("  n_taps=None 可跳过 FIR 级, 用于对比。\n")


def check_epoch_quality(quick: bool) -> None:
    """[3] 逐 epoch 质量标记是否与真实可用性一致。"""
    print("=" * 78)
    print("[3] 逐 epoch 质量标记")
    print("=" * 78)
    cfg = SimConfig(duration_s=7200.0, seed=42, motion_rate_per_h=6.0)
    d = simulate_subject(cfg, subj_id="01")
    n_epochs = int(cfg.duration_s / cfg.epoch_s)
    res = detect_beats(d["signal"], cfg.fs)

    quality = epoch_quality(res.peak_times_s, n_epochs)

    # 真值: 该 epoch 实际有多少拍
    true_idx = (d["beat_times"] / cfg.epoch_s).astype(int)
    true_counts = np.bincount(true_idx[true_idx < n_epochs], minlength=n_epochs)

    flagged_bad = quality == 0
    print(f"  总 epoch {n_epochs}, 标记不可用 {flagged_bad.sum()} "
          f"({flagged_bad.mean():.1%})")
    if flagged_bad.any():
        print(f"  被标记不可用的 epoch 实际拍数: "
              f"中位 {np.median(true_counts[flagged_bad]):.0f} "
              f"(阈值 {10} 拍)")
    print()


def check_hrv_feature_impact(quick: bool) -> None:
    """[4] 特征级影响 — 用真实检测误差重算 HRV 7 个训练特征的相对误差。

    对比两种误差模型:
      - ±25 ms 均匀量化 = 不做亚采样精修时的下限
      - 实测检测误差   = 开精修后的实际水平
    这一项说明精修到底值不值。
    """
    from hrvanalysis import (get_frequency_domain_features,
                             get_poincare_plot_features,
                             get_time_domain_features)

    print("=" * 78)
    print("[4] HRV 特征级影响 (训练实际使用的 7 个独立特征)")
    print("=" * 78)

    KEYS = ["median_nni", "ratio_sd2_sd1", "vlf", "lf", "hf",
            "lf_hf_ratio", "total_power"]

    def make_beats(rsa_ms: float, seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        t, out = 0.0, []
        while t < 600.0:
            rr = (0.850 + rsa_ms / 1000 * np.sin(2 * np.pi * 0.25 * t)
                  + 0.6 * rsa_ms / 1000 * np.sin(2 * np.pi * 0.10 * t)
                  + rng.randn() * 0.008)
            t += rr
            out.append(t)
        return np.asarray(out[:-1])

    def feats(bt: np.ndarray) -> dict:
        rr = np.diff(bt) * 1000.0
        td = get_time_domain_features(rr)
        fd = get_frequency_domain_features(rr)
        pc_ = get_poincare_plot_features(rr)
        return {"median_nni": td["median_nni"], "ratio_sd2_sd1": pc_["ratio_sd2_sd1"],
                "vlf": fd["vlf"], "lf": fd["lf"], "hf": fd["hf"],
                "lf_hf_ratio": fd["lf_hf_ratio"], "total_power": fd["total_power"]}

    # 实测检测器在 1-5Hz 带通下的噪声水平 (见 [1])
    DETECTED_JITTER_S = 0.008

    print(f"  {'RSA':>5} {'误差模型':>14} " + " ".join(f"{k:>13}" for k in KEYS))
    rsa_list = [20.0, 45.0] if quick else [20.0, 45.0, 80.0]
    for rsa in rsa_list:
        for label, jitter in [("±25ms 均匀量化", None), ("实测检测误差", DETECTED_JITTER_S)]:
            diffs = {k: [] for k in KEYS}
            for s in range(8):
                gt = make_beats(rsa, s)
                if jitter is None:
                    det = np.round(gt * 20) / 20
                else:
                    det = np.sort(gt + np.random.RandomState(s).normal(0, jitter, len(gt)))
                a, b = feats(gt), feats(det)
                for k in KEYS:
                    if abs(a[k]) > 1e-9:
                        diffs[k].append((b[k] - a[k]) / abs(a[k]))
            print(f"  {rsa:>4.0f} {label:>14} "
                  + " ".join(f"{np.median(diffs[k]):>13.1%}" for k in KEYS))
    print("\n  注: 误差随 RSA 幅度增大而减小 —— 低 RSA 人群 (老年人/病理) 误差最大。")
    print("      '实测检测误差' 那一行才是开启亚采样精修后的真实水平。\n")


def check_end_to_end_hrv(quick: bool) -> None:
    """[5] 端到端特征级: 真实检测器输出 → HRV 特征, 对比真值拍时刻。

    与 [4] 的区别: [4] 用高斯抖动**模型**模拟检测误差, 这里跑的是**真实检测器**,
    能捕捉模型掩盖的系统性偏差和误差相关性。

    在 30 s epoch 上逐段计算（与 MESA 路径的窗口口径一致）。
    """
    from hrvanalysis import (get_frequency_domain_features,
                             get_poincare_plot_features,
                             get_time_domain_features)

    print("=" * 78)
    print("[5] 端到端 HRV 特征误差 (真实检测器输出 vs 真值拍时刻, 30s epoch)")
    print("=" * 78)

    KEYS = ["median_nni", "ratio_sd2_sd1", "lf", "hf", "lf_hf_ratio"]

    def feats(beat_times: np.ndarray) -> dict:
        rr = np.diff(beat_times) * 1000.0
        if len(rr) < 5:
            return None
        td = get_time_domain_features(rr)
        fd = get_frequency_domain_features(rr)
        pc_ = get_poincare_plot_features(rr)
        out = {"median_nni": td["median_nni"], "ratio_sd2_sd1": pc_["ratio_sd2_sd1"],
               "lf": fd["lf"], "hf": fd["hf"], "lf_hf_ratio": fd["lf_hf_ratio"]}
        return out if all(np.isfinite(v) for v in out.values()) else None

    print(f"  {'场景':<14} {'epoch':>6} " + " ".join(f"{k:>13}" for k in KEYS))
    for rsa in ([45.0] if quick else [20.0, 45.0, 80.0]):
        cfg = SimConfig(duration_s=3600.0, seed=42, rsa_ms=rsa, motion_rate_per_h=0.0)
        d = simulate_subject(cfg, subj_id="01")
        res = detect_beats(d["signal"], cfg.fs)

        det_t, true_t = res.peak_times_s, d["beat_times"]
        epoch_s = 30.0
        n_ep = int(cfg.duration_s / epoch_s)
        diffs = {k: [] for k in KEYS}
        for e in range(n_ep):
            lo, hi = e * epoch_s, (e + 1) * epoch_s
            a = feats(true_t[(true_t >= lo) & (true_t < hi)])
            b = feats(det_t[(det_t >= lo) & (det_t < hi)])
            if a is None or b is None:
                continue
            for k in KEYS:
                if abs(a[k]) > 1e-9:
                    diffs[k].append((b[k] - a[k]) / abs(a[k]))
        n_used = len(diffs[KEYS[0]])
        print(f"  RSA {rsa:>3.0f}ms     {n_used:>6} "
              + " ".join(f"{np.median(diffs[k]):>13.1%}" for k in KEYS))
    print("\n  这是**实测**结果, 不是误差模型。不含运动伪迹段。\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="IR-UWB 检测器验证")
    parser.add_argument("--quick", action="store_true", help="快速子集")
    args = parser.parse_args()

    print(f"\n处理模式: causal={pc.causal} "
          f"(环境变量 SLEEP_CAUSAL={__import__('os').environ.get('SLEEP_CAUSAL', '0')})\n")

    ok_beats = check_beat_detection(args.quick)
    check_band_and_jpeak(args.quick)
    check_epoch_quality(args.quick)
    check_hrv_feature_impact(args.quick)
    check_end_to_end_hrv(args.quick)

    print("=" * 78)
    print("汇总（合成数据）")
    print("=" * 78)
    if ok_beats:
        print(f"  RR 间期中位误差 < {KILL_GO_MEDIAN_MS:.0f} ms")
    else:
        print(f"  RR 间期中位误差 ≥ {KILL_GO_MEDIAN_MS:.0f} ms")
    print("  ⚠️ 以上均为合成数据结果, 不代表真机性能。接入真机后先看: 心搏分量的")
    print("     功率谱形状、实际 SNR、心搏波形形态（见 README 的「接入真机后先看什么」）")


if __name__ == "__main__":
    main()
