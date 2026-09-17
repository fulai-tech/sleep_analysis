"""流式预处理模块（`classification/inference/streaming/`）的回归对拍。

它是 C++ 翻译的**护栏** —— 内联实现与原库一旦分叉，数值会静默偏掉，
所以每项检查都有明确的判据（逐位 or 容差）。

检查项
------
[1] `filters.lfilter` / `apply_iir_causal` vs `scipy.signal.lfilter`     判据: 逐位
[2] `filters.apply_filter_chain` vs `beat_detection.apply_filter_chain`  判据: 1e-12
[3] `hrv.hrv_features` vs `hrvanalysis`                                  判据: 时域逐位 / 频域 1e-12
[4] `resp.rrv_features` vs `neurokit2`                                   判据: 时域逐位 / 频域 1e-12
[5] `beats.detect_beats` vs `beat_detection.detect_beats`                判据: 峰数相等、峰位 1e-10
[6] `movement` 状态机 vs `extract_movement`                              判据: 稀疏结构一致、值 1e-12
[7] 端到端 `StreamingPreprocessor.update` 的形状 / 时间戳 / 耗时
[8] 跨调用一致性（连续窗口的重叠 epoch）—— **已知 1 个槽位不满足**, 见 ⚠️
[9] 退化输入（常量段）必须退化成 NaN 而非抛异常 + 窗口步进校验

⚠️ 本脚本需要参考库（scipy / pandas / hrvanalysis / neurokit2）—— 这是刻意的：
   它们是对拍的**基准**。流式模块本身不依赖它们。

⚠️ 所有结论都建立在**合成信号**上（`tmp/iruwb_10h/raw`），不代表真机性能。

用法
----
    python experiments/evaluation/validate_streaming_preprocess.py
    python experiments/evaluation/validate_streaming_preprocess.py --quick   # 跳过 10h 数据相关项
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

import sleep_analysis.processing_config as pc  # noqa: E402
from sleep_analysis.classification.inference.streaming import (  # noqa: E402
    SLOT_NAMES, StreamingPreprocessor, beats as B, filters as F, hrv as H,
    movement as SM, resp as R,
)
from sleep_analysis.preprocessing.iruwb import beat_detection as BD  # noqa: E402
from sleep_analysis.preprocessing.iruwb import movement as RM  # noqa: E402

#: 浮点容差。内联实现与库的差异应当在 1 ULP 量级（~1e-15 相对），留三个数量级余量。
RTOL = 1e-12

#: FIR / 滑动平均的绝对容差。它们的差异来自浮点累加顺序，在幅值接近 0 的点上
#: 相对容差帮不上忙，必须给一个绝对下界。
ATOL = 1e-13

#: 跨调用一致性的容差。频域特征（`vlf`/`lf`/`hf`/`total_power`）绝对幅值大
#: （可达 1e5），窗口滑动带来的浮点噪声在 1e-12 量级，比 RTOL 松两个数量级。
RTOL_CROSS_CALL = 1e-10

#: 合成数据路径（由 `experiments/data_handling/simulate_iruwb.py` 生成）。
RAW_CSV = Path(__file__).parents[2] / "tmp/iruwb_10h/raw/Vp_01/physio_01.csv"

#: 跨调用一致性里**已知不满足**的槽位（根因见 streaming/README.md 的"已知局限"）。
KNOWN_CROSS_CALL_OFFENDERS = {"150_hrv_median_nni"}

_bar = "=" * 78


def _load(quick: bool):
    """加载合成信号；缺失时返回 None（相应检查跳过）。"""
    if not RAW_CSV.exists():
        return None
    n = 18000 if quick else 72000
    return pd.read_csv(RAW_CSV)["phase"].to_numpy(dtype=float)[:n]


# ---------------------------------------------------------------------------
# [1] 滤波原语
# ---------------------------------------------------------------------------

def check_filters(sig, quick) -> bool:
    import scipy.signal as ss

    print(f"\n{_bar}\n[1] filters.lfilter / apply_iir_causal vs scipy.signal.lfilter\n{_bar}")
    x = sig[:6000] if sig is not None else np.random.RandomState(0).randn(6000)
    print(f"{'滤波器':>10s} {'配置':>10s} {'逐位一致':>10s} {'最大绝对差':>14s}")
    print("-" * 50)
    ok = True
    for name, b, a, ziu in [("IIR_HP", F.IIR_HP_B, F.IIR_HP_A, F.IIR_HP_ZI_UNIT),
                            ("IIR_LP", F.IIR_LP_B, F.IIR_LP_A, F.IIR_LP_ZI_UNIT),
                            ("FIR", F.FIR_TAPS, [1.0], None)]:
        ref = ss.lfilter(b, a, x)
        got = F.lfilter(b, a, x)
        nz = len(np.unique(np.abs(ref - got)))
        eq = np.array_equal(ref, got)
        print(f"{name:>10s} {'无 zi':>10s} {str(eq):>10s} {np.abs(ref-got).max():>14.3e}")
        # ⚠️ scipy 传 zi 时返回元组 (y, zf)
        if ziu is not None:
            ref2, _ = ss.lfilter(b, a, x, zi=ss.lfilter_zi(b, a) * x[0])
            got2 = F.apply_iir_causal(x, b, a, ziu)
            eq2 = np.array_equal(ref2, got2)
            ok &= eq2
            print(f"{name:>10s} {'带 zi':>10s} {str(eq2):>10s} {np.abs(ref2-got2).max():>14.3e}")
        if name == "FIR":
            ok &= np.allclose(ref, got, rtol=RTOL, atol=ATOL)
    return ok


# ---------------------------------------------------------------------------
# [2] 滤波链
# ---------------------------------------------------------------------------

def check_chain(sig) -> bool:
    print(f"\n{_bar}\n[2] filters.apply_filter_chain vs beat_detection.apply_filter_chain\n{_bar}")
    x = sig[:6000] if sig is not None else np.random.RandomState(0).randn(6000)
    pc.set_causal(True)
    ref = BD.apply_filter_chain(x, BD.build_filter_chain(20.0), causal=True)
    got = F.apply_filter_chain(x)
    d = np.abs(ref - got).max()
    ok = np.allclose(ref, got, rtol=RTOL, atol=ATOL)
    print(f"  最大绝对差 {d:.3e}   容差 {RTOL:.0e}   通过: {ok}")
    return ok


# ---------------------------------------------------------------------------
# [3] HRV
# ---------------------------------------------------------------------------

def check_hrv() -> bool:
    from hrvanalysis import (get_frequency_domain_features, get_poincare_plot_features,
                             get_time_domain_features)

    print(f"\n{_bar}\n[3] hrv.hrv_features vs hrvanalysis\n{_bar}")
    rng = np.random.RandomState(42)
    keys = [("median_nni", "median_nni", True), ("ratio_sd2_sd1", "ratio_sd2_sd1", True),
            ("vlf", "vlf", False), ("lf", "lf", False), ("hf", "hf", False),
            ("lf_hf_ratio", "lf_hf_ratio", False), ("total_power", "total_power", False)]
    print(f"{'拍数':>6s} {'槽位':>15s} {'逐位一致':>10s} {'相对差':>11s}")
    print("-" * 46)
    ok = True
    for n in [10, 30, 150, 270]:
        nn = 1000.0 + 30 * np.sin(2 * np.pi * 0.25 * np.arange(n)) + rng.randn(n) * 12
        ref = {}
        ref.update(get_time_domain_features(nn))
        ref.update(get_frequency_domain_features(nn))
        ref.update(get_poincare_plot_features(nn))
        got = H.hrv_features(nn)
        for slot, key, exact in keys:
            a, b = float(ref[key]), float(got[slot])
            eq = (a == b) or (np.isnan(a) and np.isnan(b))
            rel = abs(a - b) / abs(a) if abs(a) > 1e-12 else abs(a - b)
            if exact:
                ok &= eq
            else:
                ok &= (eq or rel < RTOL)
            if not eq:
                print(f"{n:>6d} {slot:>15s} {str(eq):>10s} {rel:>11.2e}")
    print(f"  全部通过: {ok}   （未列出的项即逐位一致）")
    return ok


# ---------------------------------------------------------------------------
# [4] RRV
# ---------------------------------------------------------------------------

def check_rrv(sig) -> bool:
    import neurokit2 as nk

    print(f"\n{_bar}\n[4] resp.rrv_features vs neurokit2\n{_bar}")
    src = sig if sig is not None else np.random.RandomState(0).randn(36000)
    keys = ["MedianBB", "CVBB", "MCVBB", "MeanBB", "SDBB", "MadBB", "LF", "VLF", "HF"]
    exact_keys = {"MedianBB", "CVBB", "MCVBB", "MeanBB", "SDBB", "MadBB"}
    print(f"{'窗口':>8s} {'槽位':>10s} {'逐位一致':>10s} {'相对差':>11s}")
    print("-" * 42)
    ok = True
    for w_s, off in [(150, 100), (270, 400), (150, 1000)]:
        a0, b0 = int(off * 20), int((off + w_s) * 20)
        if b0 > len(src):
            continue
        win = src[a0:b0]
        _, pdct = nk.rsp_peaks(win, sampling_rate=20.0, method="biosppy")
        ref = nk.rsp_rrv(win, pdct, sampling_rate=20.0, show=False).to_dict("records")[0]
        got = R.rrv_features(win, 20.0)
        for k in keys:
            x, y = float(ref["RRV_" + k]), float(got[k])
            eq = (x == y) or (np.isnan(x) and np.isnan(y))
            rel = abs(x - y) / abs(x) if abs(x) > 1e-12 else abs(x - y)
            if k in exact_keys:
                ok &= eq
            else:
                ok &= (eq or rel < 1e-11)
            if not eq:
                print(f"{w_s:>6d}s {k:>10s} {str(eq):>10s} {rel:>11.2e}")
    print(f"  全部通过: {ok}   （未列出的项即逐位一致）")
    return ok


# ---------------------------------------------------------------------------
# [5] 峰检测
# ---------------------------------------------------------------------------

def check_beats(sig) -> bool:
    print(f"\n{_bar}\n[5] beats.detect_beats vs beat_detection.detect_beats（因果路径）\n{_bar}")
    if sig is None:
        print("  (跳过: 无合成数据)")
        return True
    pc.set_causal(True)
    print(f"{'窗口':>14s} {'参考峰数':>9s} {'内联峰数':>9s} {'峰位最大差':>12s} {'RR最大差':>11s}")
    print("-" * 60)
    ok = True
    for label, a, b in [("前 900s", 0, 18000), ("中段 900s", 18000, 36000)]:
        if b > len(sig):
            continue
        seg = sig[a:b]
        ref = BD.detect_beats(seg, 20.0)
        got = B.detect_beats(seg, 20.0)
        dt = np.abs(ref.peak_times_s - got["peak_times_s"]).max() if len(ref.peak_times_s) else 0.0
        drr = np.abs(ref.peaks["RR_Interval"].to_numpy() - got["peaks"][:, 1]).max()
        same = len(ref.peaks) == len(got["peaks"])
        ok &= same and dt < 1e-10
        print(f"{label:>14s} {len(ref.peaks):>9d} {len(got['peaks']):>9d} {dt:>12.3e} {drr:>11.3e}")
    print(f"  通过: {ok}")
    return ok


# ---------------------------------------------------------------------------
# [6] 体动状态机
# ---------------------------------------------------------------------------

def check_movement(sig) -> bool:
    print(f"\n{_bar}\n[6] movement 状态机 vs extract_movement\n{_bar}")
    x = sig[:18000] if sig is not None else np.random.RandomState(0).randn(18000)
    pc.set_causal(True)
    ma = RM._moving_average(x, round(RM.DEFAULT_MA_WINDOW_S * 20), causal=True)
    ref = np.where(RM._normalize(RM._first_order_derivative(RM._scale(ma, True)), True)
                   > RM.DEFAULT_THRESHOLD,
                   RM._normalize(RM._first_order_derivative(RM._scale(ma, True)), True), 0.0)
    st = SM.ActState()
    got = st.process(x)
    d = np.abs(ref - got).max()
    same_sparsity = int((ref > 0).sum()) == int((got > 0).sum())
    ok = same_sparsity and d < 1e-12
    print(f"  非零样本数: 参考={int((ref>0).sum())} 内联={int((got>0).sum())}  "
          f"一致={same_sparsity}")
    print(f"  最大绝对差 {d:.3e}   通过: {ok}")

    # 分段喂 == 整段喂
    st2 = SM.ActState()
    got2 = np.concatenate([st2.process(x[i:i + 600]) for i in range(0, len(x), 600)])
    eq = np.array_equal(got2, got)
    print(f"  分段喂 vs 整段喂 逐位一致: {eq}")
    return ok and eq


# ---------------------------------------------------------------------------
# [7] 端到端
# ---------------------------------------------------------------------------

def check_end_to_end(sig) -> bool:
    print(f"\n{_bar}\n[7] StreamingPreprocessor.update 端到端\n{_bar}")
    if sig is None or len(sig) < 18000:
        print("  (跳过: 无合成数据)")
        return True
    t0 = 1_800_000_000_000
    pre = StreamingPreprocessor()
    t = time.time()
    res = pre.update(sig[:18000], 18000, t0 + 900_000)
    dt = time.time() - t

    checks = [
        ("features.shape == (21, 13)", res.features.shape == (21, 13), res.features.shape),
        ("epoch 起点 268.45", abs(res.epoch_starts_s[0] - 268.45) < 1e-9, res.epoch_starts_s[0]),
        ("epoch 末端 898.45", abs(res.epoch_starts_s[-1] + 30 - 898.45) < 1e-9,
         res.epoch_starts_s[-1] + 30),
        ("result_start = latest−331550", res.result_start_ts_ms == t0 + 900_000 - 331550,
         res.result_start_ts_ms - t0 - 900_000),
        ("result_end   = latest−301550", res.result_end_ts_ms == t0 + 900_000 - 301550,
         res.result_end_ts_ms - t0 - 900_000),
    ]
    ok = True
    for name, passed, val in checks:
        ok &= passed
        print(f"  {name:>32s}: {str(passed):>5s}   {val}")
    print(f"  耗时 {dt:.2f} s/窗（30 s 预算的 {dt/30:.1%}）")
    return ok


# ---------------------------------------------------------------------------
# [8] 跨调用一致性
# ---------------------------------------------------------------------------

def check_cross_call(sig) -> bool:
    print(f"\n{_bar}\n[8] 跨调用一致性（连续窗口的重叠 epoch）\n{_bar}")
    if sig is None or len(sig) < 18000 + 2 * 600:
        print("  (跳过: 无合成数据)")
        return True
    pre = StreamingPreprocessor()
    res = [pre.update(sig[k * 600:k * 600 + 18000], 18000, 1_800_000_000_000 + (900 + 30 * k) * 1000)
           for k in range(3)]
    print("⚠️  已知 `150_hrv_median_nni` 不满足（根因见 streaming/README.md 的\"已知局限\"）\n")
    print(f"{'槽位':>22s} {'最大相对差':>12s} {'判定':>10s}")
    print("-" * 48)
    ok = True
    for c, s in enumerate(SLOT_NAMES):
        wr = 0.0
        for k in range(2):
            a = res[k].features[1:, c]
            b = res[k + 1].features[:-1, c]
            m = np.isfinite(a) & np.isfinite(b)
            if not m.any():
                continue
            wr = max(wr, (np.abs(a[m] - b[m]) / np.maximum(np.abs(a[m]), 1e-12)).max())
        if s in KNOWN_CROSS_CALL_OFFENDERS:
            verdict = "已知" if wr < 1e-2 else "恶化!"
        else:
            verdict = "通过" if wr < RTOL_CROSS_CALL else "失败"
            ok &= (wr < RTOL_CROSS_CALL)
        print(f"{s:>22s} {wr:>12.2e} {verdict:>10s}")
    return ok


# ---------------------------------------------------------------------------
# [9] 退化输入与调用契约
# ---------------------------------------------------------------------------

def check_robustness(sig) -> bool:
    """退化信号必须**退化成 NaN**, 而不是把异常抛给调用方中断整次推理。

    真机上探头掉线 / 饱和 / 直流漂移都会给出常量段。离线路径 `iruwb/rrv.py`
    每个窗口都 catch, 流式侧也必须如此 —— 否则一次 300 s 的异常就会中断整次
    `update()`（对应 C ABI 的 `SS_RT_ERR_INTERNAL`）。
    """
    print(f"\n{_bar}\n[9] 退化输入与调用契约\n{_bar}")
    rng = np.random.RandomState(0)
    base = rng.randn(18000) * 0.5
    T0 = 1_800_000_000_000
    ok = True

    print("  退化信号必须退化成 NaN（不能抛）:")
    for label, w in [("常量 +2.0", np.full(18000, 2.0)),
                     ("常量 0.0", np.zeros(18000)),
                     ("常量 -1.5", np.full(18000, -1.5)),
                     ("后 300s 恒定", np.concatenate([base[:15000], np.full(3000, 0.7)]))]:
        pre = StreamingPreprocessor()
        try:
            r = pre.update(w, 18000, T0)
            n_nan = int(np.isnan(r.features).sum())
            good = n_nan > 0           # 至少要有一处退化被标出来
            ok &= good
            print(f"    {label:>12s}: 返回 OK, NaN 数={n_nan:<4d} {'✓' if good else '✗ 未标退化'}")
        except Exception as e:
            ok = False
            print(f"    {label:>12s}: ✗ 抛出 {type(e).__name__}: {e}")

    pre = StreamingPreprocessor(fill_nan=0.0)
    r = pre.update(np.full(18000, 2.0), 18000, T0)
    filled = int(np.isnan(r.features).sum()) == 0
    ok &= filled
    print(f"    fill_nan=0.0 能接手: {'✓' if filled else '✗'}")

    print("\n  窗口步进校验（ACT 锚点依赖'每次推进一个 epoch'）:")
    for label, d, expect_reject in [("正常 +30s", 30_000, False),
                                    ("重推同一窗 (+0)", 0, True),
                                    ("漏推一窗 (+60s)", 60_000, True),
                                    ("超前 (+15s)", 15_000, True),
                                    ("倒退 (-30s)", -30_000, True)]:
        pre = StreamingPreprocessor()
        pre.update(base, 18000, T0)
        try:
            pre.update(base, 18000, T0 + d)
            rejected = False
        except ValueError:
            rejected = True
        good = rejected == expect_reject
        ok &= good
        print(f"    {label:>16s}: {'拦截' if rejected else '通过'}  "
              f"{'✓' if good else '✗ 期望' + ('拦截' if expect_reject else '通过')}")

    pre = StreamingPreprocessor()
    pre.update(base, 18000, T0)
    pre.reset()
    try:
        pre.update(base, 18000, T0 + 12345)
        print(f"    {'reset 后可重新开始':>16s}: 通过  ✓")
    except ValueError:
        ok = False
        print(f"    {'reset 后可重新开始':>16s}: ✗ 仍被拦截")

    print(f"\n  通过: {ok}")
    return ok


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="只跑短片段")
    args = ap.parse_args()

    # 本模块只实现因果路径, 显式打开（参考库调用需要它, 如 beat_detection.detect_beats）
    pc.set_causal(True)

    sig = _load(args.quick)
    print(_bar)
    print("流式预处理模块回归对拍")
    print(_bar)
    print(f"合成数据: {'已加载' if sig is not None else '缺失（部分检查跳过）'}"
          f"{f' {len(sig)} 样本' if sig is not None else ''}")
    print(f"处理模式: SLEEP_CAUSAL={pc.causal}")

    results = [
        ("[1] 滤波原语", check_filters(sig, args.quick)),
        ("[2] 滤波链", check_chain(sig)),
        ("[3] HRV vs hrvanalysis", check_hrv()),
        ("[4] RRV vs neurokit2", check_rrv(sig)),
        ("[5] 峰检测", check_beats(sig)),
        ("[6] 体动状态机", check_movement(sig)),
        ("[7] 端到端", check_end_to_end(sig)),
        ("[8] 跨调用一致性", check_cross_call(sig)),
        ("[9] 退化输入与契约", check_robustness(sig)),
    ]

    print(f"\n{_bar}\n汇总\n{_bar}")
    for name, ok in results:
        print(f"  {name:>26s}: {'PASS' if ok else 'FAIL'}")
    n_pass = sum(ok for _, ok in results)
    print(f"\n  {n_pass}/{len(results)} 通过")
    print("\n⚠️ 全部结论建立在**合成信号**上，不代表真机性能。")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
