"""生成 IR-UWB 20 Hz 仿真数据集（用于在真实数据采集前把预处理管线跑通）。

产出布局（模拟真实采集系统的交付格式）::

    <out_dir>/
      Vp_01/
        physio_01.csv         # 20 Hz 单通道胸腔位移, 列: time_offset_s, phase
        labels_01.csv         # 30 s epoch 睡眠分期 (真实数据来自 PSG 评分)
        events_01.csv         # 事件真值: breath / beat (亚采样) + motion 时刻
        sim_config_01.json    # 本次仿真的参数快照 (可追溯)

``events_*.csv`` 是仿真独有的资产 —— 有了它才能把"检测器误差"和"特征计算误差"
分开度量。真实数据没有这一列, 但预处理管线不需要它。

用法
----
    # 1 个被试, 8 小时整夜 (默认)
    python experiments/data_handling/simulate_iruwb.py --out-dir /tmp/iruwb_raw

    # 10 个被试, 每个 8 小时
    python experiments/data_handling/simulate_iruwb.py --n-subjects 10 --out-dir /tmp/iruwb_raw

    # 快速试跑: 1 个被试 1 小时
    python experiments/data_handling/simulate_iruwb.py --hours 1 --out-dir /tmp/iruwb_raw

    # 生成"参考基线"版本 (200 Hz, 同一组事件时刻) 用于量化 20 Hz 的损失
    python experiments/data_handling/simulate_iruwb.py --fs 200 --out-dir /tmp/iruwb_ref
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

from sleep_analysis.preprocessing.iruwb.simulate import SimConfig, simulate_subject  # noqa: E402

DEFAULT_START = "2026-01-01 22:00:00"


def write_subject(out_dir: Path, subj_id: str, data: dict, start_time: str) -> None:
    subj_dir = out_dir / f"Vp_{subj_id}"
    subj_dir.mkdir(parents=True, exist_ok=True)

    fs = data["fs"]
    n = len(data["signal"])
    t = np.arange(n) / fs

    pd.DataFrame({
        "time_offset_s": t,
        "phase": data["signal"],       # 胸腔位移 (mm), 呼吸+心搏混叠
    }).to_csv(subj_dir / f"physio_{subj_id}.csv", index=False)

    stages = data["stages"]
    pd.DataFrame({
        "epoch": np.arange(len(stages["5stage"])),
        "sleep": stages["sleep"],
        "5stage": stages["5stage"],
        "4stage": stages["4stage"],
        "3stage": stages["3stage"],
    }).to_csv(subj_dir / f"labels_{subj_id}.csv", index=False)

    # 事件真值: 两个点过程用不同前缀存在一张长表里, 便于人工查看
    ev = pd.concat([
        pd.DataFrame({"event": "breath", "time_s": data["breath_times"]}),
        pd.DataFrame({"event": "beat", "time_s": data["beat_times"]}),
        pd.DataFrame({"event": "motion", "time_s": data["motion_times"]}),
    ], ignore_index=True)
    ev.to_csv(subj_dir / f"events_{subj_id}.csv", index=False)

    cfg = dict(data["config"])
    cfg["start_time"] = start_time
    (subj_dir / f"sim_config_{subj_id}.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="生成 IR-UWB 20 Hz 仿真数据")
    p.add_argument("--out-dir", type=Path, required=True, help="输出目录")
    p.add_argument("--n-subjects", type=int, default=1)
    p.add_argument("--hours", type=float, default=8.0, help="每个被试的时长 (小时)")
    p.add_argument("--fs", type=float, default=20.0,
                   help="采样率。默认 20 (真实硬件)。传更高值可生成参考基线")
    p.add_argument("--seed", type=int, default=42, help="基础随机种子")
    p.add_argument("--rsa-ms", type=float, default=45.0,
                   help="RSA (呼吸性窦性心律不齐) 幅度 (ms)。仿真用参考值: "
                        "20(低) / 45(中) / 80(高)")
    p.add_argument("--motion-rate", type=float, default=6.0,
                   help="每小时运动伪迹事件数, 0 = 无运动")
    p.add_argument("--j-sigma", type=float, default=0.04,
                   help="心搏 J 峰宽度 (秒), 即仿真里心搏波形的锐度。"
                        "真机的心搏波形形态未知, 这个值只是一个建模假设")
    p.add_argument("--start-time", type=str, default=DEFAULT_START,
                   help="录制起始时刻 (写入 sim_config, 决定 epoch 时间轴)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"生成 {args.n_subjects} 个被试 × {args.hours}h @ {args.fs} Hz → {out_dir}")
    t0 = time.time()
    for i in range(args.n_subjects):
        subj_id = f"{i + 1:02d}"
        cfg = SimConfig(
            fs=args.fs,
            duration_s=args.hours * 3600.0,
            seed=args.seed + i,
            rsa_ms=args.rsa_ms,
            motion_rate_per_h=args.motion_rate,
            card_j_sigma_s=args.j_sigma,
        )
        data = simulate_subject(cfg, subj_id=subj_id)
        write_subject(out_dir, subj_id, data, args.start_time)
        if not args.quiet:
            amp = data["config"]["resp_amp_mm"] / data["config"]["card_amp_mm"]
            print(f"  Vp_{subj_id}: {len(data['breath_times'])} 次呼吸 / "
                  f"{len(data['beat_times'])} 拍  幅度比 {amp:.0f}x  "
                  f"({len(data['signal']) * 8 / 1e6:.1f} MB)")

    elapsed = time.time() - t0
    total_mb = sum(f.stat().st_size for f in out_dir.rglob("*.csv")) / 1e6
    print(f"完成: {elapsed:.1f}s, 共 {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
