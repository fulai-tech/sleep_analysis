"""生成混合数据集的 split 文件（MESA_Sleep + SHHS1 + SHHS2 多数据集格式）。

输出格式 (key 与 MixedDataset 的 subj_id 前缀一致, 小写无下划线):
    {"mesasleep": {"train": [...], "val": [...], "test": [...]},
     "shhs1":    {"train": [...], "val": [...], "test": [...]},
     "shhs2":    {"train": [...], "val": [...], "test": [...]}}

划分规则 (与 LSTM_paper_params.py 的自动划分逻辑一致, 保证 split 文件 == 自动划分):
  - MESA: 直接复用 splits/split_mesa_1121_20260806.json 的名单 (保持与之前实验一致)
  - SHHS1/SHHS2: 按 nsrrid 联合划分 (共享参与者不会跨 train/val/test),
    np.random.seed(42) shuffle → 80/20 → 80/20 (与 LSTM_paper_params L157-193 相同)

用法:
    python experiments/data_handling/make_splits.py [--output splits/split_mixed_YYYYMMDD.json]
      [--shhs1-path DIR] [--shhs2-path DIR]   # 覆盖 study_data.json 的 SHHS 数据路径 (测试用)

依赖: 各数据集的 features_full_combined 目录已生成 (ID 集合从文件列表提取, 与 create_index 一致)。
"""

import argparse
import json
from pathlib import Path

import numpy as np

import sleep_analysis.processing_config as pc  # noqa: F401  (确保包可导入)
# import sleep_analysis.datasets 触发数据集自注册 (经 registry 导入隐式触发)


def _ids_from_dataset(ds) -> list:
    """从数据集类的 index 提取被试 ID (create_index 的 ID 提取逻辑由数据集类自己声明)。"""
    return [str(s) for s in ds.index["subj_id"]]


def split_shhs_joint(shhs1_ids: list, shhs2_ids: list, seed: int = 42):
    """复现 LSTM_paper_params.py L157-193 的 SHHS1/2 联合划分 (按 nsrrid, 防同人跨集)。"""
    pids = sorted(set(shhs1_ids) | set(shhs2_ids))
    rng = np.random.RandomState(seed)
    perm = pids.copy()
    rng.shuffle(perm)
    n_test = max(1, int(len(perm) * 0.2))
    test_pids = set(perm[:n_test])
    trainval = perm[n_test:]
    n_val = max(1, int(len(trainval) * 0.2))
    val_pids = set(trainval[:n_val])
    train_pids = set(trainval[n_val:])

    assert train_pids.isdisjoint(val_pids) and train_pids.isdisjoint(test_pids) and val_pids.isdisjoint(test_pids)

    def _part(ids):
        return {
            "train": [i for i in ids if i in train_pids],
            "val": [i for i in ids if i in val_pids],
            "test": [i for i in ids if i in test_pids],
        }

    return _part(shhs1_ids), _part(shhs2_ids), len(train_pids), len(val_pids), len(test_pids)


def main():
    parser = argparse.ArgumentParser(description="生成混合数据集 split 文件")
    parser.add_argument("--output", type=Path, default=None,
                        help="输出路径 (默认 splits/split_mixed_YYYYMMDD.json)")
    parser.add_argument("--mesa-split", type=Path,
                        default=Path("splits/split_mesa_1121_20260806.json"),
                        help="MESA 现有 split 文件 (直接复用其名单)")
    parser.add_argument("--shhs1-path", type=Path, default=None,
                        help="覆盖 study_data.json 的 shhs1_processed_path (测试用)")
    parser.add_argument("--shhs2-path", type=Path, default=None,
                        help="覆盖 study_data.json 的 shhs2_processed_path (测试用)")
    args = parser.parse_args()

    cfg = json.load(open("study_data.json"))
    from sleep_analysis.datasets.mesadataset import MesaDataset
    from sleep_analysis.datasets.shhs_dataset import ShhsDataset

    # MESA: 复用现有 split 名单
    mesa_split = json.load(open(args.mesa_split))
    mesa_ids = _ids_from_dataset(MesaDataset())
    mesa_part = {
        k: [i for i in mesa_ids if i in set(v)] for k, v in mesa_split.items()
    }
    print(f"MESA: {len(mesa_ids)} 被试, 名单 train={len(mesa_part['train'])} "
          f"val={len(mesa_part['val'])} test={len(mesa_part['test'])}")

    # SHHS1/SHHS2: 联合划分 (复现自动划分逻辑)
    shhs1_ids = _ids_from_dataset(ShhsDataset(study="shhs1", processed_path=args.shhs1_path))
    shhs2_ids = _ids_from_dataset(ShhsDataset(study="shhs2", processed_path=args.shhs2_path))
    print(f"SHHS1: {len(shhs1_ids)} 被试 | SHHS2: {len(shhs2_ids)} 被试")
    shhs1_part, shhs2_part, n_tr, n_va, n_te = split_shhs_joint(shhs1_ids, shhs2_ids)
    print(f"SHHS 联合划分: train={n_tr} val={n_va} test={n_te} (nsrrid 级别, 无跨集重叠)")
    print(f"  SHHS1: train={len(shhs1_part['train'])} val={len(shhs1_part['val'])} test={len(shhs1_part['test'])}")
    print(f"  SHHS2: train={len(shhs2_part['train'])} val={len(shhs2_part['val'])} test={len(shhs2_part['test'])}")

    split = {"mesasleep": mesa_part, "shhs1": shhs1_part, "shhs2": shhs2_part}
    output = args.output or Path(f"splits/split_mixed_{__import__('datetime').datetime.now():%Y%m%d}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(split, indent=2, ensure_ascii=False))
    print(f"[split] 写入: {output}")

    # ✅20260825: 每数据集一个划分文件 — 训练脚本按数据集自声明 split_file 组合。
    # MESA 沿用已有 splits/split_mesa_1121_20260806.json (mesa_part 由其过滤而来, 不重复写)。
    out_date = __import__('datetime').datetime.now().strftime("%Y%m%d")
    for _key, _part in [("shhs1", shhs1_part), ("shhs2", shhs2_part)]:
        out = Path(f"splits/split_{_key}_{out_date}.json")
        out.write_text(json.dumps(_part, indent=2, ensure_ascii=False))
        print(f"[split] 写入: {out}")

    # 一致性校验: 每个数据集内 train/val/test 互斥且全覆盖
    for name, part in split.items():
        all_ids = set(sum(part.values(), []))
        total = sum(len(v) for v in part.values())
        disjoint = (set(part["train"]) & set(part["val"]) == set()
                    and set(part["train"]) & set(part["test"]) == set()
                    and set(part["val"]) & set(part["test"]) == set())
        print(f"  {name}: {total} 个划分 ID, 互斥={disjoint}, 去重后={len(all_ids)}")


if __name__ == "__main__":
    main()
