"""MixedDataset — 合并多个 tpcp Dataset 为一个统一的训练集。

典型用法:
    from sleep_analysis.datasets.mesadataset import MesaDataset
    from sleep_analysis.datasets.shhs_dataset import ShhsDataset
    from sleep_analysis.datasets.mixed_dataset import MixedDataset

    ds = MixedDataset({
        "mesa": MesaDataset(),
        "shhs2": ShhsDataset(study="shhs2"),
    })
    train, test = get_random_split(ds)   # 和普通 Dataset 一样用
"""

import pandas as pd
from tpcp import Dataset

from sleep_analysis.datasets.base_sleep_dataset import BaseSleepDataset


class MixedDataset(Dataset):
    """合并多个同构 tpcp Dataset。

    每个源 dataset 的 index 中自动加 _source 列标记来源。
    迭代 / 切片 / 属性访问时根据 _source 转发到正确的底层 dataset。
    """

    # 20260822: 特征列选择走数据集自描述接口 — MixedDataset 使用标准命名约定
    # (与历史行为一致: 混合集统一按 MESA/SHHS 的列名选择)
    @classmethod
    def feature_columns(cls, modality: str):
        return BaseSleepDataset.FEATURE_COLUMNS.get(modality)

    def __init__(self, sources, *, groupby_cols=None, subset_index=None, **kwargs):
        """
        Parameters
        ----------
        sources : dict[str, Dataset]
            源数据集映射，如 {"mesa": MesaDataset(), "shhs2": ShhsDataset(study="shhs2")}
        """
        self.sources = sources
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index, **kwargs)

    # 用于前缀分隔的常量（避免 SHHS1 / SHHS2 被试 ID 碰撞）
    _SEP = "@"

    def create_index(self):
        frames = []
        for src_name, ds in self.sources.items():
            idx = ds.index.copy()
            # 前缀化 subj_id，保证跨子集全局唯一
            idx["subj_id"] = f"{src_name}{self._SEP}" + idx["subj_id"].astype(str)
            idx["_source"] = src_name
            frames.append(idx)
        return pd.concat(frames, ignore_index=True)

    # ---- 内部：按当前单被试子集，找到底层 dataset 的对应切片 ----

    def _get_sub_source(self):
        """返回底层 dataset 中当前被试的单行子集。"""
        src = self.index["_source"].iloc[0]
        full_id = str(self.index["subj_id"].iloc[0])
        # 去掉前缀还原原始 ID
        prefix = f"{src}{self._SEP}"
        raw_id = full_id[len(prefix):] if full_id.startswith(prefix) else full_id
        ds = self.sources[src]
        try:
            return ds.get_subset(subj_id=[int(raw_id)])
        except (ValueError, KeyError):
            return ds.get_subset(subj_id=[raw_id])

    # ---- 属性转发 ----

    @property
    def feature_table(self):
        return self._get_sub_source().feature_table

    @property
    def ground_truth(self):
        return self._get_sub_source().ground_truth

    @property
    def actigraph_data(self):
        return self._get_sub_source().actigraph_data

    @property
    def information(self):
        src = self.index["_source"].iloc[0]
        if hasattr(self.sources[src], "information"):
            return self._get_sub_source().information
        return pd.Series(dtype=object)

    @property
    def tst(self):
        return self._get_sub_source().tst

    @property
    def edf_path(self):
        return self._get_sub_source().edf_path
