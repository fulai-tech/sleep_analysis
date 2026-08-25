"""数据集注册表 (工厂) — 数据集模块通过 @register 自注册, 训练管线按名字查表。

加新数据集: 新建模块 → 类上 @register("注册名") → 在 datasets/__init__.py 的
数据集目录里加一行 import (触发注册)。训练脚本 / data_peparation / make_splits
零改动。

注意: registry 模块本身不 import 任何数据集模块 (避免循环导入) —
数据集模块 import 本模块拿装饰器, 注册动作由 datasets/__init__.py 触发。
"""

DATASETS: dict = {}


def register(name: str):
    """类装饰器: 把数据集类注册到 DATASETS[name]。"""

    def deco(cls):
        DATASETS[name] = cls
        return cls

    return deco


def get_dataset_class(name: str):
    """按注册名取数据集类。"""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name} (已注册: {sorted(DATASETS)})")
    return DATASETS[name]


def get_dataset(name: str, **kwargs):
    """按注册名创建数据集实例 (工厂方法)。"""
    return get_dataset_class(name)(**kwargs)
