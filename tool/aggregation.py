# -*- coding: utf-8 -*-
"""公共聚合模块 / Common aggregation utilities for federated learning.

核心思想 / Core idea
====================
联邦平均（FedAvg, McMahan et al., 2017, https://arxiv.org/abs/1602.05629）聚合的是
客户端的**完整模型状态**，即 ``state_dict()`` 中的全部张量：既包含可学习参数
（``parameters()``），也包含注册的非参数缓冲（``buffers()``）。

BatchNorm 的滑动统计量 ``running_mean`` / ``running_var`` / ``num_batches_tracked``
属于 buffer，**不会出现在** ``parameters()`` / ``named_parameters()`` 中。若聚合时
只搬运 ``parameters()`` 而遗漏 buffer，全局模型在评测阶段（``.eval()`` 使用 running
统计量）会一直沿用初始统计量（mean=0, var=1），与客户端训练阶段（使用 batch 统计量）
产生归一化错位，可能使模型退化为恒定预测器（ACC≈0.5, DEO=0）。

与 FedBN 的关系 / Relation to FedBN
===================================
FedBN (Li et al., ICLR 2021, https://arxiv.org/abs/2102.07623) 有意将**整个
BatchNorm 层**（仿射参数 γ/β **与** running 统计量）保留在本地、不参与聚合。
因此「聚合 BN 仿射参数却漏掉 BN 统计量」既不是 FedAvg 也不是 FedBN，属于实现缺陷。

本模块统一以「完整 state_dict」为聚合单位，从根上杜绝此类遗漏：所有联邦算法都应
通过本模块完成「客户端状态 → 全局模型」的搬运，而不是各自手写参数级聚合循环。
"""

from collections import OrderedDict
from typing import List, Mapping, Optional, Sequence

import torch
import torch.nn as nn


def _check_state_dicts(client_states: Sequence[Mapping[str, torch.Tensor]]) -> List[str]:
    """校验客户端状态非空且 keys 完全一致，返回公共 keys。"""
    if not client_states:
        raise ValueError("'client_states' must be a non-empty sequence of state_dicts")
    keys = list(client_states[0].keys())
    for state in client_states[1:]:
        if list(state.keys()) != keys:
            raise ValueError("all client state_dicts must share identical keys")
    return keys


def _normalize_weights(
    weights: Optional[Sequence[float]], size: int
) -> List[float]:
    """把权重规整为长度 ``size``、总和为正的浮点列表；``None`` 表示等权。"""
    if weights is None:
        return [1.0] * size
    if len(weights) != size:
        raise ValueError("'weights' and 'client_states' must have equal length")
    weights = [float(w) for w in weights]
    if sum(weights) <= 0:
        raise ValueError("aggregation weights must sum to a positive value")
    return weights


def average_state_dicts(
    client_states: Sequence[Mapping[str, torch.Tensor]],
    weights: Optional[Sequence[float]] = None,
    *,
    keys: Optional[Sequence[str]] = None,
) -> "OrderedDict[str, torch.Tensor]":
    """对若干客户端 ``state_dict`` 做加权平均，**完整包含 buffer**。

    Args:
        client_states: 客户端 ``state_dict`` 序列（keys 必须一致）。
        weights: 与 ``client_states`` 等长的非负权重；``None`` 表示等权平均。
        keys: 仅聚合这些 key（用于部分层聚合，例如 FedRep 只聚合 base 层）；
            ``None`` 表示聚合全部 key。

    Returns:
        ``OrderedDict[str, torch.Tensor]``：CPU 上的平均状态，可直接
        ``module.load_state_dict(state, strict=False)`` 写回。

    数值语义与 ``tool.utils.get_parameters`` + ``np.average`` 链路一致：
    浮点（含复数）张量按权重加权平均；整型 buffer 亦按权重平均后还原为原 dtype
    （``num_batches_tracked`` 这类簿记量对默认 momentum 的 BN 无实际影响）；
    布尔 buffer 取参考客户端的值。
    """
    all_keys = _check_state_dicts(client_states)
    if keys is None:
        aggregate_keys = all_keys
    else:
        missing = [k for k in keys if k not in client_states[0]]
        if missing:
            raise KeyError(f"keys not present in client state_dict: {missing}")
        aggregate_keys = list(keys)

    normalized_weights = _normalize_weights(weights, len(client_states))
    total_weight = float(sum(normalized_weights))

    averaged: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for name in aggregate_keys:
        reference = client_states[0][name].detach().cpu()
        if reference.is_floating_point() or reference.is_complex():
            value = torch.zeros_like(reference)
            for state, weight in zip(client_states, normalized_weights):
                value.add_(
                    state[name].detach().to(device="cpu", dtype=reference.dtype),
                    alpha=weight / total_weight,
                )
            averaged[name] = value
        elif reference.dtype == torch.bool:
            averaged[name] = reference.clone()
        else:
            value = torch.zeros_like(reference, dtype=torch.float64)
            for state, weight in zip(client_states, normalized_weights):
                value.add_(
                    state[name].detach().to(device="cpu", dtype=torch.float64),
                    alpha=weight / total_weight,
                )
            averaged[name] = value.to(dtype=reference.dtype)
    return averaged


def state_dict_keys_of_submodule(module: nn.Module, submodule_name: str) -> List[str]:
    """返回 ``module.state_dict()`` 中属于子模块 ``submodule_name`` 的 key。

    用于「部分层聚合」：例如 FedRep 只聚合 ``shared_base`` / ``bert``、分类头保持
    客户端本地个性化。返回值同时覆盖参数与 buffer，避免只聚合 ``parameters()``。
    """
    if not hasattr(module, submodule_name):
        raise AttributeError(f"module has no submodule named '{submodule_name}'")
    submodule = getattr(module, submodule_name)
    prefix = submodule_name + "."
    return [prefix + key for key in submodule.state_dict().keys()]


def assign_state_dict(
    module: nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    strict: bool = False,
) -> nn.Module:
    """把平均后的状态写回模块。

    ``strict=False`` 允许只写回部分 key（例如仅 base 层），其余保持不变。
    """
    module.load_state_dict(state, strict=strict)
    return module
