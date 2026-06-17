"""
统一种子管理器 — 保证实验可复现性

用法：
    from tool.seed_manager import set_all_seeds, get_repeat_seed

    # 在每个 repeat 开始时调用
    set_all_seeds(seed=42)

    # 或使用基于 repeat_idx 的确定性种子（推荐）
    set_all_seeds(get_repeat_seed(repeat_idx=0, base_seed=42))
"""

import os
import random
import numpy as np
import torch


# 全局基础种子（可通过环境变量覆盖）
_BASE_SEED = int(os.environ.get("FL_BASE_SEED", "42"))


def get_repeat_seed(repeat_idx: int = 0, base_seed: int = None) -> int:
    """
    为第 N 次 repeat 生成确定性种子。

    不同 repeat 使用不同但确定的种子，保证：
    - 同一 repeat_idx → 同一结果（可复现）
    - 不同 repeat_idx → 不同随机性（Mean±STD 有意义）

    Args:
        repeat_idx: 第几次重复实验（0-based）
        base_seed: 基础种子，默认从 FL_BASE_SEED 环境变量或 42 取值

    Returns:
        该 repeat 应使用的种子值
    """
    if base_seed is None:
        base_seed = _BASE_SEED
    return base_seed + repeat_idx * 1000


def set_all_seeds(seed: int = 42):
    """
    统一设置所有随机数生成器的种子。

    覆盖：Python random / NumPy / PyTorch(CPU+GPU) / CUDA

    Args:
        seed: 种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 确保 cuDNN 的确定性（可能略微降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置 Python hash 随机化（Python 3.3+）
    os.environ['PYTHONHASHSEED'] = str(seed)


def seed_worker(worker_id: int):
    """DataLoader worker 的种子设置函数，传给 worker_init_fn 参数。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_id)


def make_deterministic(base_seed: int = None):
    """
    完全确定化模式 — 用于需要严格可复现的场景。
    除了 set_all_seeds 外还禁用所有非确定性优化。

    Args:
        base_seed: 基础种子
    """
    seed = base_seed if base_seed is not None else _BASE_SEED
    set_all_seeds(seed)
