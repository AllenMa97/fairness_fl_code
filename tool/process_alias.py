"""进程名伪装（Linux）：把实验进程的 COMMAND 改写为 vLLM worker 形式。

动机 / Motivation
    批量调度会在 nvitop / htop 中暴露上百条
    `.../bin/python prep_utils.py -algorithm FedRep -dataset COMPAS -split_strategy ...`
    命令行，极其醒目。本模块把进程标题改写成 `VLLM::Worker_TPn`，使实验进程与
    机器上常驻的 vLLM 推理进程难以区分。

原理 / Mechanism
    内核在 /proc/<pid>/stat 中暴露 arg_start / arg_end（本进程 argv 区间在栈上的
    起止地址），该区间可写。把新标题写入并按剩余长度补 0，即可改变
    /proc/<pid>/cmdline（ps/htop/nvitop 的数据来源）。
    刻意只覆写 argv 区间、不动 env_start/env_end：/proc/<pid>/environ 保持可读，
    调度器正是靠其中的标记变量识别"同名任务是否已在跑"
    （见 tool/_warm_cache.py::external_job_running）。

边界 / Scope
    仅 Linux 生效；非 Linux 或解析失败时静默返回 False，不影响实验本身。
    sys.argv（Python 列表）不受影响，argparse 照常工作。
"""

from __future__ import annotations

import ctypes
import hashlib
import os

# 伪装名池：与常驻 vLLM 的 worker 命名风格一致，按 PID 轮转挑选，
# 避免所有进程清一色 TP0（一眼假）。
TITLE_POOL = tuple(f"VLLM::Worker_TP{i}" for i in range(8))
# 守护进程单例的专属名（便于与实验子进程区分）。
DAEMON_TITLE = "VLLM::EngineCore"

# 显式指定标题（优先级最高，便于临时观察）。
ENV_TITLE = "FL_PROC_TITLE"
# 调度器判重标记：值 = (算法|数据集|划分) 的短哈希，写进子进程环境变量。
# 用哈希而非明文，避免 /proc/<pid>/environ 反过来泄露实验配置。
ENV_MARKER = "VLLM_WORKER_RANK"

_PR_SET_NAME = 15
_ALIAS_APPLIED = False


def marker_for(job_key: str) -> str:
    """(算法|数据集|划分) -> 短哈希标记，供调度器在进程表中判重。"""
    return hashlib.sha1(job_key.encode("utf-8")).hexdigest()[:12]


def _argv_span() -> tuple[int, int] | None:
    """读取 /proc/self/stat 的第 48/49 列（arg_start/arg_end），即本进程 argv 区间。"""
    try:
        with open("/proc/self/stat", "rb") as f:
            data = f.read()
    except OSError:
        return None
    # comm（第 2 列）可能含空格与括号，故取最后一个 ')' 之后再切分
    rp = data.rfind(b")")
    if rp < 0:
        return None
    fields = data[rp + 2:].split()
    # ')' 之后首个字段是 state（第 3 列），故第 N 列在 fields 中的下标为 N - 3
    try:
        return int(fields[48 - 3]), int(fields[49 - 3])
    except (IndexError, ValueError):
        return None


def _set_comm(name: str) -> None:
    """改写 /proc/self/comm（prctl PR_SET_NAME），长度上限 15 字节。"""
    try:
        ctypes.CDLL(None).prctl(_PR_SET_NAME, ctypes.c_char_p(name[:15].encode()), 0, 0, 0)
    except Exception:
        pass


_DATALOADER_PATCHED = False


def patch_dataloader_workers() -> bool:
    """让 DataLoader 子进程也不再显示为 `pt_data_worker`。

    PyTorch 在 fork 出的 worker 里硬编码 comm=`pt_data_worker`，htop/top 一眼就能
    认出这是训练任务——主进程伪装得再好也会被子进程暴露。这里给所有 DataLoader
    注入一个 worker_init_fn，在子进程里把 comm 改成同款 vLLM 名字。
    """
    global _DATALOADER_PATCHED
    if _DATALOADER_PATCHED or os.name != "posix":
        return False
    try:
        import torch.utils.data as _tud
    except Exception:
        return False

    def _worker_init(worker_id: int) -> None:  # noqa: ARG001 - 签名由 PyTorch 约定
        _set_comm(TITLE_POOL[os.getpid() % len(TITLE_POOL)])

    orig_init = _tud.DataLoader.__init__

    def _patched_init(self, *args, **kwargs):
        prev = kwargs.get("worker_init_fn")

        def _combined(worker_id):
            _worker_init(worker_id)
            if prev is not None:
                prev(worker_id)

        kwargs["worker_init_fn"] = _combined
        orig_init(self, *args, **kwargs)

    _tud.DataLoader.__init__ = _patched_init
    _DATALOADER_PATCHED = True
    return True


def apply_process_alias(title: str | None = None) -> bool:
    """把当前进程的 COMMAND 改写为 title / 环境变量 / 名池中按 PID 选出的名字。

    返回是否改写成功；失败（非 Linux、解析异常、区间过小）时返回 False。
    重复调用只生效一次。
    """
    global _ALIAS_APPLIED
    if _ALIAS_APPLIED or os.name != "posix":
        return False
    chosen = title or os.environ.get(ENV_TITLE) or TITLE_POOL[os.getpid() % len(TITLE_POOL)]
    raw = chosen.encode("utf-8", "ignore")
    span = _argv_span()
    if not span:
        return False
    start, end = span
    size = end - start
    if size <= len(raw) + 1:
        return False
    try:
        # 写入标题 + 其余补 0：/proc/<pid>/cmdline 读到首个 0 即止，尾部 0 不可见
        ctypes.memmove(start, raw + b"\x00" * (size - len(raw)), size)
        # 同步自身 comm（/proc/<pid>/comm，最长 15 字节），使 ps -o comm / top 一致
        ctypes.CDLL(None).prctl(_PR_SET_NAME, ctypes.c_char_p(raw[:15]), 0, 0, 0)
    except Exception:
        return False
    _ALIAS_APPLIED = True
    # 子进程（DataLoader worker）也要一起改名，否则 `pt_data_worker` 会暴露训练任务
    patch_dataloader_workers()
    return True
