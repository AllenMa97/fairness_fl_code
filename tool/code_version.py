# -*- coding: utf-8 -*-
"""代码版本标记 / Code-version stamping for experiment traceability.

为什么要记录 / Why record it
===========================
同一个 (算法, 数据集) 组合可能在多轮迭代中反复重跑，而 ``result_path`` 下的结果
文件本身不带版本信息。一旦修复了算法实现又重跑，新旧结果就会混在一起而无法区分。
本模块给每个结果文件打上产出它的代码版本，便于事后追溯与一致性核对。

版本标识 / Version identity
==========================
1. ``fingerprint``：对源码目录做内容摘要（sorted path + sha256），**代码内容变了就一定变**，
   且不依赖 git 状态，因此在服务器上"git HEAD 落后、工作区被覆盖"的场景下依然可靠。
2. ``git_commit`` / ``git_dirty``：尽力而为地从 git 读取，仅作辅助信息；仓库不可用时为 unknown。

用法 / Usage
============
    from tool.code_version import code_version, code_version_tag
    payload["code_version"] = code_version()          # 结构化，写入 json
    stream.write(f"# code_version: {code_version_tag()}\n")   # 单行，写入 txt
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime
from pathlib import Path

# 仓库根目录：本文件位于 <root>/tool/code_version.py
_REPO_ROOT = Path(__file__).resolve().parent.parent

# 参与指纹计算的源码路径（相对仓库根）；运行时产物目录（log_path/result_path/...）不参与
_SOURCE_PATHS = (
    "algorithm",
    "module",
    "tool",
    "hypothesis",
    "main_Tabular_CLF.py",
    "main_IMG_CLF.py",
    "main_AUD_CLF.py",
    "main_SENT_CLF.py",
    "experiment.py",
)

# 进程内缓存：同一进程只探测一次
_CACHE: dict | None = None


def _source_fingerprint() -> str:
    """对源码文件集合做内容摘要，返回 12 位十六进制串。"""
    digest = hashlib.sha256()
    for name in _SOURCE_PATHS:
        target = _REPO_ROOT / name
        if target.is_dir():
            # 只统计 .py 源码；跳过软链接（如服务器上的 prep_utils.py 入口别名）、
            # 以及 tool/_resource_history 之类的运行时产物，使指纹只反映代码本身。
            # 排序统一按 POSIX 路径字符串：Path 的默认比较在 Windows 上忽略大小写、
            # 在 Linux 上区分大小写，直接 sorted(paths) 会让同一份代码算出不同指纹。
            raw_files = (
                p for p in target.rglob("*.py")
                if "__pycache__" not in p.parts
                and "_resource_history" not in p.parts
                and not p.is_symlink()
            )
            files = sorted(raw_files, key=lambda p: p.relative_to(_REPO_ROOT).as_posix())
        elif target.is_file():
            files = [target]
        else:
            continue
        for path in files:
            digest.update(path.relative_to(_REPO_ROOT).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _git(args: list[str]):
    return subprocess.run(
        ["git", *args], cwd=str(_REPO_ROOT),
        capture_output=True, text=True, timeout=10,
    )


def code_version() -> dict:
    """返回产出结果的代码版本信息。

    Returns:
        dict: ``fingerprint``（源码内容摘要）、``git_commit``（短 SHA，可能为 unknown）、
        ``git_dirty``（工作区相对 HEAD 是否有改动；git 不可用时为 None）、
        ``recorded_at``（记录时间，ISO 格式）。
    """
    global _CACHE
    if _CACHE is not None:
        return dict(_CACHE)

    info = {
        "fingerprint": "unknown",
        "git_commit": "unknown",
        "git_dirty": None,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        info["fingerprint"] = _source_fingerprint()
    except Exception:
        pass
    try:
        head = _git(["rev-parse", "--short", "HEAD"])
        if head.returncode == 0 and head.stdout.strip():
            info["git_commit"] = head.stdout.strip()
        status = _git(["status", "--porcelain"])
        if status.returncode == 0:
            info["git_dirty"] = bool(status.stdout.strip())
    except Exception:
        pass

    _CACHE = dict(info)
    return dict(info)


def code_version_tag() -> str:
    """返回单行可读标签，形如 ``fp=1a2b3c4d5e6f commit=a1b2c3d dirty=True``。"""
    version = code_version()
    return (
        f"fp={version['fingerprint']} "
        f"commit={version['git_commit']} "
        f"dirty={version['git_dirty']}"
    )
