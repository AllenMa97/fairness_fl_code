#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验入口的统一前置检查（数据 + 预训练权重）。

设计目标：`main_*.py` 是唯一入口，用户不需要事先手动执行 `setup_data.py`。
每个入口在解析完命令行参数后调用 `ensure_ready(param_dict)`，由它完成：

  1. 按 `-dataset` 检查关键文件是否存在，缺失就自动调用 `setup_data.py` 下载/解压；
  2. 文本任务（SENT_CLF）检查仓库内 BERT 离线权重，缺失则尝试下载；
  3. 打印一行就绪摘要，让用户看到「要跑什么、数据在哪、权重用的是哪份」。

失败时给出可直接复制的手工命令，而不是把问题推迟到训练循环里炸开。
"""

import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SETUP_SCRIPT = os.path.join(_REPO_ROOT, "setup_data.py")
_BERT_LOCAL_DIR = os.path.join(_REPO_ROOT, "pretrained", "bert-base-uncased")

# 需要仓库内 BERT 离线权重的入口（其余任务不加载 transformers 权重）
_TEXT_TASKS = {"SENT_CLF"}
_BERT_MODEL_NAME = "bert-base-uncased"


def _run_setup(args):
    """以子进程方式调用 setup_data.py（保持该脚本可独立使用的原有行为）。"""
    return subprocess.run([sys.executable, _SETUP_SCRIPT] + args,
                          cwd=_REPO_ROOT).returncode


def _bert_weight_ready():
    return all(os.path.isfile(os.path.join(_BERT_LOCAL_DIR, name))
               for name in ("config.json", "vocab.txt", "model.safetensors"))


def ensure_dataset(dataset_name):
    """确保单个数据集可用；缺失时自动下载。返回 (是否就绪, 说明)。"""
    from module.experiment_setup import ensure_dataset_ready

    if ensure_dataset_ready(dataset_name):
        return True, "已就绪"
    return False, f"缺少必要文件，请手动运行: python setup_data.py --datasets {dataset_name}"


def ensure_pretrained(task, param_dict):
    """文本任务确保 BERT 离线权重存在。返回 (是否就绪, 说明)。"""
    if task.upper() not in _TEXT_TASKS:
        return True, "不适用"
    # 用户显式指定了本地目录时直接尊重，不再干预
    configured = param_dict.get("bert_model_name_or_path")
    if configured and configured != _BERT_MODEL_NAME:
        return True, configured
    if _bert_weight_ready():
        return True, _BERT_LOCAL_DIR

    print(f"[SETUP] 未找到离线权重 pretrained/{_BERT_MODEL_NAME}，尝试自动下载 ...")
    if _run_setup(["--models", _BERT_MODEL_NAME]) == 0 and _bert_weight_ready():
        return True, _BERT_LOCAL_DIR
    # hub 在线兜底仍可能可用，因此这里只告警不终止
    return True, f"{_BERT_MODEL_NAME}（在线 hub，离线环境会失败）"


def ensure_ready(param_dict):
    """所有 main 入口的统一前置钩子。就绪返回 True，否则打印指引并返回 False。"""
    task = str(param_dict.get("task", "") or "")
    raw_datasets = str(param_dict.get("dataset", "") or "")
    datasets = [name.strip() for name in raw_datasets.split(",") if name.strip()]

    if not datasets:
        return True

    ok = True
    for name in datasets:
        ready, note = ensure_dataset(name)
        if ready:
            print(f"[SETUP] dataset={name}  {note}")
        else:
            print(f"[ERROR] dataset={name}  {note}")
            ok = False

    ready, note = ensure_pretrained(task, param_dict)
    if not ready:
        print(f"[ERROR] pretrained={note}")
        ok = False
    elif note not in ("不适用", "已就绪"):
        print(f"[SETUP] pretrained={note}")

    return ok
