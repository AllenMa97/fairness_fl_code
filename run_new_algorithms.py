#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量运行新增算法（12 个入口名）的实验调度脚本
Batch experiment runner for newly added federated learning algorithms (12 entry names).

算法列表 / Algorithms:
    FedBE, FedCVAE-Ens, FedCVAE-KD, FedBEns, FAFI, FedTMOS, FOL, FedLMG, AFL, GeFL, GeFL-F, Fair-FedMOE

数据集（沿用现有任务划分，按 dataset -> task 自动选择入口脚本）:
    - IMG_CLF:     CelebA, UTKFace, FairFace, LFWA+   -> main_IMG_CLF.py
    - SENT_CLF:    moji, bios                          -> main_SENT_CLF.py
    - Tabular_CLF: ADULT, COMPAS, DRUG, DUTCH          -> main_Tabular_CLF.py

说明 / Notes:
    - 算法超参数从 json/algorithm/<算法名>.json 自动加载，与本脚本解耦；
      新数据集接入后（json/dataset/<名称>.json 就绪）无需改动本脚本即可复用。
    - 每个 (算法, 数据集, 划分策略) 组合作为一次独立运行，由子进程串行执行，
      单个组合失败不影响后续组合。
    - --skip_existing 会检测 ./log_path/<dataset>/<split>/<algorithm>/ 下
      是否已有包含 "Mean±STD" 汇总的日志，有则跳过（断点续跑）。

用法 / Usage:
    python run_new_algorithms.py --dry_run                       # 只打印命令，不执行
    python run_new_algorithms.py                                 # 全部算法 x 全部数据集 x 全部策略
    python run_new_algorithms.py -algorithms FedBE,AFL           # 指定算法
    python run_new_algorithms.py -datasets moji,bios             # 指定数据集
    python run_new_algorithms.py -split_strategies Uniform       # 指定划分策略
    python run_new_algorithms.py -skip_existing                  # 跳过已完成的组合
    python run_new_algorithms.py -system_data_count 2000         # 冒烟测试（限制样本数）
"""

import os
import sys
import glob
import argparse
import subprocess

# 12 个算法入口名（与 algorithm/ 目录及 json/algorithm/ 配置一一对应）
NEW_ALGORITHMS = [
    "FedBE",
    "FedCVAE-Ens",
    "FedCVAE-KD",
    "FedBEns",
    "FAFI",
    "FedTMOS",
    "FOL",
    "FedLMG",
    "AFL",
    "GeFL",
    "GeFL-F",
    "Fair-FedMOE",
]

# 数据集 -> (任务类型, 入口脚本)
DATASET_TASK_MAP = {
    "CelebA": ("IMG_CLF", "main_IMG_CLF.py"),
    "UTKFace": ("IMG_CLF", "main_IMG_CLF.py"),
    "FairFace": ("IMG_CLF", "main_IMG_CLF.py"),
    "LFWA+": ("IMG_CLF", "main_IMG_CLF.py"),
    "moji": ("SENT_CLF", "main_SENT_CLF.py"),
    "bios": ("SENT_CLF", "main_SENT_CLF.py"),
    "ADULT": ("Tabular_CLF", "main_Tabular_CLF.py"),
    "COMPAS": ("Tabular_CLF", "main_Tabular_CLF.py"),
    "DRUG": ("Tabular_CLF", "main_Tabular_CLF.py"),
    "DUTCH": ("Tabular_CLF", "main_Tabular_CLF.py"),
}

# 划分策略（与各 main 脚本内置列表一致）
DEFAULT_SPLIT_STRATEGIES = ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"]


def parse_list_arg(value, allowed, arg_name):
    """解析逗号分隔的列表参数并校验合法性 / Parse a comma-separated list arg and validate it."""
    if not value or value.lower() == "all":
        return list(allowed)
    items = [x.strip() for x in value.split(",") if x.strip()]
    for item in items:
        if item not in allowed:
            print(f"[ERROR] Invalid {arg_name}: {item}. Allowed: {allowed}")
            sys.exit(1)
    return items


def build_command(algorithm, dataset, split_strategy, args):
    """
    拼接单次实验的命令行 / Build the command line for one experiment run.
    """
    task, entry_script = DATASET_TASK_MAP[dataset]
    cmd = [
        sys.executable,
        entry_script,
        "-algorithm", algorithm,
        "-dataset", dataset,
        "-task", task,
        "-split_strategy", split_strategy,
        "-cuda", args.cuda,
        "-exp_repeat_times", str(args.exp_repeat_times),
    ]
    if args.system_data_count is not None:
        cmd += ["-system_data_count", str(args.system_data_count)]
    if args.communication_round_I is not None:
        cmd += ["-communication_round_I", str(args.communication_round_I)]
    if args.algorithm_epoch_T is not None:
        cmd += ["-algorithm_epoch_T", str(args.algorithm_epoch_T)]
    if args.num_clients_K is not None:
        cmd += ["-num_clients_K", str(args.num_clients_K)]
    return cmd


def is_combination_finished(algorithm, dataset, split_strategy):
    """
    断点检测：判断某个组合是否已有完整的汇总结果 / Check whether a combination already has a final summary.
    判据：./log_path/<dataset>/<split>/<algorithm>/ 下任意日志包含 "Mean±STD"。
    """
    log_root = os.path.join("log_path", dataset, split_strategy, algorithm)
    if not os.path.isdir(log_root):
        return False
    for log_file in glob.glob(os.path.join(log_root, "**", "*.txt"), recursive=True):
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                if "Mean±STD" in f.read():
                    return True
        except OSError:
            continue
    return False


def main():
    parser = argparse.ArgumentParser(description="Batch runner for the 10 new FL algorithms / 10 个新算法的批量实验调度")
    parser.add_argument("-algorithms", default="all", type=str,
                        help="Comma-separated algorithm names or 'all'. Default: all")
    parser.add_argument("-datasets", default="all", type=str,
                        help="Comma-separated dataset names or 'all'. Default: all")
    parser.add_argument("-split_strategies", default="all", type=str,
                        help="Comma-separated split strategies or 'all'. Default: all")
    parser.add_argument("-cuda", default="0,1,2,3", type=str, help="CUDA device ids passed to the entry script")
    parser.add_argument("-exp_repeat_times", default=3, type=int,
                        help="Repeats per experiment with different seeds (passed through). Default: 3")
    parser.add_argument("-system_data_count", default=None, type=int,
                        help="Limit training samples for smoke tests. Default: None (all data)")
    parser.add_argument("-communication_round_I", default=None, type=int,
                        help="Override communication rounds. Default: None (use built-in defaults)")
    parser.add_argument("-algorithm_epoch_T", default=None, type=int,
                        help="Override local epochs. Default: None (use built-in defaults)")
    parser.add_argument("-num_clients_K", default=None, type=int,
                        help="Override client count. Default: None (use built-in defaults)")
    parser.add_argument("-skip_existing", action="store_true",
                        help="Skip combinations whose logs already contain a Mean±STD summary")
    parser.add_argument("-dry_run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    algorithms = parse_list_arg(args.algorithms, NEW_ALGORITHMS, "algorithm")
    datasets = parse_list_arg(args.datasets, list(DATASET_TASK_MAP.keys()), "dataset")
    split_strategies = parse_list_arg(args.split_strategies, DEFAULT_SPLIT_STRATEGIES, "split_strategy")

    # 预检：算法配置文件必须存在 / Pre-check: algorithm config json must exist
    missing = [a for a in algorithms
               if not os.path.exists(os.path.join("json", "algorithm", a + ".json"))]
    if missing:
        print(f"[ERROR] Missing algorithm config in json/algorithm/: {missing}")
        sys.exit(1)

    # 生成组合矩阵 / Build the combination matrix
    combinations = [(a, d, s)
                    for a in algorithms
                    for d in datasets
                    for s in split_strategies]
    total = len(combinations)
    print(f"[PLAN] {len(algorithms)} algorithms x {len(datasets)} datasets x "
          f"{len(split_strategies)} splits = {total} runs")

    finished = failed = executed = 0
    for idx, (algorithm, dataset, split_strategy) in enumerate(combinations, 1):
        tag = f"[{idx}/{total}] {algorithm} | {dataset} | {split_strategy}"
        cmd = build_command(algorithm, dataset, split_strategy, args)

        if args.skip_existing and is_combination_finished(algorithm, dataset, split_strategy):
            print(f"{tag} -> [SKIP] already has Mean±STD summary")
            finished += 1
            continue

        if args.dry_run:
            print(f"{tag} -> [DRY] {' '.join(cmd)}")
            continue

        print(f"{tag} -> [RUN] {' '.join(cmd)}")
        try:
            ret = subprocess.run(cmd).returncode
        except KeyboardInterrupt:
            print("[INTERRUPT] Stopped by user.")
            sys.exit(130)
        if ret == 0:
            executed += 1
        else:
            failed += 1
            print(f"{tag} -> [FAIL] exit code {ret}")

    print("=" * 60)
    if args.dry_run:
        print(f"[DONE] Dry run finished. {total} commands generated.")
    else:
        print(f"[DONE] executed: {executed}, skipped(finished): {finished}, failed: {failed}")


if __name__ == "__main__":
    main()
