"""全量扫描本机 save_path/ 下每个格子的已完成 repeat 数 / scan completed repeats per cell.

目录层级（实测）：save_path/<dataset>/<split>/<algorithm>/<hypothesis>/<K>Clients/experiment_state/<hash>/repeat_<NN>/
输出 TSV: dataset \t split \t algorithm_dir \t hypothesis \t K \t n_done \t n_partial
只认 metrics.json，不认日志 —— 用来回答「表格任务到底跑完没有」。
"""
from __future__ import annotations

import glob
import os
import socket

SAVE = "/root/fl/save_path"
DATASETS = [
    "ADULT", "BANK", "COMPAS", "DRUG", "DUTCH", "ACSIncome", "bios",
    "UTKFace", "CelebA", "LFWA+", "FairFace",
]

rows = []
for ds in DATASETS:
    for split_dir in sorted(glob.glob(os.path.join(SAVE, ds, "*"))):
        split = os.path.basename(split_dir)
        for algo_dir in sorted(glob.glob(os.path.join(split_dir, "*"))):
            algo = os.path.basename(algo_dir)
            # 兼容有/无 hypothesis 层的两种布局
            k_dirs = sorted(glob.glob(os.path.join(algo_dir, "*", "*Clients")))
            k_dirs += sorted(glob.glob(os.path.join(algo_dir, "*Clients")))
            for k_dir in k_dirs:
                hyp = os.path.basename(os.path.dirname(k_dir))
                k = os.path.basename(k_dir)[:-len("Clients")]
                reps = sorted(glob.glob(os.path.join(k_dir, "experiment_state", "*", "repeat_*")))
                done = sum(1 for r in reps if os.path.exists(os.path.join(r, "metrics.json")))
                part = sum(1 for r in reps
                           if not os.path.exists(os.path.join(r, "metrics.json"))
                           and os.path.exists(os.path.join(r, "checkpoint_latest.pt")))
                if done == 0 and part == 0:
                    continue
                rows.append((ds, split, algo, hyp, k, done, part))

host = socket.gethostname().split(".")[-1]
out = "/tmp/gapscan_%s.tsv" % host
with open(out, "w", encoding="utf-8", newline="\n") as f:
    for r in rows:
        f.write("%s\t%s\t%s\t%s\t%s\t%d\t%d\n" % r)
print("wrote", out, "cells:", len(rows))
