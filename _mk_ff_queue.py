"""为 GPU4 生成 FairFace 图像任务队列（bs=32）。

输入：
  _gap30.tsv / _gap31.tsv   两机逐格 repeat 计数
  _imgpool30.txt/_imgpool31.txt  两机历史图像命令行池
规则：
  1. 只取 done=0 的空白格（没有任何 metrics），避免与已有 hash 目录的 adopt 冲突；
  2. 命令行从历史池里按 (dataset, split, algorithm, K) 精确匹配，不手拼参数；
  3. 统一改 -batch_size/-test_batch_size 为 32（GPU4 只有 ~6 GB 空闲显存）。
输出：
  _ffq.txt      可直接投喂 launch_parallel.sh 的队列
  _ffplan.txt   人工核对用清单
"""
from __future__ import annotations

import re
import sys

IMG = ["FairFace", "CelebA", "LFWA+"]


def load_gap(path):
    cells = {}
    for line in open(path, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) != 7:
            continue
        cells[(p[0], p[1], p[2], p[4])] = (int(p[5]), int(p[6]))  # (done, part)
    return cells


# 已知病态格：多次尝试都卡在 round 2（RSS 持续增长、231 线程空转），先跳过
EXCLUDE = {
    ("FairFace", "Dirichlet01", "FL_FairBatch", "20"),
}


def key_of(cmd):
    d = {}
    toks = cmd.split()
    i = 0
    while i < len(toks) - 1:
        if toks[i].startswith("-"):
            d[toks[i]] = toks[i + 1]
            i += 2
        else:
            i += 1
    if d.get("-dataset") not in IMG:
        return None
    return (d["-dataset"], d.get("-split_strategy", ""), d.get("-algorithm", ""),
            str(d.get("-num_clients_K", "")))


def load_pool(paths):
    pool = {}
    for path in paths:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or "main_IMG_CLF.py" not in line:
                continue
            key = key_of(line)
            if key:
                pool.setdefault(key, []).append(line)
    return pool


def with_bs32(cmd):
    cmd = re.sub(r"-batch_size \d+", "-batch_size 32", cmd)
    cmd = re.sub(r"-test_batch_size \d+", "-test_batch_size 32", cmd)
    if "-resume" not in cmd:
        cmd = cmd + " -resume"
    return cmd


def main():
    ds_filter = sys.argv[1] if len(sys.argv) > 1 else "FairFace"
    c30, c31 = load_gap("_gap30.tsv"), load_gap("_gap31.tsv")
    pool = load_pool(["_imgpool30.txt", "_imgpool31.txt"])

    keys = {k for k in set(c30) | set(c31) if k[0] == ds_filter}
    rows, missing, skipped = [], [], []
    for key in keys:
        d30, p30 = c30.get(key, (0, 0))
        d31, p31 = c31.get(key, (0, 0))
        done = max(d30, d31)
        if done != 0:
            continue                      # 只挑完全空白的格
        if key in EXCLUDE:
            skipped.append(key)
            continue
        if key not in pool:
            missing.append(key)
            continue
        rows.append((key, pool[key][0], max(p30, p31)))

    # 优先级：① 已证实会卡死的算法家族排后面（FL_FairBatch 在 FairFace 上有卡死前科）
    #         ② 历史中断次数少的优先  ③ 其余按 K/split/algorithm 稳定排序
    risky = {"FL_FairBatch"}
    rows.sort(key=lambda r: (r[0][2] in risky, r[2], r[0][3].zfill(3), r[0][1], r[0][2]))
    q = [with_bs32(cmd) for _, cmd, _ in rows]
    with open("_ffq.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(q) + "\n")
    with open("_ffplan.txt", "w", encoding="utf-8", newline="\n") as f:
        for key, cmd, part in rows:
            f.write("%s\t%s\t%s\tK%s\tpart=%d\n" % (key[0], key[1], key[2], key[3], part))

    print("候选空白格=%d  已写队列=%d  无历史命令行=%d  主动跳过=%d"
          % (len(keys), len(q), len(missing), len(skipped)))
    for key in sorted(skipped):
        print("  - 跳过(病态): %s" % (key,))
    for key in sorted(missing):
        print("  ! 无命令行: %s" % (key,))
    for key, _, part in rows:
        print("  + %-10s %-12s %-24s K%-3s 历史中断%d次" % (key[0], key[1], key[2], key[3], part))


if __name__ == "__main__":
    main()
