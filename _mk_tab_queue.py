"""为主对比算法缺口生成表格任务队列（默认 ACSIncome）。

规则：
  1. 只挑 done<3 的格；done=0 的排前面（无 adopt 冲突风险）
  2. 命令行从历史池里按 (dataset, split, algorithm, K) 匹配，优先 bs=256
  3. 池子里带 -variant_tag 的调试命令已在采集阶段剔除
  4. 输出 newline="\\n"，避免 Windows CRLF 把 -resume 变成 -resume\\r

用法：python _mk_tab_queue.py <dataset> <输出队列文件> [最多条数]
"""
from __future__ import annotations

import re
import sys

VARIANTS = (
    "acspilot", "acssmoke", "acsbench", "nwbench", "nwlong", "pdfabv",
    "pdfnw", "pdfla", "smoke", "pilot", "bench", "comb",
)
# 论文主对比的 16 个算法（依《实验决策记录》D10.5 的 16 算法矩阵；
# 含 coverage 审计用到的写法别名 FederatedNova / FederatedRenyi / Simple_mFairFL）
MAIN = {
    # .30 分区 8 个
    "FedAvg", "FederatedAverageWithPo", "FedFair", "FedProto", "FL_FairBatch",
    "LoGoFair", "PDFFed", "Scaffold",
    # .31 分区 8 个
    "FedProx", "FederatedNova", "FedNova", "FedRep", "FairFed", "FedFB",
    "FederatedRenyi", "FedRenyi", "mFairFL", "Simple_mFairFL", "PDFFed_DP",
}


def load_gap(path):
    cells = {}
    for line in open(path, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) != 7:
            continue
        cells[(p[0], p[1], p[2], p[4])] = (int(p[5]), int(p[6]))
    return cells


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
    if not d.get("-dataset"):
        return None, d
    return (d["-dataset"], d.get("-split_strategy", ""), d.get("-algorithm", ""),
            str(d.get("-num_clients_K", ""))), d


def load_pool(paths):
    """同一 (dataset, split, algorithm, K) 可能有多条历史命令（bs / 并发设置不同）。

    ACSIncome 已完成跑次的规范参数由 metrics.json 反查得到：
        client_parallel=2, dataloader_num_workers=0, batch_size=256
    两机共 89 个跑次里 89 个都是这三项，故以此作为选优排序，
    避免选中调试期遗留的 `dataloader_num_workers=2` 版本。
    """
    def rank(d):
        return (
            d.get("-batch_size") != "256",
            d.get("-dataloader_num_workers") != "0",
            d.get("-client_parallel") != "2",
        )

    pool = {}
    for path in paths:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or "main_Tabular_CLF.py" not in line or "-variant_tag" in line:
                continue
            key, d = key_of(line)
            if not key:
                continue
            if key not in pool or rank(d) < rank(pool[key][2]):
                pool[key] = (d.get("-batch_size"), line, d)
    return pool


def synth(pool, key):
    """没有历史命令行时，用同数据集+同划分+同算法、不同 K 的命令派生。

    只替换 -num_clients_K 一个字段，其余参数（learning_rate / rounds / T /
    batch_size / client_parallel / dataloader_num_workers / seed）逐字照抄，
    保证与主表口径一致。
    """
    ds, split, algo, k = key
    cands = []
    for (d2, s2, a2, k2), (bs, cmd, _) in pool.items():
        if d2 == ds and s2 == split and a2 == algo and k2 != k:
            cands.append((bs != "256", k2, bs, cmd))
    if not cands:
        return None, None
    cands.sort()
    _, src_k, _, cmd = cands[0]
    new = re.sub(r"-num_clients_K\s+\d+", "-num_clients_K %s" % k, cmd)
    return new, src_k


def normalize(cmd):
    """把并发相关旋钮对齐到 ACSIncome 已完成 89 个跑次一致的口径。

    只补齐/纠正 `-client_parallel 2` 与 `-dataloader_num_workers 0`，
    其余参数不动；这两项在历史池里存在个别调试期遗留值（2 / 缺失），
    若不纠正会与主表口径不一致。
    """
    for flag, val in (("-client_parallel", "2"), ("-dataloader_num_workers", "0")):
        if re.search(r"%s\s+\S+" % flag, cmd):
            cmd = re.sub(r"%s\s+\S+" % flag, "%s %s" % (flag, val), cmd)
        else:
            cmd = cmd + " " + flag + " " + val
    return cmd


def main():
    ds = sys.argv[1]
    out = sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 6

    c30, c31 = load_gap("_gap30.tsv"), load_gap("_gap31.tsv")
    pool = load_pool(["_tabpool30.txt", "_tabpool31.txt"])

    rows, missing, skipped = [], [], []
    for key in {k for k in set(c30) | set(c31) if k[0] == ds}:
        done = max(c30.get(key, (0, 0))[0], c31.get(key, (0, 0))[0])
        if done >= 3:
            continue
        algo = key[2]
        if algo not in MAIN:
            skipped.append(key)
            continue
        if key in pool:
            rows.append((done, key, pool[key][1], "历史"))
            continue
        cmd, src_k = synth(pool, key)
        if cmd is None:
            missing.append(key)
        else:
            rows.append((done, key, cmd, "由K%s派生" % src_k))

    rows.sort(key=lambda r: (r[0], r[1][2], r[1][1], r[1][3].zfill(3)))
    rows = rows[:limit]
    cmds, fixed = [], 0
    for _, _, c, _ in rows:
        c2 = normalize(c)
        if c2 != c:
            fixed += 1
        cmds.append(c2 + ("" if "-resume" in c2 else " -resume"))
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(cmds) + "\n")

    print("数据集=%s  写队列=%d  非主算法跳过=%d  仍无命令=%d  并发口径纠正=%d"
          % (ds, len(cmds), len(skipped), len(missing), fixed))
    for key in sorted(missing)[:12]:
        print("  ! 无命令: %s/%s/%s/K%s" % key)
    for done, key, _, src in rows:
        print("  + done=%d  %-12s %-22s K%-3s [%s]" % (done, key[1], key[2], key[3], src))


if __name__ == "__main__":
    main()
