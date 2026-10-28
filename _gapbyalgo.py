"""把缺口按「算法」拆开：主对比算法 vs 消融/变体目录，回答「哪些可以不再跑」。"""
from __future__ import annotations

from collections import defaultdict

# 论文主对比的 16 个算法（依《实验决策记录》D10.5 的 16 算法矩阵；
# 含 coverage 审计用到的写法别名 FederatedNova / FederatedRenyi / Simple_mFairFL）
MAIN = {
    "FedAvg", "FederatedAverageWithPo", "FedFair", "FedProto", "FL_FairBatch",
    "LoGoFair", "PDFFed", "Scaffold",
    "FedProx", "FederatedNova", "FedNova", "FedRep", "FairFed", "FedFB",
    "FederatedRenyi", "FedRenyi", "mFairFL", "Simple_mFairFL", "PDFFed_DP",
}
TAB = {"ADULT", "BANK", "COMPAS", "DRUG", "DUTCH", "ACSIncome", "bios"}


def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) != 7:
            continue
        out[(p[0], p[1], p[2], p[4])] = (int(p[5]), int(p[6]))
    return out


c30, c31 = load("_gap30.tsv"), load("_gap31.tsv")
merged = {}
for key in set(c30) | set(c31):
    merged[key] = max(c30.get(key, (0, 0))[0], c31.get(key, (0, 0))[0])

per_algo = defaultdict(lambda: [0, 0, 0])   # 总格, 满3, 未满
per_ds = defaultdict(lambda: [0, 0])
for (ds, split, algo, k), done in merged.items():
    per_algo[algo][0] += 1
    per_ds[ds][0] += 1
    if done >= 3:
        per_algo[algo][1] += 1
        per_ds[ds][1] += 1
    else:
        per_algo[algo][2] += 1

print("== 按算法：已存在格的完成情况（未满3=待补）")
print("%-34s %5s %5s %5s %s" % ("algorithm", "总格", "满3", "未满", "归类"))
for algo in sorted(per_algo, key=lambda a: -per_algo[a][2]):
    t, f, s = per_algo[algo]
    print("%-34s %5d %5d %5d %s" % (algo, t, f, s, "主对比" if algo in MAIN else "变体/消融"))

print("\n== 按数据集")
for ds in sorted(per_ds, key=lambda d: -(per_ds[d][0] - per_ds[d][1])):
    t, f = per_ds[ds]
    print("  %-10s 总格=%3d 满3=%3d 待补=%3d  %s" % (ds, t, f, t - f, "表格" if ds in TAB else "图像"))

main_gap = sum(per_algo[a][2] for a in per_algo if a in MAIN)
var_gap = sum(per_algo[a][2] for a in per_algo if a not in MAIN)
print("\n主对比算法待补格=%d   变体/消融待补格=%d" % (main_gap, var_gap))
