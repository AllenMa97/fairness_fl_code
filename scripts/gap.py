import glob
import json
import os
import sys

MAIN = ["FedAvg", "FedProx", "FedNova", "Scaffold", "FedRep", "FL_FairBatch",
        "FairFed", "FedFB", "FedRenyi", "PDFFed", "PDFFed_DP", "FedProto",
        "LoGoFair", "mFairFL", "FederatedAverageWithPo", "FedFair"]
ALIAS = {"FederatedNova": "FedNova", "FederatedRenyi": "FedRenyi", "Simple_mFairFL": "mFairFL"}
ROUNDS = {
    "ACSIncome": 500, "FairFace": 50, "CREMA-D": 200, "CREMA-D_gender": 200,
    "CREMA-D_ar": 200, "RAVDESS": 200, "RAVDESS_ar": 200,
    "UTKFace": 50, "CelebA": 50, "LFWA+": 50, "bios": 10, "AudioMNIST": 100,
    "ADULT": 500, "BANK": 500, "DUTCH": 500, "COMPAS": 500, "DRUG": 500,
}
ORDER = ["ACSIncome", "ADULT", "BANK", "COMPAS", "DRUG", "DUTCH", "bios",
         "UTKFace", "CelebA", "LFWA+", "FairFace",
         "CREMA-D", "CREMA-D_gender", "CREMA-D_ar", "RAVDESS", "AudioMNIST"]
SPL4 = ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"]
SPK = ["Speaker"]
# 计划网格写死（不靠目录反推：从未启动过的 K 不会建目录，反推会漏列）
PLAN = {ds: (SPL4, ["20", "30", "40"]) for ds in
        ["ACSIncome", "BANK", "COMPAS", "DRUG", "DUTCH", "UTKFace", "CelebA", "LFWA+", "FairFace"]}
PLAN.update({ds: (SPL4, ["20"]) for ds in ["ADULT", "bios", "AudioMNIST"]})
PLAN.update({ds: (SPK, ["20"]) for ds in ["CREMA-D", "CREMA-D_gender", "CREMA-D_ar", "RAVDESS"]})
LABEL = {"CREMA-D": "CREMA-D (race)", "CREMA-D_gender": "CREMA-D (gender)",
         "CREMA-D_ar": "CREMA-D (arousal)", "RAVDESS_ar": "RAVDESS (arousal)"}
ROOT = "/root/fl/result_path"


def row_of(path):
    rel = path.split("result_path/", 1)[-1].split("/")
    if len(rel) < 6:
        return None
    ds, split, algo = rel[0], rel[1], ALIAS.get(rel[2], rel[2])
    if algo not in MAIN:
        return None
    try:
        payload = json.load(open(path))
    except (OSError, ValueError):
        return None
    conf = payload.get("experiment_config", {})
    vals = (payload.get("aggregate", {}).get("metrics", {}).get("ACC", {}) or {}).get("values") or []
    return [ds, split, algo, rel[4].replace("Clients", ""),
            str(conf.get("communication_round_I")), str(len(vals))]


rows = []
for path in glob.glob(os.path.join(ROOT, "**", "*.txt.json"), recursive=True):
    if "_bal" in path or "pilot" in path:
        continue
    r = row_of(path)
    if r:
        rows.append(r)

if len(sys.argv) > 1 and sys.argv[1] == "--dump":
    for r in rows:
        print("\t".join(r))
    sys.exit(0)

if len(sys.argv) > 2 and sys.argv[1] == "--merge":
    for line in open(sys.argv[2], encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 6 and "_bal" not in parts[0] and "pilot" not in parts[0]:
            rows.append(parts)

best = {}
for r in rows:
    key = (r[0], r[1], r[2], r[3])
    cur = best.get(key)
    if cur is None or int(r[5]) > int(cur[5]):
        best[key] = r


def discover(ds):
    if ds in PLAN:
        return PLAN[ds]
    splits, ks = set(), set()
    for d in glob.glob(os.path.join(ROOT, ds, "*")):
        name = os.path.basename(d)
        if os.path.isdir(d) and "pilot" not in name and "bal" not in name:
            splits.add(name)
    for (d, s, a, k) in best:
        if d == ds:
            splits.add(s)
            ks.add(k)
    if not ks:
        ks = {"20"}
    return sorted(splits), sorted(ks, key=lambda x: int(x))


def classify(ds, split, k):
    rounds = ROUNDS.get(ds)
    done, partial, miss = [], [], []
    for algo in MAIN:
        r = best.get((ds, split, algo, k))
        if r and int(r[4]) == rounds and int(r[5]) >= 3:
            done.append(algo)
        elif r:
            partial.append("%s(I=%s,n=%s)" % (algo, r[4], r[5]))
        else:
            miss.append(algo)
    return done, partial, miss


MODE = "--md" if "--md" in sys.argv else "--txt"

if MODE == "--md":
    print("## 1. 总览\n")
    print("| 数据集 | 计划列（划分×K） | 计划格 | 已完成格 | 缺口格 |")
    print("|---|---|---|---|---|")
    tot_col = tot_cell = tot_done = 0
    grid = {}
    for ds in ORDER:
        splits, ks = discover(ds)
        if not splits:
            continue
        grid[ds] = (splits, ks)
        cell = len(splits) * len(ks) * len(MAIN)
        done = 0
        for s in splits:
            for k in ks:
                done += len(classify(ds, s, k)[0])
        tot_col += len(splits) * len(ks)
        tot_cell += cell
        tot_done += done
        print("| %s | %d | %d | %d | %d |" % (LABEL.get(ds, ds), len(splits) * len(ks), cell, done, cell - done))
    print("| **合计** | | **%d** | **%d** | **%d** |" % (tot_cell, tot_done, tot_cell - tot_done))
    print("\n## 2. 逐数据集矩阵（每格 x/16 = 已完成算法数）\n")
    for ds in ORDER:
        if ds not in grid:
            continue
        splits, ks = grid[ds]
        print("### %s（协议 %s 轮，%s）\n" % (LABEL.get(ds, ds), ROUNDS.get(ds), "×".join(ks)))
        print("| 划分 | K | 完成 | 缺失算法 |")
        print("|---|---|---|---|")
        for s in splits:
            for k in ks:
                done, partial, miss = classify(ds, s, k)
                extra = (" 未达标: " + ", ".join(partial)) if partial else ""
                print("| %s | %s | %d/16 | %s%s |" % (s, k, len(done), ", ".join(miss) or "—", extra))
        print()
    print("## 3. 缺口汇总（按算法，统计还缺的格数）\n")
    print("| 算法 | 缺口格 |")
    print("|---|---|")
    per = {}
    for ds, (splits, ks) in grid.items():
        for s in splits:
            for k in ks:
                for algo in classify(ds, s, k)[2]:
                    per[algo] = per.get(algo, 0) + 1
    for algo, n in sorted(per.items(), key=lambda kv: -kv[1]):
        print("| %s | %d |" % (algo, n))
    sys.exit(0)

for ds in ORDER:
    splits, ks = discover(ds)
    if not splits:
        continue
    total = done = 0
    lines = []
    for split in splits:
        for k in ks:
            d, p, m = classify(ds, split, k)
            total += 16
            done += len(d)
            if m or p:
                lines.append("  %-12s K=%-3s 缺%s%s" % (split, k, ",".join(m) or "-",
                                                        ("  [未达标] " + ",".join(p)) if p else ""))
    print("=== %s  I=%s  完成 %d/%d ===" % (ds, ROUNDS.get(ds), done, total))
    for line in lines:
        print(line)
