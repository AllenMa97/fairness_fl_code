import glob
import json
import os
import sys

# 各数据集协议通信轮数的依据（ROUNDS 字典必须与此处一致；gap.py 完成判据以 ROUNDS 为准）：
#
#   音频（AUD_CLF）：
#     - CREMA-D / CREMA-D_gender / CREMA-D_ar / RAVDESS / RAVDESS_ar / VCTK = 200
#         ↑ FedAudio (ICASSP 2023, arXiv:2210.15707) Appendix A.1.1. 的默认值：
#           四集（CREMA-D, Google Speech Commands, IEMOCAP, Urban Sound）统一 communication_round = 200、
#           local epoch E=1；Urban Sound 单列 300 轮但本仓库不跑；CREMA-D 的 Speaker 划分、audio_conv_rnn 模型
#           也都对齐 FedAudio（见 module/partition.py 的 _speaker_partition 注释与 hypothesis/AUDIOCNNRNN.py）。
#         ↑ VCTK 与 RAVDESS 沿用同一协议：任务都是按说话人/演员划分 + 情感/说话人判别，单轮计算成本相当。
#     - AudioMNIST = 100
#         ↑ Becker et al., "AudioMNIST: Exploring Explainable AI for audio analysis on a simple benchmark",
#           Journal of the Franklin Institute, 2023. 是原始数据集作者给出的 FL 基线轮数；FedFACT-2025 等后续工作
#           在 AudioMNIST 上也沿用 I=100。注意：FedAudio 基准本身**不包含** AudioMNIST 这一集。
#
#   图像（IMG_CLF）：
#     - UTKFace / CelebA / LFWA+ / FairFace / PAD-UFES-20 = 50
#         ↑ 图像 FL 主流协议（FedAvg / FairFed / Fairness-in-FL 基线）在 CelebA / FairFace / UTKFace 上都用 I=50。
#
#   表格（Tabular_CLF）：
#     - ACSIncome / ADULT / BANK / COMPAS / DRUG / DUTCH = 500
#         ↑ FL fairness 文献在表格数据集（ADULT / COMPAS / DUTCH / DRUG / BANK）上的标准协议。
#     - bios = 10
#         ↑ FedFACT 原文在 bios 上的协议轮数。

MAIN = ["FedAvg", "FedProx", "FedNova", "Scaffold", "FedRep", "FL_FairBatch",
        "FairFed", "FedFB", "FedRenyi", "PDFFed", "PDFFed_DP", "FedProto",
        "LoGoFair", "mFairFL", "FederatedAverageWithPo", "FedFair",
        "FedFACT", "PraFFL"]
ALIAS = {"FederatedNova": "FedNova", "FederatedRenyi": "FedRenyi", "Simple_mFairFL": "mFairFL"}
ROUNDS = {
    "ACSIncome": 500, "FairFace": 50, "CREMA-D": 200, "CREMA-D_gender": 200,
    "CREMA-D_ar": 200, "RAVDESS": 200, "RAVDESS_ar": 200,
    "UTKFace": 50, "CelebA": 50, "LFWA+": 50, "bios": 10, "AudioMNIST": 100,
    "ADULT": 500, "BANK": 500, "DUTCH": 500, "COMPAS": 500, "DRUG": 500,
    "PAD-UFES-20": 50, "VCTK": 200,
}
ORDER = ["ACSIncome", "ADULT", "BANK", "COMPAS", "DRUG", "DUTCH", "bios",
         "UTKFace", "CelebA", "LFWA+", "FairFace", "PAD-UFES-20",
         "CREMA-D", "CREMA-D_gender", "CREMA-D_ar", "RAVDESS", "VCTK", "AudioMNIST"]
SPL4 = ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"]
SPK = ["Speaker"]
# 计划网格写死（不靠目录反推：从未启动过的 K 不会建目录，反推会漏列）
PLAN = {ds: (SPL4, ["20", "30", "40"]) for ds in
        ["ACSIncome", "BANK", "COMPAS", "DRUG", "DUTCH", "UTKFace", "CelebA", "LFWA+",
         "FairFace", "PAD-UFES-20"]}
PLAN.update({ds: (SPL4, ["20"]) for ds in ["ADULT", "bios", "AudioMNIST"]})
PLAN.update({ds: (SPK, ["20"]) for ds in ["CREMA-D", "CREMA-D_gender", "CREMA-D_ar",
                                          "RAVDESS", "VCTK"]})
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


def prefer(new, cur, rounds):
    """同格多份结果时的取舍：先看轮数是否等于协议轮数，再看 repeat 数。

    跨机合并时两机可能各有一份同格结果（例如误跑的 200 轮 AudioMNIST 覆盖了
    100 轮原件）。若只比 ACC.values 长度，会选中轮数不符的那份，把本来已完成
    的格判成「未达标」。所以协议轮数匹配优先，其次再比 repeat 数。
    """
    if cur is None:
        return True
    if rounds is not None:
        cur_ok = int(cur[4]) == rounds
        new_ok = int(new[4]) == rounds
        if cur_ok != new_ok:
            return new_ok
    return int(new[5]) > int(cur[5])


best = {}
for r in rows:
    key = (r[0], r[1], r[2], r[3])
    if prefer(r, best.get(key), ROUNDS.get(r[0])):
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
    print("\n## 2. 逐数据集矩阵（每格 x/%d = 已完成算法数）\n" % len(MAIN))
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
                print("| %s | %s | %d/%d | %s%s |" % (s, k, len(done), len(MAIN), ", ".join(miss) or "—", extra))
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
            total += len(MAIN)
            done += len(d)
            if m or p:
                lines.append("  %-12s K=%-3s 缺%s%s" % (split, k, ",".join(m) or "-",
                                                        ("  [未达标] " + ",".join(p)) if p else ""))
    print("=== %s  I=%s  完成 %d/%d ===" % (ds, ROUNDS.get(ds), done, total))
    for line in lines:
        print(line)
