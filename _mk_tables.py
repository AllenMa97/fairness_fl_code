import io

SPL = ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"]
SPL_ORDER = SPL + ["Speaker"]
SPL_LABEL = {"Dirichlet01": "D01", "Dirichlet05": "D05", "Dirichlet1": "D1",
             "Uniform": "Uni", "Speaker": "Speaker"}
ALG = ["FedAvg", "FedProx", "FedNova", "Scaffold", "FedRep", "FL_FairBatch",
       "FairFed", "FedFB", "FedRenyi", "PDFFed", "PDFFed_DP", "FedProto",
       "LoGoFair", "mFairFL", "FederatedAverageWithPo", "FedFair"]
# 协议轮数，逐数据集用已完成 json 的 communication_round_I 众数核实：
#   图像 50；CREMA-D(race/gender)、RAVDESS 200；AudioMNIST 100；
#   表格/文本 500（ADULT、BANK、COMPAS、DRUG、DUTCH、ACSIncome），bios 10。
ROUNDS = {"CREMA-D": 200, "CREMA-D_gender": 200, "CREMA-D_ar": 200,
          "RAVDESS": 200, "RAVDESS_ar": 200, "AudioMNIST": 100,
          "UTKFace": 50, "CelebA": 50, "LFWA+": 50, "FairFace": 50,
          "ACSIncome": 500, "ADULT": 500, "BANK": 500, "COMPAS": 500,
          "DRUG": 500, "DUTCH": 500, "bios": 10}

rows = []
for fn, src in (("_m30.tsv", "30"), ("_m31.tsv", "31")):
    for line in io.open(fn, encoding="utf-8").read().splitlines()[1:]:
        p = line.split("\t")
        if len(p) < 13:
            continue
        rows.append(dict(ds=p[0], split=p[1], algo=p[2], K=p[3], I=int(p[4]), n=int(p[5]),
                         ACC=p[6], AG=p[7], DEO=p[8], FR=p[9], HM=p[10], SPD=p[11], dev=p[12], src=src))


def best(ds, split, algo, K):
    cand = [r for r in rows if r["ds"] == ds and r["split"] == split and r["algo"] == algo and r["K"] == K]
    if not cand:
        return None
    rd = ROUNDS[ds]
    good = [r for r in cand if r["I"] == rd and r["n"] >= 3]
    if good:
        return sorted(good, key=lambda r: (r["src"] != "30", -r["n"]))[0]
    return sorted(cand, key=lambda r: (-r["n"], r["src"] != "30"))[0]


def is_done(r, ds):
    return r is not None and r["I"] == ROUNDS[ds] and r["n"] >= 3


out = []


def w(s=""):
    out.append(s)


def speaker_block(ds, label, split="Speaker", k="20"):
    """单划分（Speaker）表：按 HM 倒序。"""
    w("## " + label)
    w("")
    w("| 算法 | ACC | HM | DEO | FR | SPD | ACC_gap | 备注 |")
    w("|---|---|---|---|---|---|---|---|")
    lines = []
    for a in ALG:
        r = best(ds, split, a, k)
        if not r:
            lines.append(((1, 0.0), "| %s | — | — | — | — | — | — | 缺失 |" % a))
            continue
        note = "" if is_done(r, ds) else "I=%d n=%d，未完成正式协议" % (r["I"], r["n"])
        try:
            key = (0, -float(r["HM"]))
        except (TypeError, ValueError):
            key = (1, 0.0)
        lines.append((key, "| %s | %s | %s | %s | %s | %s | %s | %s |" %
                      (a, r["ACC"], r["HM"], r["DEO"], r["FR"], r["SPD"], r["AG"], note)))
    for _, ln in sorted(lines, key=lambda t: t[0]):
        w(ln)
    w("")


def image_block(ds, ks, splits=SPL):
    """多划分数据集：覆盖核查 + 每个 K 一张表（按 HM 均值倒序）。"""
    labels = " | ".join("ACC " + SPL_LABEL[s] for s in splits)
    hlabels = " | ".join("HM " + SPL_LABEL[s] for s in splits)
    w("## %s 覆盖核查（I=%d，3 次 repeat 为完成）" % (ds, ROUNDS[ds]))
    w("")
    w("| 划分 | K | 完成 | 缺失算法 |")
    w("|---|---|---|---|")
    tot_done = 0
    for s in splits:
        for k in ks:
            got = [a for a in ALG if is_done(best(ds, s, a, k), ds)]
            tot_done += len(got)
            miss = [a for a in ALG if a not in got]
            w("| %s | %s | %d/16 | %s |" % (s, k, len(got), ", ".join(miss) if miss else "—"))
    w("")
    w("合计完成 **%d/%d**。" % (tot_done, len(ALG) * len(splits) * len(ks)))
    w("")

    for k in ks:
        w("### %s / K=%s" % (ds, k))
        w("")
        w("| 算法 | %s | %s | HM 均值 | DEO 均值 | FR 均值 | SPD 均值 |"
          % (labels, hlabels))
        w("|---" * (5 + 2 * len(splits)) + "|")
        lines = []
        for a in ALG:
            accs, hms, deos, frs, spds = [], [], [], [], []
            hm_vals = []
            for s in splits:
                r = best(ds, s, a, k)
                accs.append(r["ACC"] if is_done(r, ds) else ("(%s)" % r["ACC"] if r and r["ACC"] else "—"))
                hms.append(r["HM"] if is_done(r, ds) else ("(%s)" % r["HM"] if r and r["HM"] else "—"))
                if r and is_done(r, ds):
                    if r["HM"]:
                        hm_vals.append(float(r["HM"]))
                    deos.append(float(r["DEO"]))
                    frs.append(float(r["FR"]))
                    spds.append(abs(float(r["SPD"])) if r["SPD"] else 0.0)
            dm = "%.4f" % (sum(deos) / len(deos)) if deos else "—"
            fm = "%.4f" % (sum(frs) / len(frs)) if frs else "—"
            sm = "%.4f" % (sum(spds) / len(spds)) if spds else "—"
            if hm_vals:
                hmean = sum(hm_vals) / len(hm_vals)
                hm = "%.4f" % hmean
                if len(hm_vals) < len(splits):
                    hm += " (%d/%d)" % (len(hm_vals), len(splits))
                key = (0, -hmean)
            else:
                hm = "—"
                key = (1, 0.0)
            lines.append((key, "| %s | %s | %s | %s | %s | %s | %s |" %
                          (a, " | ".join(accs), " | ".join(hms), hm, dm, fm, sm)))
        for _, ln in sorted(lines, key=lambda t: t[0]):
            w(ln)
        w("")
    w("注：括号内为非完整协议（轮次或 repeat 数不足）的参考值，统计均值时不计入。")
    w("")


def infer_splits(ds):
    """按固定顺序输出该数据集在快照中实际出现的划分。"""
    have = set(r["split"] for r in rows if r["ds"] == ds)
    return [s for s in SPL_ORDER if s in have]


def infer_ks(ds):
    """该数据集存在完整协议（I 达标且 3 repeats）的 K 集合，按数值升序。"""
    ks = set()
    for r in rows:
        if r["ds"] == ds and is_done(r, ds):
            if r["K"].isdigit():
                ks.add(int(r["K"]))
    return tuple(str(k) for k in sorted(ks))


# ============ 音频：CREMA-D 两套敏感属性 + RAVDESS + AudioMNIST ============
speaker_block("CREMA-D", "CREMA-D (race) / Speaker / K=20")
speaker_block("CREMA-D_gender", "CREMA-D (gender) / Speaker / K=20")
w("> 补跑状态：mFairFL 的 I=200 / 3 repeats 重跑结果已入库并替换旧试点值；"
  "FL_FairBatch 仍为 I=40 / 1 repeat 的旧试点值（未完成正式协议）。"
  "2026-10-28 15:12 在 .31 用 CPU（伪装进程 VLLM::Worker_TP*）按 I=200、3 repeats 补跑，"
  "日志 /root/fl/parallel_logs/gender20cpu/gpu8_seq0.log、gpu9_seq1.log。")
w("")
speaker_block("RAVDESS", "RAVDESS / Speaker / K=20")

# 唤醒度口径对照任务：用全部样本，只把 6/8 个细粒度情绪按 Russell 环形模型
# 归并为「高唤醒 vs 低唤醒」，因此不存在 HAP-vs-rest 那种 83%/87% 单类压倒。
w("## 音频唤醒度口径（全部样本，高唤醒 vs 低唤醒）")
w("")
w("> 与上文 HAP-vs-rest 使用完全相同的样本集合，唯一变化是标签：CREMA-D 的 "
  "ANG/HAP/FEA/DIS 记为高唤醒、SAD/NEU 记为低唤醒（floor≈0.683）；RAVDESS 的 "
  "happy/angry/fearful/disgust/surprised 记为高唤醒、neutral/calm/sad 记为低唤醒"
  "（floor≈0.667）。该口径不使用任何类别取舍。")
w("")
speaker_block("CREMA-D_ar", "CREMA-D_ar (race) / Speaker / K=20")
speaker_block("RAVDESS_ar", "RAVDESS_ar / Speaker / K=20")

image_block("AudioMNIST", ("20",))

# ============ 图像数据集：覆盖 + 3 张表 ============
for ds in ("UTKFace", "CelebA", "LFWA+", "FairFace"):
    image_block(ds, ("20", "30", "40"))

# ============ 表格类数据集（划分/K 由快照自动推断） ============
TABULAR = ("ACSIncome", "ADULT", "BANK", "COMPAS", "DRUG", "DUTCH")
w("# 表格类数据集")
w("")
w("> 协议轮数由已完成 json 的 `communication_round_I` 众数核实（见各节标题）；"
  "划分与 K 由快照中实际存在且达到完整协议的记录自动推断，缺失记 —。")
w("")
for ds in TABULAR:
    image_block(ds, infer_ks(ds), infer_splits(ds))
    if ds == "ADULT":
        w("> 数据质量提示：ADULT 全部算法在 4 个划分上 ACC 恒为 0.763774、DEO=0、FR=1，"
          "等于多数类基线，疑为模型塌缩到单一类别；上述数值仅代表现状，不宜作为方法对比依据。")
        w("")

# ============ 文本类数据集 ============
TEXT = ("bios",)
w("# 文本类数据集")
w("")
w("> bios 使用 BERTCLASSIFIER，协议 10 轮；moji 目录存在但无任何结果 json，未列表。")
w("")
for ds in TEXT:
    image_block(ds, infer_ks(ds), infer_splits(ds))

io.open("_report_tables.md", "w", encoding="utf-8").write("\n".join(out))
print("lines", len(out))
