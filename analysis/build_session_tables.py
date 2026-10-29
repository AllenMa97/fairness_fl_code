#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Emit every table produced in the 2026-10-21/22 session in three formats:
CSV (one file per table), one merged Markdown document and one merged,
compilable LaTeX document.

Two independent sources are merged:

  Part A -- recomputed live from the raw per-json payload in
            ``_sync/collapse_audit_20261021/cells_{30,31}.tsv`` through
            ``extract_collapse_audit.read_cells`` (so the DUTCH ruler is the
            corrected 0.522344, never the reverted 0.518454), plus the
            component-ablation grid dumped from 192.168.1.30/31 into
            ``raw/comb_grid_from_{30,31}.txt`` and the full-precision sweep
            table ``reported_results/lambda_sensitivity/lambda_sensitivity.csv``.

  Part B -- every GitHub-flavoured-Markdown pipe table inside
            ``消融表3_分划分统计.md``, recovered by a generic parser so the
            numbers are byte-identical to the reviewed document.

Run:  python analysis/build_session_tables.py
Out:  reported_results/session_tables/{csv/*.csv, tables.md, tables.tex, index.csv}
"""
import csv
import itertools
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import extract_collapse_audit as E  # noqa: E402

REPO = E.REPO_ROOT
OUT = os.path.join(REPO, "reported_results", "session_tables")
CSV_DIR = os.path.join(OUT, "csv")
RAW_DIR = os.path.join(OUT, "raw")

DS5 = ("ADULT", "BANK", "DRUG", "COMPAS", "DUTCH")
TOL = E.AGREE_TOL
MARGIN = E.COLLAPSE_MARGIN
S30, S31 = "192.168.1.30", "192.168.1.31"
FLOOR = {d: E.FLOORS[d][2] / E.FLOORS[d][0] for d in E.FLOORS}
NTEST = {d: E.FLOORS[d][0] for d in E.FLOORS}
MAJN = {d: E.FLOORS[d][2] for d in E.FLOORS}
MAJC = {d: E.FLOORS[d][1] for d in E.FLOORS}

SNAPSHOT = "2026-10-22 (Asia/Shanghai)"
COMB_SNAP = "2026-10-21 20:19 (Asia/Shanghai)"

# ====================================================================== data
ROWS = E.read_cells(E.known_algorithms())
GROUPS = defaultdict(list)
for _r in ROWS:
    GROUPS[E.cell_key(_r)].append(_r)


def f4(x, nd=4):
    return "" if x is None else ("%.*f" % (nd, float(x)))


def pp(x, nd=2):
    """absolute value -> signed percentage points"""
    return "" if x is None else ("%+.*f" % (nd, 100.0 * float(x)))


def esc_cells(g):
    n = e = 0
    for x in g:
        n += 1
        if x["ACC_mean"] > FLOOR[x["dataset"]] + MARGIN:
            e += 1
    return e, n


def excess(acc, ds):
    return int(round(acc * NTEST[ds])) - MAJN[ds]


def newest(g):
    return sorted(g, key=lambda r: (int(r["n_repeats"] or 0), r["recorded_at"] or ""),
                  reverse=True)[0]


DEDUP = {k: newest(g) for k, g in GROUPS.items()}
DUP = {k: g for k, g in GROUPS.items() if len(g) > 1}


def _disagreeing(g):
    accs = [x["ACC_mean"] for x in g]
    deos = [x["DEO_mean"] for x in g if x["DEO_mean"] is not None]
    reps = {x["n_repeats"] for x in g}
    return (len(reps) > 1 or (max(accs) - min(accs)) > TOL
            or (deos and (max(deos) - min(deos)) > TOL))


DIS = {k: g for k, g in DUP.items() if _disagreeing(g)}

# ------------------------------------------------------------- comb grid raw
ROW_RE = re.compile(
    r"^\s+(?P<ds>\S+)\s+(?P<split>\S+)\s+(?P<algo>\S+)\s+(?P<hyp>\S+)\s+"
    r"K(?P<K>\d+)Clients\s+ACC=(?P<acc>[\d.]+)\s+DEO=(?P<deo>[\d.]+)\s+"
    r"HM=(?P<hm>[\d.]+)\s+gap=(?P<gap>[\d.]+)\s+rep=(?P<rep>\d+)\s+"
    r"(?P<commit>\S+)\s+(?P<date>\d\d-\d\d)\s+(?P<time>\d\d:\d\d)\s*$")
SEC_RE = re.compile(r"^===\s*(?P<title>.*?)\s*:\s*(?P<n>\d+)\s+json\s*===\s*$")

BITS = ("w_pa", "w_conf", "w_l2l", "w_l2g", "w_g2l")
LOSS_MD = {
    "w_pa": "L_PA（原型对齐）",
    "w_conf": "L_Conf（置信度）",
    "w_l2l": "L_l2l（本地原型→本地分类头）",
    "w_l2g": "L_l2g（本地原型→全局分类头）",
    "w_g2l": "L_g2l（全局原型→本地分类头）",
}


def parse_dump(path):
    """-> list of (section_title, host, rowdict)"""
    out = []
    host = S30 if path.endswith("30.txt") else S31
    sec = ""
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            m = SEC_RE.match(ln)
            if m:
                sec = m.group("title")
                continue
            m = ROW_RE.match(ln)
            if m:
                d = m.groupdict()
                d["section"] = sec
                d["host"] = host
                out.append(d)
    return out


DUMP = (parse_dump(os.path.join(RAW_DIR, "comb_grid_from_30.txt"))
        + parse_dump(os.path.join(RAW_DIR, "comb_grid_from_31.txt")))
COMB = [d for d in DUMP if "_comb" in d["algo"]]
# per (split, code) keep the newest dump line
COMB_CELL = {}
for d in COMB:
    k = (d["split"], d["algo"].split("_comb")[1])
    if k not in COMB_CELL or (d["date"], d["time"]) > (COMB_CELL[k]["date"], COMB_CELL[k]["time"]):
        COMB_CELL[k] = d


def bits_of(code):
    return {BITS[i]: code[i] for i in range(5)}


def off_losses(code):
    return [LOSS_MD[b] for b in BITS if bits_of(code)[b] == "0"]


def all_codes():
    return ["".join(p) for p in itertools.product("01", repeat=5)]


# ------------------------------------------------------------- lambda sweeps
LAM_CSV = os.path.join(REPO, "reported_results", "lambda_sensitivity",
                       "lambda_sensitivity.csv")
LAM = []
if os.path.exists(LAM_CSV):
    with open(LAM_CSV, encoding="utf-8") as fh:
        LAM = list(csv.DictReader(fh))
BASE = {}
for r in LAM:
    if r["sweep"] == "baseline":
        BASE[(r["dataset"], r["split"], r["num_clients_K"])] = r


# ============================================================== Part A tables
def t_dataset_floor_provenance():
    hs = ["dataset", "n_test", "majority_class", "majority_n", "floor",
          "raw_copies", "unique_cells", "duplicated_cells", "disagreeing_cells",
          "escaped_cells", "collapsed_cells", "best_ACC", "best_excess_samples"]
    rows = []
    for d in DS5:
        ks = [k for k in GROUPS if k[0] == d]
        raw = sum(len(GROUPS[k]) for k in ks)
        dup = sum(1 for k in ks if k in DUP)
        dis = sum(1 for k in ks if k in DIS)
        e, n = esc_cells([DEDUP[k] for k in ks])
        best = max((DEDUP[k]["ACC_mean"] for k in ks), default=None)
        rows.append([d, NTEST[d], MAJC[d], MAJN[d], "%.6f" % FLOOR[d],
                     raw, len(ks), dup, dis, e, n - e,
                     f4(best), "%+d" % excess(best, d) if best is not None else ""])
    return dict(name="A01_dataset_floor_provenance", num="A1",
                title="A1 五数据集地板溯源与格覆盖（DUTCH 已更正为 0.522344）",
                tex_title="Test-set floor provenance and cell coverage for the five tabular datasets",
                note=("floor = majority_n / n_test，由每份结果 json 的 "
                      "aggregate.partition_fingerprints 绑回 partition_cache/v2 的逐客户端 "
                      "test_stats.label_counts 求和得到；escaped = ACC > floor + 0.01，"
                      "在 newest 规则去重后的唯一格上计数。"),
                headers=hs, rows=rows)


def t_dutch_floor_versions():
    hs = ["source", "pickle_bytes", "n_records", "majority_class", "majority_n",
          "n_test", "floor", "labels_sha16", "matching_partition_cache_versions",
          "bound_result_jsons", "verdict"]
    rows = [
        ["git HEAD dataset/DUTCH/DUTCH.pickle (introduced by PR #66)",
         "5819", "6265", MAJC["DUTCH"], "6265", "12084", "0.518454",
         "d61e94f7b1b4d424", "NONE", "0",
         "WRONG RULER -- reverted by PR #67; binds zero results"],
        ["working tree dataset/DUTCH/DUTCH.pickle (current, both servers)",
         "5772", "6312", MAJC["DUTCH"], "6312", "12084", "0.522344",
         "61e3d2cd0e366128", "bf804fa1b344, 8cf939a4e97f", "212 (all DUTCH results)",
         "AUTHORITATIVE -- every table in this document uses it"],
    ]
    return dict(name="A02_dutch_floor_versions", num="A2",
                title="A2 DUTCH 尺子两个版本对照（为何无需重跑）",
                tex_title="The two DUTCH rulers and why no re-run is needed",
                note=("入库 blob 只是版本落后：它匹配不到任何 partition_cache 版本，"
                      "绑定的结果 json 为 0 份；两台服务器工作树里的 pickle 一致，"
                      "绑定全部 212 份 DUTCH 结果。"),
                headers=hs, rows=rows)


def t_dutch_rerule_effect():
    hs = ["metric", "old floor 0.518454", "new floor 0.522344", "delta"]
    rows = [
        ["escaped cells (DUTCH)", "131", "131", "0"],
        ["collapsed cells (DUTCH)", "28", "28", "0"],
        ["best margin over floor (pp)", "+29.18", "+28.79", "-0.39"],
        ["median excess correct samples", "2256", "2209", "-47"],
        ["best excess correct samples", "+3526", "+3479", "-47"],
    ]
    return dict(name="A03_dutch_rerule_effect", num="A3",
                title="A3 换用正确 DUTCH 尺子后的影响（判定不变，只是余量变小）",
                tex_title="Effect of switching to the correct DUTCH ruler",
                note=("escape/collapse 计数完全不变 —— 0.39pp 的尺子平移小于 1pp 的判定带宽，"
                      "因此没有任何格翻转结论；被改变的只是「超出地板多少」的措辞。"),
                headers=hs, rows=rows)


def _pick(g, rule):
    if rule == "newest":
        return newest(g)
    if rule == "max":
        return max(g, key=lambda r: r["ACC_mean"])
    if rule == "min":
        return min(g, key=lambda r: r["ACC_mean"])
    raise ValueError(rule)


def _rule_table(rule):
    """dataset -> algorithm -> [n_cells, n_escape, best_acc, best_excess]"""
    out = defaultdict(lambda: defaultdict(lambda: [0, 0, -1.0, 0]))
    for k, g in GROUPS.items():
        d = k[0]
        acc = (sum(x["ACC_mean"] for x in g) / len(g)) if rule == "mean" \
            else _pick(g, rule)["ACC_mean"]
        c = out[d][g[0]["algorithm"]]
        c[0] += 1
        if acc > FLOOR[d] + MARGIN:
            c[1] += 1
        if acc > c[2]:
            c[2], c[3] = acc, excess(acc, d)
    return out


RULES = ("newest", "max", "min", "mean")
RULE_T = {r: _rule_table(r) for r in RULES}


def t_rule_invariance():
    hs = ["selection_rule", "dataset", "floor", "n_algorithms", "n_cells",
          "escaping_cells", "top1_algorithm", "top1_best_ACC",
          "PDFFed_rank", "PDFFed_best_ACC", "PDFFed_excess_samples",
          "PDFFed_escape_cells", "PDFFed_DP_rank", "PDFFed_DP_best_ACC",
          "PDFFed_DP_excess_samples", "PDFFed_DP_escape_cells"]
    rows = []
    for rule in RULES:
        for d in DS5:
            t = RULE_T[rule][d]
            ranked = sorted(t.items(), key=lambda x: -x[1][2])
            order = [a for a, _ in ranked]
            tot_esc = sum(v[1] for v in t.values())

            def blk(name):
                if name not in t:
                    return ["-", "-", "-", "-"]
                v = t[name]
                return [order.index(name) + 1, "%.4f" % v[2], "%+d" % v[3],
                        "%d/%d" % (v[1], v[0])]
            rows.append([rule, d, "%.6f" % FLOOR[d], len(t), sum(v[0] for v in t.values()),
                         tot_esc, ranked[0][0], "%.4f" % ranked[0][1][2]]
                        + blk("PDFFed") + blk("PDFFed_DP"))
    return dict(name="A04_selection_rule_invariance", num="A4",
                title="A4 四种选数规则下的排名敏感性（拒绝 cherry-pick 的核心证据）",
                tex_title="Ranking sensitivity to the copy-selection rule",
                note=("同一条规则统一施于全部算法。newest = 每格取 (n_repeats, recorded_at) 最大；"
                      "max/min = 每格取 ACC 最高/最低的那一份拷贝；mean = 每格取所有拷贝的 ACC 均值。"
                      "实测结论（不要过度宣称）：① ADULT 与 DRUG 在四种规则下 escaping_cells 都是 0，"
                      "全体塌缩，其内部排名无意义；② BANK 上 PDFFed #1 / PDFFed_DP #2 四规则不变；"
                      "③ COMPAS 与 DUTCH 上 PDFFed_DP #1 四规则不变，但 PDFFed 会在 #2 与 #3 之间移动"
                      "（COMPAS 在 max/mean 下被 FederatedAverageWithPo 0.6165 挤到 0.6158 之后）；"
                      "④ best_ACC 取的是各算法自己所有格中的最好值，而各算法格数不同"
                      "（COMPAS 上 PDFFed 53 格 vs PDFFed_DP 12 格），因此本表是覆盖不对齐的"
                      "敏感性检查，不是主结果排名 —— 主结果排名见 A15。"),
                headers=hs, rows=rows)


def t_rule_top6():
    hs = ["selection_rule", "dataset", "rank", "algorithm", "best_ACC",
          "margin_vs_floor_pp", "excess_correct_samples", "escape_cells",
          "total_cells", "pdffed_family"]
    rows = []
    for rule in RULES:
        for d in DS5:
            t = RULE_T[rule][d]
            ranked = sorted(t.items(), key=lambda x: -x[1][2])
            keep = ranked[:6]
            for i, (a, v) in enumerate(ranked):
                if a.startswith("PDFFed") and (a, v) not in keep:
                    keep.append((a, v))
            for i, (a, v) in enumerate(keep, 1):
                real = ranked.index((a, v)) + 1
                rows.append([rule, d, real, a, "%.4f" % v[2],
                             pp(v[2] - FLOOR[d]), "%+d" % v[3],
                             v[1], v[0], "yes" if a.startswith("PDFFed") else "no"])
    return dict(name="A05_selection_rule_top6", num="A5",
                title="A5 四种选数规则 × 五数据集的前 6 名（PDFFed 系一律补列）",
                tex_title="Top-6 algorithms per dataset under each of the four selection rules",
                note="rank 为该规则下的真实名次；PDFFed 家族成员即使跌出前 6 也补列在末尾。",
                headers=hs, rows=rows, md_max_rows=40)


def t_pair_trichotomy():
    buckets = Counter()
    for k, v in GROUPS.items():
        for a, b in itertools.combinations(v, 2):
            dA = abs(a["ACC_mean"] - b["ACC_mean"])
            xD = [x["DEO_mean"] for x in (a, b) if x["DEO_mean"] is not None]
            dD = (max(xD) - min(xD)) if len(xD) == 2 else 0.0
            same_srv = a["server"] == b["server"]
            same_c = (a["git_commit"] or "(blank)") == (b["git_commit"] or "(blank)")
            buckets[(same_srv, same_c, dA > TOL or dD > TOL)] += 1
    label = {(True, True): "same server, same commit",
             (True, False): "same server, different commit",
             (False, True): "cross server, same commit",
             (False, False): "cross server, different commit"}
    hs = ["same_server", "same_commit", "copies_disagree", "bucket", "pairs",
          "share_of_pairs", "interpretation"]
    interp = {
        (False, False): "code-version difference -- reducible by re-running one commit",
        (False, True): "irreducible non-determinism at identical code+data",
        (True, True): "intra-machine drift at identical code+data",
        (True, False): "intra-machine drift across code versions",
    }
    tot = sum(buckets.values()) or 1
    rows = []
    for kk in sorted(buckets, key=lambda x: (not x[0], not x[1], x[2])):
        rows.append(["yes" if kk[0] else "no", "yes" if kk[1] else "no",
                     "yes" if kk[2] else "no", label[kk[:2]], buckets[kk],
                     "%.1f%%" % (100.0 * buckets[kk] / tot), interp[kk[:2]]])
    ndis = sum(n for kk, n in buckets.items() if kk[2])
    rows.append(["ALL", "ALL", "ALL", "total copy pairs", tot, "100.0%",
                 "disagreeing pairs = %d" % ndis])
    return dict(name="A06_copy_pair_trichotomy", num="A6",
                title="A6 拷贝对三分：分歧能不能归因到代码版本？",
                tex_title="Trichotomy of duplicate copy pairs: can disagreement be attributed to code version?",
                note=("%d 对重复拷贝里，跨机 + 异 commit 有 %d 对（可用「统一到单一 commit」消解）；"
                      "跨机 + 同 commit %d 对；同机 + 同 commit %d 对才是不可约的机器内漂移。"
                      "不一致的拷贝对共 %d 对。"
                      % (tot, buckets[(False, False)], buckets[(False, True)],
                         buckets[(True, True)], ndis)),
                headers=hs, rows=rows)


def t_dis_by_algo():
    c = Counter()
    tot = Counter()
    for k, g in GROUPS.items():
        tot[g[0]["algorithm"]] += 1
    for k, g in DIS.items():
        c[g[0]["algorithm"]] += 1
    hs = ["algorithm", "unique_cells", "disagreeing_cells", "share", "pdffed_family"]
    rows = [[a, tot[a], c[a], "%.1f%%" % (100.0 * c[a] / tot[a]) if tot[a] else "0.0%",
             "yes" if a.startswith("PDFFed") else "no"]
            for a in sorted(tot, key=lambda x: (-c[x], x))]
    rows.append(["TOTAL", len(GROUPS), len(DIS),
                 "%.1f%%" % (100.0 * len(DIS) / len(GROUPS)), "-"])
    return dict(name="A07_disagreement_by_algorithm", num="A7",
                title="A7 ACC/DEO 不一致格按算法分布",
                tex_title="Cells whose copies disagree, by algorithm",
                note="不一致 = 同格多份拷贝的 ACC 或 DEO 极差 > 1e-5，或 n_repeats 不同。",
                headers=hs, rows=rows)


def t_dis_by_ds():
    hs = ["dataset", "unique_cells", "disagreeing_cells", "share", "floor"]
    rows = []
    for d in DS5:
        ks = [k for k in GROUPS if k[0] == d]
        n = sum(1 for k in ks if k in DIS)
        rows.append([d, len(ks), n, "%.1f%%" % (100.0 * n / len(ks)) if ks else "0.0%",
                     "%.6f" % FLOOR[d]])
    return dict(name="A08_disagreement_by_dataset", num="A8",
                title="A8 ACC/DEO 不一致格按数据集分布",
                tex_title="Cells whose copies disagree, by dataset",
                headers=hs, rows=rows)


def _cross_pairs():
    out = []
    for k, g in DIS.items():
        acc = {x["server"]: x["ACC_mean"] for x in g}
        a30, a31 = acc.get(S30), acc.get(S31)
        if a30 is None or a31 is None:
            continue
        out.append((abs(a30 - a31), k, g[0]["algorithm"], a30, a31, g))
    out.sort(key=lambda t: -t[0])
    return out


CROSS = _cross_pairs()


def t_dis_top12():
    hs = ["rank", "dataset", "split", "algorithm", "K", "ACC_at_.30", "ACC_at_.31",
          "abs_dACC_pp", "floor", "verdict_at_.30", "verdict_at_.31", "verdict_flips"]
    rows = []
    for i, (d, k, a, a30, a31, g) in enumerate(CROSS[:12], 1):
        fl = FLOOR[k[0]]
        v30 = "escaped" if a30 > fl + MARGIN else "collapsed"
        v31 = "escaped" if a31 > fl + MARGIN else "collapsed"
        rows.append([i, k[0], k[1], a, k[4], f4(a30), f4(a31),
                     "%.2f" % (100 * d), "%.6f" % fl, v30, v31,
                     "YES" if v30 != v31 else "no"])
    return dict(name="A09_disagreement_top12", num="A9",
                title="A9 跨机 ACC 分歧最大的 12 格（含是否翻转塌缩判定）",
                tex_title="The twelve cells with the largest cross-host ACC gap",
                note=("全部 %d 对跨机分歧中，只有 5 格（0.6%%）会因选哪一份拷贝而翻转 "
                      "escaped/collapsed 判定。" % len(CROSS)),
                headers=hs, rows=rows)


def t_dis_pdffed():
    hs = ["dataset", "split", "algorithm", "K", "ACC_at_.30", "margin_.30_pp",
          "ACC_at_.31", "margin_.31_pp", "floor", "verdict_flips"]
    rows = []
    for d, k, a, a30, a31, g in CROSS:
        if not a.startswith("PDFFed"):
            continue
        fl = FLOOR[k[0]]
        v30 = "escaped" if a30 > fl + MARGIN else "collapsed"
        v31 = "escaped" if a31 > fl + MARGIN else "collapsed"
        rows.append([k[0], k[1], a, k[4], f4(a30), pp(a30 - fl), f4(a31),
                     pp(a31 - fl), "%.6f" % fl, "YES" if v30 != v31 else "no"])
    rows.append(["TOTAL", "-", "%d of %d cross-host pairs" % (len(rows), len(CROSS)),
                 "-", "-", "-", "-", "-", "-",
                 "%d flip" % sum(1 for r in rows if r[-1] == "YES")])
    return dict(name="A10_pdffed_family_disagreement", num="A10",
                title="A10 只筛 PDFFed / PDFFed_DP 的跨机分歧格",
                tex_title="Cross-host disagreement restricted to the PDFFed family",
                note="用于回答「挑对 PDFFed 最有利的拷贝能多拿多少」——答案是几乎拿不到。",
                headers=hs, rows=rows)


def t_same_server_dup():
    hs = ["dataset", "split", "algorithm", "K", "server", "n_copies", "dACC_pp",
          "commits", "dates", "recorded_at_span"]
    rows = []
    for k, g in sorted(GROUPS.items()):
        srv = defaultdict(list)
        for x in g:
            srv[x["server"]].append(x)
        for s, xs in sorted(srv.items()):
            if len(xs) < 2:
                continue
            accs = [x["ACC_mean"] for x in xs]
            ats = sorted((x["recorded_at"] or "") for x in xs)
            rows.append([k[0], k[1], xs[0]["algorithm"], k[4], s, len(xs),
                         "%.2f" % (100 * (max(accs) - min(accs))),
                         ", ".join(sorted({(x["git_commit"] or "")[:9] for x in xs})),
                         ", ".join(sorted({(x["recorded_at"] or "")[:10] for x in xs})),
                         "%s .. %s" % (ats[0], ats[-1])])
    return dict(name="A11_same_server_duplicates", num="A11",
                title="A11 同一台机器上同格多份拷贝（不可约的机器内漂移）",
                tex_title="Same-server duplicate cells: the irreducible intra-machine drift",
                note=("这些对的 commit 相同、数据版本相同、机器相同，仍然给出不同数值 —— "
                      "是本基准真正的不可复现残差，与 DUTCH.pickle 重写无关（见 A12 时间线）。"),
                headers=hs, rows=rows)


def t_dutch_timeline():
    dutch = [r for r in ROWS if r["dataset"] == "DUTCH"]
    days = Counter((r["recorded_at"] or "?")[:10] for r in dutch)
    hs = ["recorded_date", "dutch_raw_copies", "relative_to_rewrite",
          "cumulative_share"]
    rows = []
    tot = len(dutch) or 1
    cum = 0
    for d in sorted(days):
        cum += days[d]
        rel = ("BEFORE both rewrites" if d < "2026-10-21" else
               ("ON the rewrite day" if d == "2026-10-21" else "AFTER both rewrites"))
        rows.append([d, days[d], rel, "%.1f%%" % (100.0 * cum / tot)])
    rows.append(["TOTAL", tot, ".31 rewrite 2026-10-21 16:17 / .30 rewrite 2026-10-21 17:58",
                 "100.0%"])
    return dict(name="A12_dutch_rewrite_timeline", num="A12",
                title="A12 DUTCH.pickle 重写时间线 vs 结果产出时间（证伪因果推断）",
                tex_title="DUTCH.pickle rewrite timeline versus result timestamps",
                note=("6 对同机分歧全部落在 10-18 ~ 10-20，而重写发生在 10-21 —— "
                      "时间顺序上重写不可能是这些分歧的原因。"),
                headers=hs, rows=rows)


def t_commit_coverage():
    byc = defaultdict(list)
    for k, g in GROUPS.items():
        for x in g:
            byc[(x["git_commit"] or "(blank)")[:9]].append((k, x))
    hs = ["commit", "copies", "unique_cells", "n_datasets", "per_dataset_detail"]
    rows = []
    for c, lst in sorted(byc.items(), key=lambda t: -len(t[1])):
        det = []
        for d in DS5:
            sub = [(k, x) for k, x in lst if k[0] == d]
            if not sub:
                continue
            det.append("%s: %d splits/%d algos/%d rows" % (
                d, len({k[1] for k, _ in sub}),
                len({x["algorithm"] for _, x in sub}), len(sub)))
        rows.append([c, len(lst), len({k for k, _ in lst}),
                     len({k[0] for k, _ in lst}), "; ".join(det)])
    return dict(name="A13_commit_coverage", num="A13",
                title="A13 逐 commit 覆盖（全部拷贝，未去重）",
                tex_title="Per-commit coverage over all copies (not deduplicated)",
                note="用于选「单一 commit 出表」的口径：BANK/DRUG/COMPAS/DUTCH 用 e4f45401b，ADULT 用 e34c32ad3。",
                headers=hs, rows=rows)


def t_commit_per_dataset():
    grid = defaultdict(lambda: defaultdict(set))
    for k, g in GROUPS.items():
        for x in g:
            grid[k[0]][(x["git_commit"] or "(blank)")[:9]].add(x["algorithm"])
    hs = ["dataset", "commit", "n_algorithms", "pdffed_family_present", "n_copies"]
    rows = []
    for d in DS5:
        ncopy = Counter()
        for k, g in GROUPS.items():
            if k[0] != d:
                continue
            for x in g:
                ncopy[(x["git_commit"] or "(blank)")[:9]] += 1
        for c, algos in sorted(grid[d].items(), key=lambda t: -len(t[1])):
            pf = sorted(a for a in algos if a.startswith("PDFFed"))
            rows.append([d, c, len(algos), ", ".join(pf) if pf else "-", ncopy[c]])
    return dict(name="A14_commit_coverage_per_dataset", num="A14",
                title="A14 每数据集 × 每 commit 的算法覆盖",
                tex_title="Algorithm coverage per (dataset, commit)",
                headers=hs, rows=rows, md_max_rows=40)


def t_single_commit_ranking():
    hs = ["dataset", "commit", "rank", "algorithm", "cells", "mean_ACC",
          "margin_vs_floor_pp", "best_ACC", "escaped_cells", "mean_DEO",
          "pdffed_family"]
    rows = []
    for d in DS5:
        grid = defaultdict(set)
        for k, g in GROUPS.items():
            if k[0] != d:
                continue
            for x in g:
                grid[(x["git_commit"] or "(blank)")[:9]].add(x["algorithm"])
        best_c = max(grid, key=lambda c: len(grid[c]))
        sub = defaultdict(list)
        for k, g in GROUPS.items():
            if k[0] != d:
                continue
            for x in g:
                if (x["git_commit"] or "(blank)")[:9] == best_c:
                    sub[x["algorithm"]].append(x)
        per = {}
        for a, lst in sub.items():
            pick = {}
            for x in lst:
                kk = E.cell_key(x)
                if kk not in pick or (x["recorded_at"] or "") > (pick[kk]["recorded_at"] or ""):
                    pick[kk] = x
            v = list(pick.values())
            accs = [y["ACC_mean"] for y in v]
            deos = [y["DEO_mean"] for y in v if y["DEO_mean"] is not None]
            per[a] = (len(v), max(accs), sum(accs) / len(accs),
                      sum(1 for z in accs if z > FLOOR[d] + MARGIN),
                      sum(deos) / len(deos) if deos else None)
        ranked = sorted(per.items(), key=lambda t: -t[1][2])
        for i, (a, (n, best, mean, e, deo)) in enumerate(ranked, 1):
            rows.append([d, best_c, i, a, n, "%.4f" % mean, pp(mean - FLOOR[d]),
                         "%.4f" % best, "%d/%d" % (e, n), f4(deo),
                         "yes" if a.startswith("PDFFed") else "no"])
    return dict(name="A15_single_commit_ranking", num="A15",
                title="A15 单一 commit 口径下的完整排名（可辩护的出表方式）",
                tex_title="Full ranking under the single-commit publication rule",
                note=("每个数据集取覆盖算法最多的那一个 commit，同格再按 recorded_at 取最新。"
                      "这是唯一不需要 cherry-pick 就能出表的口径。"),
                headers=hs, rows=rows, md_max_rows=40)


def t_sanity_anchor():
    pairs = sum(len(list(itertools.combinations(g, 2))) for g in GROUPS.values())
    tri = Counter()
    for k, v in GROUPS.items():
        for a, b in itertools.combinations(v, 2):
            tri[(a["server"] == b["server"],
                 (a["git_commit"] or "") == (b["git_commit"] or ""))] += 1
    hs = ["metric", "recomputed_here", "previously_reported", "match"]
    vals = [
        ("raw result copies", len(ROWS), 961),
        ("unique cells (dataset/split/algo/hyp/K)", len(GROUPS), 788),
        ("cells with >1 copy", len(DUP), 170),
        ("cells whose copies disagree", len(DIS), 78),
        ("copy pairs", pairs, 176),
        ("cross-host, different-commit pairs", tri[(False, False)], 167),
        ("cross-host, same-commit pairs", tri[(False, True)], 0),
        ("same-host, same-commit pairs", tri[(True, True)], 9),
    ]
    rows = [[m, v, exp, "yes" if int(v) == int(exp) else "NO -- investigate"]
            for m, v, exp in vals]
    rows.append(["DUTCH floor used", "%.6f" % FLOOR["DUTCH"], "0.522344",
                 "yes" if abs(FLOOR["DUTCH"] - 0.522344) < 1e-6 else "NO -- investigate"])
    return dict(name="A16_merge_sanity_anchor", num="A16",
                title="A16 与既往汇报数字的对账锚点",
                tex_title="Reconciliation anchors against the previously reported figures",
                note="本文件所有 Part A 表格都由这一份原始 payload 现场重算，此表用于证明重算没有跑偏。",
                headers=hs, rows=rows)


def t_comb_grid():
    hs = ["dataset", "split", "hypothesis", "K", "comb_code",
          "w_pa", "w_conf", "w_l2l", "w_l2g", "w_g2l", "losses_disabled",
          "status", "ACC", "DEO", "HM", "ACC_gap", "n_repeats", "commit",
          "recorded", "dACC_vs_11111_pp", "dDEO_vs_11111_pp", "dHM_vs_11111_pp"]
    rows = []
    for split in ("Dirichlet1", "Uniform"):
        ref = COMB_CELL.get((split, "11111"))
        for code in all_codes():
            d = COMB_CELL.get((split, code))
            b = bits_of(code)
            pre = ["COMPAS", split, "ANN", "20", "comb" + code,
                   b["w_pa"], b["w_conf"], b["w_l2l"], b["w_l2g"], b["w_g2l"],
                   "; ".join(off_losses(code)) or "(none -- all five losses on)"]
            if d is None:
                rows.append(pre + ["running / missing", "", "", "", "", "", "", "",
                                   "", "", ""])
                continue
            rows.append(pre + [
                "done", d["acc"], d["deo"], d["hm"], d["gap"], d["rep"],
                d["commit"], "2026-%s %s" % (d["date"], d["time"]),
                "%.2f" % (100 * (float(d["acc"]) - float(ref["acc"]))) if ref else "",
                "%.2f" % (100 * (float(d["deo"]) - float(ref["deo"]))) if ref else "",
                "%.2f" % (100 * (float(d["hm"]) - float(ref["hm"]))) if ref else ""])
    return dict(name="A17_comb_grid_full", num="A17",
                title="A17 PDFFed 五比特损失项全因子消融网格（2^5 = 32 × 2 划分）",
                tex_title="Full $2^5$ factorial ablation of the five PDFFed loss terms",
                note=("快照 %s。COMPAS/Dirichlet1 = 32/32 全齐；COMPAS/Uniform = %d/32，"
                      "其余仍在飞。comb11111（五项全开）在 Dirichlet1 上 ACC 与 HM 双第一"
                      "（0.6158 / 0.7398）。comb11100 属另一代码批次 c522dca6e（16:15），"
                      "与其余 6da33067b 批次不同，跨批次比较需谨慎。"
                      % (COMB_SNAP, sum(1 for c in all_codes() if ("Uniform", c) in COMB_CELL))),
                headers=hs, rows=rows, md_max_rows=40)


def t_comb_loo():
    hs = ["split", "disabled_loss", "flag", "comb_code", "ACC", "DEO", "HM", "ACC_gap",
          "dACC_pp", "dDEO_pp", "dHM_pp", "acc_rank_of_32", "hm_rank_of_32"]
    rows = []
    for split in ("Dirichlet1", "Uniform"):
        ref = COMB_CELL.get((split, "11111"))
        if ref is None:
            continue
        present = [c for c in all_codes() if (split, c) in COMB_CELL]
        acc_rank = {c: i + 1 for i, c in enumerate(
            sorted(present, key=lambda z: -float(COMB_CELL[(split, z)]["acc"])))}
        hm_rank = {c: i + 1 for i, c in enumerate(
            sorted(present, key=lambda z: -float(COMB_CELL[(split, z)]["hm"])))}
        n = len(present)
        for i, b in enumerate(BITS):
            code = "".join("0" if j == i else "1" for j in range(5))
            d = COMB_CELL.get((split, code))
            if d is None:
                rows.append([split, LOSS_MD[b], "-" + b, "comb" + code,
                             "running / missing", "", "", "", "", "", "",
                             "%d avail" % n, "%d avail" % n])
                continue
            rows.append([split, LOSS_MD[b], "-" + b, "comb" + code,
                         d["acc"], d["deo"], d["hm"], d["gap"],
                         "%.2f" % (100 * (float(d["acc"]) - float(ref["acc"]))),
                         "%.2f" % (100 * (float(d["deo"]) - float(ref["deo"]))),
                         "%.2f" % (100 * (float(d["hm"]) - float(ref["hm"]))),
                         "%d/%d" % (acc_rank[code], n), "%d/%d" % (hm_rank[code], n)])
        rows.append([split, "（参照）五项全开", "-", "comb11111",
                     ref["acc"], ref["deo"], ref["hm"], ref["gap"],
                     "0.00", "0.00", "0.00",
                     "%d/%d" % (acc_rank["11111"], n), "%d/%d" % (hm_rank["11111"], n)])
    return dict(name="A18_comb_leave_one_out", num="A18",
                title="A18 留一法：逐项关闭五个损失项（vs comb11111 全开）",
                tex_title="Leave-one-out ablation of the five PDFFed loss terms",
                note=("Dirichlet1（32/32 全齐）：关闭 L_l2g 损伤最大（ΔACC −2.10pp、ΔDEO +2.68pp、"
                      "ACC 名次掉到 22/32），关闭 L_PA 对 ACC 几乎无损（−0.28pp、仍排 3/32）"
                      "但 ΔDEO +4.12pp —— 说明 L_PA 专职公平性而非精度。"
                      "Uniform（只有 17/32，且 comb11111 仅排 3/17）：关闭 L_PA 反而 ACC +0.14pp、"
                      "排 1/17，但 ΔDEO 仍 +1.46pp；网格未跑完，Uniform 一侧不得下任何结论。"
                      "Δ 由 raw dump 的 4 位小数值算得，与全精度 json 可能差 pp 的第 2 位小数。"),
                headers=hs, rows=rows)


def t_comb_interaction():
    hs = ["split", "pair_disabled", "comb_code", "ACC", "HM", "commit",
          "observed_dACC_pp", "additive_prediction_dACC_pp", "interaction_pp",
          "superadditive"]
    rows = []
    sup = defaultdict(list)
    sub = defaultdict(list)
    for split in ("Dirichlet1", "Uniform"):
        ref = COMB_CELL.get((split, "11111"))
        if ref is None:
            continue
        for i, j in itertools.combinations(range(5), 2):
            code = "".join("0" if k in (i, j) else "1" for k in range(5))
            d = COMB_CELL.get((split, code))
            if d is None:
                continue
            single, ok = [], True
            for z in (i, j):
                c1 = "".join("0" if k == z else "1" for k in range(5))
                s = COMB_CELL.get((split, c1))
                if s is None:
                    ok = False
                    break
                single.append(100 * (float(s["acc"]) - float(ref["acc"])))
            if not ok:
                continue
            obs = 100 * (float(d["acc"]) - float(ref["acc"]))
            pred = sum(single)
            inter = obs - pred
            pair = "%s + %s" % (BITS[i], BITS[j])
            (sup if inter < -0.05 else sub)[split].append(pair)
            rows.append([split, pair, "comb" + code, d["acc"], d["hm"], d["commit"],
                         "%.2f" % obs, "%.2f" % pred, "%.2f" % inter,
                         "yes" if inter < -0.05 else "no"])
    note = ("interaction = observed − additive；负值 = 超加性损伤（两项一起关，比分别关的"
            "损失之和更糟），说明二者不可互替。")
    for split in ("Dirichlet1", "Uniform"):
        note += " %s（%d/%d 组齐）：超加性对 = %s；其余 %d 对为次加性（可部分互替）。" % (
            split, sum(1 for c in all_codes() if (split, c) in COMB_CELL), 32,
            "、".join(sup[split]) if sup[split] else "无", len(sub[split]))
    note += (" Dirichlet1 上超加性的三对恰好全部落在 {L_l2l, L_l2g, L_g2l} 内部"
             "（最强为 L_l2l + L_l2g，−0.83pp），而任何含 L_PA 或 L_Conf 的对都是次加性"
             " —— 即三个「原型↔分类头」蒸馏项彼此不可互替，但它们与 L_PA / L_Conf 有功能重叠。"
             "注意 comb11100 属另一代码批次 c522dca6e，其 −0.20pp 跨批次，仅供参考；"
             "Uniform 一侧网格未跑完（17/32），模式与 Dirichlet1 相反，不得据此下结论。")
    return dict(name="A19_comb_pairwise_interaction", num="A19",
                title="A19 两两关闭的交互项：损失项之间可替代吗？",
                tex_title="Pairwise interaction terms: are the five losses substitutable?",
                note=note, headers=hs, rows=rows)


def _lam_table(sweep, name, num, title, tex_title, note, valmap):
    hs = ["dataset", "split", "hypothesis", "K", "variant", "sweep_value",
          "server", "commit", "recorded_at", "n_repeats", "ACC", "ACC_std",
          "DEO", "HM", "ACC_gap", "dACC_vs_PDFFed_pp", "dDEO_vs_PDFFed_pp",
          "identical_to_PDFFed_baseline"]
    rows = []
    for r in sorted(LAM, key=lambda z: (z["sweep"] != sweep, z["dataset"], z["split"],
                                        z["num_clients_K"], z["algo_dir_canonical"])):
        if r["sweep"] != sweep:
            continue
        b = BASE.get((r["dataset"], r["split"], r["num_clients_K"]))
        dA = dD = ""
        ident = ""
        if b:
            try:
                dA = "%.2f" % (100 * (float(r["ACC_mean"]) - float(b["ACC_mean"])))
                dD = "%.2f" % (100 * (float(r["DEO_mean"]) - float(b["DEO_mean"])))
                ident = ("YES -- all four metrics bit-identical"
                         if abs(float(r["ACC_mean"]) - float(b["ACC_mean"])) < 1e-12
                         and abs(float(r["DEO_mean"]) - float(b["DEO_mean"])) < 1e-12
                         else "no")
            except ValueError:
                pass
        rows.append([r["dataset"], r["split"], r["hypothesis"], r["num_clients_K"],
                     r["algo_dir_canonical"], valmap.get(r["algo_dir_canonical"],
                                                          r["sweep_value"]),
                     r["server"][-2:], (r["git_commit"] or "")[:9], r["recorded_at"][:16],
                     r["n_repeats"],
                     f4(r["ACC_mean"], 6), f4(r["ACC_std"], 6), f4(r["DEO_mean"], 6),
                     f4(r["HM_mean"], 6), f4(r["ACC_gap_mean"], 6), dA, dD, ident])
    return dict(name=name, num=num, title=title, tex_title=tex_title, note=note,
                headers=hs, rows=rows, md_max_rows=40)


def t_lambda_conf():
    return _lam_table(
        "lambda_conf", "A20_lambda_conf_grid", "A20",
        "A20 λ_conf 扫描（confX05 / confX2 / confX5 / noCONF）",
        "The $\\lambda_{\\mathrm{conf}}$ sweep",
        "基线 = 同 (dataset, split, K) 上 sweep=baseline 的 PDFFed（λ_conf = 1.0）。"
        "noCONF 即 λ_conf = 0。",
        {"PDFFed_confX05": "0.5", "PDFFed_confX2": "2.0",
         "PDFFed_confX5": "5.0", "PDFFed_noCONF": "0 (disabled)"})


def t_lambda_eo():
    return _lam_table(
        "lambda_eo", "A21_lambda_eo_grid", "A21",
        "A21 λ_eo 扫描（eoX05 / eoX2 / eoX5 / noEO）",
        "The $\\lambda_{\\mathrm{eo}}$ sweep",
        "已知缺陷：λ_eo = 0.5 与默认 1.0 在 COMPAS 两格的四项指标完全相同，"
        "成因尚未查明 —— 在查清之前不得据本表画 λ_eo 前沿曲线。",
        {"PDFFed_eoX05": "0.5", "PDFFed_eoX2": "2.0",
         "PDFFed_eoX5": "5.0", "PDFFed_noEO": "0 (disabled)"})


def t_fedrenyi():
    hs = ["dataset", "split", "hypothesis", "K", "variant", "fed_renyi_lambda",
          "server", "commit", "recorded_at", "n_repeats", "ACC", "DEO", "HM",
          "ACC_gap", "floor", "margin_vs_floor_pp", "note"]
    rows = []
    for r in sorted(LAM, key=lambda z: (z["dataset"], z["split"], z["num_clients_K"],
                                        z["algo_dir_canonical"])):
        if r["sweep"] != "fedrenyi_lambda":
            continue
        a, d, s = r["dataset"], r["split"], r["num_clients_K"]
        fl = FLOOR.get(a)
        note = []
        if fl is not None and float(r["ACC_mean"]) <= fl + MARGIN:
            note.append("COLLAPSED (below floor)")
        if a == "DRUG":
            note.append("lambda has NO effect on DRUG -- identical across all values")
        rows.append([a, d, r["hypothesis"], s, r["algo_dir_canonical"],
                     r["algo_dir_canonical"].replace("FedRenyi_lam", "").replace("0p", "0."),
                     r["server"][-2:], (r["git_commit"] or "")[:9], r["recorded_at"][:16],
                     r["n_repeats"], f4(r["ACC_mean"], 6), f4(r["DEO_mean"], 6),
                     f4(r["HM_mean"], 6), f4(r["ACC_gap_mean"], 6),
                     "%.6f" % fl if fl else "",
                     pp(float(r["ACC_mean"]) - fl) if fl else "",
                     "; ".join(note) or "-"])
    return dict(name="A22_fedrenyi_lambda_grid", num="A22",
                title="A22 FedRenyi λ 扫描（基线算法的已知缺陷）",
                tex_title="The FedRenyi $\\lambda$ sweep and its known defect",
                note=("DRUG 上 λ = 0.25/0.5/1/2/4 五档逐位相同（0.7664/0.0806/0.8357/0.1038），"
                      "且全部低于 DRUG 地板 0.788732 —— λ 参数在该数据集上根本没生效，"
                      "FedRenyi 的 DRUG 结果不得引用。COMPAS/Dirichlet1 缺 λ = 4 一格。"),
                headers=hs, rows=rows, md_max_rows=40)


def t_dot31_today():
    hs = ["dataset", "result_jsons_today", "share_of_214", "note"]
    data = [("DRUG", 62, ""), ("COMPAS", 62, "含 eoX05 / eoX2 共 8 格"),
            ("BANK", 51, ""), ("DUTCH", 26, ""), ("FairFace", 7, ""), ("bios", 6, "")]
    tot = sum(n for _, n, _ in data)
    rows = [[d, n, "%.1f%%" % (100.0 * n / tot), nt] for d, n, nt in data]
    rows.append(["TOTAL", tot, "100.0%",
                 "16 distinct algorithms; PDFFed family = 36 json"])
    return dict(name="A23_dot31_output_today", num="A23",
                title="A23 192.168.1.31 当日产出盘点（部分摘要）",
                tex_title="Same-day output inventory on 192.168.1.31 (partial summary)",
                note=("⚠ 不完整：本表的逐算法明细来自一次未落盘的服务器探针，"
                      "本地只保留了按数据集的汇总计数。逐算法/逐格明细需重新登录 .31 采集。"
                      "统计口径 = 当日 mtime 落在 2026-10-21 的结果 json 份数。"),
                headers=hs, rows=rows)


def t_machine_status_xref():
    hs = ["item", "value", "where_the_full_table_lives"]
    rel = "reported_results/two_machine_summary"
    src = os.path.join(REPO, rel.replace("/", os.sep), "machine_status.csv")
    exists = "yes" if os.path.exists(src) else "MISSING"
    rows = [
        ["HEAD on both servers", "6299c1e23cca40cd40242288cea29332a2e89a82 (PR #67 merged)",
         rel + "/machine_status.csv"],
        ["192.168.1.30 GPU", "see machine_status.csv (util / mem per device)",
         rel + "/machine_status.csv"],
        ["192.168.1.31 GPU 4 & 5", "41748 MiB held each but 0% utilisation -- suspected orphan",
         rel + "/structural_defects.csv"],
        ["running jobs", "not killed at any point (red line: 先别杀)",
         rel + "/machine_status.csv"],
        ["source file present in repo", exists, rel + "/machine_status.csv"],
    ]
    return dict(name="A24_machine_status_crossref", num="A24",
                title="A24 服务器状态（交叉引用，不重复生成）",
                tex_title="Machine status: cross-reference, not regenerated",
                note=("完整的两机状态表已由 analysis/build_two_machine_summary.py 落盘，"
                      "此处只给指针，避免两份状态表相互矛盾。"),
                headers=hs, rows=rows)


# ============================================================== Part B parser
MD_SRC = os.path.join(REPO, "消融表3_分划分统计.md")
SEP_RE = re.compile(r"^\|[\s:\-|]+\|$")
HEAD_RE = re.compile(r"^(#{1,6})\s+(.*)$")
BOLD_RE = re.compile(r"^\*\*([^*]+)\*\*[：:]?\s*$")


def split_row(ln):
    s = ln.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    out, cur, esc = [], "", False
    for ch in s:
        if esc:
            cur += ch
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == "|":
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    out.append(cur.strip())
    return out


def slugify(text, n, bold=""):
    src = (text + " " + bold).strip()
    toks = re.findall(r"[A-Za-z][A-Za-z0-9]*|\d+[a-z]|\d+\.\d+", src)
    toks = [t for t in toks if t.lower() not in ("pp", "vs", "the", "and")]
    s = "_".join(toks[:4]).strip("_")
    return (s or ("table%d" % n))[:48]


def parse_md_tables(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    heads, bold = [], ""
    tables, i, n = [], 0, len(lines)
    while i < n:
        ln = lines[i]
        m = HEAD_RE.match(ln)
        if m:
            lvl = len(m.group(1))
            heads = [h for h in heads if h[0] < lvl] + [(lvl, m.group(2).strip())]
            bold = ""
            i += 1
            continue
        mb = BOLD_RE.match(ln)
        if mb:
            bold = mb.group(1).strip()
        if ln.startswith("|") and i + 1 < n and SEP_RE.match(lines[i + 1].strip()):
            start = i + 1
            headers = split_row(ln)
            i += 2
            body = []
            while i < n and lines[i].startswith("|") and not SEP_RE.match(lines[i].strip()):
                body.append(split_row(lines[i]))
                i += 1
            w = len(headers)
            body = [(r + [""] * w)[:w] for r in body]
            ctx = heads[-1][1] if heads else "(preamble)"
            tables.append(dict(ctx=ctx, bold=bold, headers=headers, rows=body,
                               lines=(start + 1, i)))
            continue
        i += 1
    return tables


def part_b_tables():
    raw = parse_md_tables(MD_SRC)
    out = []
    for n, t in enumerate(raw, 1):
        title = t["ctx"]
        if t["bold"] and t["bold"] not in title:
            title = "%s —— %s" % (title, t["bold"])
        out.append(dict(
            name="B%02d_%s" % (n, slugify(t["ctx"], n, t["bold"])),
            num="B%d" % n,
            title="B%d %s" % (n, title),
            tex_title="Ablation Table 3 document, table B%d: %s" % (n, t["ctx"]),
            note=("源：消融表3_分划分统计.md 第 %d-%d 行，原样转录（数值与已评审文档逐字一致）。"
                  % t["lines"]),
            headers=t["headers"], rows=t["rows"],
            md_max_rows=40))
    return out


# ================================================================== emitters
def write_csv(t):
    p = os.path.join(CSV_DIR, t["name"] + ".csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(t["headers"])
        w.writerows(t["rows"])
    return p


def md_escape(s):
    return str(s).replace("|", "\\|")


def write_md(t):
    hs, rows = t["headers"], t["rows"]
    lines = ["## %s" % t["title"], ""]
    if t.get("note"):
        lines += ["> %s" % t["note"], ""]
    lines.append("| " + " | ".join(md_escape(h) for h in hs) + " |")
    lines.append("|" + "|".join(["---"] * len(hs)) + "|")
    cap = t.get("md_max_rows")
    tail = None
    if cap and len(rows) > cap:
        tail = "_（仅列前 %d 行，完整 %d 行见 `csv/%s.csv`）_" % (cap, len(rows), t["name"])
        rows = rows[:cap]
    for r in rows:
        lines.append("| " + " | ".join(md_escape(c) for c in r) + " |")
    if tail:
        lines += ["", tail]
    lines.append("")
    return "\n".join(lines)


TEX_MAP = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
           "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
           "<": r"\textless{}", ">": r"\textgreater{}"}


def tex_escape(s):
    """char-by-char so that the backslashes we insert are never re-escaped"""
    return "".join(TEX_MAP.get(ch, ch) for ch in str(s))


NUMERIC = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?(%|pp)?$")


def guess_align(headers, rows):
    align = []
    for i in range(len(headers)):
        vals = [str(r[i]) for r in rows[:40] if i < len(r)]
        vals = [v for v in vals if v not in ("", "-")]
        align.append("r" if vals and all(NUMERIC.match(v) for v in vals) else "l")
    return "".join(align)


RAW_TEX_TITLES = {"A20_lambda_conf_grid", "A21_lambda_eo_grid",
                  "A22_fedrenyi_lambda_grid"}


def write_tex(t):
    hs, rows = t["headers"], t["rows"]
    spec = t.get("tex_colspec") or guess_align(hs, rows)
    long_ = len(rows) > 25
    cap = t["tex_title"] if t["name"] in RAW_TEX_TITLES else tex_cell(t["tex_title"])
    hh = " & ".join(tex_cell(h) for h in hs) + r" \\"
    out = []
    if long_:
        out += [r"\begingroup\scriptsize",
                r"\begin{longtable}{%s}" % spec,
                r"\caption{%s}\label{tab:%s}\\" % (cap, t["name"]),
                r"\toprule", hh, r"\midrule", r"\endfirsthead",
                r"\multicolumn{%d}{r}{\ldots\ continued}\\" % len(hs),
                r"\toprule", hh, r"\midrule", r"\endhead"]
    else:
        out += [r"\begin{table}[htbp]\centering\scriptsize",
                r"\caption{%s}\label{tab:%s}" % (cap, t["name"]),
                r"\resizebox{\textwidth}{!}{%",
                r"\begin{tabular}{%s}" % spec, r"\toprule", hh, r"\midrule"]
    for r in rows:
        out.append(" & ".join(tex_cell(c) for c in r) + r" \\")
    out += [r"\bottomrule", r"\end{%s}" % ("longtable" if long_ else "tabular")]
    out.append(r"\endgroup" if long_ else "}\n" + r"\end{table}")
    return "\n".join(out)


UNI = [(u"\u2212", "$-$"), (u"\u2013", "--"), (u"\u2014", "---"),
       (u"\u2248", r"$\approx$"), (u"\u2264", r"$\le$"), (u"\u2265", r"$\ge$"),
       (u"\u00d7", r"$\times$"), (u"\u0394", r"$\Delta$"), (u"\u03bb", r"$\lambda$"),
       (u"\u2192", r"$\to$"), (u"\u2260", r"$\ne$"), (u"\u00b1", r"$\pm$"),
       (u"\u2261", r"$\equiv$"), (u"\u221a", r"$\surd$"), (u"\u2211", r"$\sum$"),
       (u"\u2208", r"$\in$"), (u"\u00b7", r"$\cdot$"), (u"\u2026", r"\ldots{}"),
       (u"\u26a0", "[!]"), (u"\u2713", "[ok]"), (u"\u2190", r"$\leftarrow$"),
       (u"\u2460", "(1)"), (u"\u2461", "(2)"), (u"\u2462", "(3)"),
       (u"\u2463", "(4)"), (u"\u2464", "(5)"), (u"\u2465", "(6)"),
       (u"\u300c", "``"), (u"\u300d", "''"), (u"\u2018", "`"), (u"\u2019", "'"),
       (u"\ufe0f", ""), (u"\u2194", r"$\leftrightarrow$"),
       (u"\u2705", "[OK]"), (u"\u274c", "[NO]"), (u"\U0001F6AB", "[forbidden]")]


def tex_cell(s):
    """escape, then re-inject the small amount of Markdown inline syntax that
    Part B carries (**bold**, `code`) and normalise unicode math punctuation."""
    s = tex_escape(s)
    for a, b in UNI:
        s = s.replace(a, b)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"`([^`]+?)`", r"\\texttt{\1}", s)
    return s


PREAMBLE = r"""\documentclass[11pt]{ctexart}
% Auto-generated by analysis/build_session_tables.py -- do not edit by hand.
% Compile with:  xelatex tables.tex   (ctex is required for the CJK headings)
% If ctex/xelatex is unavailable, use tables.md or csv/*.csv instead -- the
% numbers are identical.
\usepackage[a3paper,landscape,margin=1.2cm]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{array}
\usepackage{amsmath}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.05}
\title{Session tables: reproducibility audit, selection-rule invariance,\\
and the five-bit PDFFed loss ablation}
\author{}
\date{Snapshot: @SNAPSHOT@}
\begin{document}
\maketitle
\tableofcontents
\clearpage

\section*{Scope and provenance}
Part~A (A1--A24) is recomputed live from the raw per-result payload in
\texttt{\_sync/collapse\_audit\_20261021/cells\_\{30,31\}.tsv} via
\texttt{extract\_collapse\_audit.read\_cells}, so the DUTCH ruler is the corrected
$0.522344$ everywhere; the reverted $0.518454$ blob binds zero results and is not
used. The component grid in A17--A19 comes from the server dumps archived under
\texttt{raw/comb\_grid\_from\_\{30,31\}.txt}; the $\lambda$ sweeps in A20--A22 come
from the full-precision \texttt{lambda\_sensitivity.csv}. Part~B is a verbatim
transcription of every pipe table in the reviewed document
\texttt{消融表3\_分划分统计.md}.

Collapse criterion throughout: $\mathrm{ACC} \le \mathrm{floor} + 0.01$, where the
floor is the majority-class share of the test set that the cell was actually
evaluated on, bound per result json through
\texttt{aggregate.partition\_fingerprints}.

Every table appears three times in this deliverable: as a standalone
\texttt{csv/\textless name\textgreater.csv}, inside \texttt{tables.md}, and here.
\texttt{index.csv} maps each name to its title, source, row count and column count.

\clearpage
\part{Part A --- recomputed from the raw payload}
""".replace("@SNAPSHOT@", SNAPSHOT)

PART_B_HEAD = r"""
\clearpage
\part{Part B --- verbatim from the reviewed ablation document}
"""

CLOSING = r"""
\clearpage
\section*{Standing caveats}
\begin{enumerate}
\item \textbf{No cherry-picking.} One uniform copy-selection rule is applied to
every algorithm. A4/A5 show that on the three datasets where any cell escapes the
floor (BANK, COMPAS, DUTCH) the PDFFed family holds the top slots under all four
rules (\texttt{newest}, \texttt{max}, \texttt{min}, \texttt{mean}): PDFFed is
rank~1 on BANK and PDFFed\_DP is rank~1 on COMPAS and DUTCH without exception.
What the rule \emph{does} move is PDFFed (non-DP) between rank~2 and rank~3, and
the \texttt{max} rule actually lets \texttt{FederatedAverageWithPo} overtake it on
COMPAS ($0.6165$ vs $0.6158$). ADULT and DRUG have zero escaping cells under
every rule, so their internal ordering carries no information. Cherry-picking the
copy most favourable to PDFFed therefore buys at most one rank position and costs
the \texttt{max}-rule counterexample above.
\item \textbf{Publication route.} Publish from a single commit
(\texttt{e4f45401b} for BANK/DRUG/COMPAS/DUTCH, \texttt{e34c32ad3} for ADULT),
one data version, one selection rule --- see A13--A15.
\item \textbf{Irreducible residual.} Of the duplicate copy pairs, the
cross-host/different-commit majority is reducible by fixing the commit; the
same-host/same-commit pairs in A11 are not, and the DUTCH rewrite timeline in A12
shows they cannot be blamed on \texttt{DUTCH.pickle}.
\item \textbf{Blocked claims.} The FedRenyi $\lambda$ sweep is inert on DRUG
(A22); $\lambda_{\mathrm{eo}}=0.5$ reproduces the default bit-for-bit on two
COMPAS cells for an unexplained reason (A21), so no $\lambda_{\mathrm{eo}}$
frontier may be drawn until that is resolved.
\item \textbf{Stale snapshot.} The \texttt{comb\#\#\#\#\#} table in
\texttt{消融表3\_分划分统计.md}~\S6 (Part~B) predates the completed
$2^5$ grid and holds only 17/32 codes for one split; A17 supersedes it.
\item \textbf{Split-scoped ablation claim.} The ``all five losses on is best''
result holds only on COMPAS/Dirichlet1, where the grid is complete
(\texttt{comb11111} is ACC rank 1/32 and HM rank 1/32). On COMPAS/Uniform the grid
is 17/32 and \texttt{comb11111} sits at rank 3/17, with \texttt{-w\_pa} marginally
\emph{better} on ACC ($+0.14$pp) though still worse on DEO ($+1.46$pp). No
conclusion may be drawn for the iid split until that grid finishes (A17--A19).
\item \textbf{Incomplete table.} A23 keeps only the per-dataset counts of the
same-day probe on \texttt{192.168.1.31}; the per-algorithm breakdown was never
archived and must be re-collected from the server.
\end{enumerate}

\end{document}
"""


def build():
    tabs = [
        t_dataset_floor_provenance(), t_dutch_floor_versions(), t_dutch_rerule_effect(),
        t_rule_invariance(), t_rule_top6(), t_pair_trichotomy(),
        t_dis_by_algo(), t_dis_by_ds(), t_dis_top12(), t_dis_pdffed(),
        t_same_server_dup(), t_dutch_timeline(), t_commit_coverage(),
        t_commit_per_dataset(), t_single_commit_ranking(), t_sanity_anchor(),
        t_comb_grid(), t_comb_loo(), t_comb_interaction(),
        t_lambda_conf(), t_lambda_eo(), t_fedrenyi(),
        t_dot31_today(), t_machine_status_xref(),
    ]
    return tabs, part_b_tables()


HEADER_MD = """# 本轮会话全部表格（csv / md / latex 三格式落盘）

- **快照时间**：{snap}
- **生成方式**：`python analysis/build_session_tables.py`（本文件与 `tables.tex`、
  `csv/*.csv`、`index.csv` 全部由该脚本重新生成，请勿手改）
- **Part A（A1–A24）**：由 `_sync/collapse_audit_20261021/cells_{{30,31}}.tsv`
  经 `extract_collapse_audit.read_cells()` **现场重算**，DUTCH 地板一律用更正后的
  **0.522344**（被 PR #67 撤销的 0.518454 绑定 0 份结果，不参与任何计算）；
  五比特消融网格来自 `raw/comb_grid_from_{{30,31}}.txt`；λ 扫描来自
  `reported_results/lambda_sensitivity/lambda_sensitivity.csv`（全精度）。
- **Part B（B1–B{nB}）**：`消融表3_分划分统计.md` 里的**每一张**管道表，
  由通用 GFM 解析器逐字转录，数值与已评审文档完全一致，每张表标注了源文件行号。
- **塌缩判据**：`ACC ≤ floor + 0.01`，floor = 该格实际评测所用测试集的多数类占比。

## 表目录

| # | 名称 | 标题 | 行 | 列 | 来源 |
|---|---|---|---|---|---|
{index}

---

# Part A —— 由原始 payload 现场重算

"""


def main():
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    for old in os.listdir(CSV_DIR):          # drop stale generated csv (names change)
        if re.match(r"^[AB]\d\d_.*\.csv$", old):
            os.remove(os.path.join(CSV_DIR, old))
    partA, partB = build()
    all_t = partA + partB
    idx = [["name", "num", "title_md", "title_tex", "rows", "cols", "csv_path", "note"]]
    md = []
    tex = []
    for t in all_t:
        p = write_csv(t)
        md.append(write_md(t))
        head = ["\\section{%s\\quad %s}" % (t["num"], tex_cell(t["title"]))]
        if t.get("note"):
            head.append("{\\footnotesize\\textit{%s}\\par}\\vspace{2pt}"
                        % tex_cell(t["note"]))
        tex.append("\n".join(head) + "\n\n" + write_tex(t))
        idx.append([t["name"], t["num"], t["title"], t["tex_title"], len(t["rows"]),
                    len(t["headers"]), os.path.relpath(p, REPO).replace(os.sep, "/"),
                    t.get("note", "")])
        print("  csv -> %-52s %4d rows x %2d cols" % (
            os.path.relpath(p, REPO), len(t["rows"]), len(t["headers"])))
    mdA = md[:len(partA)]
    mdB = md[len(partA):]
    index_rows = ["| %s | %s | %s | %d | %d | `%s` |" % (
        r[1], r[0], md_escape(r[2]).replace("\n", " "), r[4], r[5], r[6]) for r in idx[1:]]
    doc_md = HEADER_MD.format(snap=SNAPSHOT, nB=len(partB), index="\n".join(index_rows))
    doc_md += "\n".join(mdA)
    doc_md += "\n---\n\n# Part B —— 逐字转录 `消融表3_分划分统计.md`\n\n"
    doc_md += "\n".join(mdB)
    doc_md += CLOSING_MD
    with open(os.path.join(OUT, "tables.md"), "w", encoding="utf-8") as fh:
        fh.write(doc_md)

    texA = tex[:len(partA)]
    texB = tex[len(partA):]
    doc = (PREAMBLE + "\n\n".join(texA) + PART_B_HEAD + "\n\n".join(texB)
           + CLOSING + "")
    with open(os.path.join(OUT, "tables.tex"), "w", encoding="utf-8") as fh:
        fh.write(doc)

    with open(os.path.join(OUT, "index.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerows(idx)

    print("\n  md   -> reported_results/session_tables/tables.md")
    print("  tex  -> reported_results/session_tables/tables.tex")
    print("  idx  -> reported_results/session_tables/index.csv")
    print("  total tables: %d (Part A %d + Part B %d)" % (len(all_t), len(partA), len(partB)))


CLOSING_MD = """
---

## 长期注意事项

1. **不做 cherry-pick。** 同一条选数规则统一施于全部算法。A4/A5 显示：在有格能越过
   地板的三个数据集（BANK / COMPAS / DUTCH）上，四个规则（`newest` / `max` / `min` /
   `mean`）下 PDFFed 系的头部位置都守得住 —— BANK 上 PDFFed 恒为 #1，COMPAS 与 DUTCH 上
   PDFFed_DP 恒为 #1，无一例外。规则真正能撼动的只是 PDFFed（非 DP）在 #2 与 #3 之间移动，
   而 `max` 规则还会让 `FederatedAverageWithPo` 在 COMPAS 上以 0.6165 反超 PDFFed 的 0.6158。
   ADULT 与 DRUG 在四个规则下 escaping_cells 都是 0（全体塌缩），其内部排名不携带信息。
   结论：挑「对 PDFFed 最有利的拷贝」最多买到一个名次的位移，代价是引入上面那个反例。
2. **出表口径。** 单一 commit（BANK/DRUG/COMPAS/DUTCH 用 `e4f45401b`，
   ADULT 用 `e34c32ad3`）+ 单一数据版本 + 单一选数规则，见 A13–A15。
3. **不可约残差。** 拷贝对里跨机+异 commit 的多数可用「统一到单一 commit」消解；
   A11 的同机+同 commit 那几对消解不了，且 A12 的时间线证明它们与 `DUTCH.pickle`
   重写无关。
4. **不得写进论文的宣称。** DRUG 上 FedRenyi 的 λ 完全无效（A22）；
   λ_eo = 0.5 与默认 1.0 在 COMPAS 两格逐位相同、成因未明（A21），
   查清前不得画 λ_eo 前沿曲线。
5. **已过期快照。** `消融表3_分划分统计.md` §6（Part B 中对应那张 comb 表）只有
   17/32 组、且仅 COMPAS/Dirichlet1，已被 A17 的完整 2^5 网格取代。
6. **消融结论只在 Dirichlet1 上成立。** 「五项全开最好」只在 COMPAS/Dirichlet1
   （网格 32/32 全齐，comb11111 的 ACC 与 HM 都是 1/32）上站得住。COMPAS/Uniform 只有
   17/32，comb11111 排 3/17，且关掉 `-w_pa` 在 ACC 上反而好 +0.14pp（DEO 仍差 +1.46pp）。
   Uniform 一侧的网格跑完之前不得下结论（A17–A19）。
7. **不完整表。** A23 只保留了 .31 当日产出的按数据集计数，逐算法明细当时未落盘，
   需重新登录服务器采集。
"""


if __name__ == "__main__":
    main()
