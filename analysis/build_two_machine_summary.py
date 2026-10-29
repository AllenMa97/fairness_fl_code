#!/usr/bin/env python3
"""Build the two-machine (.30 / .31) experiment summary in CSV + Markdown + LaTeX.

Inputs
------
``reported_results/two_machine_summary/raw/cells_{30,31}.tsv``
    Per-cell inventory dumped on each server by scanning ``result_path`` and
    binding every result json's ``aggregate.partition_fingerprints`` back to
    ``partition_cache/v2/<DS>/<fp>/metadata.json`` to recover the *actual*
    test-set label distribution the run evaluated on.

Outputs
-------
``reported_results/two_machine_summary/``
    ``summary.md``    every table in Markdown + the conclusions
    ``summary.tex``   the same, compilable (article + booktabs + longtable)
    ``<table>.csv``   one long-form CSV per table

Collapse criterion
------------------
A cell is a *constant predictor* (collapsed) when ``ACC <= floor + 0.01``, where
``floor`` is the majority-class share of the test set that the cell was actually
evaluated on. This is the criterion fixed by
``reported_results/collapse_audit/appendix_collapse_disclosure.md`` S2. The older
heuristic (``DEO == 0 and SPD == 0 and FR == 1``) is NOT used here: it produces
false positives on healthy high-ACC cells and misses every collapse whose
fairness metrics are merely small rather than exactly zero.

Merge rule
----------
One cell key is ``(dataset, split, algo, hypothesis, K)``. When both machines
hold a copy, the one with the newest ``code_version.recorded_at`` wins, matching
the "newest" selection rule fixed in the disclosure appendix S9.7.
"""

from __future__ import annotations

import csv
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "reported_results", "two_machine_summary")
RAW = os.path.join(OUT, "raw")

SNAPSHOT = "2026-10-22 09:33 (Asia/Shanghai)"
MARGIN = 0.01
HOSTS = ("30", "31")

# Ablation / hyper-parameter-sweep variants and algorithms imported from other
# projects. They share the result tree with the main comparison but run on 1-2
# partitions only, so they must never be ranked against the nsp=4 main calibre.
ABLATION = re.compile(
    r"(comb\d|confX|eoX|pvc|pve|pvFULL|noG2L|noL2G|noL2L|noEO|noPA|noCONF"
    r"|lam\d|lam0p|eps\d|PraFFL|FedFACT|DOSFL|OSFL|FedMix|NaiveMix|ProxProb)"
)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


P = ">{\\raggedright\\arraybackslash}p{%s}"


def colspec(widths):
    """LaTeX column specification for tables holding long free text."""
    return "".join(P % w for w in widths)


# --------------------------------------------------------------------------
# load + merge
# --------------------------------------------------------------------------
def load():
    raw = []
    for h in HOSTS:
        path = os.path.join(RAW, "cells_%s.tsv" % h)
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                r["host"] = h
                raw.append(r)

    byk = defaultdict(list)
    for r in raw:
        byk[(r["dataset"], r["split"], r["algo"], r["hyp"], r["K"])].append(r)

    merged, disagree, crosshost = [], [], 0
    for k, rs in byk.items():
        rs.sort(key=lambda r: r["recorded_at"] or "")
        merged.append(rs[-1])
        if len(rs) > 1:
            if len({r["host"] for r in rs}) > 1:
                crosshost += 1
            accs = [num(r["ACC"]) for r in rs]
            if max(accs) - min(accs) > 1e-5:
                disagree.append((k, rs, max(accs) - min(accs)))
    disagree.sort(key=lambda t: -t[2])
    return raw, merged, byk, disagree, crosshost


RAW_ROWS, CELLS, BYK, DISAGREE, CROSSHOST = load()


def valid(r):
    return r["floor"] != "NA"


def collapsed(r):
    return r["collapse"] == "COLLAPSE"


BY_DS = defaultdict(list)
for r in CELLS:
    BY_DS[r["dataset"]].append(r)


def best_cell(ds, algo):
    c = [r for r in BY_DS[ds] if r["algo"] == algo and valid(r)]
    return max(c, key=lambda r: num(r["ACC"])) if c else None


# --------------------------------------------------------------------------
# table 1: machine status (from the SNAPSHOT probe, not derivable from the tsv)
# --------------------------------------------------------------------------
T1_HEADERS = ["item", "dot30", "dot31"]
T1_ROWS = [
    ["probe time", "2026-10-22 09:33:40", "2026-10-22 09:33:57"],
    ["git HEAD", "6299c1e23 (Merge PR #67)", "6299c1e23 (Merge PR #67)"],
    ["dirty files", "26", "31"],
    ["running launchers",
     "pareto30gpu (GPU0,2) 2d01h; k20fill30 (GPU1,3) 2d14h; "
     "redo/dutch30 (GPU0,1) 2d21h; redo/ab30 (GPU2,3,4) 2d21h",
     "k20fill31 (GPU1,2) 2d15h; img31fill (GPU3) 1d00h; "
     "img31fill_b2 (GPU1) 20h; cpu31bank (CPU) 20h"],
    ["our own jobs (VLLM_WORKER_RANK marker)", "0", "0"],
    ["GPU util %", "0=93 1=99 2=99 3=99 4=0 5=19", "0=99 1=100 2=100 3=3 4=99 5=99"],
    ["GPU mem MiB", "0=36297 1=37584 2=18494 3=23184 4=45745 5=14112",
     "0=12716 1=29072 2=45374 3=23385 4=41748 5=41748"],
    ["idle-but-occupied GPU", "GPU4: 45745 MiB @ 0%", "GPU3: 23385 MiB @ 3%"],
    ["result txt files", "943", "1143"],
    ["txt written in last 3 h", "6", "3"],
    ["disk", "/dev/nvme0n1p2 1.4T/1.8T 85%", "/dev/nvme0n1p2 1.3T/1.8T 74%"],
    ["sent launcher (ours, terminated)",
     "done.tsv: sent_batch 9 OK / 51 FAILED; sent_bios_b 17 OK / 19 FAILED",
     "done.tsv empty"],
]

# --------------------------------------------------------------------------
# table 2: floor + usability per dataset
# --------------------------------------------------------------------------
T2_HEADERS = ["dataset", "floor", "n_test", "cells", "cells_with_floor", "collapsed",
              "collapse_pct", "no_floor", "pdffed_best_acc", "pdffed_best_excess_pp",
              "pdffed_best_cell", "pdffed_verdict", "pdffed_dp_best_acc",
              "pdffed_dp_best_excess_pp", "pdffed_dp_best_cell", "pdffed_dp_verdict"]


def _best_fields(b):
    if not b:
        return ["-", "-", "-", "no result"]
    return ["%.3f" % num(b["ACC"]), "%+.1f" % num(b["excess_pp"]),
            "K%s/%s" % (b["K"], b["split"]),
            "COLLAPSE" if collapsed(b) else "valid"]


def build_t2():
    rows = []
    for ds in sorted(BY_DS):
        lst = BY_DS[ds]
        known = [r for r in lst if valid(r)]
        ncol = sum(1 for r in known if collapsed(r))
        fls = sorted({num(r["floor"]) for r in known})
        floor = "%.4f" % fls[0] if len(fls) == 1 else "MULTI"
        ntest = sorted({r["n_test"] for r in known})
        rows.append([ds, floor, ntest[0] if len(ntest) == 1 else "MULTI",
                     len(lst), len(known), ncol,
                     "%.1f" % (100.0 * ncol / len(known)) if known else "NA",
                     len(lst) - len(known)]
                    + _best_fields(best_cell(ds, "PDFFed"))
                    + _best_fields(best_cell(ds, "PDFFed_DP")))
    rows.sort(key=lambda r: (float("inf") if r[6] == "NA" else float(r[6]), r[0]))
    return rows


T2_ROWS = build_t2()

# --------------------------------------------------------------------------
# grouped main-calibre tables (shared by T3 and T4)
# --------------------------------------------------------------------------
USABLE_DS = [
    ds for ds in sorted(BY_DS)
    if any(r["algo"] in ("PDFFed", "PDFFed_DP") and r["collapse"] == "valid"
           for r in BY_DS[ds])
]


def build_groups():
    """(dataset, hyp, K) -> [(algo, [cell rows])] restricted to usable datasets,
    ablation variants excluded."""
    cell = defaultdict(list)
    for r in CELLS:
        if r["dataset"] not in USABLE_DS or ABLATION.search(r["algo"]):
            continue
        cell[(r["dataset"], r["hyp"], r["K"], r["algo"])].append(r)
    groups = defaultdict(list)
    for (ds, hyp, K, al), rs in cell.items():
        groups[(ds, hyp, K)].append((al, rs))
    out = {}
    for key, algos in groups.items():
        nsp_max = max(len(rs) for _, rs in algos)
        if nsp_max < 3:          # too few partitions to rank anything
            continue
        tab = {}
        for al, rs in algos:
            ncol = sum(1 for r in rs if collapsed(r))
            tab[al] = dict(
                hm=mean([num(r["HM"]) for r in rs]),
                acc=mean([num(r["ACC"]) for r in rs]),
                gap=mean([num(r["ACC_gap"]) for r in rs]),
                deo=mean([num(r["DEO"]) for r in rs]),
                exc_avg=mean([num(r["excess_pp"]) for r in rs]),
                exc_max=max(num(r["excess_pp"]) for r in rs),
                nsp=len(rs), nsp_max=nsp_max, ncol=ncol,
                commit=rs[0]["commit"], host=rs[0]["host"],
            )
        out[key] = (nsp_max, tab)
    return out


GROUPS = build_groups()
GKEYS = sorted(GROUPS, key=lambda k: (k[0], k[1], int(k[2])))


def ranks(tab, algo):
    """1-based ranks of `algo` inside `tab`: HM/ACC descending, gap/DEO ascending."""
    def rk(metric, rev):
        order = sorted(tab.items(), key=lambda kv: (kv[1][metric] if kv[1][metric] == kv[1][metric] else -1e9), reverse=rev)
        return [a for a, _ in order].index(algo) + 1
    return rk("hm", True), rk("acc", True), rk("gap", False), rk("deo", False)


# --------------------------------------------------------------------------
# table 3: PDFFed family head-to-head, one row per (dataset, K, variant)
# --------------------------------------------------------------------------
T3_HEADERS = ["dataset", "hyp", "K", "variant", "n_algos_main", "nsp_max",
              "hm_rank", "acc_rank", "gap_rank", "deo_rank",
              "hm", "acc", "acc_gap", "deo",
              "valid_splits", "exc_avg_pp", "exc_max_pp",
              "dhm_vs_fedavg", "dacc_vs_fedavg", "dhm_vs_best_baseline",
              "best_baseline_by_hm", "src"]
T3_ROWS = []
for key in GKEYS:
    ds, hyp, K = key
    nsp_max, tab = GROUPS[key]
    main = {a: v for a, v in tab.items() if v["nsp"] == nsp_max}
    if not main:
        continue
    fa = main.get("FedAvg")
    base = {a: v for a, v in main.items() if not a.startswith("PDFFed")}
    bb = max(base.items(), key=lambda kv: kv[1]["hm"]) if base else None
    for variant in ("PDFFed", "PDFFed_DP"):
        if variant not in main:
            continue
        t = main[variant]
        rh, ra, rg, rd = ranks(main, variant)
        T3_ROWS.append([
            ds, hyp, K, variant, len(main), nsp_max, rh, ra, rg, rd,
            "%.3f" % t["hm"], "%.3f" % t["acc"], "%.4f" % t["gap"], "%.4f" % t["deo"],
            "%d/%d" % (t["nsp"] - t["ncol"], t["nsp"]),
            "%+.2f" % t["exc_avg"], "%+.2f" % t["exc_max"],
            "%+.3f" % (t["hm"] - fa["hm"]) if fa else "NA",
            "%+.3f" % (t["acc"] - fa["acc"]) if fa else "NA",
            "%+.3f" % (t["hm"] - bb[1]["hm"]) if bb else "NA",
            bb[0] if bb else "-",
            ".%s@%s" % (t["host"], t["commit"]),
        ])

# --------------------------------------------------------------------------
# table 4: full main-calibre detail, one row per (dataset, hyp, K, algo)
# --------------------------------------------------------------------------
T4_HEADERS = ["dataset", "hyp", "K", "floor", "nsp_max", "rank", "algo",
              "hm", "acc", "acc_gap", "deo", "nsp", "collapsed_splits",
              "exc_avg_pp", "exc_max_pp", "src", "is_pdffed"]
T4_ROWS = []
DS_FLOOR = {}
for ds, lst in BY_DS.items():
    fls = sorted({num(r["floor"]) for r in lst if valid(r)})
    DS_FLOOR[ds] = "%.4f" % fls[0] if len(fls) == 1 else "MULTI"
for key in GKEYS:
    ds, hyp, K = key
    nsp_max, tab = GROUPS[key]
    ordered = sorted(tab.items(), key=lambda kv: -(kv[1]["hm"] if kv[1]["hm"] == kv[1]["hm"] else -1e9))
    for i, (al, t) in enumerate(ordered, 1):
        T4_ROWS.append([
            ds, hyp, K, DS_FLOOR[ds],
            nsp_max, i, al,
            "%.3f" % t["hm"], "%.3f" % t["acc"], "%.4f" % t["gap"], "%.4f" % t["deo"],
            t["nsp"], t["ncol"],
            "%+.2f" % t["exc_avg"], "%+.2f" % t["exc_max"],
            ".%s@%s" % (t["host"], t["commit"]),
            "yes" if al.startswith("PDFFed") else "",
        ])

# --------------------------------------------------------------------------
# table 5: cross-machine / cross-commit disagreement
# --------------------------------------------------------------------------
T5_HEADERS = ["dataset", "split", "algo", "hyp", "K", "n_copies", "hosts",
              "d_acc", "copies_host_commit_acc"]
T5_ROWS = []
for k, rs, sp in DISAGREE:
    T5_ROWS.append([
        k[0], k[1], k[2], k[3], k[4], len(rs),
        "+".join(sorted({r["host"] for r in rs})),
        "%.4f" % sp,
        " | ".join(".%s@%s:%s" % (r["host"], (r["commit"] or "blank")[:9], r["ACC"])
                   for r in sorted(rs, key=lambda r: r["host"])),
    ])

# --------------------------------------------------------------------------
# table 6: cells that ran with balance_sampling = 1
# --------------------------------------------------------------------------
T6_HEADERS = ["dataset", "algo", "hyp", "K", "split", "acc", "floor",
              "excess_pp", "collapse", "commit", "host"]
T6_ROWS = [
    [r["dataset"], r["algo"], r["hyp"], r["K"], r["split"], r["ACC"], r["floor"],
     r["excess_pp"], r["collapse"], r["commit"] or "blank", "." + r["host"]]
    for r in sorted(CELLS, key=lambda r: (r["dataset"], r["algo"], r["split"]))
    if r["balance"] == "1"
]

# --------------------------------------------------------------------------
# table 7: merge / coverage statistics
# --------------------------------------------------------------------------
KNOWN = [r for r in CELLS if valid(r)]
T7_HEADERS = ["metric", "value", "note"]
T7_ROWS = [
    ["raw cells .30", sum(1 for r in RAW_ROWS if r["host"] == "30"), "result_path scan"],
    ["raw cells .31", sum(1 for r in RAW_ROWS if r["host"] == "31"), "result_path scan"],
    ["raw cells total", len(RAW_ROWS), ""],
    ["unique cells after merge", len(CELLS), "key = dataset/split/algo/hyp/K"],
    ["cross-machine duplicate cells", CROSSHOST, "same key present on both hosts"],
    ["cross-copy ACC disagreements", len(DISAGREE), "spread > 1e-5"],
    ["max single-cell ACC spread", "%.4f" % DISAGREE[0][2] if DISAGREE else "0",
     "/".join(DISAGREE[0][0]) if DISAGREE else ""],
    ["cells without floor", len(CELLS) - len(KNOWN), "json missing or fingerprint not cached"],
    ["cells judged collapsed", sum(1 for r in KNOWN if collapsed(r)), "ACC <= floor + 1pp"],
    ["cells judged valid", sum(1 for r in KNOWN if not collapsed(r)), ""],
    ["overall collapse rate", "%.1f%%" % (100.0 * sum(1 for r in KNOWN if collapsed(r)) / len(KNOWN)),
     "over cells with a floor"],
    ["cells with balance_sampling=1", len(T6_ROWS), "0 in the 2026-10-21 audit"],
    ["datasets scanned", len(BY_DS), ""],
    ["datasets usable for the main table", len(USABLE_DS), ", ".join(USABLE_DS)],
    ["main-calibre (dataset, hyp, K) groups", len(GKEYS), "nsp_max >= 3, ablations excluded"],
    ["distinct git commits in evidence", len({r["commit"] for r in CELLS if r["commit"]}),
     ", ".join(sorted({(r["commit"] or "blank")[:9] for r in CELLS}))],
    ["snapshot", SNAPSHOT, "status probe time; result scan finished 09:28"],
]

# --------------------------------------------------------------------------
# table 8: structural defects
# --------------------------------------------------------------------------
def count_identical(a, b):
    """(dataset, K) groups where two algorithms agree to 1e-4 on ACC/gap/DEO."""
    idx = defaultdict(dict)
    for r in CELLS:
        if r["algo"] in (a, b):
            idx[(r["dataset"], r["K"], r["split"], r["hyp"])][r["algo"]] = r
    return sorted(k for k, v in idx.items()
                  if len(v) == 2 and all(
                      abs(num(v[a][m]) - num(v[b][m])) < 1e-4
                      for m in ("ACC", "ACC_gap", "DEO")))


FA_EQ = count_identical("FairFed", "FedAvg")
MF_EQ = count_identical("mFairFL", "FedAvg")
COMPAS_K20_ALGOS = len({r["algo"] for r in CELLS if r["dataset"] == "COMPAS" and r["K"] == "20"})
CELEBA_ALGOS = sorted({r["algo"] for r in CELLS if r["dataset"] == "CelebA" and not ABLATION.search(r["algo"])})
UTK_ALGOS = sorted({r["algo"] for r in CELLS if r["dataset"] == "UTKFace"})
FOREIGN = sorted({r["algo"] for r in CELLS if r["host"] == "31"} &
                 {"PraFFL", "FedFACT", "DOSFL", "OSFL", "FedMix", "NaiveMix", "ProxProbability"})
NOFLOOR = defaultdict(int)
for r in CELLS:
    if not valid(r):
        NOFLOOR[r["dataset"]] += 1

T8_HEADERS = ["id", "category", "finding", "quantified evidence", "impact on reporting"]
T8_ROWS = [
    ["D1", "dataset-level collapse",
     "9 of 15 datasets are wholly or nearly wholly collapsed; UTKFace and FairFace "
     "have EVERY algorithm below the floor, i.e. the failure is in the "
     "label/partition/evaluator, not in any algorithm.",
     "ADULT 100%, DRUG 100%, FairFace 100%, UTKFace 100%, CREMA-D 96%, "
     "CREMA-D_gender 94%, RAVDESS 94%, BANK 87%, LFWA+ 82%",
     "ADULT and DRUG must be removed from the main table (appendix S7). "
     "UTKFace / FairFace / LFWA+ / CREMA-D* / RAVDESS produce no usable number "
     "for any algorithm."],
    ["D2", "baseline is a no-op",
     "FairFed reproduces FedAvg bit-for-bit, so the FairFed column carries no "
     "information about FairFed.",
     "%d (dataset, K, split) cells identical on ACC+ACC_gap+DEO: %s"
     % (len(FA_EQ), ", ".join("%s/K%s/%s" % (d, k, s) for d, k, s, _ in FA_EQ[:8])),
     "FairFed cannot be used as a comparison point on those datasets until the "
     "implementation is fixed."],
    ["D3", "baseline is a no-op",
     "mFairFL reproduces FedAvg bit-for-bit on some datasets.",
     "%d cells identical: %s" % (len(MF_EQ), ", ".join("%s/K%s/%s" % (d, k, s) for d, k, s, _ in MF_EQ[:8])),
     "Same as D2 for mFairFL."],
    ["D4", "hyper-parameter not wired",
     "FedRenyi lambda has no effect: lam0p25 / lam0p5 / lam1 / lam2 give identical "
     "numbers on DRUG.",
     "4 variant dirs, identical ACC/ACC_gap/DEO on DRUG",
     "The Renyi fairness-strength ablation cannot be reported."],
    ["D5", "result-tree pollution",
     "COMPAS and DRUG 20Clients directories mix PDFFed ablation variants and "
     "lambda sweeps with the main comparison, at nsp=1-2 only.",
     "COMPAS K=20 holds %d distinct algorithm directories" % COMPAS_K20_ALGOS,
     "Any ranking that does not filter these is meaningless; they are excluded "
     "here by the ABLATION regex."],
    ["D6", "coverage hole",
     "CelebA has no PDFFed (non-DP) and no LoGoFair result at all; UTKFace K=30/40 "
     "has no PDFFed-family result.",
     "CelebA main algorithms present: %s | UTKFace algorithms present: %s"
     % (", ".join(CELEBA_ALGOS), ", ".join(UTK_ALGOS)),
     "CelebA can only support a PDFFed_DP claim, and only 3 algorithms reach "
     "nsp=4 there, so the group is too thin to rank."],
    ["D7", "foreign algorithms",
     ".31 result tree contains algorithms belonging to another project, some "
     "without ACC_gap/HM at all.",
     "%s" % ", ".join(FOREIGN),
     "They show up as NaN rows and must be filtered before any ranking."],
    ["D8", "provenance",
     "Cells whose json is missing or whose partition fingerprint is not in the "
     "local cache cannot be assigned a floor or a commit.",
     "%d cells: %s" % (len(CELLS) - len(KNOWN),
                       ", ".join("%s=%d" % (k, v) for k, v in sorted(NOFLOOR.items()))),
     "They are excluded from every collapse statistic and cannot enter a "
     "single-commit table."],
    ["D9", "reproducibility",
     "The same cell key run on two machines / two commits disagrees on ACC.",
     "%d cells disagree, max spread %.4f (%s)"
     % (len(DISAGREE), DISAGREE[0][2] if DISAGREE else 0,
        "/".join(DISAGREE[0][0]) if DISAGREE else "-"),
     "Fix per appendix S9.1: publish from ONE commit, do not re-run. "
     "Selection rule fixed a priori to newest recorded_at."],
    ["D10", "metric interpretation",
     "PDFFed wins HM/ACC but ranks mid-to-low on ACC_gap and DEO in the same "
     "groups, so the HM win is driven by accuracy, not by narrowing unfairness.",
     "see pdffed_head_to_head.csv columns gap_rank / deo_rank",
     "Any claim of a fairness advantage must be made on CelebA PDFFed_DP only "
     "(DEO rank 1), and must not be inferred from HM."],
    ["D11", "mitigation did not help",
     "balance_sampling=1 was turned on for 20 cells (0 in the 2026-10-21 audit) "
     "but every one of them is still collapsed.",
     "%d cells, all on RAVDESS (floor 0.8667) and CREMA-D_gender (floor 0.8293)"
     % len(T6_ROWS),
     "Label-balanced sampling does not rescue high-floor datasets; the collapse "
     "cause is elsewhere."],
]


# --------------------------------------------------------------------------
# emitters
# --------------------------------------------------------------------------
TABLES = [
    dict(name="machine_status", num="T1",
         title="Machine status at snapshot",
         tex_colspec=colspec(["0.17\\textwidth", "0.38\\textwidth", "0.38\\textwidth"]),
         note="Probed live on both hosts at %s. Not derivable from the cell inventory." % SNAPSHOT,
         headers=T1_HEADERS, rows=T1_ROWS),
    dict(name="dataset_floor_availability", num="T2",
         title="Test-set floor, collapse rate and best PDFFed-family cell per dataset",
         note="floor = majority-class share of the test set the cells were actually "
              "evaluated on (bound per result json via aggregate.partition_fingerprints). "
              "A cell is COLLAPSE when ACC <= floor + 0.01. Sorted by collapse rate. "
              "ADULT/BANK/DRUG/COMPAS/DUTCH floors reproduce "
              "appendix_collapse_disclosure.md S3 exactly.",
         headers=T2_HEADERS, rows=T2_ROWS),
    dict(name="pdffed_head_to_head", num="T3",
         title="PDFFed family vs the field, main calibre only",
         note="Main calibre = algorithms reaching nsp_max partitions in that "
              "(dataset, hyp, K) group, with ablation/sweep variants and foreign "
              "algorithms excluded. Ranks are 1-based inside the group: HM and ACC "
              "descending, ACC_gap and DEO ascending (lower is better). "
              "Groups with nsp_max < 3 are dropped because ranks are not comparable.",
         headers=T3_HEADERS, rows=T3_ROWS),
    dict(name="main_calibre_detail", num="T4",
         title="Full main-calibre detail per (dataset, hyp, K)",
         tex_font=r"\scriptsize", tex_tabcolsep="1.5pt",
         note="One row per algorithm per group, ranked by HM. collapsed_splits counts "
              "how many of that algorithm's partitions are constant predictors. "
              "exc_avg_pp / exc_max_pp are ACC above the floor in percentage points.",
         headers=T4_HEADERS, rows=T4_ROWS),
    dict(name="cross_machine_disagreement", num="T5",
         title="Cells whose copies disagree on ACC",
         note="Same cell key present more than once with an ACC spread above 1e-5. "
              "Sorted by spread. copies_host_commit_acc lists every copy.",
         headers=T5_HEADERS, rows=T5_ROWS),
    dict(name="balance_sampling_cells", num="T6",
         title="Cells run with balance_sampling = 1",
         note="The 2026-10-21 audit found 0 such cells out of 788; there are now %d, "
              "and all of them are still collapsed." % len(T6_ROWS),
         headers=T6_HEADERS, rows=T6_ROWS),
    dict(name="merge_stats", num="T7",
         title="Merge and coverage statistics",
         tex_colspec=colspec(["0.26\\textwidth", "0.09\\textwidth", "0.58\\textwidth"]),
         note="Counting the two inventories in raw/ after the newest-recorded_at merge.",
         headers=T7_HEADERS, rows=T7_ROWS),
    dict(name="structural_defects", num="T8",
         title="Structural defects that block publication",
         tex_long=True, tex_font=r"\small",
         tex_colspec=colspec(["0.03\\textwidth", "0.11\\textwidth", "0.30\\textwidth",
                              "0.25\\textwidth", "0.24\\textwidth"]),
         note="Each row is independently evidenced by the cell inventory.",
         headers=T8_HEADERS, rows=T8_ROWS),
]

MD_TITLE = {
    "machine_status": "T1 两台机器运行状态",
    "dataset_floor_availability": "T2 地板、塌缩率与 PDFFed 系最好格（按塌缩率升序）",
    "pdffed_head_to_head": "T3 PDFFed 系 vs 全场（主口径）",
    "main_calibre_detail": "T4 主口径逐组明细",
    "cross_machine_disagreement": "T5 跨机/跨版本 ACC 不一致",
    "balance_sampling_cells": "T6 balance_sampling=1 的格",
    "merge_stats": "T7 合并与覆盖统计",
    "structural_defects": "T8 阻碍出表的结构性缺陷",
}

TEX_TITLE = {
    "machine_status": "Machine status at snapshot %s" % SNAPSHOT,
    "dataset_floor_availability": "Test-set floor, collapse rate and best PDFFed-family cell per dataset",
    "pdffed_head_to_head": "PDFFed family vs the field, main calibre only",
    "main_calibre_detail": "Full main-calibre detail per (dataset, hyp, K)",
    "cross_machine_disagreement": "Cells whose copies disagree on ACC",
    "balance_sampling_cells": "Cells run with \\texttt{balance\\_sampling}=1",
    "merge_stats": "Merge and coverage statistics",
    "structural_defects": "Structural defects that block publication",
}


def write_csv(t):
    path = os.path.join(OUT, t["name"] + ".csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(t["headers"])
        w.writerows(t["rows"])
    return path


def md_escape(s):
    return str(s).replace("|", "\\|")


def write_md(t):
    hs = t["headers"]
    lines = ["## %s" % MD_TITLE[t["name"]], ""]
    if t.get("note"):
        lines += ["> %s" % t["note"], ""]
    lines.append("| " + " | ".join(md_escape(h) for h in hs) + " |")
    lines.append("|" + "|".join(["---"] * len(hs)) + "|")
    rows = t["rows"]
    tail = None
    if t["name"] == "cross_machine_disagreement":
        rows = rows[:15]
        tail = "_（仅列 TOP15，完整 %d 行见 CSV）_" % len(t["rows"])
    if t["name"] == "main_calibre_detail":
        rows = [r for r in rows if int(r[5]) <= 12]
        tail = "_（每组仅列 HM 前 12，完整 %d 行见 CSV）_" % len(t["rows"])
    for r in rows:
        lines.append("| " + " | ".join(md_escape(c) for c in r) + " |")
    if tail:
        lines += ["", tail]
    lines.append("")
    return "\n".join(lines)


def tex_escape(s):
    s = str(s)
    for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("<", r"\textless{}"), (">", r"\textgreater{}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


NUMERIC = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?(%|pp)?$")


def guess_align(headers, rows):
    align = []
    for i, h in enumerate(headers):
        vals = [str(r[i]) for r in rows[:40] if i < len(r)]
        if vals and all(NUMERIC.match(v) for v in vals if v not in ("", "-")):
            align.append("r")
        else:
            align.append("l")
    return "".join(align)


def write_tex(t):
    hs, rows = t["headers"], t["rows"]
    if t["name"] == "main_calibre_detail":
        rows = [r for r in rows if int(r[5]) <= 12]
    spec = t.get("tex_colspec") or guess_align(hs, rows)
    resize = t.get("tex_resize", True) and not t.get("tex_colspec")
    long_ = len(rows) > 25 or t.get("tex_long", False)
    env = "longtable" if long_ else "tabular"
    out = []
    if long_:
        # wide tables need a smaller font and tighter column padding to stay
        # inside \textwidth; longtable cannot be wrapped in \resizebox.
        out += [r"\begingroup" + t.get("tex_font", r"\small"),
                r"\setlength{\tabcolsep}{%s}" % t.get("tex_tabcolsep", "3pt"),
                r"\begin{longtable}{%s}" % spec,
                r"\caption{%s}\label{tab:%s}\\" % (TEX_TITLE[t["name"]], t["name"]),
                r"\toprule"]
    else:
        out += [r"\begin{table}[htbp]\centering\small",
                r"\caption{%s}\label{tab:%s}" % (TEX_TITLE[t["name"]], t["name"])]
        if resize:
            out.append(r"\resizebox{\textwidth}{!}{%")
        out += [r"\begin{tabular}{%s}" % spec, r"\toprule"]
    out.append(" & ".join(tex_escape(h) for h in hs) + r" \\")
    out.append(r"\midrule")
    if long_:
        out.append(" & ".join(tex_escape(h) for h in hs) + r" \\")
        out.append(r"\midrule")
        out.append(r"\endfirsthead")
        out.append(r"\multicolumn{%d}{r}{\ldots\ continued}\\" % len(hs))
        out.append(r"\toprule")
        out.append(" & ".join(tex_escape(h) for h in hs) + r" \\")
        out.append(r"\midrule")
        out.append(r"\endhead")
    for r in rows:
        out.append(" & ".join(tex_escape(c) for c in r) + r" \\")
    out += [r"\bottomrule", r"\end{%s}" % env]
    if long_:
        out.append(r"\endgroup")
    else:
        if resize:
            out.append("}")
        out.append(r"\end{table}")
    return "\n".join(out)


CONCLUSIONS_MD = """## 底线结论

按地板判据（`ACC ≤ floor + 1pp`），当前 **{uniq} 个唯一格**里只有 **{usable} 个数据集**能进主结果表：
{usable_list}。

1. **PDFFed（非 DP）确实拿第一，但赢在 ACC 不在公平性。** {p1} 个主口径组里 PDFFed 有
   {p1_first} 组 HM#1、{p1_acc} 组 ACC#1。优势最大的是 DUTCH（对 FedAvg ΔHM +0.11~+0.15，
   对最强基线 mFairFL +0.071），其次 AudioMNIST（对 Po +0.025）。但同组内 `gap_rank`
   与 `deo_rank` 系统性靠后（COMPAS DEO 13/16、DUTCH ACC_gap 10/15），**HM#1 是 ACC 抬起来的**。
2. **CelebA 的 PDFFed_DP 是唯一「DEO#1 + HM#1」的组合**（DEO 0.018/0.013 vs FedAvg 0.253/0.306），
   代价是 ACC 掉 3.7~10.4pp；且该组只有 3 个算法达到 nsp=4，样本量不足以支撑排名结论。
3. **BANK 的优势在噪声量级。** floor 0.8776，PDFFed 平均只越线 +1.19~+1.46pp
   （n_test=13211 → 1pp≈132 样本），且 4 个划分里仍有 1~2 个塌缩。按审计 §7 只能保留越线结果，
   并**同时披露**其余 13 个算法 ≤+1.02pp、10 个 0 越线，且给出 floor 作参照。
4. **五个模态零可用结果**：文本（bios）、人脸（UTKFace / FairFace / LFWA+）、
   语音情感（CREMA-D / CREMA-D_gender / RAVDESS）。bios floor = **0.5449**，
   PDFFed_DP 0.520 与 PDFFed 0.485 都在地板之下 —— 两者都是塌缩格，
   此前「PDFFed_DP 在文本上有真实优势」的说法**不成立**，那个「最低 ACC_gap」正是塌缩伪公平。
5. **ADULT、DRUG 按审计 §7 整体移出主结果表**（100% 塌缩，0 格越线）。
6. **出表方式**：不重跑，按单一 commit 出表（BANK/DRUG/COMPAS/DUTCH 用 `e4f45401b`，
   ADULT 用 `e34c32ad3`），同格取 `recorded_at` 最新。跨机分歧 {dis} 处已列于 T5。

### 待办（阻塞项）

- CelebA 补 PDFFed（非 DP）与 LoGoFair，并把 FedAvg/FairFed 补齐到 nsp=4。
- 修 FairFed ≡ FedAvg、mFairFL ≡ FedAvg、FedRenyi λ 未生效三个基线缺陷（见 T8 D2/D3/D4）。
- UTKFace / FairFace 全算法低于地板，需先排查标签映射与评测器，属数据集级失效（T8 D1）。
- {nofloor} 格无版本指纹，无法归入任何 commit 口径（T8 D8）。
"""


def p1_ranks():
    """(groups, HM#1 count, ACC#1 count) for PDFFed non-DP in the main calibre."""
    p1 = [r for r in T3_ROWS if r[3] == "PDFFed"]
    return len(p1), sum(1 for r in p1 if r[6] == 1), sum(1 for r in p1 if r[7] == 1)


def conclusions():
    np1, nhm, nacc = p1_ranks()
    return CONCLUSIONS_MD.format(
        uniq=len(CELLS),
        usable=len(USABLE_DS),
        usable_list="、".join(USABLE_DS),
        p1=np1,
        p1_first=nhm,
        p1_acc=nacc,
        dis=len(DISAGREE),
        nofloor=len(CELLS) - len(KNOWN),
    )


HEADER_MD = """# 两台机器（192.168.1.30 / 192.168.1.31）实验结果汇总

- **快照时间**：{snap}（状态探针）；结果扫描完成于 09:28
- **判据**：塌缩格 = `ACC <= floor + 0.01`，`floor` = 该格实际评测所用测试集的多数类占比，
  由每份结果 json 的 `aggregate.partition_fingerprints` 绑回
  `partition_cache/v2/<DS>/<fp>/metadata.json` 的逐客户端 `test_stats.label_counts` 求和得到
  （口径来自 `reported_results/collapse_audit/appendix_collapse_disclosure.md` §2）
- **合并规则**：cell key = `dataset/split/algo/hyp/K`；两台都有时取 `recorded_at` 最新的一份
  （审计 §9.7 的 newest 规则）
- **主口径**：同一 `(dataset, hyp, K)` 组内达到 `nsp_max` 个划分的算法；消融/超参扫描变体
  （`comb*`/`confX*`/`eoX*`/`pvc*`/`pve*`/`no*`/`lam*`/`eps*`）与他项目算法
  （PraFFL/FedFACT/DOSFL/OSFL/FedMix/NaiveMix/ProxProbability）一律排除
- **数据源**：`raw/cells_30.tsv`（{n30} 行）、`raw/cells_31.tsv`（{n31} 行）
- **重算**：`python analysis/build_two_machine_summary.py`

> 注意：此前用过的 `DEO=0 & SPD=0 & FR=1` 塌缩判据**不采用**。它在高 ACC 的健康格上产生假阳性
> （BANK 的 FedProx/PDFFed_DP/Scaffold、LFWA+ 的 FedRenyi、DRUG 的 LoGoFair），
> 又漏掉所有公平指标只是「很小」而非「恰好为 0」的真塌缩
> （BANK 的 FedFB 0.453、FairFace 与 UTKFace 全体）。

""".format(snap=SNAPSHOT,
           n30=sum(1 for r in RAW_ROWS if r["host"] == "30"),
           n31=sum(1 for r in RAW_ROWS if r["host"] == "31"))


def main():
    os.makedirs(OUT, exist_ok=True)
    md = [HEADER_MD]
    tex_tables = []
    for t in TABLES:
        p = write_csv(t)
        md.append(write_md(t))
        head = ["\\section*{%s\\quad %s}" % (t["num"], tex_escape(t["title"]))]
        if t.get("note"):
            head.append("{\\footnotesize\\textit{%s}\\par}" % tex_escape(t["note"]))
        tex_tables.append("\n".join(head) + "\n\n" + write_tex(t))
        print("  csv -> %s (%d rows)" % (os.path.relpath(p, REPO), len(t["rows"])))
    md.append(conclusions())
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print("  md  -> reported_results/two_machine_summary/summary.md")

    np1, nhm, nacc = p1_ranks()
    concl = (CONCLUSION_TEX.replace("@P1@", str(np1))
             .replace("@P1HM@", str(nhm)).replace("@P1ACC@", str(nacc)))
    doc = PREAMBLE + "\n\n".join(tex_tables) + "\n\n" + concl + "\n\n" + POSTAMBLE
    with open(os.path.join(OUT, "summary.tex"), "w", encoding="utf-8") as fh:
        fh.write(doc)
    print("  tex -> reported_results/two_machine_summary/summary.tex")


PREAMBLE = r"""\documentclass[11pt]{article}
% Auto-generated by analysis/build_two_machine_summary.py -- do not edit by hand.
\usepackage[a4paper,margin=1.6cm,landscape]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{array}
\usepackage{amsmath}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.05}
\title{Two-machine experiment summary (192.168.1.30 / 192.168.1.31)}
\author{}
\date{Snapshot: 2026-10-22 09:33 (Asia/Shanghai)}
\begin{document}
\maketitle

\section*{Criterion}
A cell is a \emph{constant predictor} (collapsed) when
$\mathrm{ACC} \le \mathrm{floor} + 0.01$, where \emph{floor} is the
majority-class share of the test set that the cell was actually evaluated on.
The floor is bound per result json through
\texttt{aggregate.partition\_fingerprints} $\rightarrow$
\texttt{partition\_cache/v2/\textless DS\textgreater/\textless fp\textgreater/metadata.json} $\rightarrow$ per-client
\texttt{test\_stats.label\_counts}. The criterion is the one fixed by
\texttt{reported\_results/collapse\_audit/appendix\_collapse\_disclosure.md}~\S2.
Cell key is \texttt{dataset/split/algo/hyp/K}; when both hosts hold a copy the
newest \texttt{recorded\_at} wins (\S9.7). Main calibre means the algorithms
reaching \texttt{nsp\_max} partitions in that group, with ablation/sweep
variants and foreign algorithms excluded.
"""

CONCLUSION_TEX = r"""\section*{Bottom line}
Under the floor criterion only \textbf{DUTCH, COMPAS, AudioMNIST, CelebA
(PDFFed\_DP only) and BANK (with mandatory disclosure)} can enter the main
result table.

\begin{enumerate}
\item PDFFed (non-DP) does rank first, but it wins on accuracy, not on
fairness: of its @P1@ main-calibre groups it takes HM rank 1 in @P1HM@ and
ACC rank 1 in @P1ACC@,
while \texttt{gap\_rank} and \texttt{deo\_rank} sit mid-to-low in the same
groups (COMPAS DEO 13/16, DUTCH ACC\_gap 10/15). The largest genuine margin is
DUTCH ($\Delta$HM $+0.11\ldots+0.15$ over FedAvg, $+0.071$ over the strongest
baseline mFairFL), then AudioMNIST ($+0.025$ over Po).
\item CelebA PDFFed\_DP is the only combination that is simultaneously DEO rank
1 and HM rank 1 (DEO $0.018/0.013$ vs FedAvg $0.253/0.306$), at the cost of
$3.7$--$10.4$pp of accuracy; only 3 algorithms reach nsp=4 there, so the group
is too thin to support a ranking claim.
\item The BANK margin is at noise scale: floor $0.8776$, PDFFed exceeds it by
only $+1.19$ to $+1.46$pp on average ($n_{\text{test}}=13211$, so 1pp
$\approx132$ samples) and 1--2 of its 4 partitions are still collapsed. Per
\S7 the crossing results may be kept only alongside the disclosure that the
other 13 algorithms stay $\le+1.02$pp with 10 of them never crossing.
\item Five modalities currently yield nothing usable: text (bios), faces
(UTKFace / FairFace / LFWA+) and speech emotion (CREMA-D / CREMA-D\_gender /
RAVDESS). bios floor is $0.5449$; PDFFed\_DP $0.520$ and PDFFed $0.485$ are
both \emph{below} it, i.e. both are collapsed cells, and their small ACC\_gap
is spurious fairness from constant prediction.
\item ADULT and DRUG leave the main table entirely (100\% collapsed, zero
crossing cells) per \S7.
\item Publication route: do not re-run; publish from a single commit
(\texttt{e4f45401b} for BANK/DRUG/COMPAS/DUTCH, \texttt{e34c32ad3} for ADULT),
selecting the newest \texttt{recorded\_at} per cell. Cross-host disagreements
are enumerated in Table~\ref{tab:cross_machine_disagreement}.
\end{enumerate}
"""

POSTAMBLE = r"\end{document}"


if __name__ == "__main__":
    main()
