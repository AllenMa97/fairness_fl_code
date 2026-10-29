#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw ACC and fairness versus client number (20/30/40).

Reads ``reported_results/clients30_raw.tsv`` (flattened from the .30 server) and
plots, per tabular dataset, how ACC and the fairness gap move as the number of
clients grows.  Each dataset is one figure with a 2 x 4 grid: the top row is ACC
(higher is better), the bottom row the fairness gap (lower is better; DEO for
``--fairness eo``, |SPD| for ``--fairness dp``), and the four columns are the
four data-heterogeneity partitions.

``--layout row`` collapses the four partitions into a single line per method by
averaging ACC / fairness across partitions, then lays the five datasets side by
side with a shared legend (the FedFACT multi-panel style).

File selection: some (dataset, partition, method) cells hold more than one
result json (a re-run with a different code fingerprint, or the FedRenyi lambda
ladder).  To keep each curve on one code version, the loader prefers the file
whose ``fingerprint`` matches the 30/40-client sweep batch; otherwise it falls
back to the lowest file number.  A fingerprint-consistency audit is printed to
stdout and written to ``reported_results/client_number_fp_audit.tsv`` so curves
that cross a code-version boundary at 20 -> 30 clients can be spotted.

Usage::

    python analysis/plot_client_number.py                  # per dataset, DEO
    python analysis/plot_client_number.py --fairness dp    # |SPD| axis
    python analysis/plot_client_number.py --layout row     # summary, shared legend
"""

from __future__ import annotations

import csv
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fig_variants

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO_ROOT, "reported_results", "clients30_raw.tsv")
AUDIT = os.path.join(REPO_ROOT, "reported_results", "client_number_fp_audit.tsv")
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "figures")

DATASETS = ("COMPAS", "DRUG", "DUTCH", "ADULT", "BANK")
SPLITS = ("Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform")
SPLIT_TICK = [r"$\alpha{=}0.1$", r"$\alpha{=}0.5$", r"$\alpha{=}1$", "Uniform"]
CCS = ("20Clients", "30Clients", "40Clients")
CC_X = {"20Clients": 0, "30Clients": 1, "40Clients": 2}
CC_TICK = [20, 30, 40]

CURVES = ("PDFFed", "PDFFed_DP", "FedRenyi", "FedAvg", "FedProx", "FairFed",
          "mFairFL", "FL_FairBatch")

# What actually reaches the canvas: ``CURVES`` drives the data selection and the
# fingerprint audit, which stay complete even when a variant hides a method.
DRAW = fig_variants.active(CURVES)

COLORS = {
    "PDFFed": "tab:blue", "PDFFed_DP": "tab:orange", "FedRenyi": "tab:red",
    "FedAvg": "#7f7f7f", "FedProx": "#8c564b", "FairFed": "#9467bd",
    "mFairFL": "#17becf", "FL_FairBatch": "#bcbd22",
}
MARKERS = {
    "PDFFed": "*", "PDFFed_DP": "o", "FedRenyi": "^", "FedAvg": "s",
    "FedProx": "D", "FairFed": "v", "mFairFL": "P", "FL_FairBatch": "X",
}
LINEWIDTH = {"PDFFed": 2.6}
LINESTYLE = {"PDFFed_DP": (0, (4, 2))}
LABEL = {"PDFFed": "PDFFed", "PDFFed_DP": "PDFFed-DP", "FedRenyi": "FedRenyi",
         "FedAvg": "FedAvg", "FedProx": "FedProx", "FairFed": "FairFed",
         "mFairFL": "mFairFL", "FL_FairBatch": "FL-FairBatch"}

_FILE_NUM = re.compile(r"^(\d+)")


def load():
    rows = []
    with open(RAW, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r.get("algo_dir") in CURVES:
                rows.append(r)
    return rows


def fnum(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else float("nan")


def _file_stem(r):
    m = _FILE_NUM.match(r["file"])
    return int(m.group(1)) if m else 9999


def select(rows):
    """Return {(ds, sp, algo, cc): row} using consistency-preferring picks."""
    by = defaultdict(list)
    for r in rows:
        by[(r["dataset"], r["split"], r["algo_dir"], r["kdir"])].append(r)

    sel = {}
    for ds in DATASETS:
        for sp in SPLITS:
            for algo in CURVES:
                fps = {cc: {r["fp"] for r in by.get((ds, sp, algo, cc), [])}
                       for cc in CCS}
                sweep = (fps["30Clients"] & fps["40Clients"])
                if sweep:
                    target = sorted(sweep)[0]
                elif fps["30Clients"]:
                    target = sorted(fps["30Clients"])[0]
                elif fps["40Clients"]:
                    target = sorted(fps["40Clients"])[0]
                else:
                    target = None
                for cc in CCS:
                    cands = by.get((ds, sp, algo, cc), [])
                    if not cands:
                        continue
                    pool = [r for r in cands if r["fp"] == target] or cands
                    pool = sorted(pool, key=_file_stem)
                    sel[(ds, sp, algo, cc)] = pool[0]
    return sel


def metric_val(r, metric):
    if metric == "acc":
        return fnum(r, "ACC")
    if metric == "dp":
        return abs(fnum(r, "SPD"))
    return fnum(r, "DEO")


def series(sel, ds, sp, algo, metric):
    xs, ys = [], []
    for cc in CCS:
        r = sel.get((ds, sp, algo, cc))
        if r is None:
            continue
        xs.append(CC_X[cc])
        ys.append(metric_val(r, metric))
    return xs, ys


def avg_series(sel, ds, algo, metric):
    xs, ys = [], []
    for cc in CCS:
        vals = []
        for sp in SPLITS:
            r = sel.get((ds, sp, algo, cc))
            if r is not None and not np.isnan(metric_val(r, metric)):
                vals.append(metric_val(r, metric))
        if vals:
            xs.append(CC_X[cc])
            ys.append(float(np.mean(vals)))
    return xs, ys


def draw_curve(ax, xs, ys, algo):
    color = COLORS[algo]
    ax.plot(xs, ys, color=color, lw=LINEWIDTH.get(algo, 1.4),
            ls=LINESTYLE.get(algo, "-"), marker=MARKERS[algo], markersize=6,
            markerfacecolor="white" if algo != "PDFFed" else color,
            markeredgecolor=color, markeredgewidth=1.1, label=LABEL[algo],
            zorder=3)


def _set_cc_axis(ax):
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(CC_TICK)
    ax.tick_params(labelsize=9)
    ax.grid(True, axis="y", alpha=0.25, lw=0.6)
    ax.set_xlim(-0.3, 2.3)


def audit(rows, sel):
    """Print + persist a fingerprint-consistency matrix for the two own methods."""
    lines = ["dataset\tsplit\talgo\t20Clients\t30Clients\t40Clients\tconsistent"]
    for ds in DATASETS:
        for sp in SPLITS:
            for algo in ("PDFFed", "PDFFed_DP"):
                fps = []
                for cc in CCS:
                    r = sel.get((ds, sp, algo, cc))
                    fps.append((r["fp"][:6] if r else "MISSING"))
                consistent = "yes" if len(set(f for f in fps if f != "MISSING")) == 1 else "NO"
                lines.append("\t".join([ds, sp, algo, *fps, consistent]))
    with open(AUDIT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    bad = [l for l in lines[1:] if l.rstrip().endswith("\tNO")]
    print("[audit] fingerprint consistency (20/30/40 clients) ->", AUDIT)
    print("[audit] cells whose code version changes across client counts: %d"
          % len(bad))


def main():
    metric = "eo"
    if "--fairness" in sys.argv:
        metric = sys.argv[sys.argv.index("--fairness") + 1]
        if metric not in ("eo", "dp"):
            sys.exit("--fairness must be 'eo' or 'dp'")
    layout = "panel"
    if "--layout" in sys.argv:
        layout = sys.argv[sys.argv.index("--layout") + 1]
        if layout not in ("panel", "row"):
            sys.exit("--layout must be 'panel' or 'row'")

    rows = load()
    sel = select(rows)
    audit(rows, sel)

    ylab = "Fairness gap ($|$DEO$|$)" if metric == "eo" else \
        "Fairness gap ($|$SPD$|$)"

    if layout == "row":
        n = len(DATASETS)
        fig, axes = plt.subplots(2, n, figsize=(2.9 * n, 4.4), sharex=True)
        for j, ds in enumerate(DATASETS):
            for i, (mkey, ylabel) in enumerate((("acc", "ACC"), (metric, ylab))):
                ax = axes[i][j]
                for algo in DRAW:
                    xs, ys = avg_series(sel, ds, algo, mkey)
                    if len(xs) > 1:
                        draw_curve(ax, xs, ys, algo)
                ax.set_title(ds, fontsize=10.5)
                ax.set_ylabel(ylabel if j == 0 else "", fontsize=9)
                ax.tick_params(labelsize=8)
                ax.set_xlim(-0.3, 2.3)
                if i == 1:
                    ax.set_xticks([0, 1, 2])
                    ax.set_xticklabels(CC_TICK, fontsize=8)
                    ax.set_xlabel("clients", fontsize=8)
                ax.grid(True, axis="y", alpha=0.25, lw=0.6)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=7.5, frameon=False,
                   loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=8,
                   handletextpad=0.35, columnspacing=1.0)
        fig.subplots_adjust(left=0.07, right=0.985, top=0.94, bottom=0.16,
                            wspace=0.18, hspace=0.30)
        stem = "client_row_" + metric + fig_variants.suffix()
        fig.savefig(os.path.join(OUT_DIR, stem + ".png"), dpi=300)
        fig.savefig(os.path.join(OUT_DIR, stem + ".pdf"))
        plt.close(fig)
        print("wrote", os.path.join(OUT_DIR, stem + ".png"))
        print("wrote", os.path.join(OUT_DIR, stem + ".pdf"))
        return

    for ds in DATASETS:
        fig, axes = plt.subplots(2, 4, figsize=(13.0, 5.4), sharex=True)
        for j, sp in enumerate(SPLITS):
            for i, (mkey, ylabel) in enumerate((("acc", "ACC"), (metric, ylab))):
                ax = axes[i][j]
                for algo in DRAW:
                    xs, ys = series(sel, ds, sp, algo, mkey)
                    if len(xs) > 1:
                        draw_curve(ax, xs, ys, algo)
                _set_cc_axis(ax)
                if i == 0:
                    ax.set_title(SPLIT_TICK[j], fontsize=10.5)
                if j == 0:
                    ax.set_ylabel(ylabel, fontsize=9)
                if i == 1:
                    ax.set_xlabel("clients", fontsize=8)
        fig.suptitle(ds, fontsize=12, y=0.99)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=7, frameon=False, ncol=8,
                   loc="lower center", bbox_to_anchor=(0.5, 0.02),
                   handletextpad=0.35, columnspacing=1.0)
        fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.14,
                            wspace=0.20, hspace=0.35)
        stem = "client_%s_%s%s" % (ds, metric, fig_variants.suffix())
        fig.savefig(os.path.join(OUT_DIR, stem + ".png"), dpi=300)
        fig.savefig(os.path.join(OUT_DIR, stem + ".pdf"))
        plt.close(fig)
        print("wrote", os.path.join(OUT_DIR, stem + ".png"))


if __name__ == "__main__":
    main()
