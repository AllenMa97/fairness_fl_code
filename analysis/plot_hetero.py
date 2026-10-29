#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw ACC and fairness versus data-heterogeneity (Dirichlet alpha).

Reads ``reported_results/hetero_raw.tsv`` (flattened from the .31 server) and
plots, per tabular dataset, how ACC and the fairness gap move as the client
partition goes from highly heterogeneous (Dirichlet alpha=0.1) to homogeneous
(Uniform, equivalent to alpha -> +inf).

Each dataset is one figure with two stacked panels: ACC on top (higher is
better) and the fairness gap below (lower is better, DEO for ``--fairness eo``,
|SPD| for ``--fairness dp``).  PDFFed is drawn bold with a star; the baselines
are thin coloured lines so the reader can judge heterogeneity robustness at a
glance.

``--layout row`` puts all datasets side by side in one figure with a shared
legend, the FedFACT multi-panel style.

Usage::

    python analysis/plot_hetero.py                  # one file per dataset, DEO
    python analysis/plot_hetero.py --fairness dp    # |SPD| axis
    python analysis/plot_hetero.py --layout row     # all datasets, shared legend
"""

from __future__ import annotations

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fig_variants

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO_ROOT, "reported_results", "hetero_raw.tsv")
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "figures")

DATASETS = ("COMPAS", "DRUG", "DUTCH", "ADULT", "BANK")
# Heterogeneity axis, low to high.  Uniform is the homogeneous limit.
SPLITS = ("Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform")
SPLIT_X = {"Dirichlet01": 0, "Dirichlet05": 1, "Dirichlet1": 2, "Uniform": 3}
SPLIT_TICK = [r"$\alpha{=}0.1$", r"$\alpha{=}0.5$", r"$\alpha{=}1$",
              "Uniform"]

# The methods drawn as curves.  PDFFed and PDFFed_DP are the paper's own
# methods; the rest are the standard baselines kept for the heterogeneity
# comparison.  LoGoFair and FederatedAverageWithPo are private/broken and left
# out; the ablation subdirectories (PDFFed_eoX05, ...) and the FedRenyi lambda
# ladder are not on this axis either, only the single-configuration runs.
CURVES = fig_variants.active(
    ("PDFFed", "PDFFed_DP", "FedRenyi", "FedAvg", "FedProx", "FairFed",
     "mFairFL", "FL_FairBatch"))

COLORS = {
    "PDFFed": "tab:blue",
    "PDFFed_DP": "tab:orange",
    "FedRenyi": "tab:red",
    "FedAvg": "#7f7f7f",
    "FedProx": "#8c564b",
    "FairFed": "#9467bd",
    "mFairFL": "#17becf",
    "FL_FairBatch": "#bcbd22",
}
MARKERS = {
    "PDFFed": "*", "PDFFed_DP": "o", "FedRenyi": "^", "FedAvg": "s",
    "FedProx": "D", "FairFed": "v", "mFairFL": "P", "FL_FairBatch": "X",
}
LINEWIDTH = {"PDFFed": 2.6}
LINESTYLE = {"PDFFed_DP": (0, (4, 2))}


def load():
    rows = []
    with open(RAW, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if not r.get("algo_dir"):
                continue
            if r["algo_dir"] not in CURVES:
                continue
            rows.append(r)
    return rows


def fnum(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else float("nan")


def series(rows, dataset, algo, metric):
    """Return (xs, ys) for one method across the four heterogeneity levels."""
    xs, ys = [], []
    for sp in SPLITS:
        hit = [r for r in rows if r["dataset"] == dataset and r["split"] == sp
               and r["algo_dir"] == algo]
        if not hit:
            continue
        r = hit[0]
        val = fnum(r, "ACC") if metric == "acc" else (
            abs(fnum(r, "SPD")) if metric == "dp" else fnum(r, "DEO"))
        xs.append(SPLIT_X[sp])
        ys.append(val)
    return xs, ys


def draw_curve(ax, xs, ys, algo, metric):
    color = COLORS[algo]
    lw = LINEWIDTH.get(algo, 1.4)
    ls = LINESTYLE.get(algo, "-")
    marker = MARKERS[algo]
    label = {"PDFFed": "PDFFed", "PDFFed_DP": "PDFFed-DP",
             "FedRenyi": "FedRenyi", "FedAvg": "FedAvg",
             "FedProx": "FedProx", "FairFed": "FairFed",
             "mFairFL": "mFairFL", "FL_FairBatch": "FL-FairBatch"}[algo]
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, marker=marker, markersize=6,
            markerfacecolor="white" if algo != "PDFFed" else color,
            markeredgecolor=color, markeredgewidth=1.1, label=label, zorder=3)


def draw_panel(ax, rows, datasets, metric, title_size=11.0):
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(SPLIT_TICK)
    ax.tick_params(labelsize=9)
    ax.grid(True, axis="y", alpha=0.25, lw=0.6)
    for i, ds in enumerate(datasets):
        ax.axvline(i, color="0.85", lw=0.8, zorder=0)


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

    if layout == "row":
        n = len(DATASETS)
        fig, axes = plt.subplots(2, n, figsize=(2.9 * n, 4.4), sharex=True)
        for j, ds in enumerate(DATASETS):
            for i, (mkey, ylabel) in enumerate((("acc", "ACC"),
                                                (metric, "Fairness gap"))):
                ax = axes[i][j]
                for algo in CURVES:
                    xs, ys = series(rows, ds, algo, mkey)
                    if len(xs) > 1:
                        draw_curve(ax, xs, ys, algo, mkey)
                ax.set_title(ds, fontsize=10.5)
                ax.set_xlim(-0.35, 3.35)
                ax.set_ylabel(ylabel if j == 0 else "", fontsize=9)
                ax.tick_params(labelsize=8)
                if i == 1:
                    ax.set_xticks([0, 1, 2, 3])
                    ax.set_xticklabels(SPLIT_TICK, fontsize=8)
                ax.grid(True, axis="y", alpha=0.25, lw=0.6)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=7.5, frameon=False,
                   loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=8,
                   handletextpad=0.35, columnspacing=1.0)
        fig.subplots_adjust(left=0.07, right=0.985, top=0.94, bottom=0.16,
                            wspace=0.18, hspace=0.30)
        stem = "hetero_row_" + metric + fig_variants.suffix()
        fig.savefig(os.path.join(OUT_DIR, stem + ".png"), dpi=300)
        fig.savefig(os.path.join(OUT_DIR, stem + ".pdf"))
        plt.close(fig)
        print("wrote", os.path.join(OUT_DIR, stem + ".png"))
        print("wrote", os.path.join(OUT_DIR, stem + ".pdf"))
        return

    for ds in DATASETS:
        fig, axes = plt.subplots(2, 1, figsize=(6.0, 5.2), sharex=True)
        for i, (mkey, ylabel) in enumerate((("acc", "ACC"),
                                            (metric, "Fairness gap"))):
            ax = axes[i]
            for algo in CURVES:
                xs, ys = series(rows, ds, algo, mkey)
                if len(xs) > 1:
                    draw_curve(ax, xs, ys, algo, mkey)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xlim(-0.35, 3.35)
            ax.tick_params(labelsize=9)
            ax.grid(True, axis="y", alpha=0.25, lw=0.6)
            if i == 0:
                ax.set_title(ds, fontsize=12)
        axes[1].set_xticks([0, 1, 2, 3])
        axes[1].set_xticklabels(SPLIT_TICK)
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels, fontsize=7, frameon=False, ncol=4,
                       loc="best", handletextpad=0.4, columnspacing=0.9)
        fig.subplots_adjust(left=0.14, right=0.97, top=0.93, bottom=0.10,
                            hspace=0.22)
        stem = "hetero_%s_%s%s" % (ds, metric, fig_variants.suffix())
        fig.savefig(os.path.join(OUT_DIR, stem + ".png"), dpi=300)
        fig.savefig(os.path.join(OUT_DIR, stem + ".pdf"))
        plt.close(fig)
        print("wrote", os.path.join(OUT_DIR, stem + ".png"))


if __name__ == "__main__":
    main()
