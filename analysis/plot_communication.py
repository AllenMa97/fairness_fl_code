#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw communication-effectiveness (convergence) curves, one setting per figure.

Reads ``analysis/curves2/curve_<DS>_<SPLIT>.csv`` (extracted from the .31 server
training logs with correct "next GMT" round attribution) and plots, for a single
tabular dataset x heterogeneity split, how ACC and the fairness metric evolve
over communication rounds for PDFFed versus the baselines.

Each setting (dataset x split) is its own figure with two stacked panels: ACC on
top (higher is better) and the fairness metric below (lower is better; DEO for
``--fairness eo``, |SPD| for ``--fairness dp``).  PDFFed is drawn bold with a
star and PDFFed-DP dashed; the baselines are thin coloured lines.  Because the
baselines log a global test only every ``test_eval_freq = 10`` rounds (49
points) while PDFFed/PDFFed-DP test every round (501 points), the density
difference is kept as-is.

Usage::

    python analysis/plot_communication.py                  # one file per setting, DEO
    python analysis/plot_communication.py --fairness dp    # |SPD| axis
"""

from __future__ import annotations

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fig_variants

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURVE_DIR = os.path.join(REPO_ROOT, "analysis", "curves2")
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "figures")

DATASETS = ("COMPAS", "DRUG", "DUTCH", "ADULT", "BANK")
SPLITS = ("Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform")
SPLIT_LABEL = {"Dirichlet01": r"$\alpha{=}0.1$", "Dirichlet05": r"$\alpha{=}0.5$",
               "Dirichlet1": r"$\alpha{=}1$", "Uniform": "Uniform"}
METRIC_LABEL = {"eo": "DEO", "dp": "|SPD|"}

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
LABELS = {"PDFFed": "PDFFed", "PDFFed_DP": "PDFFed-DP",
          "FedRenyi": "FedRenyi", "FedAvg": "FedAvg", "FedProx": "FedProx",
          "FairFed": "FairFed", "mFairFL": "mFairFL",
          "FL_FairBatch": "FL-FairBatch"}


def load(ds, sp):
    """Return {algo: {round: {ACC,DEO,SPD}}} for one (dataset, split)."""
    path = os.path.join(CURVE_DIR, "curve_%s_%s.csv" % (ds, sp))
    if not os.path.exists(path):
        return {}
    rows = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            algo = r["algo"]
            if algo not in CURVES:
                continue
            rnd = int(r["round"])
            rows.setdefault(algo, {})[rnd] = {
                "ACC": _f(r, "ACC"), "DEO": _f(r, "DEO"), "SPD": _f(r, "SPD"),
            }
    return rows


def _f(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else float("nan")


def series(rows, algo, metric):
    xs, ys = [], []
    for rnd in sorted(rows.get(algo, {})):
        v = rows[algo][rnd]
        val = v["ACC"] if metric == "acc" else (
            abs(v["SPD"]) if metric == "dp" else v["DEO"])
        xs.append(rnd)
        ys.append(val)
    return xs, ys


def draw_curve(ax, xs, ys, algo):
    color = COLORS[algo]
    lw = LINEWIDTH.get(algo, 1.4)
    ls = LINESTYLE.get(algo, "-")
    marker = MARKERS[algo]
    dense = algo in ("PDFFed", "PDFFed_DP")
    markevery = 25 if dense else 1
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, marker=marker,
            markersize=6 if algo != "PDFFed" else 7, markevery=markevery,
            markerfacecolor="white" if algo not in ("PDFFed",) else color,
            markeredgecolor=color, markeredgewidth=1.1, label=LABELS[algo],
            zorder=3)


def main():
    metric = "eo"
    if "--fairness" in sys.argv:
        metric = sys.argv[sys.argv.index("--fairness") + 1]
        if metric not in ("eo", "dp"):
            sys.exit("--fairness must be 'eo' or 'dp'")

    ylabel = METRIC_LABEL[metric]

    for ds in DATASETS:
        for sp in SPLITS:
            rows = load(ds, sp)
            if not rows:
                continue
            fig, axes = plt.subplots(2, 1, figsize=(11.0, 5.2), sharex=True)
            for i, (mkey, lab) in enumerate((("acc", "ACC"),
                                             (metric, ylabel))):
                ax = axes[i]
                for algo in CURVES:
                    xs, ys = series(rows, algo, mkey)
                    if len(xs) > 1:
                        draw_curve(ax, xs, ys, algo)
                ax.set_ylabel(lab, fontsize=11)
                ax.set_xlim(-10, 510)
                ax.tick_params(labelsize=9)
                ax.grid(True, axis="y", alpha=0.25, lw=0.6)
                if i == 0:
                    ax.set_title(f"{ds}  {SPLIT_LABEL[sp]}", fontsize=12)
            axes[1].set_xlabel("Communication round", fontsize=10)
            handles, labels = axes[0].get_legend_handles_labels()
            axes[0].legend(handles, labels, fontsize=7.5, frameon=False, ncol=8,
                           loc="best", handletextpad=0.35, columnspacing=0.8)
            fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.09,
                                hspace=0.20)
            stem = "comm_%s_%s_%s%s" % (ds, sp, metric, fig_variants.suffix())
            fig.savefig(os.path.join(OUT_DIR, stem + ".png"), dpi=300)
            fig.savefig(os.path.join(OUT_DIR, stem + ".pdf"))
            plt.close(fig)
            print("wrote", os.path.join(OUT_DIR, stem + ".png"))


if __name__ == "__main__":
    main()
