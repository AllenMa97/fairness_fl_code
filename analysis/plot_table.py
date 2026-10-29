#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw the tabular main-result (table) comparison: ACC vs fairness metric.

Reads ``reported_results/hetero_raw.tsv`` (flattened from the .31 server result
jsons) and, for every tabular dataset x heterogeneity split, plots all sixteen
algorithms as points on the accuracy-fairness plane -- the same layout the audio
figures use.  Each (dataset, split) is one figure with two stacked panels: DEO
on top, |SPD| below, ACC on the shared x-axis.  PDFFed is a large blue star,
PDFFed-DP an orange circle; the fourteen baselines are smaller coloured markers.

This is the figure counterpart of the paper's main results table: one file per
cell of the dataset x split grid, so every reported combination is shown.

``--layout row`` collapses the grid the other way round: one figure per
heterogeneity split, the five datasets side by side as columns, DEO and |SPD| as
the two rows, and a single shared legend under the figure.  That is the
FedFACT-style multi-panel summary, meant for the camera-ready page rather than
for the appendix.

Usage::

    python analysis/plot_table.py                  # one file per (dataset, split)
    python analysis/plot_table.py --layout row     # five-up summary, one per split
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
RAW = os.path.join(REPO_ROOT, "reported_results", "hetero_raw.tsv")
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "figures")

DATASETS = ("COMPAS", "DRUG", "DUTCH", "ADULT", "BANK")
SPLITS = ("Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform")
SPLIT_LABEL = {"Dirichlet01": r"$\alpha{=}0.1$", "Dirichlet05": r"$\alpha{=}0.5$",
               "Dirichlet1": r"$\alpha=1$", "Uniform": "Uniform"}

ALGORITHMS = list(fig_variants.active(
    ["PDFFed", "PDFFed_DP", "FedAvg", "FedProx", "FedNova",
     "Scaffold", "FedProto", "FedRep", "FairFed", "FedFB",
     "FedFair", "FedRenyi", "mFairFL", "FL_FairBatch",
     "LoGoFair", "FederatedAverageWithPo"]))

_TAB20 = plt.cm.tab20.colors
COLORS = {"PDFFed": "tab:blue", "PDFFed_DP": "tab:orange"}
_rest = [a for a in ALGORITHMS if a not in COLORS]
for _i, _a in enumerate(_rest):
    COLORS[_a] = _TAB20[(_i + 4) % len(_TAB20)]

MARKERS = {"PDFFed": "*", "PDFFed_DP": "o", "FedAvg": "s", "FedProx": "D",
           "FedNova": "^", "Scaffold": "v", "FedProto": "P", "FedRep": "X",
           "FairFed": "<", "FedFB": ">", "FedFair": "8", "FedRenyi": "p",
           "mFairFL": "h", "FL_FairBatch": "H", "LoGoFair": "d",
           "FederatedAverageWithPo": "*"}
LABELS = {"PDFFed": "PDFFed", "PDFFed_DP": "PDFFed-DP", "FedAvg": "FedAvg",
          "FedProx": "FedProx", "FedNova": "FedNova", "Scaffold": "Scaffold",
          "FedProto": "FedProto", "FedRep": "FedRep", "FairFed": "FairFed",
          "FedFB": "FedFB", "FedFair": "FedFair", "FedRenyi": "FedRenyi",
          "mFairFL": "mFairFL", "FL_FairBatch": "FL-FairBatch",
          "LoGoFair": "LoGoFair", "FederatedAverageWithPo": "FedAvg+Po"}


def load():
    rows = []
    with open(RAW, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["algo_dir"] in ALGORITHMS:
                rows.append(r)
    return rows


def _f(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else float("nan")


def point(rows, ds, sp, algo):
    hit = [r for r in rows if r["dataset"] == ds and r["split"] == sp
           and r["algo_dir"] == algo]
    if not hit:
        return None
    r = hit[0]
    return (_f(r, "ACC"), _f(r, "DEO"), _f(r, "SPD"),
            _f(r, "ACC_sd"), _f(r, "DEO_sd"), _f(r, "SPD_sd"))


def draw_panel(ax, rows, ds, sp, mkey, lab, show_ylabel=True):
    """Scatter every algorithm of one (dataset, split) cell on one panel.

    Each point carries the +/- 1 sd over the three repeats as a thin error bar
    in the algorithm's own colour, drawn behind the marker.  Error bars are
    drawn only when both the point and its sd are finite.
    """
    for algo in ALGORITHMS:
        p = point(rows, ds, sp, algo)
        if p is None:
            continue
        acc, deo, spd, acc_sd, deo_sd, spd_sd = p
        val = abs(spd) if mkey == "SPD" else deo
        if (acc != acc) or (val != val):
            continue
        big = algo == "PDFFed"
        xerr = acc_sd if acc_sd == acc_sd else None
        sd = abs(spd_sd) if mkey == "SPD" else deo_sd
        yerr = sd if sd == sd else None
        if xerr is not None or yerr is not None:
            ax.errorbar(acc, val, xerr=xerr, yerr=yerr, fmt="none",
                        ecolor=COLORS[algo], alpha=0.55, elinewidth=1.0,
                        capsize=2.0, capthick=0.8, zorder=2)
        ax.scatter(acc, val, color=COLORS[algo], marker=MARKERS[algo],
                   s=240 if big else 90, label=LABELS[algo], zorder=3,
                   edgecolors="black" if big else "none",
                   linewidths=1.0 if big else 0.0)
    ax.set_ylabel(lab if show_ylabel else "", fontsize=10)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=8.5)


def main():
    layout = "panel"
    if "--layout" in sys.argv:
        layout = sys.argv[sys.argv.index("--layout") + 1]
        if layout not in ("panel", "row"):
            sys.exit("--layout must be 'panel' or 'row'")

    rows = load()

    if layout == "row":
        for sp in SPLITS:
            fig, axes = plt.subplots(2, len(DATASETS),
                                     figsize=(2.85 * len(DATASETS), 5.0),
                                     sharex=False)
            for j, ds in enumerate(DATASETS):
                for i, (mkey, lab) in enumerate((("DEO", "DEO"),
                                                 ("SPD", "|SPD|"))):
                    ax = axes[i][j]
                    draw_panel(ax, rows, ds, sp, mkey, lab,
                               show_ylabel=(j == 0))
                    if i == 0:
                        ax.set_title(ds, fontsize=10.5)
                    if i == 1:
                        ax.set_xlabel("ACC", fontsize=9)
            handles, labels = axes[0][0].get_legend_handles_labels()
            fig.legend(handles, labels, fontsize=7.5, frameon=False,
                       loc="lower center", bbox_to_anchor=(0.5, 0.004),
                       ncol=8, handletextpad=0.35, columnspacing=1.0)
            fig.suptitle("Main results, %s" % SPLIT_LABEL[sp], fontsize=11.5)
            fig.subplots_adjust(left=0.058, right=0.99, top=0.90, bottom=0.155,
                                wspace=0.16, hspace=0.22)
            stem = "table_row_%s%s" % (sp, fig_variants.suffix())
            fig.savefig(os.path.join(OUT_DIR, stem + ".png"), dpi=300)
            fig.savefig(os.path.join(OUT_DIR, stem + ".pdf"))
            plt.close(fig)
            print("wrote", os.path.join(OUT_DIR, stem + ".png"))
        return

    for ds in DATASETS:
        for sp in SPLITS:
            fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.8), sharex=True)
            for i, (mkey, lab) in enumerate((("DEO", "DEO"), ("SPD", "|SPD|"))):
                draw_panel(axes[i], rows, ds, sp, mkey, lab)
                if i == 0:
                    axes[i].set_title(f"{ds}  {SPLIT_LABEL[sp]}", fontsize=12)
            axes[1].set_xlabel("ACC", fontsize=11)
            handles, labels = axes[0].get_legend_handles_labels()
            axes[0].legend(handles, labels, fontsize=6.5, frameon=False, ncol=4,
                           loc="best", handletextpad=0.3, columnspacing=0.7,
                           markerscale=0.8)
            fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.09,
                                hspace=0.24)
            stem = "table_%s_%s%s" % (ds, sp, fig_variants.suffix())
            fig.savefig(os.path.join(OUT_DIR, stem + ".png"), dpi=300)
            fig.savefig(os.path.join(OUT_DIR, stem + ".pdf"))
            plt.close(fig)
            print("wrote", os.path.join(OUT_DIR, stem + ".png"))


if __name__ == "__main__":
    main()
