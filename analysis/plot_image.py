#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw the image-modality main-result comparison: ACC vs fairness metric.

Reads ``reported_results/modality_raw.tsv`` -- the merged, per-cell flattening
of both servers' ``result_path/<dataset>/<split>/<algo>/<model>/20Clients/``
jsons, produced by ``analysis/collect_result_jsons.py`` -- and draws one figure
per (dataset, split) with 20 clients: DEO on the upper panel, |SPD| on the
lower panel, ACC shared on the x axis.  Every marker carries the +/- 1 sd over
the repeats as a thin error bar.

Only cells with at least ``MIN_ALGOS`` of the sixteen compared algorithms are
drawn; the rest are too sparse to rank.  PDFFed is a large blue star,
PDFFed-DP an orange circle, baselines thin coloured markers -- the same visual
language as ``plot_table.py`` and ``plot_audio.py``.

Dataset caveat: these corpora are the weakest in the suite.  Against the
majority-class floor the audit marks UTKFace (floor 0.538) and FairFace
(0.529) as COLLAPSE for every cell, and LFWA+ (0.613) as COLLAPSE with its best
PDFFed cell at 0.606.  On CelebA the K=20 cells drawn here also sit at chance
(PDFFed 0.50-0.56) while FedRenyi / FedRep reach 0.69-0.73, so the near-zero
DEO of the PDFFed runs in these panels is the collapse floor, not a fairness
win.  CelebA's single valid cell is K=30 / Dirichlet1 (PDFFed-DP 0.691 against
the 0.504 floor), which is why it is rendered separately with
``--clients 30 --only CelebA Dirichlet1``.

Usage::

    python analysis/plot_image.py                  # DEO axis, 20 clients
    python analysis/plot_image.py --fairness dp    # |SPD| axis
    python analysis/plot_image.py --clients 30 --only CelebA Dirichlet1
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
RAW = os.path.join(REPO_ROOT, "reported_results", "modality_raw.tsv")
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "figures")

DATASETS = ("CelebA", "LFWA+", "FairFace", "UTKFace")
SPLITS = ("Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform")
SPLIT_LABEL = {"Dirichlet01": r"$\alpha{=}0.1$", "Dirichlet05": r"$\alpha{=}0.5$",
               "Dirichlet1": r"$\alpha=1$", "Uniform": "Uniform"}
CLIENTS = "20"
MIN_ALGOS = 10

ALGORITHMS = list(fig_variants.active(
    ["PDFFed", "PDFFed_DP", "FedAvg", "FedProx", "FedNova", "Scaffold",
     "FedProto", "FedRep", "FairFed", "FedFB", "FedFair", "FedRenyi",
     "mFairFL", "FL_FairBatch", "LoGoFair", "FederatedAverageWithPo"]))

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


def load(clients):
    cells = {}
    with open(RAW, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["clients"] != clients:
                continue
            if row["dataset"] not in DATASETS:
                continue
            if row["algo"] not in ALGORITHMS:
                continue
            key = (row["dataset"], row["split"])
            cells.setdefault(key, {})[row["algo"]] = row
    return cells


def _f(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def draw_panel(ax, label, rows, metric):
    for algo in ALGORITHMS:
        row = rows.get(algo)
        if row is None:
            continue
        acc, acc_sd = _f(row, "ACC"), _f(row, "ACC_sd")
        if metric == "dp":
            val, val_sd = abs(_f(row, "SPD")), abs(_f(row, "SPD_sd"))
        else:
            val, val_sd = _f(row, "DEO"), _f(row, "DEO_sd")
        if acc != acc or val != val:
            continue
        big = algo == "PDFFed"
        ax.errorbar(acc, val, xerr=acc_sd if acc_sd == acc_sd else None,
                    yerr=val_sd if val_sd == val_sd else None, fmt="none",
                    ecolor=COLORS[algo], alpha=0.55, elinewidth=1.0,
                    capsize=2.0, capthick=0.8, zorder=2)
        ax.scatter(acc, val, color=COLORS[algo], marker=MARKERS[algo],
                   s=220 if big else 80, label=LABELS[algo], zorder=3,
                   edgecolors="black" if big else "none",
                   linewidths=1.0 if big else 0.0)
    ax.set_ylabel(label, fontsize=11)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=9)


def main():
    metric = "eo"
    clients = CLIENTS
    only = None
    if "--fairness" in sys.argv:
        metric = sys.argv[sys.argv.index("--fairness") + 1]
        if metric not in ("eo", "dp"):
            sys.exit("--fairness must be 'eo' or 'dp'")
    if "--clients" in sys.argv:
        clients = sys.argv[sys.argv.index("--clients") + 1]
    if "--only" in sys.argv:
        i = sys.argv.index("--only")
        only = (sys.argv[i + 1], sys.argv[i + 2])
    ylabel = "|SPD|" if metric == "dp" else "DEO"
    ktag = "" if clients == CLIENTS else "_k" + clients

    cells = load(clients)
    for dataset in DATASETS:
        for split in SPLITS:
            if only is not None and (dataset, split) != only:
                continue
            rows = cells.get((dataset, split))
            floor = MIN_ALGOS if only is None else 2
            if not rows or len(rows) < floor:
                continue
            fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.6), sharex=True)
            for i, (mkey, lab) in enumerate((("eo", "DEO"), ("dp", "|SPD|"))):
                draw_panel(axes[i], lab, rows, mkey)
            axes[1].set_xlabel("ACC", fontsize=11)
            axes[0].set_title("%s  %s  (%s clients)"
                              % (dataset, SPLIT_LABEL[split], clients), fontsize=12)
            handles, labels = axes[0].get_legend_handles_labels()
            axes[0].legend(handles, labels, fontsize=6.5, frameon=False, ncol=4,
                           loc="best", handletextpad=0.3, columnspacing=0.7,
                           markerscale=0.8)
            fig.subplots_adjust(left=0.17, right=0.97, top=0.93, bottom=0.09,
                                hspace=0.24)
            stem = "image_%s_%s_%s%s%s" % (dataset.replace("+", "p"), split,
                                            metric, ktag, fig_variants.suffix())
            fig.savefig(os.path.join(OUT_DIR, stem + ".png"), dpi=300)
            fig.savefig(os.path.join(OUT_DIR, stem + ".pdf"))
            plt.close(fig)
            print("wrote", os.path.join(OUT_DIR, stem + ".png"))


if __name__ == "__main__":
    main()
