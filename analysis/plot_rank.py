#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw the cross-dataset rank summary.

Reads ``reported_results/modality_raw.tsv`` -- the merged per-cell flattening of
both servers -- and ranks the sixteen algorithms inside every cell: by ACC
(higher is better) and by DEO (lower is better).  A cell is any
(dataset, split) pair at the protocol client count (20) in which at least
``MIN_ALGOS`` of the compared algorithms were actually run; sparse cells are
dropped so that no algorithm gains rank from an empty field.

The figure has two panels: mean ACC rank and mean HM rank over the same datasets,
ordered by HM rank -- HM is the quantity the summary is about, and ACC is shown
next to it so a reader can see where the two disagree.  Lower is better in both,
and both files draw the best algorithm on top.  Four single-panel files
(``*_hm``, ``*_acc``, ``*_deo``, ``*_spd``) draw one metric each, sorted by that
metric.  DEO and |SPD| are ranked the same way as in the summary: absolute value,
lower is better.

The title reports the number of datasets in the pool and names them.  An
algorithm's mean rank is taken over the cells it actually covers, so coverage is
not identical across algorithms; the console table prints each algorithm's cell
count next to its mean, which is where that check belongs rather than on the
figure.

Ties inside a cell receive the average rank, so two algorithms sitting at the
same floor value split the credit instead of one being arbitrarily favoured.

``--drop-datasets`` re-runs the same ranking on a subset of corpora and writes
``rank_summary_dropN.*`` instead, so the full-pool figure is never overwritten::

    python analysis/plot_rank.py --drop-datasets ADULT,CREMA-D,bios

Usage::

    python analysis/plot_rank.py
"""

from __future__ import annotations

import csv
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fig_variants

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO_ROOT, "reported_results", "modality_raw.tsv")
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "figures")

CLIENTS = "20"
MIN_ALGOS = 10
SKIP_SPLITS = {"CREMA-D": {"Dirichlet1", "Uniform", "Dirichlet05", "Dirichlet01"}}

ALGORITHMS = list(fig_variants.active(
    ["PDFFed", "PDFFed_DP", "FedAvg", "FedProx", "FedNova", "Scaffold",
     "FedProto", "FedRep", "FairFed", "FedFB", "FedFair", "FedRenyi",
     "mFairFL", "FL_FairBatch", "LoGoFair", "FederatedAverageWithPo"]))
LABELS = {"PDFFed": "PDFFed", "PDFFed_DP": "PDFFed-DP",
          "FederatedAverageWithPo": "FedAvg+Po", "FL_FairBatch": "FL-FairBatch"}
# CREMA-D appears twice in the results tree, once per sensitive attribute; the
# gender variant is the same corpus and is reported under one name.
DISPLAY = {"CREMA-D_gender": "CREMA-D"}


def _f(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def load_cells(drop=frozenset(), min_algos=MIN_ALGOS):
    cells = {}
    with open(RAW, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["clients"] != CLIENTS:
                continue
            if row["algo"] not in ALGORITHMS:
                continue
            if row["dataset"] in drop:
                continue
            if row["split"] in SKIP_SPLITS.get(row["dataset"], set()):
                continue
            key = (row["dataset"], row["split"])
            cells.setdefault(key, {})[row["algo"]] = row
    return {k: v for k, v in cells.items() if len(v) >= min_algos}


def average_ranks(values, higher_is_better):
    """Rank mapping with average ranks for ties."""
    ordered = sorted(values.items(), key=lambda kv: kv[1],
                     reverse=higher_is_better)
    ranks = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        span = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = span
        i = j + 1
    return ranks


def collect(cells):
    acc_ranks, deo_ranks, hm_ranks, spd_ranks = {}, {}, {}, {}
    for rows in cells.values():
        acc = {a: _f(r, "ACC") for a, r in rows.items()}
        deo = {a: abs(_f(r, "DEO")) for a, r in rows.items()}
        hm = {a: _f(r, "HM") for a, r in rows.items()}
        hm = {a: v for a, v in hm.items() if v == v}
        spd = {a: abs(_f(r, "SPD")) for a, r in rows.items()}
        spd = {a: v for a, v in spd.items() if v == v}
        for algo, rank in average_ranks(acc, True).items():
            acc_ranks.setdefault(algo, []).append(rank)
        for algo, rank in average_ranks(deo, False).items():
            deo_ranks.setdefault(algo, []).append(rank)
        for algo, rank in average_ranks(hm, True).items():
            hm_ranks.setdefault(algo, []).append(rank)
        for algo, rank in average_ranks(spd, False).items():
            spd_ranks.setdefault(algo, []).append(rank)
    return acc_ranks, deo_ranks, hm_ranks, spd_ranks


def title_size(text, width_in, base):
    """Shrink the title font until the longest title line fits the figure width.

    Approximation: a DejaVu Sans glyph averages 0.0066 in per point, which is
    accurate enough to keep the text off the canvas edge without measuring it.
    """
    longest = max(len(line) for line in text.split("\n"))
    return max(6.5, min(base, width_in / (0.0066 * max(longest, 1))))


def draw_bars(ax, table, order, xlabel, show_dots=True):
    """Horizontal bars with the standard error of the mean and the raw ranks.

    The whisker is the standard error (sd / sqrt(n)), not the standard
    deviation: the spread of ranks across corpora is large by construction and
    a +-1 sd whisker would read as if the mean itself were that uncertain.  The
    faint dots are the individual per-cell ranks, which is the honest version of
    the same information -- e.g. PDFFed sits at rank 1 on several tabular cells
    and at 10+ on DRUG, a bimodality no summary statistic shows.
    """
    means, sems, ranks = [], [], []
    for algo in order:
        vals = list(table.get(algo, []))
        ranks.append(vals)
        means.append(sum(vals) / len(vals) if vals else float("nan"))
        sems.append(statistics.stdev(vals) / len(vals) ** 0.5
                    if len(vals) > 1 else 0.0)
    colors = ["tab:blue" if a == "PDFFed" else
              "tab:orange" if a == "PDFFed_DP" else "#9e9e9e" for a in order]
    ax.barh(range(len(order)), means, color=colors, height=0.7,
            xerr=sems, error_kw={"ecolor": "#555555", "elinewidth": 1.0,
                                 "capsize": 2.5, "capthick": 1.0}, zorder=2)
    if show_dots:
        for i, vals in enumerate(ranks):
            dx = [((j * 37) % 11) / 10.0 - 0.5 for j in range(len(vals))]
            ax.scatter(vals, [i + 0.30 * d for d in dx], s=6,
                       color=colors[i], alpha=0.45, linewidths=0, zorder=3)
    for i, (value, sem) in enumerate(zip(means, sems)):
        if value == value:
            ax.text(value + sem + 0.25, i, "%.2f" % value, va="center",
                    fontsize=7.5, color="#333333")
    ax.set_xlim(0.6, 17.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABELS.get(a, a) for a in order], fontsize=9)
    # Row 0 sits at the top: a shared y axis is inverted once here, so an
    # ``invert_yaxis`` call per panel would cancel itself out on the second one.
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.grid(True, axis="x", alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=9)


def main():
    drop = set()
    if "--drop-datasets" in sys.argv:
        raw = sys.argv[sys.argv.index("--drop-datasets") + 1]
        drop = {x.strip() for x in raw.split(",") if x.strip()}
    full_grid = "--full-grid" in sys.argv
    cells = load_cells(drop, len(ALGORITHMS) if full_grid else MIN_ALGOS)
    acc_ranks, deo_ranks, hm_ranks, spd_ranks = collect(cells)
    order = sorted(hm_ranks, key=lambda a: (sum(hm_ranks[a]) / len(hm_ranks[a]),
                                            sum(acc_ranks[a]) / len(acc_ranks[a])))
    corpora = sorted({DISPLAY.get(key[0], key[0]) for key in cells})
    n_datasets = len(corpora)
    headline = ("Cross-dataset rank over %d datasets\n(%s)"
                % (n_datasets, ", ".join(corpora)))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4), sharey=True)
    draw_bars(axes[0], hm_ranks, order, "HM mean rank  (lower is better)")
    draw_bars(axes[1], acc_ranks, order,
              "ACC mean rank  (lower is better)", show_dots=False)
    fig.suptitle(headline, fontsize=title_size(headline, 10.0, 10.5))
    fig.subplots_adjust(left=0.13, right=0.995, top=0.82, bottom=0.16,
                        wspace=0.06)

    stem = "rank_summary" + ("_%dds" % n_datasets if drop else "") \
        + ("_full" if full_grid else "") + fig_variants.suffix()

    def save(figure, name):
        figure.savefig(os.path.join(OUT_DIR, name + ".png"), dpi=300,
                       bbox_inches="tight", pad_inches=0.03)
        figure.savefig(os.path.join(OUT_DIR, name + ".pdf"),
                       bbox_inches="tight", pad_inches=0.03)
        plt.close(figure)
        print("wrote", os.path.join(OUT_DIR, name + ".png"))

    save(fig, stem)

    for key, table, label in (("hm", hm_ranks, "HM"), ("acc", acc_ranks, "ACC"),
                              ("deo", deo_ranks, "DEO"),
                              ("spd", spd_ranks, "|SPD|")):
        single_order = sorted(table, key=lambda a: sum(table[a]) / len(table[a]))
        fig_s, ax_s = plt.subplots(figsize=(7.2, 3.1))
        draw_bars(ax_s, table, single_order,
                  "%s mean rank  (lower is better)" % label)
        fig_s.suptitle(headline, fontsize=title_size(headline, 7.2, 9),
                       y=0.985)
        fig_s.subplots_adjust(left=0.19, right=0.99, top=0.80, bottom=0.18)
        save(fig_s, stem + "_" + key)
    print(headline.replace("\n", "  |  "))
    print("datasets:", ", ".join("%s(%d)" % (name, sum(1 for k in cells if k[0] == name))
                                 for name in sorted({k[0] for k in cells})))
    print("%-22s %5s %8s %8s %8s %8s %8s"
          % ("algorithm", "cells", "ACC", "DEO", "HM", "HM_sem", "pooled"))
    for algo in order:
        hm = hm_ranks.get(algo, [])
        acc = sum(acc_ranks[algo]) / len(acc_ranks[algo])
        deo = sum(deo_ranks[algo]) / len(deo_ranks[algo])
        hmv = sum(hm) / len(hm) if hm else float("nan")
        hsem = statistics.stdev(hm) / len(hm) ** 0.5 if len(hm) > 1 else 0.0
        print("%-22s %5d %8.2f %8.2f %8.2f %8.2f %8.2f"
              % (LABELS.get(algo, algo), len(acc_ranks[algo]), acc, deo, hmv,
                 hsem, (acc + deo + hmv) / 3.0))


if __name__ == "__main__":
    main()
