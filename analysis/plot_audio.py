#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw the audio main-result comparison: ACC vs fairness metric.

Two corpora are covered.

CREMA-D
    Reads ``reported_results/aud_raw.tsv`` (flattened from the .31 server
    ``result_path/<DS>/Speaker/<algo>/AUDIOCNNRNN/20Clients/*.txt.json``) and
    plots, per sensitive attribute (CREMA-D = race, CREMA-D_gender = gender),
    how each algorithm trades off ACC against the fairness metric (DEO for
    ``--fairness eo``, |SPD| for ``--fairness dp``).  The audio protocol has no
    Dirichlet ladder (the split strategy is fixed to ``Speaker``) and a fixed
    client count of 20, so the comparison collapses to a single scatter per
    attribute.

AudioMNIST
    Reads ``reported_results/audio_mnist_raw.tsv``, the same flattening of
    ``result_path/AudioMNIST/<split>/<algo>/AUDIOCNNRNN/20Clients/*.txt.json``
    on the .30 server.  Here the split strategy does vary (Dirichlet 0.1 /
    0.5 / 1 and Uniform), so one figure is drawn per split.  Every point
    carries the +/- 1 sd over the three repeats as a thin error bar.

Each figure has two stacked panels (DEO on top, |SPD| below) sharing the ACC
axis.  PDFFed is a large blue star, PDFFed-DP an orange circle; the other
baselines are smaller coloured markers.

Usage::

    python analysis/plot_audio.py                  # one file per attribute/split, DEO
    python analysis/plot_audio.py --fairness dp    # |SPD| axis
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
RAW = os.path.join(REPO_ROOT, "reported_results", "aud_raw.tsv")
AM_RAW = os.path.join(REPO_ROOT, "reported_results", "audio_mnist_raw.tsv")
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "figures")

DATASETS = [("CREMA-D", "race"), ("CREMA-D_gender", "gender")]
AM_SPLITS = ("Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform")
AM_SPLIT_LABEL = {"Dirichlet01": r"$\alpha{=}0.1$", "Dirichlet05": r"$\alpha{=}0.5$",
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
            rows.append(r)
    return rows


def load_am():
    rows = []
    with open(AM_RAW, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rows.append(r)
    return rows


def _f(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else float("nan")


def series(rows, ds, algo, metric):
    hit = [r for r in rows if r["dataset"] == ds and r["algo"] == algo]
    if not hit:
        return None, None
    r = hit[0]
    acc = _f(r, "ACC")
    val = abs(_f(r, "SPD")) if metric == "dp" else _f(r, "DEO")
    return acc, val


def am_point(rows, split, algo, metric):
    """(ACC, value, ACC sd, value sd) for one AudioMNIST cell."""
    hit = [r for r in rows if r["split"] == split and r["algo"] == algo]
    if not hit:
        return None
    r = hit[0]
    acc, acc_sd = _f(r, "ACC"), _f(r, "ACC_sd")
    if metric == "dp":
        val, val_sd = abs(_f(r, "SPD")), abs(_f(r, "SPD_sd"))
    else:
        val, val_sd = _f(r, "DEO"), _f(r, "DEO_sd")
    return acc, val, acc_sd, val_sd


def draw_panel(ax, lab, points):
    """Scatter one panel from ``(algo, acc, val, acc_sd, val_sd)`` tuples.

    The +/- 1 sd is drawn as a thin error bar in the algorithm's own colour
    behind the marker; a missing sd simply drops the bar.
    """
    for algo, acc, val, acc_sd, val_sd in points:
        if (acc != acc) or (val != val):
            continue
        big = algo == "PDFFed"
        xerr = acc_sd if acc_sd == acc_sd else None
        yerr = val_sd if val_sd == val_sd else None
        if xerr is not None or yerr is not None:
            ax.errorbar(acc, val, xerr=xerr, yerr=yerr, fmt="none",
                        ecolor=COLORS[algo], alpha=0.55, elinewidth=1.0,
                        capsize=2.0, capthick=0.8, zorder=2)
        ax.scatter(acc, val, color=COLORS[algo], marker=MARKERS[algo],
                   s=220 if big else 80, label=LABELS[algo], zorder=3,
                   edgecolors="black" if big else "none",
                   linewidths=1.0 if big else 0.0)
    ax.set_ylabel(lab, fontsize=11)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=9)


def finish(fig, axes, title, stem):
    axes[1].set_xlabel("ACC", fontsize=11)
    axes[0].set_title(title, fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, fontsize=6.5, frameon=False, ncol=4,
                   loc="best", handletextpad=0.3, columnspacing=0.7,
                   markerscale=0.8)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.09,
                        hspace=0.24)
    fig.savefig(os.path.join(OUT_DIR, stem + ".png"), dpi=300)
    fig.savefig(os.path.join(OUT_DIR, stem + ".pdf"))
    plt.close(fig)
    print("wrote", os.path.join(OUT_DIR, stem + ".png"))


def main():
    metric = "eo"
    if "--fairness" in sys.argv:
        metric = sys.argv[sys.argv.index("--fairness") + 1]
        if metric not in ("eo", "dp"):
            sys.exit("--fairness must be 'eo' or 'dp'")

    rows = load()
    for ds, attr in DATASETS:
        fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.6), sharex=True)
        for i, mkey in enumerate(("eo", "dp")):
            points = []
            for algo in ALGORITHMS:
                acc, val = series(rows, ds, algo, mkey)
                if acc is None or val is None:
                    continue
                points.append((algo, acc, val, float("nan"), float("nan")))
            draw_panel(axes[i], "|SPD|" if mkey == "dp" else "DEO", points)
        finish(fig, axes, f"CREMA-D ({attr})  —  Speaker split",
               "audio_%s_%s%s" % (ds, metric, fig_variants.suffix()))

    if not os.path.exists(AM_RAW):
        print("skip AudioMNIST: %s not found" % AM_RAW)
        return
    am = load_am()
    for split in AM_SPLITS:
        fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.6), sharex=True)
        for i, mkey in enumerate(("eo", "dp")):
            points = []
            for algo in ALGORITHMS:
                p = am_point(am, split, algo, mkey)
                if p is None:
                    continue
                points.append((algo,) + p)
            draw_panel(axes[i], "|SPD|" if mkey == "dp" else "DEO", points)
        finish(fig, axes, "AudioMNIST  %s" % AM_SPLIT_LABEL[split],
               "audio_AudioMNIST_%s_%s%s" % (split, metric, fig_variants.suffix()))


if __name__ == "__main__":
    main()
