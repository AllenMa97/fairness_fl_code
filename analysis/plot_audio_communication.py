#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw the communication-effectiveness curves for the speech datasets.

Three corpora are covered, each extracted from the training logs with the same
"next global test" round attribution as the tabular curves:

    analysis/curves2/curve_CREMA-D_Speaker.csv          sensitive attribute = race
    analysis/curves2/curve_CREMA-D_gender_Speaker.csv   sensitive attribute = gender
    analysis/curves2/curve_RAVDESS_Speaker.csv          RAVDESS, speaker split

Each figure has ACC on top and the fairness gap below (DEO for
``--fairness eo``, |SPD| for ``--fairness dp``).

Same visual language as ``plot_communication.py``: PDFFed bold with a star,
PDFFed-DP dashed, baselines thin.  The three methods that collapse to the
majority class on CREMA-D (FL-FairBatch, FedProto, Scaffold) are drawn as-is,
per the agreed "report honestly, no annotation" decision.

Log coverage caveat: on CREMA-D only PDFFed and PDFFed-DP carry a full
200-point trajectory; the fourteen baselines log a global test at rounds
50/100/150 only, so their lines have three markers.  On RAVDESS the baselines
log every fifth round (39 points) and PDFFed every round (200 points).  That is
a property of the stored logs, not of the plot.

AudioMNIST is drawn separately, one figure per split (Dirichlet 0.1 / 0.5 /
1 and Uniform), with PDFFed and PDFFed-DP over 100 rounds.  Its baselines have
no per-round metric line at all -- they write a single global test at the end
of the run -- so they appear in the final-value figures (``plot_audio.py``)
rather than here.

Usage::

    python analysis/plot_audio_communication.py              # DEO
    python analysis/plot_audio_communication.py --fairness dp
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

ATTRS = (("CREMA-D_race", "curve_CREMA-D_Speaker.csv", "CREMA-D (race)"),
         ("CREMA-D_gender", "curve_CREMA-D_gender_Speaker.csv", "CREMA-D (gender)"),
         ("RAVDESS", "curve_RAVDESS_Speaker.csv", "RAVDESS (speaker split)"))

# AudioMNIST is a separate family: the split strategy varies (Dirichlet 0.1 /
# 0.5 / 1 and Uniform) and only PDFFed / PDFFed-DP wrote a per-round metric
# line, so each figure carries two lines over 100 rounds; the baselines stopped
# at the single global test and are reported in the final-value figures instead.
AM_ATTRS = (("AudioMNIST_Dirichlet01", "curve_AudioMNIST_Dirichlet01.csv",
             r"AudioMNIST ($\alpha{=}0.1$)"),
            ("AudioMNIST_Dirichlet05", "curve_AudioMNIST_Dirichlet05.csv",
             r"AudioMNIST ($\alpha{=}0.5$)"),
            ("AudioMNIST_Dirichlet1", "curve_AudioMNIST_Dirichlet1.csv",
             r"AudioMNIST ($\alpha=1$)"),
            ("AudioMNIST_Uniform", "curve_AudioMNIST_Uniform.csv",
             "AudioMNIST (Uniform)"))

METRIC_LABEL = {"eo": "DEO", "dp": "|SPD|"}

CURVES = fig_variants.active(
    ("PDFFed", "PDFFed_DP", "FedRenyi", "FedAvg", "FedProx", "FairFed",
     "mFairFL", "FL_FairBatch"))

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
LABELS = {"PDFFed": "PDFFed", "PDFFed_DP": "PDFFed-DP",
          "FedRenyi": "FedRenyi", "FedAvg": "FedAvg", "FedProx": "FedProx",
          "FairFed": "FairFed", "mFairFL": "mFairFL",
          "FL_FairBatch": "FL-FairBatch"}


def load(fname):
    rows = {}
    path = os.path.join(CURVE_DIR, fname)
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            algo = r["algo"]
            if algo not in CURVES:
                continue
            rows.setdefault(algo, {})[int(r["round"])] = {
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
    markevery = 10 if dense else 1
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, marker=marker,
            markersize=6 if algo != "PDFFed" else 7, markevery=markevery,
            markerfacecolor="white" if algo != "PDFFed" else color,
            markeredgecolor=color, markeredgewidth=1.1, label=LABELS[algo],
            zorder=3)


def main():
    metric = "eo"
    if "--fairness" in sys.argv:
        metric = sys.argv[sys.argv.index("--fairness") + 1]
        if metric not in ("eo", "dp"):
            sys.exit("--fairness must be 'eo' or 'dp'")

    ylabel = METRIC_LABEL[metric]
    for attr, fname, title, xmax, figw in (
            [(a, f, t, 205, 11.0) for a, f, t in ATTRS] +
            [(a, f, t, 105, 8.6) for a, f, t in AM_ATTRS]):
        rows = load(fname)
        if not rows:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(figw, 5.2), sharex=True)
        for i, (mkey, lab) in enumerate((("acc", "ACC"), (metric, ylabel))):
            ax = axes[i]
            for algo in CURVES:
                xs, ys = series(rows, algo, mkey)
                if len(xs) > 1:
                    draw_curve(ax, xs, ys, algo)
            ax.set_ylabel(lab, fontsize=11)
            ax.set_xlim(-5, xmax)
            ax.tick_params(labelsize=9)
            ax.grid(True, axis="y", alpha=0.25, lw=0.6)
            if i == 0:
                ax.set_title(title, fontsize=12)
        axes[1].set_xlabel("Communication round", fontsize=10)
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels, fontsize=7.5, frameon=False, ncol=8,
                       loc="best", handletextpad=0.35, columnspacing=0.8)
        fig.subplots_adjust(left=0.10 if figw < 10 else 0.07,
                            right=0.99, top=0.93, bottom=0.09, hspace=0.20)
        stem = "comm_%s_%s%s" % (attr, metric, fig_variants.suffix())
        fig.savefig(os.path.join(OUT_DIR, stem + ".png"), dpi=300)
        fig.savefig(os.path.join(OUT_DIR, stem + ".pdf"))
        plt.close(fig)
        print("wrote", os.path.join(OUT_DIR, stem + ".png"))


if __name__ == "__main__":
    main()
