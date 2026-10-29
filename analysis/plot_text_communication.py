#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw the bios (text) communication curves, one figure per split.

Reads ``analysis/curves2/curve_bios_<split>.csv``, extracted from the
BERT-classifier runs on the two machines.

Protocol caveat: every bios run in the archive is a **ten-round** run
(``communication_round_I = 10``), not the 500 rounds used for the tabular
datasets, so the horizontal axis is 1..10 and each line carries at most ten
points.  The last round is occasionally missing from the log copy (the
baseline writes it only after the final global test, which some runs never
reached), which is why a few lines stop at round 9.

Result caveat: per the collapse audit the text modality produces no usable
number -- PDFFed's best cell is 0.485 against a per-split majority-class floor
and is marked COLLAPSE.  The figure is drawn as-is so the appendix can show
the trajectory behind that verdict; no floor line is drawn because the floor
differs between the four splits.

Coverage caveat: the Uniform split has no PDFFed run in the log archive (the
.30 server only ever trained the fourteen baselines there, and the .31 logs
were removed), so that split is skipped rather than drawn without the method
under study.

Usage::

    python analysis/plot_text_communication.py              # DEO
    python analysis/plot_text_communication.py --fairness dp
"""

from __future__ import annotations

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import curve_style
import fig_variants

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURVE_DIR = os.path.join(REPO_ROOT, "analysis", "curves2")
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "figures")

SPLITS = ("Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform")
SPLIT_LABEL = {"Dirichlet01": r"$\alpha{=}0.1$", "Dirichlet05": r"$\alpha{=}0.5$",
               "Dirichlet1": r"$\alpha=1$", "Uniform": "Uniform"}
METRIC_LABEL = {"eo": "DEO", "dp": "|SPD|"}


def load(split):
    path = os.path.join(CURVE_DIR, "curve_bios_%s.csv" % split)
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            algo = r["algo"]
            if algo not in curve_style.CURVES:
                continue
            out.setdefault(algo, {})[int(float(r["round"]))] = {
                "ACC": _f(r, "ACC"), "DEO": _f(r, "DEO"), "SPD": _f(r, "SPD")}
    return out


def _f(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else float("nan")


def main():
    metric = "eo"
    if "--fairness" in sys.argv:
        metric = sys.argv[sys.argv.index("--fairness") + 1]
        if metric not in ("eo", "dp"):
            sys.exit("--fairness must be 'eo' or 'dp'")
    ylabel = METRIC_LABEL[metric]

    for split in SPLITS:
        rows = load(split)
        if not rows:
            continue
        if "PDFFed" not in rows:
            print("skip %s: no PDFFed run in the archive" % split)
            continue
        fig, axes = plt.subplots(2, 1, figsize=(9.0, 4.8), sharex=True)
        fit_key = "DEO" if metric == "eo" else "SPD"
        for i, (key, lab) in enumerate((("ACC", "ACC"), (fit_key, ylabel))):
            ax = axes[i]
            for algo in curve_style.CURVES:
                d = rows.get(algo, {})
                if len(d) < 2:
                    continue
                ks = sorted(d)
                ys = [abs(d[k][key]) if key == "SPD" else d[k][key] for k in ks]
                curve_style.draw_curve(ax, ks, ys, algo, markevery_dense=1)
            ax.set_xlim(0.4, 10.6)
            ax.set_ylabel(lab, fontsize=11)
            ax.tick_params(labelsize=9)
            ax.grid(True, axis="y", alpha=0.25, lw=0.6)
            if i == 0:
                ax.set_title("bios (text)  %s" % SPLIT_LABEL[split], fontsize=12)
        axes[1].set_xlabel("Communication round", fontsize=10)
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels, fontsize=7.5, frameon=False, ncol=8,
                       loc="best", handletextpad=0.35, columnspacing=0.8)
        fig.subplots_adjust(left=0.085, right=0.985, top=0.92, bottom=0.11,
                            hspace=0.22)
        stem = "comm_bios_%s_%s%s" % (split, metric, fig_variants.suffix())
        fig.savefig(os.path.join(OUT_DIR, stem + ".png"), dpi=300)
        fig.savefig(os.path.join(OUT_DIR, stem + ".pdf"))
        plt.close(fig)
        print("wrote", os.path.join(OUT_DIR, stem + ".png"))


if __name__ == "__main__":
    main()
