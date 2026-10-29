#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw the loss-term ablation of PDFFed.

Reads ``analysis/ablation/ablation_results.csv`` (4 dataset x split cells,
5 variants each, 3 repeats per cell) and compares the full objective against
four single-term removals:

    PDFFed           full objective
    PDFFed_noPA      L_PA   (prototype alignment) removed
    PDFFed_noCONF    L_Conf (confidence calibration) removed
    PDFFed_noL2L     L2L    (local-to-local contrastive) removed
    PDFFed_noEO      global EO calibration removed

Three panels -- ACC, DEO, HM -- because the ablation's claim is a trade-off:
L_PA is the only term whose removal degrades *both* accuracy and fairness at
once, which only becomes visible when the two axes are shown next to HM.

Output: ``analysis/figures/ablation_bars.png`` (+ pdf).

Usage::

    python analysis/plot_ablation.py
"""

from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "analysis", "ablation", "ablation_results.csv")
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "figures")

CELLS = (("COMPAS", "Dirichlet1"), ("COMPAS", "Uniform"),
         ("DRUG", "Dirichlet1"), ("DRUG", "Uniform"))
CELL_LABEL = {("COMPAS", "Dirichlet1"): "COMPAS " + r"$\alpha{=}1$",
              ("COMPAS", "Uniform"): "COMPAS Uniform",
              ("DRUG", "Dirichlet1"): "DRUG " + r"$\alpha{=}1$",
              ("DRUG", "Uniform"): "DRUG Uniform"}
VARIANTS = (("PDFFed", "full", "#1f77b4"),
            ("PDFFed_noPA", "w/o $L_{PA}$", "#ff7f0e"),
            ("PDFFed_noCONF", "w/o $L_{Conf}$", "#2ca02c"),
            ("PDFFed_noL2L", "w/o $L_{2L}$", "#d62728"),
            ("PDFFed_noEO", "w/o EO calib.", "#9467bd"))
PANELS = (("ACC", "ACC"), ("DEO", "DEO"), ("HM", "HM"))


def load():
    rows = {}
    with open(SRC, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            rows[(r["dataset"], r["split"], r["variant"])] = r
    return rows


def _f(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else float("nan")


def main():
    rows = load()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(13.0, 3.9), sharex=True)
    width = 0.8 / len(VARIANTS)
    for ax, (col, lab) in zip(axes, PANELS):
        for j, (var, vlab, color) in enumerate(VARIANTS):
            vals, pos = [], []
            for k, cell in enumerate(CELLS):
                r = rows.get(cell + (var,))
                if r is None:
                    continue
                vals.append(_f(r, col))
                pos.append(k + (j - (len(VARIANTS) - 1) / 2) * width)
            ax.bar(pos, vals, width=width * 0.9, color=color, label=vlab,
                   zorder=2)
        ax.set_ylabel(lab, fontsize=11)
        ax.set_xticks(range(len(CELLS)))
        ax.set_xticklabels([CELL_LABEL[c] for c in CELLS], fontsize=9)
        ax.tick_params(labelsize=9)
        ax.grid(True, axis="y", alpha=0.25, lw=0.6)
    axes[0].set_title("PDFFed loss-term ablation", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8.5, frameon=False, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, 0.004),
               handletextpad=0.4, columnspacing=1.4)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.90, bottom=0.20,
                        wspace=0.20)
    for ext in (".png", ".pdf"):
        fig.savefig(os.path.join(OUT_DIR, "ablation_bars" + ext),
                    dpi=300 if ext == ".png" else None)
    plt.close(fig)
    print("wrote", os.path.join(OUT_DIR, "ablation_bars.png"))


if __name__ == "__main__":
    main()
