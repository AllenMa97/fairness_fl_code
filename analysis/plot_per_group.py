#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw the per-group accuracy curves (worst group and between-group gap).

Reads ``analysis/curves2/pg_<DS>_<SPLIT>.csv`` -- the global test log lines
carry ``Group ACC disparity (ACC_gap): v, per-group accuracy: {g: v, ...}``
right above the ``ACC: ... DEO: ...`` line, so the per-group values can be
attributed to the same communication round as the aggregated metrics.

Two stacked panels per (dataset, split):

  top     worst-group accuracy ``min_g acc_g`` (higher is better).  This is the
          quantity the fairness literature calls "minimum group accuracy"; the
          average accuracy can hide a collapsed minority group, this cannot.
  bottom  between-group gap ``max_g acc_g - min_g acc_g`` (lower is better).

PDFFed is bold with a star, PDFFed-DP dashed, the six baselines thin, exactly
as in ``plot_communication.py`` (both use ``curve_style``).

Note on density: PDFFed/PDFFed-DP log a test every round (500 points), the
baselines every ``test_eval_freq = 10`` rounds (49 points); the denser curves
are drawn as-is.

Usage::

    python analysis/plot_per_group.py
    python analysis/plot_per_group.py --no-dp
"""

from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import curve_style
import fig_variants

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURVE_DIR = os.path.join(REPO_ROOT, "analysis", "curves2")
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "figures")

DATASETS = ("COMPAS", "DRUG", "DUTCH", "ADULT", "BANK")
SPLITS = ("Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform")
SPLIT_LABEL = {"Dirichlet01": r"$\alpha{=}0.1$", "Dirichlet05": r"$\alpha{=}0.5$",
               "Dirichlet1": r"$\alpha{=}1$", "Uniform": "Uniform"}


def load(ds, sp):
    """-> {algo: {round: (worst_group_acc, gap)}} for one (dataset, split)."""
    path = os.path.join(CURVE_DIR, "pg_%s_%s.csv" % (ds, sp))
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            algo = r["algo"]
            if algo not in curve_style.CURVES:
                continue
            g0, g1 = _f(r, "g0"), _f(r, "g1")
            if g0 != g0 or g1 != g1:
                continue
            out.setdefault(algo, {})[int(r["round"])] = (min(g0, g1), abs(g1 - g0))
    return out


def _f(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else float("nan")


def series(rows, algo, metric):
    xs, ys = [], []
    for rnd in sorted(rows.get(algo, {})):
        worst, gap = rows[algo][rnd]
        xs.append(rnd)
        ys.append(worst if metric == "worst" else gap)
    return xs, ys


def main():
    for ds in DATASETS:
        for sp in SPLITS:
            rows = load(ds, sp)
            if not rows:
                continue
            fig, axes = plt.subplots(2, 1, figsize=(11.0, 5.2), sharex=True)
            for i, (mkey, lab) in enumerate((
                    ("worst", "Worst-group accuracy"),
                    ("gap", "Group accuracy gap"))):
                ax = axes[i]
                for algo in curve_style.CURVES:
                    xs, ys = series(rows, algo, mkey)
                    if len(xs) > 1:
                        curve_style.draw_curve(ax, xs, ys, algo)
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
            fig.subplots_adjust(left=0.075, right=0.99, top=0.93, bottom=0.09,
                                hspace=0.20)
            stem = "pergroup_%s_%s%s" % (ds, sp, fig_variants.suffix())
            fig.savefig(os.path.join(OUT_DIR, stem + ".png"), dpi=300)
            fig.savefig(os.path.join(OUT_DIR, stem + ".pdf"))
            plt.close(fig)
            print("wrote", os.path.join(OUT_DIR, stem + ".png"))


if __name__ == "__main__":
    main()
