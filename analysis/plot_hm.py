#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw the HM (utility-fairness harmonic mean) trajectory per setting.

Reads ``analysis/curves2/pg_<DS>_<SPLIT>.csv`` -- the per-group extraction also
carries the ``HM`` column, and those files have the widest algorithm coverage
(the two machines' logs merged), so PDFFed is present in every cell.  It plots
HM against the communication round for PDFFed and the seven trajectory
baselines.

HM = 2 * ACC * (1 - DEO) / (ACC + 1 - DEO) is the single scalar the paper uses
to rank methods: it collapses the accuracy / fairness trade-off into one number,
so a method cannot win by sacrificing one side.  Reading it per round shows
whether a method is fair-and-accurate from the start or only at convergence.

Usage::

    python analysis/plot_hm.py
    python analysis/plot_hm.py --no-dp
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
    """-> {algo: {round: HM}} for one (dataset, split).

    FL-FairBatch does not log HM, so for that run it is recomputed from the
    logged ACC and DEO with the paper's definition
    ``2*ACC*(1-DEO) / (ACC + 1 - DEO)``.
    """
    path = os.path.join(CURVE_DIR, "pg_%s_%s.csv" % (ds, sp))
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            algo = r["algo"]
            if algo not in curve_style.CURVES:
                continue
            v = r.get("HM", "")
            if v in ("", None):
                acc, deo = r.get("ACC", ""), r.get("DEO", "")
                if acc in ("", None) or deo in ("", None):
                    continue
                acc, deo = float(acc), float(deo)
                if acc + 1.0 - deo <= 0:
                    continue
                v = 2.0 * acc * (1.0 - deo) / (acc + 1.0 - deo)
            out.setdefault(algo, {})[int(r["round"])] = float(v)
    return out


def main():
    for ds in DATASETS:
        for sp in SPLITS:
            rows = load(ds, sp)
            if not rows:
                continue
            fig, ax = plt.subplots(figsize=(11.0, 3.6))
            for algo in curve_style.CURVES:
                d = rows.get(algo, {})
                if len(d) > 1:
                    ks = sorted(d)
                    curve_style.draw_curve(ax, ks, [d[k] for k in ks], algo)
            ax.set_xlim(-10, 510)
            ax.set_xlabel("Communication round", fontsize=10)
            ax.set_ylabel("HM", fontsize=11)
            ax.set_title(f"{ds}  {SPLIT_LABEL[sp]}", fontsize=12)
            ax.tick_params(labelsize=9)
            ax.grid(True, axis="y", alpha=0.25, lw=0.6)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, fontsize=7.5, frameon=False, ncol=8,
                      loc="best", handletextpad=0.35, columnspacing=0.8)
            fig.subplots_adjust(left=0.075, right=0.99, top=0.91, bottom=0.14)
            stem = "hm_%s_%s%s" % (ds, sp, fig_variants.suffix())
            fig.savefig(os.path.join(OUT_DIR, stem + ".png"), dpi=300)
            fig.savefig(os.path.join(OUT_DIR, stem + ".pdf"))
            plt.close(fig)
            print("wrote", os.path.join(OUT_DIR, stem + ".png"))


if __name__ == "__main__":
    main()
