#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw the FedRenyi lambda sensitivity, one figure per metric.

Reads ``reported_results/lambda_sensitivity/lambda_sensitivity.csv`` and plots
how FedRenyi's fairness strength trades off against accuracy and against the
equality-of-opportunity gap as its group re-weighting coefficient lambda moves
over {0.25, 0.5, 1, 2, 4}.

Only three cells were swept (COMPAS alpha=1, DRUG alpha=1, DRUG Uniform), so
each figure carries one line per available cell, markers with +/- 1 sd over the
three repeats, and a dotted horizontal line for that cell's FedRenyi baseline
(lambda as configured in the main experiments).  It is a sanity-check figure
for the appendix, not a main result: on DRUG the curve is flat, which is itself
the finding -- FedRenyi's lambda barely moves the operating point there.

ACC and DEO live on different scales (0.55-0.62 vs 0.04-0.40), so a shared
figure forces them onto one y-axis and squeezes the lambda ticks; each metric
therefore gets its own single-panel figure with the full width.

Usage::

    python analysis/plot_lambda.py
"""

from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "reported_results", "lambda_sensitivity",
                   "lambda_sensitivity.csv")
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "figures")

CURVES = (("COMPAS", "Dirichlet1", "tab:blue", r"COMPAS $\alpha{=}1$"),
          ("DRUG", "Dirichlet1", "tab:red", r"DRUG $\alpha{=}1$"),
          ("DRUG", "Uniform", "tab:green", "DRUG Uniform"))
PANELS = (("ACC_mean", "ACC_std", "ACC", "lambda_sensitivity_ACC"),
          ("DEO_mean", "DEO_std", "DEO", "lambda_sensitivity_DEO"))


def load():
    sweep, base = [], {}
    with open(SRC, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["algorithm"] != "FedRenyi":
                continue
            cell = (r["dataset"], r["split"])
            if r["variant"] == "baseline":
                base[cell] = r
            elif r["variant"].startswith("FedRenyi_lam"):
                sweep.append((cell, float(r["sweep_value"]), r))
    return sweep, base


def _f(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else float("nan")


def draw(sweep, base, mkey, skey, mlab, stem):
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for ds, sp, color, lab in CURVES:
        cell = (ds, sp)
        pts = sorted([(lam, r) for (c, lam, r) in sweep if c == cell])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [_f(p[1], mkey) for p in pts]
        es = [_f(p[1], skey) for p in pts]
        ax.errorbar(xs, ys, yerr=es, color=color, marker="o", ms=5,
                    lw=1.8, capsize=3, label=lab, zorder=3)
        b = base.get(cell)
        if b is not None:
            ax.axhline(_f(b, mkey), color=color, ls=":", lw=1.1, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    ax.set_xlabel(r"FedRenyi $\lambda$", fontsize=10)
    ax.set_ylabel(mlab, fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.set_title(r"FedRenyi $\lambda$ sweep -- %s (dotted: main-experiment $\lambda$)"
                 % mlab, fontsize=11)
    ax.legend(fontsize=8.5, frameon=False, loc="best")
    fig.subplots_adjust(left=0.10, right=0.985, top=0.90, bottom=0.13)
    for ext in (".png", ".pdf"):
        fig.savefig(os.path.join(OUT_DIR, stem + ext),
                    dpi=300 if ext == ".png" else None)
    plt.close(fig)
    print("wrote", os.path.join(OUT_DIR, stem + ".png"))


def main():
    sweep, base = load()
    for mkey, skey, mlab, stem in PANELS:
        draw(sweep, base, mkey, skey, mlab, stem)


if __name__ == "__main__":
    main()
