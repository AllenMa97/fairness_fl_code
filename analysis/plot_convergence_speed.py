#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw the convergence-speed comparison (rounds to reach a fixed target).

For every (dataset, split) and every algorithm the per-round trajectory in
``analysis/curves2/pg_<DS>_<SPLIT>.csv`` is reduced to one number: the first
communication round at which the smoothed curve reaches

    ACC  >= 95% of its own final value          (higher is better)
    DEO  <= 105% of its own final value + 0.005 (lower is better)

The target is relative to each method's own endpoint, so the panel answers
"how many rounds until this method is done?" rather than "who ends highest".
The ``+0.005`` slack keeps the DEO target non-degenerate when a method ends at
exactly zero disparity.

One figure per dataset: two panels (ACC target, DEO target) with the four
heterogeneity splits on the x-axis and one bar per algorithm.  All trajectories
are first resampled on the 10-round grid that every method shares (PDFFed logs
each round, the baselines every tenth), so the reported rounds are directly
comparable across algorithms.

Usage::

    python analysis/plot_convergence_speed.py
    python analysis/plot_convergence_speed.py --no-dp
"""

from __future__ import annotations

import csv
import os
import statistics

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
BARS = curve_style.CURVES
HORIZON = 450          # a curve must reach this round to count as fully trained


def load(ds, sp):
    """-> {algo: [(round, ACC, DEO), ...]} for one (dataset, split)."""
    path = os.path.join(CURVE_DIR, "pg_%s_%s.csv" % (ds, sp))
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            algo = r["algo"]
            if algo not in curve_style.CURVES:
                continue
            acc, deo = _f(r, "ACC"), _f(r, "DEO")
            if acc != acc or deo != deo:
                continue
            out.setdefault(algo, []).append((int(r["round"]), acc, deo))
    for v in out.values():
        v.sort()
    return out


def _f(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else float("nan")


def common_grid(points):
    """Resample one trajectory on the 10-round grid every method shares.

    PDFFed logs a global test every round while the baselines log every tenth,
    so an unsampled comparison would reward the sparse curves purely because
    their dips are invisible.  Evaluating everything on rounds 10, 20, ...
    removes that artefact; the returned x values stay in communication rounds.
    """
    if not points:
        return []
    return [p for p in points if p[0] % 10 == 0]


def smooth(vals, window):
    half = max(1, window // 2)
    out = []
    for i in range(len(vals)):
        lo, hi = max(0, i - half), min(len(vals), i + half + 1)
        out.append(statistics.median(vals[lo:hi]))
    return out


def rounds_to_settle(points, col, mode):
    """First round after which the smoothed metric stays at the target.

    "Stays" is the key word: a method that starts high, dips and then recovers
    has not converged at round 1, so the criterion is evaluated from the end of
    the run backwards and the earliest round that holds for the whole remaining
    trajectory is returned.  This removes the early-transient artefact that a
    plain first-crossing rule produces for PDFFed.
    """
    xs = [p[0] for p in points]
    ys = smooth([p[col] for p in points], 5 if len(points) > 100 else 3)
    tail = ys[-max(3, len(ys) // 10):]
    final = statistics.mean(tail)
    target = 0.95 * final if mode == "high" else 1.05 * final + 0.005
    ok = (lambda y: y >= target) if mode == "high" else (lambda y: y <= target)
    i = len(ys) - 1
    while i > 0 and ok(ys[i - 1]):
        i -= 1
    return xs[i]


def main():
    for ds in DATASETS:
        cells = {sp: load(ds, sp) for sp in SPLITS}
        if not any(cells.values()):
            continue
        fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.4), sharex=True)
        for i, (col, mode, lab) in enumerate((
                (1, "high", "Rounds to 95% of own final ACC"),
                (2, "low", "Rounds to 105% of own final DEO"))):
            ax = axes[i]
            width = 0.8 / len(BARS)
            for j, algo in enumerate(BARS):
                vals, pos = [], []
                for k, sp in enumerate(SPLITS):
                    pts = common_grid(cells[sp].get(algo))
                    if len(pts) < 10 or pts[-1][0] < HORIZON:
                        continue
                    vals.append(rounds_to_settle(pts, col, mode))
                    pos.append(k + (j - (len(BARS) - 1) / 2) * width)
                if vals:
                    ax.bar(pos, vals, width=width * 0.92,
                           color=curve_style.COLORS[algo],
                           label=curve_style.LABELS[algo], zorder=2)
            ax.set_ylabel(lab, fontsize=10)
            ax.set_xticks(range(len(SPLITS)))
            ax.set_xticklabels([SPLIT_LABEL[s] for s in SPLITS], fontsize=9)
            ax.tick_params(labelsize=9)
            ax.grid(True, axis="y", alpha=0.25, lw=0.6)
        axes[0].set_title(ds, fontsize=12)
        axes[1].set_xlabel("Heterogeneity split", fontsize=10)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=7.5, frameon=False, ncol=8,
                   loc="lower center", bbox_to_anchor=(0.5, 0.005),
                   handletextpad=0.35, columnspacing=0.8)
        fig.subplots_adjust(left=0.11, right=0.99, top=0.93, bottom=0.175,
                            hspace=0.16)
        stem = "speed_%s%s" % (ds, fig_variants.suffix())
        fig.savefig(os.path.join(OUT_DIR, stem + ".png"), dpi=300)
        fig.savefig(os.path.join(OUT_DIR, stem + ".pdf"))
        plt.close(fig)
        print("wrote", os.path.join(OUT_DIR, stem + ".png"))


if __name__ == "__main__":
    main()
