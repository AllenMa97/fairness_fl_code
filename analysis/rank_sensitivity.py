#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Which evaluation cells carry PDFFed's cross-dataset rank?

The rank summary in ``plot_rank.py`` pools 41 (dataset, split) cells at the
protocol client count.  This script asks a different question: *how much of
PDFFed's pooled rank is produced by which cells?*  It is a diagnostic, not a
result -- dropping cells after seeing their outcome is cell-level cherry
picking, so the output is meant to be read as a robustness map rather than as a
license to report the best subset.

Three views are printed:

1. **Per-cell rank.**  PDFFed's rank inside every cell on ACC / DEO / HM,
   alongside the cell's majority floor and the algorithm count, so a weak rank
   can be attributed to either a collapse or a genuine loss.
2. **Leave-one-cell-out.**  Remove each cell alone and report the change in
   PDFFed's pooled score.  Cells whose removal helps are the ones where PDFFed
   is below its own average; cells whose removal hurts are the ones carrying it.
   The same table is also aggregated by dataset and by split, because a claim
   should be defensible at the granularity of "we do not evaluate on a
   collapsed corpus", not at the granularity of one lucky cell.
3. **Greedy removal path.**  Repeatedly drop the cell whose removal most
   improves PDFFed's pooled position, until at most ``MAX_DROP`` cells are gone
   or only ``MIN_KEEP`` remain.  The path is plotted so that the number of
   deletions needed to reach a given rank is visible.  A rank that only appears
   after ten deletions out of 41 is not a finding.

The pooled score of an algorithm is the mean of its three mean ranks
(ACC, DEO, HM); the pooled *position* is 1 + the number of algorithms with a
strictly better pooled score.  Position is the quantity the greedy step
maximises, with the pooled score itself as tie-break.

Output: ``analysis/figures/rank_sensitivity.png`` (+ pdf), and the printed
tables.

Usage::

    python analysis/rank_sensitivity.py
"""

from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fig_variants
import plot_rank as R

OUT_DIR = os.path.join(R.REPO_ROOT, "analysis", "figures")
FLOOR_CSV = os.path.join(R.REPO_ROOT, "reported_results", "two_machine_summary",
                        "dataset_floor_availability.csv")
METRICS = ("ACC", "DEO", "HM")
TARGET = "PDFFed"
MAX_DROP = 12
MIN_KEEP = 15
ALGO = R.ALGORITHMS


def load_floors():
    """{dataset: majority-class floor}, NaN when the corpus is multi-label."""
    floors = {}
    with open(FLOOR_CSV, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                floors[row["dataset"]] = float(row["floor"])
            except (TypeError, ValueError):
                floors[row["dataset"]] = float("nan")
    return floors


def rank_tables(cells):
    """{cell: {metric: {algo: rank}}} using the ranking rules of plot_rank."""
    tables = {}
    for key, rows in cells.items():
        acc = {a: R._f(r, "ACC") for a, r in rows.items()}
        deo = {a: abs(R._f(r, "DEO")) for a, r in rows.items()}
        hm = {a: R._f(r, "HM") for a, r in rows.items()}
        hm = {a: v for a, v in hm.items() if v == v}
        tables[key] = {"ACC": R.average_ranks(acc, True),
                       "DEO": R.average_ranks(deo, False),
                       "HM": R.average_ranks(hm, True)}
    return tables


def score(tables, keys, algo):
    """(pooled score, per-metric means) of ``algo`` over ``keys``."""
    per = {}
    for metric in METRICS:
        vals = [tables[k][metric][algo] for k in keys
                if algo in tables[k][metric]]
        per[metric] = sum(vals) / len(vals) if vals else float("nan")
    return sum(per[m] for m in METRICS) / 3.0, per


def position(tables, keys, algo):
    """1 + number of algorithms whose pooled score beats ``algo``'s."""
    own, _ = score(tables, keys, algo)
    better = 0
    for other in ALGO:
        if other == algo:
            continue
        if not any(other in tables[k][m] for k in keys for m in METRICS):
            continue
        if score(tables, keys, other)[0] < own - 1e-9:
            better += 1
    return better + 1


def fmt_cell(key):
    return "%s/%s" % key


def main():
    cells = R.load_cells()
    tables = rank_tables(cells)
    keys_all = sorted(cells)
    base_score, base_per = score(tables, keys_all, TARGET)
    base_pos = position(tables, keys_all, TARGET)
    print("pooled base: %d cells, PDFFed score %.2f  position %d/%d"
          % (len(keys_all), base_score, base_pos, len(ALGO)))
    print("            ACC %.2f  DEO %.2f  HM %.2f\n"
          % (base_per["ACC"], base_per["DEO"], base_per["HM"]))

    # ---- 1. per-cell rank ------------------------------------------------
    floors = load_floors()
    print("== per-cell rank of %s (rank / n_algos in that cell) ==" % TARGET)
    print("%-34s %5s %8s %11s %11s %11s  %s"
          % ("cell", "n", "floor", "ACC", "DEO", "HM", "PDFFed ACC vs floor"))
    for key in keys_all:
        n = len(cells[key])
        vals = []
        for metric in METRICS:
            table = tables[key][metric]
            vals.append("%.1f/%d" % (table[TARGET], len(table))
                        if TARGET in table else "-")
        floor = floors.get(key[0], float("nan"))
        acc = R._f(cells[key].get(TARGET, {}), "ACC")
        if TARGET not in cells[key]:
            verdict = "not run"
        elif acc != acc or floor != floor:
            verdict = "n/a"
        elif acc < floor:
            verdict = "BELOW (%.3f)" % acc
        else:
            verdict = "+%.1fpp" % ((acc - floor) * 100)
        print("%-34s %5d %8s %11s %11s %11s  %s"
              % (fmt_cell(key), n,
                 "%.3f" % floor if floor == floor else "MULTI", *vals, verdict))

    # ---- 2. leave-one-cell-out ------------------------------------------
    loo = []
    for key in keys_all:
        rest = [k for k in keys_all if k != key]
        s, per = score(tables, rest, TARGET)
        loo.append((key, s - base_score, per["ACC"] - base_per["ACC"],
                    per["DEO"] - base_per["DEO"], per["HM"] - base_per["HM"]))
    loo.sort(key=lambda t: t[1])

    print("\n== leave one cell out (delta pooled score, negative = removal helps) ==")
    print("%-34s %8s %8s %8s %8s" % ("removed cell", "dScore", "dACC", "dDEO", "dHM"))
    for key, ds, da, dd, dh in loo:
        print("%-34s %8.2f %8.2f %8.2f %8.2f" % (fmt_cell(key), ds, da, dd, dh))

    def group_delta(pick):
        kept = [k for k in keys_all if not pick(k)]
        if not kept:
            return None
        s, _ = score(tables, kept, TARGET)
        return len(keys_all) - len(kept), s - base_score

    print("\n== drop a whole dataset / split (delta pooled score) ==")
    for axis, picker in (
            ("dataset", lambda k: k[0]),
            ("split", lambda k: k[1])):
        groups = sorted({picker(k) for k in keys_all})
        out = []
        for g in groups:
            res = group_delta(lambda k, g=g: picker(k) == g)
            if res and res[0] < len(keys_all):
                out.append((g, res[0], res[1]))
        out.sort(key=lambda t: t[2])
        print("-- by %s --" % axis)
        for g, n, ds in out:
            print("   %-16s drop %2d cells  dScore %6.2f" % (g, n, ds))

    # ---- 3. a-priori defensible subsets ----------------------------------
    # Each rule below is stated without looking at who wins it, so the reader
    # can accept or reject the rule on its own merits.
    max_n = max(len(rows) for rows in cells.values())

    def below_floor(key, algo):
        acc = R._f(cells[key].get(algo, {}), "ACC")
        floor = floors.get(key[0], float("nan"))
        if acc != acc or floor != floor:
            return None
        return acc < floor

    def cell_collapsed(key):
        verdicts = [below_floor(key, a) for a in cells[key]]
        verdicts = [v for v in verdicts if v is not None]
        return bool(verdicts) and sum(verdicts) * 2 >= len(verdicts)

    rules = [
        ("all cells", lambda k: True),
        ("full grid only (n=%d)" % max_n, lambda k: len(cells[k]) == max_n),
        ("no Speaker split", lambda k: k[1] != "Speaker"),
        ("no multi-label (bios)", lambda k: k[0] != "bios"),
        ("no collapsed cell", lambda k: not cell_collapsed(k)),
        ("PDFFed above floor", lambda k: below_floor(k, TARGET) is False),
        ("no Speaker + no bios + no collapsed",
         lambda k: k[1] != "Speaker" and k[0] != "bios" and not cell_collapsed(k)),
    ]
    print("\n== named subsets (rule stated without reference to the winner) ==")
    print("%-44s %6s %6s %8s %8s  %s"
          % ("subset", "cells", "PDFFed", "score", "position", "leader"))
    for name, keep in rules:
        kept = [k for k in keys_all if keep(k)]
        if len(kept) < 3:
            print("%-44s %6d   (too small)" % (name, len(kept)))
            continue
        s, _ = score(tables, kept, TARGET)
        p = position(tables, kept, TARGET)
        mine = sum(1 for k in kept if TARGET in tables[k]["ACC"])
        lead = min((a for a in ALGO if a != TARGET),
                   key=lambda a: score(tables, kept, a)[0])
        ls, _ = score(tables, kept, lead)
        print("%-44s %6d %6d %8.2f %8d  %s %.2f"
              % (name, len(kept), mine, s, p, R.LABELS.get(lead, lead), ls))

    dropped = [k for k in keys_all if cell_collapsed(k)]
    print("cells dropped as collapsed (>= half the algorithms below the floor):")
    for key in dropped:
        verdicts = [below_floor(key, a) for a in cells[key]]
        verdicts = [v for v in verdicts if v is not None]
        print("   %-30s %2d/%2d below floor"
              % (fmt_cell(key), sum(verdicts), len(verdicts)))

    # ---- 4. greedy removal path -----------------------------------------
    kept = list(keys_all)
    path = [(0, base_score, base_pos)]
    steps = []
    while len(kept) > MIN_KEEP and len(steps) < MAX_DROP:
        best = None
        for key in kept:
            cand = [k for k in kept if k != key]
            s, _ = score(tables, cand, TARGET)
            p = position(tables, cand, TARGET)
            rank = (p, s)
            if best is None or rank < best[0]:
                best = (rank, key, s, p)
        if best is None:
            break
        cur = (path[-1][1], path[-1][2])
        if (best[3], best[2]) >= (cur[1], cur[0]):
            break
        _, key, s, p = best
        kept.remove(key)
        steps.append((key, s, p))
        path.append((len(steps), s, p))

    print("\n== greedy removal path ==")
    print("%3s %-34s %8s %8s %8s" % ("#", "removed cell", "score", "position", "cells"))
    for i, (key, s, p) in enumerate(steps, 1):
        print("%3d %-34s %8.2f %8d %8d"
              % (i, fmt_cell(key), s, p, len(keys_all) - i))

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))

    ax = axes[0]
    ys = range(len(loo))
    colors = ["#c62828" if t[1] < 0 else "#607d8b" for t in loo]
    ax.barh(list(ys), [t[1] for t in loo], color=colors, height=0.7)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([fmt_cell(t[0]) for t in loo], fontsize=7.2)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(r"$\Delta$ pooled score when the cell is removed", fontsize=10)
    ax.set_title("leave-one-cell-out  (left of 0 = removal helps)", fontsize=11)
    ax.grid(True, axis="x", alpha=0.25, lw=0.6)

    ax = axes[1]
    xs = [t[0] for t in path]
    ax.plot(xs, [t[1] for t in path], marker="o", color="tab:blue", lw=1.8,
            markersize=5, label="pooled score")
    ax.set_xlabel("cells removed", fontsize=10)
    ax.set_ylabel("PDFFed pooled score (mean rank, lower better)", fontsize=9.5)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax2 = ax.twinx()
    ax2.plot(xs, [t[2] for t in path], marker="s", color="tab:orange", lw=1.5,
             markersize=4.5, label="position")
    ax2.set_ylabel("position among %d algorithms" % len(ALGO), fontsize=9.5)
    ax2.invert_yaxis()
    ax.set_title("greedy removal path", fontsize=11)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8.5,
              frameon=False, loc="upper right")

    suffix = fig_variants.suffix()
    fig.subplots_adjust(left=0.20, right=0.92, top=0.89, bottom=0.12, wspace=0.55)
    for ext in (".png", ".pdf"):
        fig.savefig(os.path.join(OUT_DIR, "rank_sensitivity" + suffix + ext),
                    dpi=300 if ext == ".png" else None)
    plt.close(fig)
    print("\nwrote", os.path.join(OUT_DIR, "rank_sensitivity" + suffix + ".png"))


if __name__ == "__main__":
    main()
