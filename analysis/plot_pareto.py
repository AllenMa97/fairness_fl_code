#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw the PDFFed / FedRenyi Pareto front from the two-server result sweep.

Reads the flattened server dumps under
``reported_results/pareto/raw/{from_30,from_31,from_31_base,from_30_comb,
from_31_dutch_base,from_31_dutch_comb}.tsv`` and writes one figure per grid,
plus a long-form CSV::

    analysis/figures/pareto_<dataset>_<split>.{png,pdf}
    reported_results/pareto/pareto_points.csv

The x axis is ``1 - DEO`` by default so that the top-right corner of every panel
is the good corner, as is conventional for fairness-utility planes; ``--fairness
dp`` swaps in ``1 - |SPD|``.  Both axes are still read straight from
``aggregate.metrics`` of the run payload, so nothing is recomputed here beyond
that flip.

PDFFed is a switch grid, not a single run: the directory name
``PDFFed_combabcde`` encodes ``(w_pa, w_conf, w_l2l, w_l2g, w_g2l)`` and
``11111`` is the all-terms-on configuration, bit-identical to the main PDFFed
run and therefore the one the paper reports.  In the default style only that
configuration is drawn, as one point; ``--style curve`` feeds the whole grid in
and connects its non-dominated subset, which is how a knob sweep becomes a
visible front.  COMPAS carries the full 32-setting grid; DUTCH and DRUG carry
the same 17-setting subset (all-off, the five leave-one-out settings, the five
leave-two-out settings, the five single-term settings and all-on).

Because the paper claims joint optimisation rather than dominance on either axis
alone, the panel also carries faint iso-HM curves: level sets of
``HM = 2*ACC*(1-DEO) / (ACC + 1 - DEO)``, the joint metric.  A single star can
then be read against the best contour any method reaches.

The single-configuration baselines come from ``raw/from_31_base.tsv`` and
``raw/from_31_dutch_base.tsv``, the same machine the PDFFed sweeps ran on, and
each gets its own colour and marker with a legend entry.  A baseline outside the
frame is pinned to the edge with a caret marker rather than being allowed to
stretch the axis, because one off-scale fairness value would squash every point
into a corner.

The FedRenyi ``lambda`` ladder (0.25/0.5/1/2/4) is drawn with every available
point, whatever its code fingerprint, but the values that land on the same
(ACC, DEO) are drawn once and the whole ladder shares a single legend entry, so
the legend stays at two rows.  The fingerprint of each point stays in the CSV.

Both partitions of all three tabular datasets (COMPAS, DUTCH, DRUG) are drawn.

``--style curve`` writes a second, FedFACT-style figure instead: each method
that has a knob sweep is reduced to its non-dominated subset and connected, and
the single-configuration baselines stay points.  That is the layout of FedFACT
(Zhang et al., 2025) Figures 1-3, where the fairness constraint is swept and the
resulting frontier is drawn as a line.

``--style ellipse`` is the same panel with each point sitting inside its
+-1 seed-std ellipse instead of a bare marker, so a displacement inside one
standard deviation cannot pass for a ranking.

``--fairness dp`` swaps the horizontal axis to the demographic-parity gap: the
runs log both ``DEO`` (equal-opportunity gap, absolute) and ``SPD`` (statistical
parity difference, signed), so the second panel is ``1 - |SPD|``.  The sign is
dropped because a group gap is a distance, not a direction; the signed mean
stays in the raw tables.  That plane carries the same contour family with
``FR = 1 - |SPD|``, labelled ``HM_SPD`` rather than ``HM``: it is the harmonic
mean of accuracy and the plotted fairness coordinate, which for a given run is
not the HM reported in the results table (that one uses ``1 - DEO``).

The all-terms-on point is labelled so it can be told apart from the baselines.
In the default style nothing is connected: a polyline through settings that
differ by less than their own seed noise would suggest structure that is not in
the data.  No error bars or halos are drawn in that style either.

``--layout row`` puts the four grids side by side in one row with a single
shared legend, the layout of FedFACT's multi-panel figures.  The HM contours
stay but are thinned to three levels per panel: at 3.4 inches wide a six-level
fan turns into hatching.

Usage::

    python analysis/plot_pareto.py                 # one star per method
    python analysis/plot_pareto.py --style curve   # FedFACT-style frontier
    python analysis/plot_pareto.py --style ellipse # with +-1 seed-std ellipses
    python analysis/plot_pareto.py --style curve --fairness dp  # DP axis
    python analysis/plot_pareto.py --style curve --layout row   # four-up
"""

from __future__ import annotations

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

import fig_variants

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(REPO_ROOT, "reported_results", "pareto", "raw")
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "figures")
CSV_OUT = os.path.join(REPO_ROOT, "reported_results", "pareto", "pareto_points.csv")

SERVERS = (("from_30.tsv", "192.168.1.30"), ("from_31.tsv", "192.168.1.31"))
BASE_SERVERS = (("from_31_base.tsv", "192.168.1.31"),
                ("from_31_dutch_base.tsv", "192.168.1.31"),
                ("from_31_drug_base.tsv", "192.168.1.31"))
COMB_SERVERS = (("from_30_comb.tsv", "192.168.1.30"),
                ("from_31_dutch_comb.tsv", "192.168.1.31"),
                ("from_30_drug_comb.tsv", "192.168.1.30"),
                ("from_31_drug_comb.tsv", "192.168.1.31"))
DRUG_SERVERS = (("from_31_drug_sweep.tsv", "192.168.1.31"),)
# The DUTCH FedRenyi lambda ladder finished after the other dumps, so it has a
# file of its own; the name carries no ``_base``/``_comb`` suffix, which is what
# makes ``load_rows`` treat it as a sweep.
RENYI_SERVERS = (("from_31_dutch_renyi.tsv", "192.168.1.31"),)
GRIDS = (("COMPAS", "Dirichlet1"), ("COMPAS", "Uniform"),
         ("DUTCH", "Dirichlet1"), ("DUTCH", "Uniform"),
         ("DRUG", "Dirichlet1"), ("DRUG", "Uniform"))
# The one-row layout is sized for four panels at 3.4 inches each.
ROW_GRIDS = GRIDS[:4]

# The two fairness axes the runs support.  ``DEO`` is the equal-opportunity gap
# already stored unsigned; ``SPD`` is the signed statistical-parity difference,
# so its magnitude is the demographic-parity gap.  ``xlabel`` is the axis
# caption, ``csv`` the column stem written to the long table.
FAIRNESS = {
    "eo": {"xlabel": r"$1-\mathrm{DEO}$", "csv": "DEO",
           "note": "equal opportunity"},
    "dp": {"xlabel": r"$1-|\mathrm{SPD}|$", "csv": "absSPD",
           "note": "demographic parity"},
}

# Display names: the split key stays ``Dirichlet1`` everywhere in the data
# files, only the panel title spells out the concentration parameter.
SPLIT_LABEL = {"Dirichlet1": r"Dirichlet $\alpha=1$", "Uniform": "Uniform"}

# Contour levels for the curve panels.  Their frame is tighter than the scatter
# panels', which leaves the lower right corner empty, so the next levels down are
# drawn there as well.  The ``dp`` table carries the same contour family built
# from ``1 - |SPD|`` instead of ``1 - DEO``; the two families are the same
# function of the plane coordinates, but their numbers are not the HM column of
# the results table, so the labels differ (see ``HM_NAME``).
HM_LEVELS = {
    "eo": {
        ("COMPAS", "Dirichlet1"): (0.68, 0.70, 0.72, 0.74, 0.76, 0.78),
        ("COMPAS", "Uniform"): (0.66, 0.68, 0.70, 0.72, 0.74, 0.76),
        # DUTCH's frame is four times taller than COMPAS's and the completed
        # switch grid spans it corner to corner, so the fan is filled in 0.03
        # steps instead of 0.05 and is extended down to the low levels that
        # cross the otherwise empty lower-left corner (HM ~ 0.51 there).  The
        # top gap is 0.06: at 0.85 and 0.88 both labels land on the same corner
        # sliver of the top-right and touch.
        ("DUTCH", "Dirichlet1"): (0.52, 0.55, 0.58, 0.61, 0.64, 0.67, 0.70,
                                  0.73, 0.76, 0.79, 0.82, 0.88),
        ("DUTCH", "Uniform"): (0.52, 0.55, 0.58, 0.61, 0.64, 0.67, 0.70,
                               0.73, 0.76, 0.79, 0.82, 0.88),
        # DRUG's frame is a narrow strip (ACC spans 0.25, fairness 0.17) that the
        # completed 17-setting grid fills, so the levels run 0.70 -> 0.86 on the
        # EO plane; 0.68 and 0.88 only graze one edge of Uniform/Dirichlet1
        # respectively and are left out.
        ("DRUG", "Dirichlet1"): (0.70, 0.74, 0.78, 0.82, 0.86),
        ("DRUG", "Uniform"): (0.72, 0.76, 0.80, 0.84),
    },
    "dp": {
        ("COMPAS", "Dirichlet1"): (0.68, 0.70, 0.72, 0.74, 0.76),
        ("COMPAS", "Uniform"): (0.68, 0.70, 0.72, 0.74, 0.76),
        ("DUTCH", "Dirichlet1"): (0.45, 0.48, 0.54, 0.60, 0.66, 0.72, 0.78,
                                  0.84),
        ("DUTCH", "Uniform"): (0.45, 0.48, 0.54, 0.60, 0.66, 0.72, 0.78,
                               0.84),
        # ``1 - |SPD|`` reaches past 1 on DRUG, so the DP frame is taller than
        # its EO counterpart and carries one more level at the top.
        ("DRUG", "Dirichlet1"): (0.74, 0.78, 0.82, 0.86, 0.90),
        ("DRUG", "Uniform"): (0.74, 0.78, 0.82, 0.86, 0.90),
    },
}

# The four-up row keeps the contours but only a few per panel: at 3.4 inches
# wide a six-level fan turns into hatching.  COMPAS gets three because its
# frame is only 0.12 tall and already carries eleven baselines plus the switch
# front; DUTCH gets five, because its frame is four times taller and the lower
# half would otherwise be empty -- the low levels enter through the left edge
# and the high ones through the top, so the fan covers both empty corners.
ROW_HM_LEVELS = {
    "eo": {
        ("COMPAS", "Dirichlet1"): (0.68, 0.72, 0.76),
        ("COMPAS", "Uniform"): (0.66, 0.70, 0.74),
        ("DUTCH", "Dirichlet1"): (0.56, 0.64, 0.72, 0.80, 0.86),
        ("DUTCH", "Uniform"): (0.56, 0.64, 0.72, 0.80, 0.86),
    },
    "dp": {
        ("COMPAS", "Dirichlet1"): (0.68, 0.72, 0.76),
        ("COMPAS", "Uniform"): (0.68, 0.72, 0.76),
        # Four rather than five: the DP labels carry the longer ``HM_SPD``
        # prefix, and five of them along the top edge of a 3.4-inch panel would
        # run into each other.
        ("DUTCH", "Dirichlet1"): (0.56, 0.68, 0.78, 0.86),
        ("DUTCH", "Uniform"): (0.56, 0.68, 0.78, 0.86),
    },
}

# The contour family is a harmonic mean of ACC and the horizontal coordinate.
# On the EO plane that coordinate is ``1 - DEO`` and the levels coincide with
# the HM column of the results table; on the DP plane it is ``1 - |SPD|`` and
# they do not, so the two are labelled differently.
HM_NAME = {"eo": "HM", "dp": r"HM$_{\mathrm{SPD}}$"}


FEDRENYI_LAMBDA = {"FedRenyi_lam0p25": 0.25, "FedRenyi_lam0p5": 0.5,
                   "FedRenyi_lam1": 1.0, "FedRenyi_lam2": 2.0,
                   "FedRenyi_lam4": 4.0}


# ``FederatedAverageWithPo`` is a private diagnostic run, not part of the paper's
# baseline set.  ``LoGoFair`` is skipped until its post-processing is fixed: its
# DP-only threshold search drives both groups' thresholds past every score, so
# it predicts the negative class everywhere (DEO = 0 by construction).
BASELINE_SKIP = ("FederatedAverageWithPo", "LoGoFair")
BASELINE_COLORS = ("#8c564b", "#9467bd", "#e377c2", "#7f7f7f", "#bcbd22",
                   "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896",
                   "#c5b0d5", "#c49c94")
# No circle, triangle or star here: those are reserved for PDFFed (circle or
# star, depending on the style) and FedRenyi (triangle), so every method stays
# separable by shape alone, which is what a greyscale print needs.
BASELINE_MARKERS = ("s", "v", "D", "P", "X", "h", "<", ">", "p", "d", "8",
                    "H")


def load_rows():
    rows = []
    for fname, host in SERVERS + BASE_SERVERS + COMB_SERVERS + DRUG_SERVERS + RENYI_SERVERS:
        path = os.path.join(RAW_DIR, fname)
        if not os.path.isfile(path):
            sys.exit("missing %s -- copy the server dump there first" % path)
        if fname.endswith("_base.tsv"):
            kind = "baseline"
        elif fname.endswith("_comb.tsv"):
            kind = "comb"
        else:
            kind = "sweep"
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                r["host"] = host
                r["kind"] = kind
                rows.append(r)
    return rows


def fnum(row, key):
    v = row.get(key, "")
    return float(v) if v not in ("", None) else float("nan")


def gap(row, metric):
    """Return ``(gap, gap_sd)`` for the requested fairness metric.

    ``eo`` reads the stored equal-opportunity gap directly.  ``dp`` takes the
    magnitude of the signed statistical-parity difference: a group gap is a
    distance, so the sign is dropped on the plane and kept only in the raw
    table.  Both columns are means over the same three seeds.
    """
    if metric == "dp":
        return abs(fnum(row, "SPD")), fnum(row, "SPD_sd")
    return fnum(row, "DEO"), fnum(row, "DEO_sd")


def comb_points(rows, dataset, split, metric="eo"):
    """Return [(bits, acc, deo, acc_sd, deo_sd, host, fp)] for the PDFFed grid.

    ``bits`` is the five character switch string of ``PDFFed_combabcde``, in the
    order ``(w_pa, w_conf, w_l2l, w_l2g, w_g2l)``.  ``bits == "11111"`` is the
    all-terms-on configuration, i.e. the method as reported in the paper.
    """
    out = []
    for r in rows:
        if r["kind"] != "comb":
            continue
        if r["dataset"] != dataset or r["split"] != split:
            continue
        if r["kdir"] != "20Clients" or r["n_rep"] != "3":
            continue
        bits = r["algo_dir"].rsplit("comb", 1)[-1]
        g, gsd = gap(r, metric)
        out.append((bits, fnum(r, "ACC"), g, fnum(r, "ACC_sd"), gsd,
                    r["host"], r["fp"]))
    return sorted(out, key=lambda t: t[0])


def fedrenyi_points(rows, dataset, split, metric="eo"):
    """Return [(lam, acc, deo, acc_sd, deo_sd, host, fp)] for one grid.

    Every available ``lambda`` point is kept regardless of its code fingerprint.
    Where the sweep is insensitive to ``lambda`` (the COMPAS/Uniform ladder) a
    fingerprint-mixed curve is still the honest picture; the fingerprint travels
    with each point into the CSV and the panel footnote.
    """
    out = []
    for algo, lam in FEDRENYI_LAMBDA.items():
        hits = [r for r in rows if r["dataset"] == dataset and r["split"] == split
                and r["algo_dir"] == algo and r["k_clients_ok"]]
        if not hits:
            continue
        r = hits[0]
        g, gsd = gap(r, metric)
        out.append((lam, fnum(r, "ACC"), g, fnum(r, "ACC_sd"), gsd,
                    r["host"], r["fp"]))
    return sorted(out, key=lambda t: t[0])


def pdfed_points(rows, dataset, split, metric="eo"):
    """PDFFed's reported run: ``comb11111`` when that directory exists.

    Both grids that are drawn were run with the switch-named entry point, so the
    all-terms-on directory is the run the paper reports.  ``rows`` without any
    ``comb`` entry fall back to the plain ``PDFFed`` directory and the knob reads
    ``default`` in the CSV.
    """
    grid = [p for p in comb_points(rows, dataset, split, metric) if p[0] == "11111"]
    if grid:
        return grid
    out = []
    for r in rows:
        if (r["dataset"] != dataset or r["split"] != split
                or r["kind"] != "sweep" or not r["k_clients_ok"]):
            continue
        if r["algo_dir"] != "PDFFed":
            continue
        g, gsd = gap(r, metric)
        out.append(("default", fnum(r, "ACC"), g, fnum(r, "ACC_sd"), gsd,
                    r["host"], r["fp"]))
    return out[-1:]


def baseline_points(rows, dataset, split, metric="eo"):
    """Return [(name, acc, fairness, acc_sd, fair_sd)] for one grid, one row per algorithm.

    Baselines ran a single configuration each, so each is one point.  Runs named
    in ``BASELINE_SKIP`` are left out of both the figure and the CSV.
    ``fairness`` is already ``1 - DEO`` so that it shares the axis convention of
    the sweeps, and its spread is ``DEO_sd`` unchanged by the flip.
    """
    out = {}
    for r in rows:
        if r["kind"] != "baseline":
            continue
        if r["dataset"] != dataset or r["split"] != split:
            continue
        if r["kdir"] != "20Clients" or r["n_rep"] != "3":
            continue
        if r["algo_dir"] in BASELINE_SKIP:
            continue
        if not fig_variants.keep(r["algo_dir"]):
            continue
        g, gsd = gap(r, metric)
        out[r["algo_dir"]] = (fnum(r, "ACC"), 1.0 - g,
                              fnum(r, "ACC_sd"), gsd)
    return [(name, acc, fair, asd, fsd) for name, (acc, fair, asd, fsd)
            in sorted(out.items())]


def baseline_style(names):
    """Give every baseline its own colour and marker, stable across panels."""
    return {name: (BASELINE_COLORS[i % len(BASELINE_COLORS)],
                   BASELINE_MARKERS[i % len(BASELINE_MARKERS)])
            for i, name in enumerate(names)}


def to_fairness(pts):
    """Map the stored gap to ``1 - gap`` so the best corner is the top right."""
    return [(p[0], p[1], 1.0 - p[2], p[3], p[4], p[5], p[6]) for p in pts]


def is_collapsed(pts, tol=1e-6):
    """True when the sweep's points coincide, i.e. lambda has no effect at all."""
    if len(pts) < 3:
        return False
    accs = [p[1] for p in pts]
    deos = [p[2] for p in pts]
    return max(accs) - min(accs) < tol and max(deos) - min(deos) < tol


def is_mostly_collapsed(pts, tol=1e-6, noise=1.5):
    """True when >=3 of the sweep's points coincide and the rest sit inside noise.

    Some grids collapse this way: three of the five lambda values land on the
    same number and the two that differ stay within 1.5 seed standard
    deviations, so the sweep carries no Pareto structure even though it is not
    bit-identical.
    """
    if len(pts) < 3:
        return False
    accs = [p[1] for p in pts]
    deos = [p[2] for p in pts]
    modal = max(set(accs), key=accs.count)
    if sum(1 for a in accs if abs(a - modal) < tol) < 3:
        return False
    return (max(accs) - min(accs) < noise * max(p[3] for p in pts)
            and max(deos) - min(deos) < noise * max(p[4] for p in pts))


MIN_SPAN = {"ACC": 0.05, "FAIR": 0.04}


def axis_limits(vals, min_span):
    """Pad one axis so that seed noise alone cannot fill the frame.

    ``vals`` already spans mean +- std.  The span is widened to ``min_span``
    when the data are tighter than that, which is what stops a panel whose
    points all sit inside their own seed noise from looking like a landscape of
    distinct trade-offs.
    """
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, min_span)
    mid = 0.5 * (lo + hi)
    half = 0.56 * span
    return mid - half, mid + half


def draw_hm_iso(ax, xlim, ylim, levels=(0.70, 0.72, 0.74, 0.76, 0.78),
                name="HM"):
    """Faint iso-contours of the joint objective.

    On the EO plane the joint metric of the paper is the harmonic mean ``HM =
    2*ACC*FR / (ACC+FR)`` with ``FR = 1 - DEO``, so its level sets are
    hyperbolas in this plane.  Drawing them lets a single star carry the joint
    claim: the point sits on the highest reachable contour even though it is not
    the fairest one on the plane.

    The same family is drawn on the DP plane with ``FR = 1 - |SPD|``, which is a
    different number for the same operating point, so ``name`` prefixes the
    labels (``HM`` on the EO plane, ``HM_SPD`` on the DP plane) and the two can
    never be mistaken for one another.

    Only the visible part of each level set is drawn, and the label sits at the
    left end of that visible arc.  The fan opens to the right, so the space
    under the left end is empty: the text reads as the name of the curve instead
    of floating in the middle of the panel.  Levels whose arc would land outside
    the frame are skipped, which is why the highest contours only appear as a
    short segment in the top-right corner.
    """
    xs = np.linspace(xlim[0], xlim[1], 900)
    for c in levels:
        denom = 2.0 * xs - c
        ys = np.where(np.abs(denom) > 1e-12, c * xs / denom, np.nan)
        inside = np.where(np.isfinite(ys) & (ys >= ylim[0]) & (ys <= ylim[1]))[0]
        if inside.size < 2:
            continue
        ax.plot(xs[inside], ys[inside], color="#9e9e9e", lw=0.8, ls=(0, (5, 4)),
                alpha=0.65, zorder=1, label="_nolegend_")
        i = inside[0]
        # the arc can enter through the right edge when it is a corner sliver
        near_right = xs[i] > xlim[1] - 0.18 * (xlim[1] - xlim[0])
        below = ys[i] - ylim[0] > 0.14 * (ylim[1] - ylim[0])
        ax.annotate("%s %.2f" % (name, c), (xs[i], ys[i]),
                    textcoords="offset points",
                    xytext=((-5, -8) if near_right else (5, -8)) if below
                    else ((-5, 7) if near_right else (5, 7)),
                    ha="right" if near_right else "left",
                    va="top" if below else "bottom",
                    fontsize=7, color="#8a8a8a")


def dedupe(pts, ndigits=6):
    """Drop points that coincide on the plane.

    Several ``lambda`` values land on the same (ACC, DEO) to six decimals.  They
    stay in the CSV but are drawn once, so the panel does not show a stack of
    markers that looks like one point.
    """
    seen, out = set(), []
    for p in pts:
        key = (round(p[1], ndigits), round(p[2], ndigits))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def pareto_front(pts):
    """Keep the non-dominated points, ordered from least to most fair.

    "Non-dominated" is on this plane: no other configuration is at least as
    accurate *and* at least as fair.  Connecting the surviving points is what
    separates a trade-off frontier from a cloud of settings.
    """
    out = []
    for p in pts:
        if any(q[1] >= p[1] and q[2] >= p[2] and (q[1], q[2]) != (p[1], p[2])
               for q in pts):
            continue
        out.append(p)
    return sorted(out, key=lambda t: t[2])


def draw_fedrenyi(ax, pts, sd=False):
    """One marker per distinct (ACC, fairness) point, one legend entry.

    The five ``lambda`` values are a ladder, not five methods, so they share a
    single legend entry instead of spending five rows of the legend.
    """
    uniq = dedupe(pts)
    if sd:
        draw_sd_ellipses(ax, uniq, "#3f8f4f")
    ax.scatter([p[2] for p in uniq], [p[1] for p in uniq], s=150, marker="^",
               color="#3f8f4f", edgecolors="white", linewidths=0.7, zorder=3,
               label=r"FedRenyi ($\lambda$ = 0.25 $\to$ 4)")


def draw_sd_ellipses(ax, pts, color, alpha=0.18):
    """One translucent +-1 seed-std ellipse per point, on the mean coordinates.

    The ellipses are the honest version of the marker: on this plane the mean is
    a summary of three runs, and the axes are wide enough that a displacement
    inside one standard deviation should not read as a ranking.
    """
    for _k, acc, fair, asd, fsd, *_rest in pts:
        ax.add_patch(Ellipse((fair, acc), 2.0 * fsd, 2.0 * asd, facecolor=color,
                             edgecolor=color, lw=0.5, alpha=alpha, zorder=1.5))


def draw_pdfed(ax, pts, sd=False):
    """Draw PDFFed as one marker: the reported configuration only.

    That is ``comb11111``, which is bit-identical to the plain PDFFed run; a
    grid whose ``comb`` rows are missing falls back to the ``PDFFed`` directory.
    The rest of the switch grid stays off this plane.
    """
    bits, acc, fair, *_rest = pts[0]
    if sd:
        draw_sd_ellipses(ax, pts, "tab:blue")
    ax.scatter([fair], [acc], s=290, marker="*", color="tab:blue",
               edgecolors="white", linewidths=0.8, zorder=4,
               label=("PDFFed (all terms on)" if bits == "11111" else "PDFFed"))
    # A point near the right edge gets its label on the inside, otherwise the
    # text runs off the canvas on the narrow DRUG strips.
    x_lo, x_hi = ax.get_xlim()
    right = fair > x_lo + 0.82 * (x_hi - x_lo)
    ax.annotate("PDFFed", (fair, acc), textcoords="offset points",
                xytext=(-9, 4) if right else (0, 13),
                ha="right" if right else "center", fontsize=8.5,
                color="tab:blue")


def reorder_legend(ax):
    """Put PDFFed first, then FedRenyi, baselines last."""
    handles, labels = ax.get_legend_handles_labels()
    rank = []
    for i, label in enumerate(labels):
        if label.startswith("PDFFed"):
            rank.append(0)
        elif label.startswith("FedRenyi"):
            rank.append(1)
        else:
            rank.append(2)
    order = sorted(range(len(labels)), key=lambda i: (rank[i], i))
    return [handles[i] for i in order], [labels[i] for i in order]


def draw_baselines(ax, pts, style, xlim, ylim, sd=False):
    """Scatter the single-configuration baselines, one colour and marker each.

    A point outside the frame is pinned to the edge with a caret marker and its
    true coordinate printed beside it, instead of being allowed to stretch the
    axis and squash every sweep into a corner.  With ``sd`` the point also gets
    its +-1 seed-std ellipse in the same colour.
    """
    x_lo, x_hi = xlim
    y_lo, y_hi = ylim
    dx, dy = 0.012 * (x_hi - x_lo), 0.03 * (y_hi - y_lo)
    pinned = []
    for name, acc, fair, asd, fsd in pts:
        color, marker = style[name]
        if sd:
            draw_sd_ellipses(ax, [(name, acc, fair, asd, fsd)], color, alpha=0.13)
        px, py, mark = fair, acc, marker
        pinned_x = pinned_y = False
        if fair > x_hi:
            px, mark, pinned_x = x_hi - dx, ">", True
        elif fair < x_lo:
            px, mark, pinned_x = x_lo + dx, "<", True
        if acc > y_hi:
            py, mark, pinned_y = y_hi - dy, "^", True
        elif acc < y_lo:
            py, mark, pinned_y = y_lo + dy, "v", True
        ax.scatter([px], [py], s=90, marker=mark, color=color,
                   edgecolors="white", linewidths=0.6, zorder=2, label=name)
        if pinned_x or pinned_y:
            text = ("%s %.3f" % (name, fair)) if pinned_x else name
            near_right = fair > x_hi - 0.12 * (x_hi - x_lo)
            dx_pt, ha = (-6, "right") if near_right else (6, "left")
            ax.annotate(text, (px, py), textcoords="offset points",
                        xytext=(dx_pt, 5), fontsize=6.6, color=color, ha=ha)
            pinned.append(name)
    return pinned


def frame_values(comb, fr, bl, bl_sd=False):
    """Collect the coordinate values the frame has to cover.

    Mean +- seed std rather than the mean alone, so the panel is never tighter
    than the measurement it summarises.  ``bl_sd`` extends the same courtesy to
    the baselines, which is only needed when their ellipses are drawn.
    """
    fair_v, acc_v = [], []
    for pts in (comb, fr):
        for _k, acc, fair, asd, dsd, _h, _f in pts:
            fair_v += [fair - dsd, fair + dsd]
            acc_v += [acc - asd, acc + asd]
    for _n, acc, fair, asd, fsd in bl:
        if bl_sd:
            fair_v += [fair - fsd, fair + fsd]
            acc_v += [acc - asd, acc + asd]
        else:
            fair_v.append(fair)
            acc_v.append(acc)
    return fair_v, acc_v


def draw_sweeps(ax, comb_all, fr, cloud=False):
    """Plot the two knob sweeps: one marker style each, so shape tells them apart.

    With ``cloud`` the whole PDFFed grid is also drawn, as hollow circles in the
    method's own colour, behind the front.  It keeps a panel honest when the
    frontier collapses: DRUG's seventeen settings leave only one or two
    non-dominated points, and without the grid underneath the panel reads as if
    seventeen runs had never happened.
    """
    grid = dedupe(comb_all)
    front = pareto_front(grid)
    if cloud:
        on_front = {(p[1], p[2]) for p in front}
        off = [p for p in grid if (p[1], p[2]) not in on_front]
        ax.scatter([p[2] for p in off], [p[1] for p in off], s=38, marker="o",
                   facecolors="none", edgecolors="tab:blue", linewidths=0.8,
                   zorder=3)
    ax.plot([p[2] for p in front], [p[1] for p in front], color="tab:blue",
            lw=1.8, marker="o", markersize=6.5, markeredgecolor="white",
            markeredgewidth=0.7, zorder=4, label="PDFFed")

    ffr = pareto_front(dedupe(fr))
    if len(ffr) >= 2:
        ax.plot([p[2] for p in ffr], [p[1] for p in ffr], color="#3f8f4f",
                lw=1.6, marker="^", markersize=7.5, markeredgecolor="white",
                markeredgewidth=0.7, zorder=3, label="FedRenyi")
    elif ffr:
        ax.plot([p[2] for p in ffr], [p[1] for p in ffr], ls="none",
                marker="^", markersize=7.5, color="#3f8f4f",
                markeredgecolor="white", markeredgewidth=0.7, zorder=4,
                label="FedRenyi")


def draw_curve(ax, comb_all, fr, bl, bl_style, xlim, ylim, levels=None,
               hm_name="HM", cloud=False):
    """FedFACT-style panel: connect each method's own knob sweep.

    FedFACT (Zhang et al., 2025) Figures 1-3 plot every baseline as a single
    point and their own method as a curve, traced by sweeping its fairness
    constraint.  The analogue here is the five-term switch grid for PDFFed and
    the ``lambda`` ladder for FedRenyi, so each family is reduced to its
    non-dominated subset and connected in order of increasing fairness.
    Baselines ran a single configuration each and stay points.
    """
    if levels is None:
        draw_hm_iso(ax, xlim, ylim, name=hm_name)
    else:
        draw_hm_iso(ax, xlim, ylim, levels=levels, name=hm_name)
    draw_baselines(ax, bl, bl_style, xlim, ylim)
    draw_sweeps(ax, comb_all, fr, cloud=cloud)


def draw_panel(ax, ds, sp, pdfed, fr, bl, bl_style, lims, metric, style_name,
               show_ylabel=True, levels=(), title_size=13.0, label_size=11.0,
               tick_size=9.5, cloud=False):
    """Draw one fairness-accuracy panel.

    ``lims`` is fixed by the caller so every panel of a row is measured against
    the same rule; ``levels=()`` draws no contours, and ``metric`` picks which
    joint objective they encode (see ``HM_NAME``).
    """
    x_lo, x_hi = lims[0]
    y_lo, y_hi = lims[1]
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    if style_name == "curve":
        draw_curve(ax, pdfed, fr, bl, bl_style, (x_lo, x_hi), (y_lo, y_hi),
                   levels=levels, hm_name=HM_NAME[metric], cloud=cloud)
    else:
        sd = style_name == "ellipse"
        if levels:
            draw_hm_iso(ax, (x_lo, x_hi), (y_lo, y_hi), levels=levels,
                        name=HM_NAME[metric])
        elif metric == "eo":
            draw_hm_iso(ax, (x_lo, x_hi), (y_lo, y_hi))
        draw_baselines(ax, bl, bl_style, (x_lo, x_hi), (y_lo, y_hi), sd=sd)
        draw_pdfed(ax, pdfed, sd=sd)
        if is_collapsed(fr):
            lam, acc, fair, asd, dsd, host, fp = fr[0]
            if sd:
                draw_sd_ellipses(ax, fr[:1], "#3f8f4f")
            ax.scatter([fair], [acc], s=120, marker="D", color="#3f8f4f",
                       edgecolors="white", linewidths=0.7, zorder=3,
                       label=r"FedRenyi, $\lambda$ collapsed")
            near_right = fair > x_lo + 0.72 * (x_hi - x_lo)
            ax.annotate(r"all $\lambda\in\{0.25,\ldots,4\}$ coincide",
                        (fair, acc), textcoords="offset points",
                        xytext=(-11, -16) if near_right else (11, -16),
                        ha="right" if near_right else "left",
                        fontsize=8, color="#3f8f4f")
        elif fr:
            draw_fedrenyi(ax, fr, sd=sd)

    ax.set_title("%s / %s" % (ds, SPLIT_LABEL.get(sp, sp)),
                 fontsize=title_size, pad=10)
    ax.set_xlabel(FAIRNESS[metric]["xlabel"], fontsize=label_size)
    if show_ylabel:
        ax.set_ylabel("ACC", fontsize=label_size)
    ax.grid(False)
    ax.tick_params(labelsize=tick_size)
    # Same precision on every panel: a locator left to itself falls back to
    # 0.005 steps on the tight ACC axes and prints three decimals there while
    # the neighbouring panels print two.  ``steps`` without 2.5 keeps the steps
    # at 0.01 or coarser, so two decimals never collide.
    #
    # Each axis needs its own locator: a Locator instance keeps a pointer to the
    # last axis it was attached to, so a single shared instance computes the x
    # ticks from the y interval, which lands every one of them outside the
    # frame and leaves the x axis blank.
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=9, steps=[1, 2, 5, 10]))
        axis.set_major_formatter(FormatStrFormatter("%.2f"))


def main():
    style_name = "scatter"
    if "--style" in sys.argv:
        style_name = sys.argv[sys.argv.index("--style") + 1]
        if style_name not in ("scatter", "curve", "ellipse"):
            sys.exit("--style must be 'scatter', 'curve' or 'ellipse'")
    metric = "eo"
    if "--fairness" in sys.argv:
        metric = sys.argv[sys.argv.index("--fairness") + 1]
        if metric not in FAIRNESS:
            sys.exit("--fairness must be 'eo' (equal opportunity) or "
                     "'dp' (demographic parity)")
    layout = "panel"
    if "--layout" in sys.argv:
        layout = sys.argv[sys.argv.index("--layout") + 1]
        if layout not in ("panel", "row"):
            sys.exit("--layout must be 'panel' (one file per grid) or "
                     "'row' (all grids in one row, shared legend)")
    fair_spec = FAIRNESS[metric]
    cloud = "--grid-cloud" in sys.argv
    cloud_tag = "_cloud" if cloud else ""

    rows = load_rows()
    for r in rows:
        r["k_clients_ok"] = r.get("kdir") == "20Clients" and r.get("n_rep") == "3"

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_out = (CSV_OUT if metric == "eo" else
               CSV_OUT.replace(".csv", "_dp.csv"))
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    base_all = {key: baseline_points(rows, *key, metric=metric)
                for key in GRIDS}
    bl_style = baseline_style(sorted({n for pts in base_all.values() for n, *_ in pts}))

    # One frame per grid, measured over the points that are actually drawn.
    # Fixing the frame is what stops autoscaling from magnifying seed noise until
    # noise looks like trade-off structure.
    panels = []
    knobs_all = {}
    lims = {}
    for ds, sp in GRIDS:
        pdfed = to_fairness(pdfed_points(rows, ds, sp, metric=metric))
        fr = to_fairness(fedrenyi_points(rows, ds, sp, metric=metric))
        if not pdfed:
            sys.exit("no PDFFed row for %s/%s" % (ds, sp))
        # In the curve style PDFFed gets the same treatment as FedRenyi: every
        # switch setting is fed in and only the non-dominated subset is
        # connected, so its own knob sweep becomes a visible front.  The CSV
        # still reports the all-on configuration, which is the paper's method.
        knobs = (to_fairness(comb_points(rows, ds, sp, metric=metric))
                 if style_name == "curve" else pdfed)
        knobs_all[(ds, sp)] = knobs
        if style_name == "curve":
            shown_p = pareto_front(dedupe(knobs))
            shown_fr = pareto_front(dedupe(fr))
            # The cloud draws every setting, so the frame has to hold them all.
            if cloud:
                shown_p = dedupe(knobs)
        else:
            shown_p, shown_fr = pdfed, fr
        panels.append((ds, sp, pdfed, fr))
        fair_v, acc_v = frame_values(shown_p, shown_fr, base_all[(ds, sp)],
                                     bl_sd=(style_name == "ellipse"))
        # The frame is not capped at 1: DUTCH has a baseline at 1 - DEO = 0.997,
        # and a frame that stops exactly at 1 would clip half of its marker.
        x_lo, x_hi = axis_limits(fair_v, MIN_SPAN["FAIR"])
        lims[(ds, sp)] = ((x_lo, x_hi),
                          axis_limits(acc_v, MIN_SPAN["ACC"]))

    csv_rows = []
    collapse_log = []
    for ds, sp, pdfed, fr in panels:
        collapsed = is_collapsed(fr)
        mostly = (not collapsed) and is_mostly_collapsed(fr)
        if collapsed:
            collapse_log.append((ds, sp, "collapsed", fr[0][6]))
        elif mostly:
            collapse_log.append((ds, sp, "mostly collapsed", fr[0][6]))

        ref_fps = {p[6] for p in pdfed} if len(pdfed) > 1 else set()
        bits, acc, fair, asd, dsd, host, fp = pdfed[0]
        on_grid = bits == "11111"
        csv_rows.append([ds, sp, "PDFFed", "comb" if on_grid else "default",
                         bits if on_grid else "", acc, asd, 1.0 - fair, dsd, "",
                         host, fp,
                         "all_terms_on" if on_grid else "default_config"])
        for lam, acc, fair, asd, dsd, host, fp in fr:
            note_csv = ("lambda_collapsed" if collapsed
                        else "lambda_mostly_collapsed" if mostly else "")
            if fp not in ref_fps and ref_fps:
                note_csv = ";".join(x for x in (note_csv, "off_sweep_fp") if x)
            csv_rows.append([ds, sp, "FedRenyi", "fed_renyi_lambda", lam, acc, asd,
                             1.0 - fair, dsd, "", host, fp, note_csv])
        for name, acc, fair, asd, fsd in base_all[(ds, sp)]:
            csv_rows.append([ds, sp, name, "single", "", acc, asd, 1.0 - fair,
                             fsd, "", "192.168.1.31", "", ""])

    if layout == "row":
        # The row is a paper figure at 3.4 inches per panel; six of them would be
        # 20 inches wide, so it stays at the four grids it was composed for.
        row_panels = [p for p in panels if (p[0], p[1]) in ROW_GRIDS]
        n = len(row_panels)
        fig, axes = plt.subplots(1, n, figsize=(3.42 * n, 3.35))
        if n == 1:
            axes = [axes]
        for i, (ax, (ds, sp, _rep, fr)) in enumerate(zip(axes, row_panels)):
            levels = (ROW_HM_LEVELS[metric].get((ds, sp), ())
                      if style_name == "curve" else ())
            draw_panel(ax, ds, sp, knobs_all[(ds, sp)], fr, base_all[(ds, sp)],
                       bl_style, lims[(ds, sp)], metric, style_name,
                       show_ylabel=(i == 0), levels=levels,
                       title_size=10.5, label_size=9.0, tick_size=8.0,
                       cloud=cloud)
            if i:
                ax.set_ylabel("")
        handles, labels = reorder_legend(axes[0])
        fig.legend(handles, labels, fontsize=7.0, loc="lower center",
                   bbox_to_anchor=(0.5, 0.004), ncol=len(handles),
                   handletextpad=0.3, columnspacing=0.9, borderpad=0.35,
                   frameon=True, fancybox=True, facecolor="white",
                   edgecolor="0.78", framealpha=0.9)
        fig.subplots_adjust(left=0.055, right=0.983, top=0.90, bottom=0.225,
                            wspace=0.14)
        stem = "pareto_row_" + metric + cloud_tag + fig_variants.suffix()
        png = os.path.join(OUT_DIR, stem + ".png")
        pdf = os.path.join(OUT_DIR, stem + ".pdf")
        fig.savefig(png, dpi=300)
        fig.savefig(pdf)
        plt.close(fig)
        print("wrote", png)
        print("wrote", pdf)
    else:
        for ds, sp, _rep, fr in panels:
            fig, ax = plt.subplots(figsize=(5.9, 4.95))
            levels = (HM_LEVELS[metric].get((ds, sp))
                      if style_name == "curve" else ())
            draw_panel(ax, ds, sp, knobs_all[(ds, sp)], fr, base_all[(ds, sp)],
                       bl_style, lims[(ds, sp)], metric, style_name,
                       levels=levels, cloud=cloud)
            # Two legend rows: PDFFed and FedRenyi share the first column, the
            # eleven baselines fill the rest.  The anchor is in figure
            # coordinates so the block is centred on the page rather than on the
            # axes, which sits slightly right of centre because of the wide left
            # margin.
            ax.legend(*reorder_legend(ax), fontsize=6.5, frameon=False,
                      loc="upper center", bbox_to_anchor=(0.5, 0.083),
                      bbox_transform=fig.transFigure, ncol=7,
                      handletextpad=0.35, columnspacing=0.8)
            fig.subplots_adjust(left=0.12, right=0.975, top=0.934, bottom=0.19)

            stem = (("pareto_curve_%s_%s" if style_name == "curve" else
                     "pareto_sd_%s_%s" if style_name == "ellipse" else
                     "pareto_%s_%s") % (ds, sp))
            if metric != "eo":
                stem += "_" + metric
            stem += cloud_tag + fig_variants.suffix()
            png = os.path.join(OUT_DIR, stem + ".png")
            pdf = os.path.join(OUT_DIR, stem + ".pdf")
            fig.savefig(png, dpi=300)
            fig.savefig(pdf)
            plt.close(fig)
            print("wrote", png)
            print("wrote", pdf)

    # The long table is the canonical dump of the whole plane, so the variant
    # runs that hide a method must not overwrite it with a filtered copy.
    if not fig_variants.hidden():
        with open(csv_out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset", "split", "algorithm", "knob", "value",
                        "ACC_mean", "ACC_std", fair_spec["csv"] + "_mean",
                        fair_spec["csv"] + "_std", "other_knob",
                        "server", "fingerprint", "note"])
            w.writerows(csv_rows)
        print("wrote", csv_out, "(%d rows)" % len(csv_rows))

    if collapse_log:
        print("\nFedRenyi lambda insensitive (no Pareto structure):")
        for ds, sp, kind, fp in collapse_log:
            print("  %-18s %-16s fp=%s" % ("%s/%s" % (ds, sp), kind, fp))
    # ``DUTCH/Uniform`` was frozen at four lambdas (0.25, 0.5, 2, 4) after the
    # lambda=1 run was abandoned at round 11/500 and purged from both disks, so
    # the five-lambda check must not report it as an unfinished gap.  Re-running
    # it would contradict decision D22 in ``实验决策记录.md``; the target count is
    # spelled out here so a later inspection does not mistake the shortfall for
    # lost work.
    frozen_lambdas = {("DUTCH", "Uniform"): 4}
    missing = []
    for ds, sp in GRIDS:
        want = frozen_lambdas.get((ds, sp), 5)
        have = len([1 for r in rows if r["dataset"] == ds and r["split"] == sp
                    and r["algo_dir"].startswith("FedRenyi_lam") and r["k_clients_ok"]])
        if have < want:
            missing.append((ds, sp))
    if missing:
        print("\nFedRenyi lambda still short of its target count on:",
              ", ".join("%s/%s" % m for m in missing))
    finalised = ["%s/%s=%d" % (ds, sp, n) for (ds, sp), n in frozen_lambdas.items()
                 if (ds, sp) not in missing]
    if finalised:
        print("FedRenyi lambda ladder finalised by decision D22 (not a gap):",
              ", ".join(finalised))


if __name__ == "__main__":
    main()
