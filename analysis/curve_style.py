#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared visual language for the per-round curve figures.

``plot_communication.py``, ``plot_per_group.py`` and ``plot_hm.py`` all show the
same eight algorithms on a communication-round axis, so they must not drift
apart in colour, marker or line weight.  This module holds that one palette and
the ``fig_variants`` switch that removes PDFFed-DP for the appendix copies.

The eight algorithms are the ones that actually log a global test trajectory on
every dataset; the remaining eight baselines only appear in the point figures
(``plot_table.py``, ``plot_audio.py``) because their runs are single-cell.
"""

import fig_variants

CURVES = fig_variants.active(
    ("PDFFed", "PDFFed_DP", "FedRenyi", "FedAvg", "FedProx", "FairFed",
     "mFairFL", "FL_FairBatch"))

COLORS = {
    "PDFFed": "tab:blue",
    "PDFFed_DP": "tab:orange",
    "FedRenyi": "tab:red",
    "FedAvg": "#7f7f7f",
    "FedProx": "#8c564b",
    "FairFed": "#9467bd",
    "mFairFL": "#17becf",
    "FL_FairBatch": "#bcbd22",
}
MARKERS = {
    "PDFFed": "*", "PDFFed_DP": "o", "FedRenyi": "^", "FedAvg": "s",
    "FedProx": "D", "FairFed": "v", "mFairFL": "P", "FL_FairBatch": "X",
}
LINEWIDTH = {"PDFFed": 2.6}
LINESTYLE = {"PDFFed_DP": (0, (4, 2))}
LABELS = {
    "PDFFed": "PDFFed", "PDFFed_DP": "PDFFed-DP", "FedRenyi": "FedRenyi",
    "FedAvg": "FedAvg", "FedProx": "FedProx", "FairFed": "FairFed",
    "mFairFL": "mFairFL", "FL_FairBatch": "FL-FairBatch",
}
# PDFFed and PDFFed-DP test every round (500 points); the six baselines inside
# CURVES test every 10 rounds.  Markers are thinned on the dense curves only.
DENSE = ("PDFFed", "PDFFed_DP")


def draw_curve(ax, xs, ys, algo, markevery_dense=25):
    """Draw one algorithm's round trajectory in the shared style."""
    color = COLORS[algo]
    dense = algo in DENSE
    ax.plot(xs, ys, color=color, lw=LINEWIDTH.get(algo, 1.4),
            ls=LINESTYLE.get(algo, "-"), marker=MARKERS[algo],
            markersize=6 if algo != "PDFFed" else 7,
            markevery=markevery_dense if dense else 1,
            markerfacecolor="white" if algo != "PDFFed" else color,
            markeredgecolor=color, markeredgewidth=1.1, label=LABELS[algo],
            zorder=3)
