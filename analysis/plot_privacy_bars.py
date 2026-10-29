#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw the property-inference attack results, one figure per heterogeneity level.

``analysis/privacy/run_attack.py`` used to render all fifteen upload channels for
both Dirichlet regimes into a single 2 x 2 panel.  At that width the channel
labels overlapped and no bar could be compared against its neighbour.  This
module draws four figures instead -- one per (regime, attack readout) -- with
the channels on the y axis, so every label is horizontal and every value is
annotated.  MAE and AUC got their own figure too: two panels side by side left
each bar only half a page width.

The bar order is the semantic one of the experiment: from "no information" and
the DP-defended uploads, through the prototype channels, to the raw-data oracle
that bounds the attack.  ``run_attack.py`` imports ``BAR_ORDER`` and
``make_split_figures`` from here so the pipeline and a standalone redraw of
``privacy_attack_results.csv`` cannot drift apart.

Usage::

    python analysis/plot_privacy_bars.py
"""

from __future__ import annotations

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_IN = os.path.join(REPO_ROOT, "analysis", "privacy", "privacy_attack_results.csv")
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "privacy")

# Channel order: no-information and DP-defended uploads first, the raw-data
# oracle last.  The colour ramp runs from grey (no leakage) through the
# prototype family to red (explicit group statistics) and black (raw data).
BAR_ORDER = [
    ("no_info_mean", "No info\n(constant)", "#aaaaaa"),
    ("proto_shared_dp_3.0", "Shared+DP\nsigma=3.0", "#e7f1fa"),
    ("proto_shared_dp_1.0", "Shared+DP\nsigma=1.0", "#b8d4ea"),
    ("proto_shared_dp_0.5", "Shared+DP\nsigma=0.5", "#6baed6"),
    ("prototypes_shared_padded", "Prototype\nshared+padded", "#2171b5"),
    ("prototypes_shared", "Prototype\nshared zero-fill", "#4292c6"),
    ("prototypes_padded", "Prototype\nlocal+padded", "#74add1"),
    ("prototypes", "Prototype\nlocal zero-fill", "#9ecae1"),
    ("class_prototypes_shared", "Class proto shared\n(FedProto)", "#a1d99b"),
    ("class_prototypes", "Class proto local\n(FedProto)", "#41ab5d"),
    ("loGo_fairness_stats_shared", "Fairness stats shared\n(LoGoFair)", "#fdd0a2"),
    ("loGo_fairness_stats", "Fairness stats local\n(LoGoFair)", "#fd8d3c"),
    ("fedrenyi_group_props", "Group props\n(FedRenyi)", "#ef6548"),
    ("group_stats", "Group stats\n(FedFB/FairFed)", "#d62728"),
    ("raw_oracle", "Raw data\n(upper bound)", "#444444"),
]

REGIME_TITLE = {"0.1": r"Dirichlet $\alpha=0.1$ (extreme group skew)",
                "1.0": r"Dirichlet $\alpha=1.0$ (moderate group skew)"}
REGIME_FILE = {"0.1": "alpha0p1", "1.0": "alpha1p0"}


def make_split_figures(rows, out_dir=OUT_DIR, prefix="privacy_attack_bars"):
    """One figure per (regime, metric): fifteen horizontal bars, nothing else.

    ``rows`` are dicts carrying ``regime``, ``method``, ``mae_mean``, ``mae_std``,
    ``auc_mean`` and ``auc_std``.  Channels missing from a regime are skipped
    rather than drawn as a zero bar, which would read as "no leakage".
    """
    by = {}
    for r in rows:
        try:
            by[(str(r["regime"]), r["method"])] = r
        except KeyError:
            continue

    written = []
    for regime in sorted({k[0] for k in by}):
        methods = [o for o in BAR_ORDER if (regime, o[0]) in by]
        if not methods:
            continue
        for mkey, skey, mlabel, refline in [
                ("mae_mean", "mae_std", "Attack MAE on protected-group share",
                 None),
                ("auc_mean", "auc_std",
                 "Attack AUC: 'African-American majority?'", 0.5)]:
            labels = [o[1] for o in methods]
            colors = [o[2] for o in methods]
            ys = np.arange(len(methods))[::-1]
            vals = [float(by[(regime, o[0])][mkey]) for o in methods]
            errs = [float(by[(regime, o[0])].get(skey) or 0.0) for o in methods]

            fig, ax = plt.subplots(figsize=(8.4, 7.0))
            ax.barh(ys, vals, xerr=errs, color=colors, edgecolor="black",
                    linewidth=0.5, capsize=3, error_kw=dict(lw=0.8))
            span = max(v + e for v, e in zip(vals, errs))
            for y, v, e in zip(ys, vals, errs):
                ax.text(v + e + 0.012 * span, y, "%.3f" % v, va="center",
                        fontsize=8.5, color="#333333")
            if refline is not None:
                ax.axvline(refline, ls="--", c="gray", lw=1)
                ax.text(refline, -0.9, "chance", fontsize=8, color="gray",
                        ha="center", va="bottom")
            ax.set_yticks(ys)
            ax.set_yticklabels(labels, fontsize=9.5)
            ax.set_xlim(0, span * 1.16)
            ax.set_ylim(-1.4, len(methods) - 0.4)
            ax.set_xlabel(mlabel + ("  (lower = less leakage)"
                                    if mkey == "mae_mean"
                                    else "  (higher = more leakage)"), fontsize=10)
            ax.grid(True, axis="x", alpha=0.25, lw=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", labelsize=9)
            ax.set_title("Property inference attack -- %s\nCOMPAS, "
                         "honest-but-curious server, K=20"
                         % REGIME_TITLE.get(regime, regime), fontsize=11)
            fig.subplots_adjust(left=0.245, right=0.97, top=0.88, bottom=0.09)
            stem = "%s_%s_%s" % (prefix,
                                 REGIME_FILE.get(regime, "alpha" + regime),
                                 "mae" if mkey == "mae_mean" else "auc")
            png = os.path.join(out_dir, stem + ".png")
            fig.savefig(png, dpi=300)
            plt.close(fig)
            written.append(png)
            print("wrote", png)
    return written


def load_rows(path=CSV_IN):
    rows = []
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            try:
                float(r["mae_mean"])
                float(r["auc_mean"])
            except (KeyError, ValueError, TypeError):
                continue
            rows.append(r)
    return rows


def main():
    if not os.path.isfile(CSV_IN):
        sys.exit("missing " + CSV_IN)
    make_split_figures(load_rows())


if __name__ == "__main__":
    main()
