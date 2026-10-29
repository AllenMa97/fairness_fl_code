#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw the differential-privacy privacy-utility trade-off of PDFFed_DP.

Reads ``analysis/privacy/privacy_attack_results.csv`` (produced by
``analysis/privacy/run_attack.py``) and isolates the DP-noise channel:
the shared+padded prototype (the strongest, most favourable-to-the-attacker
setting) perturbed with Gaussian noise of scale sigma = 0.5x / 1.0x / 3.0x the
per-dimension prototype std, versus the no-noise baseline (sigma = 0).

Two regimes are shown (columns): Dirichlet alpha=0.1 (extreme group skew) and
alpha=1.0 (moderate skew).  The top row plots the property-inference attack
AUC -- the lower it drops toward the random 0.5 line, the more private the
upload.  The bottom row plots utility retention: ``util_disc`` (cosine of the
aggregate class-discriminative direction, the quantity the fairness mechanism
actually consumes) and ``util_selfcos`` (per-prototype fidelity, the main price
of noise).  Together they show noise monotonically suppresses leakage while the
aggregate direction used by the fairness mechanism stays essentially intact.

Usage::

    python analysis/plot_privacy_dp.py
"""

from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(REPO_ROOT, "analysis", "privacy",
                        "privacy_attack_results.csv")
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "figures")

# Noise strength axis: sigma = {0 (no noise), 0.5, 1.0, 3.0} x prototype std.
SIGMA_X = {"0": 0, "0.5": 1, "1.0": 2, "3.0": 3}
SIGMA_TICK = ["0\n(no noise)", "0.5", "1.0", "3.0"]
REGIMES = [("0.1", r"Dirichlet $\alpha{=}0.1$ (extreme skew)"),
           ("1.0", r"Dirichlet $\alpha{=}1.0$ (moderate skew)")]


def load():
    """Return {(regime, sigma): {auc, mae, disc, selfcos, aggcorr}}."""
    rows = {}
    with open(CSV_PATH, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            m = r["method"]
            if m == "prototypes_shared_padded":
                sig = "0"
            elif m.startswith("proto_shared_dp_"):
                sig = m[len("proto_shared_dp_"):]
            else:
                continue
            if sig not in SIGMA_X:
                continue
            rows[(r["regime"], sig)] = {
                "auc": float(r["auc_mean"]),
                "mae": float(r["mae_mean"]),
                # sigma=0 has no utility columns; a no-noise upload is lossless.
                "disc": float(r["util_disc"]) if r["util_disc"] else 1.0,
                "selfcos": float(r["util_selfcos"]) if r["util_selfcos"] else 1.0,
            }
    return rows


def series(rows, regime, field):
    xs, ys = [], []
    for sig in ("0", "0.5", "1.0", "3.0"):
        key = (regime, sig)
        if key in rows:
            xs.append(SIGMA_X[sig])
            ys.append(rows[key][field])
    return xs, ys


def main():
    rows = load()
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.2), sharex=True)
    for j, (alpha, rtitle) in enumerate(REGIMES):
        ax_auc = axes[0][j]
        ax_util = axes[1][j]

        # --- top: attack AUC (lower = more private) ---
        xs, ys = series(rows, alpha, "auc")
        ax_auc.plot(xs, ys, color="tab:red", lw=2.0, marker="o", markersize=6,
                    markerfacecolor="white", markeredgecolor="tab:red",
                    markeredgewidth=1.3, label="Attack AUC", zorder=3)
        ax_auc.axhline(0.5, ls="--", c="gray", lw=1.0, zorder=2)
        ax_auc.text(3.05, 0.52, "random", fontsize=8, color="gray", va="bottom")
        ax_auc.set_ylabel("Property-inference AUC", fontsize=10)
        ax_auc.set_title(rtitle, fontsize=10.5)
        ax_auc.set_ylim(0.45, 1.05)
        ax_auc.grid(True, axis="y", alpha=0.25, lw=0.6)

        # --- bottom: utility retention (higher = better) ---
        xs_d, ys_d = series(rows, alpha, "disc")
        xs_s, ys_s = series(rows, alpha, "selfcos")
        ax_util.plot(xs_d, ys_d, color="tab:blue", lw=2.0, marker="^",
                     markersize=6, markerfacecolor="white",
                     markeredgecolor="tab:blue", markeredgewidth=1.3,
                     label="util_disc (aggregate direction)", zorder=3)
        ax_util.plot(xs_s, ys_s, color="tab:green", lw=2.0, marker="s",
                     markersize=6, markerfacecolor="white",
                     markeredgecolor="tab:green", markeredgewidth=1.3,
                     label="util_selfcos (per-prototype)", zorder=3)
        ax_util.set_ylabel("Utility retention", fontsize=10)
        ax_util.set_ylim(-0.05, 1.08)
        ax_util.grid(True, axis="y", alpha=0.25, lw=0.6)

        for ax in (ax_auc, ax_util):
            ax.set_xlim(-0.2, 3.2)
            ax.set_xticks([0, 1, 2, 3])
            ax.tick_params(labelsize=9)
        ax_util.set_xticklabels(SIGMA_TICK)
        ax_util.set_xlabel(r"Noise scale $\sigma$ (x prototype std)", fontsize=10)

    axes[0][0].legend(fontsize=7.5, frameon=False, loc="upper right",
                      handletextpad=0.4, columnspacing=0.9)
    axes[1][0].legend(fontsize=7.5, frameon=False, loc="center right",
                      handletextpad=0.4, columnspacing=0.9)
    fig.suptitle("PDFFed-DP privacy-utility trade-off under property-inference "
                 "attack (COMPAS)", fontsize=11.5, y=0.99)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12,
                        wspace=0.20, hspace=0.28)
    out_png = os.path.join(OUT_DIR, "privacy_dp_tradeoff.png")
    out_pdf = os.path.join(OUT_DIR, "privacy_dp_tradeoff.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print("wrote", out_png)
    print("wrote", out_pdf)


if __name__ == "__main__":
    main()
