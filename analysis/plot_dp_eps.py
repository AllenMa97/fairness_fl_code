#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw the differential-privacy budget sweep of PDFFed-DP.

Reads ``reported_results/modality_raw.tsv`` and picks every cell that carries
the ``PDFFed_DP_eps<eps>`` variants -- at the time of writing that is UTKFace /
Dirichlet05 / 20 clients with eps = 1, 8, 20, 50, 100, 1000 and 1e6 (the last
one is effectively no noise and is labelled ``no DP``).  The upper panel is
ACC, the lower panel DEO, both against eps on a log axis, with +/- 1 sd over
the repeats.

Two reference lines are drawn: the majority-class floor of the dataset
(``reported_results/two_machine_summary/dataset_floor_availability.csv``), and
FedAvg's value for the same cell.  The floor is what makes the figure readable:
UTKFace is flagged as a dataset-level failure in the collapse audit, and every
point here -- PDFFed-DP at any eps and FedAvg alike -- sits at chance.  The
sweep therefore shows that the noise channel is nearly free (DEO stays flat at
0.003-0.03 across five orders of magnitude of eps) but says nothing about
utility, because this corpus never leaves the floor.

Usage::

    python analysis/plot_dp_eps.py
"""

from __future__ import annotations

import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO_ROOT, "reported_results", "modality_raw.tsv")
FLOORS = os.path.join(REPO_ROOT, "reported_results", "two_machine_summary",
                      "dataset_floor_availability.csv")
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "figures")

EPS_ALGO = re.compile(r"^PDFFed_DP_eps(.+)$")
CLIENTS = "20"
REFERENCE_ALGO = "FedAvg"


def _f(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def load():
    cells = {}
    with open(RAW, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["clients"] != CLIENTS:
                continue
            hit = EPS_ALGO.match(row["algo"])
            if hit:
                try:
                    eps = float(hit.group(1))
                except ValueError:
                    continue
                key = (row["dataset"], row["split"])
                cells.setdefault(key, {"eps": {}, "reference": {}})
                cells[key]["eps"][eps] = row
            elif row["algo"] == REFERENCE_ALGO:
                key = (row["dataset"], row["split"])
                cells.setdefault(key, {"eps": {}, "reference": {}})
                cells[key]["reference"][row["algo"]] = row
    return {k: v for k, v in cells.items() if v["eps"]}


def floor_of(dataset):
    with open(FLOORS, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row["dataset"] == dataset:
                try:
                    return float(row["floor"])
                except (KeyError, ValueError):
                    return None
    return None


def main():
    cells = load()
    for (dataset, split), payload in sorted(cells.items()):
        eps_sorted = sorted(payload["eps"])
        acc = [_f(payload["eps"][e], "ACC") for e in eps_sorted]
        acc_sd = [_f(payload["eps"][e], "ACC_sd") for e in eps_sorted]
        deo = [_f(payload["eps"][e], "DEO") for e in eps_sorted]
        deo_sd = [_f(payload["eps"][e], "DEO_sd") for e in eps_sorted]

        fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True)
        for ax, values, errors, label in ((axes[0], acc, acc_sd, "ACC"),
                                          (axes[1], deo, deo_sd, "DEO")):
            ax.errorbar(eps_sorted, values, yerr=errors, marker="o",
                        color="tab:orange", lw=2.0, ms=6, capsize=3,
                        label="PDFFed-DP")
            reference = payload["reference"].get(REFERENCE_ALGO)
            if reference is not None:
                ref_val = _f(reference, "ACC" if label == "ACC" else "DEO")
                ax.axhline(ref_val, color="#7f7f7f", ls="--", lw=1.2,
                           label="%s (%s)" % (REFERENCE_ALGO, label))
            ax.set_ylabel(label, fontsize=11)
            ax.grid(True, alpha=0.25, lw=0.6)
            ax.tick_params(labelsize=9)
        floor = floor_of(dataset)
        if floor is not None:
            axes[0].axhline(floor, color="tab:red", ls=":", lw=1.2,
                            label="majority floor %.3f" % floor)
        axes[0].set_title("%s  %s  (%s clients)  --  DP budget sweep"
                          % (dataset, split, CLIENTS), fontsize=12)
        axes[1].set_xlabel(r"privacy budget $\epsilon$  (1e6 = no DP)", fontsize=10)
        axes[0].set_xscale("log")
        axes[0].legend(fontsize=7.5, frameon=False, loc="best")
        axes[1].legend(fontsize=7.5, frameon=False, loc="best")
        fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.10,
                            hspace=0.12)
        stem = "dp_eps_%s_%s" % (dataset, split)
        fig.savefig(os.path.join(OUT_DIR, stem + ".png"), dpi=300)
        fig.savefig(os.path.join(OUT_DIR, stem + ".pdf"))
        plt.close(fig)
        print("wrote", os.path.join(OUT_DIR, stem + ".png"))


if __name__ == "__main__":
    main()
