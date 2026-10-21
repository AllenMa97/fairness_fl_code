#!/usr/bin/env python3
"""Majority-class collapse audit for the tabular benchmark (appendix evidence).

Reads the per-cell inventory pulled from .30 / .31 into
``_sync/collapse_audit_20261021/cells_<host>.tsv`` (gitignored payload, produced
by the server-side walker) and joins it against the majority-class floor of each
test split. A cell is called *collapsed* when its aggregated accuracy does not
beat ``floor + 1pp`` -- i.e. the model is no better than always predicting the
majority class, which also drives DEO to ~0 and makes the fairness numbers
meaningless.

Outputs (tracked, under reported_results/):
    collapse_audit.csv    one row per unique cell
    collapse_audit.json   same rows + per-dataset / per-algorithm summary

Floor provenance: ``partition_cache/v2/<DS>/*/metadata.json`` on both servers,
reached through each result json's ``aggregate.partition_fingerprints``. Summing
the per-client ``test_stats.label_counts`` gives the label distribution the runs
actually evaluated on (2026-10-21). This matters for DUTCH: the git-committed
``DUTCH.pickle`` (5819/6265, floor 0.518454) matches no cached version and was
never used by any run, while all 212 DUTCH results are bound to the 5772/6312
version (floor 0.522344). See appendix S9.3.

Usage:
    python analysis/extract_collapse_audit.py
"""
from __future__ import annotations

import csv
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAYLOAD_DIR = os.path.join(REPO_ROOT, "_sync", "collapse_audit_20261021")
OUT_DIR = os.path.join(REPO_ROOT, "reported_results", "collapse_audit")

# dataset -> (n_test, majority_class, majority_n). floor = majority_n / n_test
#
# Authoritative source: `partition_cache/v2/<DS>/*/metadata.json` on both servers.
# Every result json records `aggregate.partition_fingerprints`, and each
# fingerprint's metadata carries `ordered_labels_sha256` /
# `ordered_protected_sha256` plus per-client `test_stats.label_counts`. Summing
# those counts gives the test-set label distribution that the run actually
# evaluated on, so the floor below is bound to the data the numbers were
# produced from rather than to whatever blob happens to sit in git.
#
# For ADULT / BANK / DRUG / COMPAS every cached version shares one
# `ordered_labels_sha256` and one `ordered_protected_sha256`, and both match the
# committed pickle -- the versions differ only in train-side sample order.
# DUTCH is the exception: all 212 DUTCH results are bound to the single cached
# version whose test labels are 5772/6312 (floor 0.522344), which matches the
# regenerated working-tree pickle. The git-committed DUTCH blob (5819/6265,
# floor 0.518454) matches NO cached version, i.e. no run ever used it, so it is
# stale and must not be used as the yardstick.
# See reported_results/collapse_audit/appendix_collapse_disclosure.md S9.
FLOORS = {
    "ADULT": (16281, "0", 12435),
    "BANK": (13211, "0", 11594),
    "DRUG": (284, "1", 224),
    "COMPAS": (478, "1", 258),
    "DUTCH": (12084, "1", 6312),
}

# A cell at or below floor + COLLAPSE_MARGIN is treated as a constant predictor.
COLLAPSE_MARGIN = 0.01

# Two copies of one cell agree when their ACC/DEO spread stays under this.
AGREE_TOL = 1e-5

# Result dirs name some algorithms differently from algorithm/*.py.
ALIAS = {
    "FedRenyi": "FederatedRenyi",
    "FedProto": "FederatedProto",
    "FedRep": "FederatedRep",
    "FedSum": "FederatedSum",
    "FedNova": "FederatedNova",
    "FedGen": "FederatedGen",
}

HOSTS = {"cells_30.tsv": "192.168.1.30", "cells_31.tsv": "192.168.1.31"}


def canonical_dir(algo_seg: str) -> str:
    """Fold the duplicated-prefix dirs (``PDFFed_PDFFed_x``) the old queue produced."""
    head, _, tail = algo_seg.partition("_")
    if tail.startswith(head + "_"):
        return canonical_dir(head + "_" + tail[len(head) + 1:])
    return algo_seg


def known_algorithms() -> set:
    d = os.path.join(REPO_ROOT, "algorithm")
    try:
        return {f[:-3] for f in os.listdir(d) if f.endswith(".py")}
    except OSError:
        return set()


def split_algorithm(algo_dir: str, known: set):
    """Longest known algorithm prefix wins, so PDFFed_DP stays distinct from PDFFed."""
    cands = [k for k in known if algo_dir == k or algo_dir.startswith(k + "_")]
    if cands:
        algo = max(cands, key=len)
    else:
        head = algo_dir.split("_")[0]
        algo = ALIAS.get(head, head)
    variant = algo_dir[len(algo):].lstrip("_")
    return algo, variant


def read_cells(known: set):
    rows = []
    for fname, host in HOSTS.items():
        path = os.path.join(PAYLOAD_DIR, fname)
        if not os.path.exists(path):
            print(f"[WARN] missing payload {os.path.relpath(path, REPO_ROOT)}", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as fh:
            for raw in csv.DictReader(fh, delimiter="\t"):
                if raw["dataset"] not in FLOORS or raw["ACC_mean"] == "":
                    continue
                algo_dir = canonical_dir(raw["algo_seg"])
                algo, variant = split_algorithm(algo_dir, known)
                n_test, maj_cls, maj_n = FLOORS[raw["dataset"]]
                floor = maj_n / n_test
                acc = float(raw["ACC_mean"])
                deo = float(raw["DEO_mean"]) if raw["DEO_mean"] != "" else None
                rows.append({
                    "dataset": raw["dataset"],
                    "split": raw["split"],
                    "algorithm": algo,
                    "variant": variant,
                    "algo_dir": algo_dir,
                    "hypothesis": raw["hypothesis"],
                    "num_clients_K": raw["K"],
                    "exp": raw["exp"],
                    "server": raw["server"] or host,
                    "n_repeats": raw["n_repeats"],
                    "ACC_mean": acc,
                    "ACC_std": raw["ACC_std"],
                    "DEO_mean": deo,
                    "ACC_gap_mean": raw["ACC_gap_mean"],
                    "FR_mean": raw["FR_mean"],
                    "HM_mean": raw["HM_mean"],
                    "balance_sampling": raw["balance_sampling"],
                    "git_commit": raw["git_commit"],
                    "recorded_at": raw["recorded_at"],
                    "n_test": n_test,
                    "majority_class": maj_cls,
                    "majority_n": maj_n,
                    "floor": floor,
                    "margin_over_floor": acc - floor,
                    # Sample-level excess over "always predict the majority class".
                    "excess_correct": round(acc * n_test) - maj_n,
                    "collapsed": acc <= floor + COLLAPSE_MARGIN,
                    "DEO_zero": bool(deo is not None and deo == 0.0),
                })
    return rows


def cell_key(r: dict) -> tuple:
    return (r["dataset"], r["split"], r["algo_dir"], r["hypothesis"], r["num_clients_K"])


def dedupe(rows):
    """One row per cell; most repeats then latest aggregation wins."""
    groups = defaultdict(list)
    for r in rows:
        groups[cell_key(r)].append(r)

    def spread(group, key):
        vals = [g[key] for g in group if g[key] is not None]
        return (max(vals) - min(vals)) if vals else 0.0

    out = []
    for group in groups.values():
        group.sort(key=lambda r: (int(r["n_repeats"] or 0), r["recorded_at"]), reverse=True)
        chosen = dict(group[0])
        chosen["n_copies"] = len(group)
        chosen["copy_servers"] = ",".join(sorted({g["server"] for g in group}))
        same_rep = len({g["n_repeats"] for g in group}) == 1
        chosen["copies_agree"] = "yes" if (
            same_rep and spread(group, "ACC_mean") <= AGREE_TOL
            and spread(group, "DEO_mean") <= AGREE_TOL) else "NO"
        out.append(chosen)
    return out


def summarize(rows):
    per_ds, per_algo = {}, defaultdict(lambda: defaultdict(list))
    for r in rows:
        per_ds.setdefault(r["dataset"], []).append(r)
        per_algo[r["dataset"]][r["algorithm"]].append(r)

    ds_sum = {}
    for ds, rs in per_ds.items():
        accs = [r["ACC_mean"] for r in rs]
        exc = [r["excess_correct"] for r in rs]
        ds_sum[ds] = {
            "n_cells": len(rs),
            "floor": round(rs[0]["floor"], 6),
            "n_test": rs[0]["n_test"],
            "best_ACC": round(max(accs), 6),
            "median_ACC": round(statistics.median(accs), 6),
            "worst_ACC": round(min(accs), 6),
            "best_margin_pp": round(100 * (max(accs) - rs[0]["floor"]), 3),
            "median_margin_pp": round(100 * (statistics.median(accs) - rs[0]["floor"]), 3),
            "collapsed": sum(1 for r in rs if r["collapsed"]),
            "escape": sum(1 for r in rs if not r["collapsed"]),
            "DEO_zero": sum(1 for r in rs if r["DEO_zero"]),
            "median_excess_correct": statistics.median(exc),
            "best_excess_correct": max(exc),
            "n_algorithms": len(per_algo[ds]),
        }

    algo_sum = {}
    for ds, algos in per_algo.items():
        for algo, rs in algos.items():
            best = max(rs, key=lambda r: r["ACC_mean"])
            algo_sum.setdefault(ds, {})[algo] = {
                "n_cells": len(rs),
                "best_ACC": round(best["ACC_mean"], 6),
                "best_margin_pp": round(100 * best["margin_over_floor"], 3),
                "best_excess_correct": best["excess_correct"],
                "collapsed": sum(1 for r in rs if r["collapsed"]),
                "escape": sum(1 for r in rs if not r["collapsed"]),
                "best_split": best["split"],
                "best_variant": best["variant"],
                "best_K": best["num_clients_K"],
            }
    return ds_sum, algo_sum


def main() -> int:
    known = known_algorithms()
    raw = read_cells(known)
    if not raw:
        print(f"[ERROR] no cells read from {PAYLOAD_DIR}", file=sys.stderr)
        return 1
    rows = dedupe(raw)
    rows.sort(key=lambda r: (r["dataset"], r["split"], r["algorithm"], r["variant"],
                             r["hypothesis"], int(r["num_clients_K"] or 0)))

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "collapse_audit.csv")
    json_path = os.path.join(OUT_DIR, "collapse_audit.json")

    fields = ["dataset", "split", "algorithm", "variant", "algo_dir", "hypothesis",
              "num_clients_K", "exp", "server", "n_copies", "copy_servers",
              "copies_agree", "n_repeats", "ACC_mean", "ACC_std", "DEO_mean",
              "ACC_gap_mean", "FR_mean", "HM_mean", "n_test", "majority_class",
              "majority_n", "floor", "margin_over_floor", "excess_correct",
              "collapsed", "DEO_zero", "balance_sampling", "git_commit",
              "recorded_at"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    ds_sum, algo_sum = summarize(rows)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({
            "collapse_margin": COLLAPSE_MARGIN,
            "floors": {k: {"n_test": v[0], "majority_class": v[1], "majority_n": v[2],
                           "floor": round(v[2] / v[0], 6)} for k, v in FLOORS.items()},
            "floor_provenance": "partition_cache/v2/<DS>/*/metadata.json test_stats.label_counts "
                                "on 192.168.1.30/.31, bound via aggregate.partition_fingerprints "
                                "(2026-10-21); DUTCH floor 0.522344 is the version all 212 runs used",
            "n_payload_rows": len(raw),
            "n_cells": len(rows),
            "per_dataset": ds_sum,
            "per_algorithm": algo_sum,
            "cells": rows,
        }, fh, ensure_ascii=False, indent=2)

    print(f"[OK] {len(raw)} payload rows -> {len(rows)} unique cells")
    print(f"[OK] {os.path.relpath(csv_path, REPO_ROOT)}")
    print(f"[OK] {os.path.relpath(json_path, REPO_ROOT)}")

    print("\n=== PER DATASET (collapsed = ACC <= floor + 1pp) ===")
    hdr = (f"{'ds':7} {'cells':>5} {'floor':>7} {'best':>7} {'median':>7} "
           f"{'bestPP':>7} {'medPP':>7} {'collapse':>9} {'escape':>7} {'DEO=0':>6} "
           f"{'medExc':>7} {'bestExc':>8} {'algos':>6}")
    print(hdr)
    for ds in ("ADULT", "BANK", "DRUG", "COMPAS", "DUTCH"):
        if ds not in ds_sum:
            continue
        s = ds_sum[ds]
        print(f"{ds:7} {s['n_cells']:>5} {s['floor']:>7.4f} {s['best_ACC']:>7.4f} "
              f"{s['median_ACC']:>7.4f} {s['best_margin_pp']:>+7.2f} "
              f"{s['median_margin_pp']:>+7.2f} {s['collapsed']:>9} {s['escape']:>7} "
              f"{s['DEO_zero']:>6} {s['median_excess_correct']:>7.0f} "
              f"{s['best_excess_correct']:>8} {s['n_algorithms']:>6}")

    print("\n=== BEST MARGIN PER ALGORITHM (pp over floor) ===")
    for ds in ("ADULT", "BANK", "DRUG", "COMPAS", "DUTCH"):
        if ds not in algo_sum:
            continue
        items = sorted(algo_sum[ds].items(), key=lambda kv: -kv[1]["best_margin_pp"])
        print(f"-- {ds} (floor {ds_sum[ds]['floor']:.4f}, {ds_sum[ds]['n_algorithms']} algorithms)")
        for algo, a in items:
            print(f"     {algo:22} n={a['n_cells']:>3} best={a['best_ACC']:.4f} "
                  f"({a['best_margin_pp']:+.2f}pp, {a['best_excess_correct']:+d} samples) "
                  f"escape={a['escape']}/{a['n_cells']} @{a['best_split']}/{a['best_variant'] or '-'}/K{a['best_K']}")

    disagree = [r for r in rows if r["copies_agree"] == "NO"]
    print(f"\n[INFO] cells present on both servers: "
          f"{sum(1 for r in rows if r['n_copies'] > 1)}; disagreeing: {len(disagree)}")
    bs = Counter(r["balance_sampling"] or "(not recorded)" for r in rows)
    print(f"[INFO] balance_sampling values: {dict(bs)}")
    partial = sum(1 for r in rows if int(r["n_repeats"] or 0) < 3)
    print(f"[INFO] cells with fewer than 3 repeats: {partial}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
