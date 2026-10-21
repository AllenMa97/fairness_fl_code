#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract the PDFFed lambda-sensitivity sweep from server result payloads.

Reads the raw ``result_path`` trees pulled back from the two compute servers
(``_sync/lambda_20261021/from_30`` and ``.../from_31``) and flattens every
aggregate result json into one tidy CSV plus one JSON index.

The sweep covers, on COMPAS/DRUG x Dirichlet1/Uniform:

* ``lambda_conf`` in {0, 0.5, 1, 2, 5}   (tags noCONF / confX05 / <baseline> / confX2 / confX5)
* ``lambda_eo``   in {0, 0.5, 1, 2, 5}   (tags noEO   / eoX05   / <baseline> / eoX2   / eoX5)
* loss-term ablations noPA / noL2L / noL2G / noG2L
* the FedRenyi ``lambda`` reference sweep (0.25 / 0.5 / 1 / 2 / 4)

Variant directories that were written before the ``variant_tag`` normalization
fix carry a duplicated ``PDFFed_PDFFed_`` prefix; both spellings collapse to the
same canonical label here so the sweep stays contiguous.

Usage::

    python analysis/extract_lambda_sensitivity.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAYLOAD_ROOT = os.path.join(REPO_ROOT, "_sync", "lambda_20261021")
OUT_DIR = os.path.join(REPO_ROOT, "reported_results", "lambda_sensitivity")

# Server payload dirs are named from_30 / from_31; the suffix is the server host.
SERVER_OF_DIR = {"from_30": "192.168.1.30", "from_31": "192.168.1.31"}

# Weight keys whose values define a sweep point. Recorded verbatim when present.
WEIGHT_KEYS = (
    "w_pa", "w_conf", "w_l2l", "w_l2g", "w_g2l",
    "lambda_eo", "fed_renyi_lambda", "FedRenyi_\u03bb", "dp_epsilon",
)

METRIC_KEYS = ("ACC", "ACC_gap", "DEO", "FR", "HM", "SPD")

# algo_dir -> (sweep name, sweep value). ``None`` value means "read from config".
TAG_SWEEP = {
    "noCONF": ("lambda_conf", 0.0),
    "confX05": ("lambda_conf", 0.5),
    "confX2": ("lambda_conf", 2.0),
    "confX5": ("lambda_conf", 5.0),
    "noEO": ("lambda_eo", 0.0),
    "eoX05": ("lambda_eo", 0.5),
    "eoX2": ("lambda_eo", 2.0),
    "eoX5": ("lambda_eo", 5.0),
    "noPA": ("ablation", "noPA"),
    "noL2L": ("ablation", "noL2L"),
    "noL2G": ("ablation", "noL2G"),
    "noG2L": ("ablation", "noG2L"),
}

FEDRENYI_RE = re.compile(r"^FedRenyi_lam(\d+)p?(\d*)$")
COMB_RE = re.compile(r"^comb([01]{5})$")


def canonical_algo_dir(algo_dir: str) -> str:
    """Collapse the pre-fix duplicated ``<algo>_<algo>_<tag>`` spelling."""
    m = re.match(r"^(?P<a>[A-Za-z0-9]+)_(?P=a)_(?P<tag>.+)$", algo_dir)
    return f"{m.group('a')}_{m.group('tag')}" if m else algo_dir


def classify(algo_dir: str):
    """Return (algorithm, variant, sweep, sweep_value) for an algo directory."""
    canon = canonical_algo_dir(algo_dir)

    if canon == "PDFFed":
        return "PDFFed", "baseline", "baseline", 1.0
    if canon == "PDFFed_DP":
        return "PDFFed", "DP", "dp_variant", None

    m = FEDRENYI_RE.match(canon)
    if m:
        whole, frac = m.group(1), m.group(2)
        val = float(f"{whole}.{frac}") if frac else float(whole)
        return "FedRenyi", canon, "fedrenyi_lambda", val

    if canon.startswith("PDFFed_"):
        tag = canon[len("PDFFed_"):]
        mc = COMB_RE.match(tag)
        if mc:
            return "PDFFed", tag, "ablation_combo", tag
        if tag in TAG_SWEEP:
            sweep, val = TAG_SWEEP[tag]
            return "PDFFed", tag, sweep, val
        return "PDFFed", tag, "other", None

    return canon, canon, "other", None


def iter_payload_json():
    """Yield (server, payload_base, json_path) for every result json on disk."""
    if not os.path.isdir(PAYLOAD_ROOT):
        return
    for payload_dir in sorted(os.listdir(PAYLOAD_ROOT)):
        base = os.path.join(PAYLOAD_ROOT, payload_dir)
        rp = os.path.join(base, "result_path")
        if not os.path.isdir(rp):
            continue
        server = SERVER_OF_DIR.get(payload_dir, payload_dir)
        for root, _dirs, files in os.walk(rp):
            for name in sorted(files):
                if name.endswith(".json"):
                    yield server, base, os.path.join(root, name)


def flatten(server: str, base: str, path: str):
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)

    cfg = doc.get("experiment_config", {}) or {}
    agg = doc.get("aggregate", {}) or {}
    metrics = agg.get("metrics", {}) or {}

    rel = os.path.relpath(path, base)
    parts = rel.replace("\\", "/").split("/")
    # result_path/<dataset>/<split>/<algo_dir>/<hypothesis>/<K>Clients/<file>.json
    parts = [p for p in parts if p != "result_path"]
    if len(parts) < 5:
        return None
    dataset, split, algo_dir, hypothesis, kdir = parts[0], parts[1], parts[2], parts[3], parts[4]
    algorithm, variant, sweep, sweep_value = classify(algo_dir)

    row = {
        "server": server,
        "dataset": dataset,
        "split": split,
        "hypothesis": hypothesis,
        "num_clients_K": int(re.sub(r"\D", "", kdir) or 0),
        "algo_dir_raw": algo_dir,
        "algo_dir_canonical": canonical_algo_dir(algo_dir),
        "algorithm": algorithm,
        "variant": variant,
        "sweep": sweep,
        "sweep_value": "" if sweep_value is None else sweep_value,
        "n_repeats": len(agg.get("repeat_indices", []) or []),
        "repeat_seeds": ",".join(str(s) for s in (agg.get("repeat_seeds", []) or [])),
        "source_path": rel.replace("\\", "/"),
        "git_commit": (doc.get("code_version", {}) or {}).get("git_commit", ""),
        "recorded_at": (doc.get("code_version", {}) or {}).get("recorded_at", ""),
    }

    for key in WEIGHT_KEYS:
        row[f"cfg_{key}"] = cfg.get(key, "")

    for mk in METRIC_KEYS:
        blk = metrics.get(mk, {}) or {}
        row[f"{mk}_mean"] = blk.get("mean", "")
        row[f"{mk}_std"] = blk.get("std", "")

    gpu = agg.get("gpu_seconds", []) or []
    row["gpu_seconds_mean"] = (sum(gpu) / len(gpu)) if gpu else ""
    comm = agg.get("communication_cost", []) or []
    row["comm_cost_mean"] = (sum(comm) / len(comm)) if comm else ""

    return row


def main() -> int:
    rows = [r for r in (flatten(*p) for p in iter_payload_json()) if r]
    if not rows:
        print(f"[ERROR] no payload json found under {PAYLOAD_ROOT}", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: (r["dataset"], r["split"], r["sweep"],
                             str(r["sweep_value"]), r["algorithm"], r["variant"],
                             r["num_clients_K"]))

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "lambda_sensitivity.csv")
    json_path = os.path.join(OUT_DIR, "lambda_sensitivity.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"cells": rows, "n_cells": len(rows)}, fh, ensure_ascii=False, indent=2)

    print(f"[OK] {len(rows)} cells -> {os.path.relpath(csv_path, REPO_ROOT)}")
    print(f"[OK] {len(rows)} cells -> {os.path.relpath(json_path, REPO_ROOT)}")

    # Compact console summary: sweep coverage per dataset/split.
    cov = {}
    for r in rows:
        cov.setdefault((r["dataset"], r["split"]), {}).setdefault(r["sweep"], set()).add(str(r["variant"]))
    print("\n=== SWEEP COVERAGE (dataset/split -> sweep: variants) ===")
    for (ds, sp), sweeps in sorted(cov.items()):
        print(f"{ds}/{sp}")
        for sw, variants in sorted(sweeps.items()):
            print(f"    {sw:18s} {len(variants)}  {','.join(sorted(variants))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
