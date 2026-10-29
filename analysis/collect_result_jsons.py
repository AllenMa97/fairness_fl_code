#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect per-cell result jsons from the two training servers into one TSV.

The servers store one aggregated json per (dataset, split, algorithm, model,
client count) run under
``result_path/<dataset>/<split>/<algo>/<model>/<N>Clients/<k>.txt.json``.
Each json already aggregates the repeats, so a cell is identified by
(dataset, split, algo, clients); the same cell may exist on both machines and
in several ``k.txt.json`` files.

Two modes::

    # on a server: flatten one result_path tree
    python analysis/collect_result_jsons.py scan  /root/fl/result_path /root/fl/reported_results/modality.tsv

    # locally: merge the per-machine TSVs, newest run per cell wins
    python analysis/collect_result_jsons.py merge reported_results/modality_raw.tsv a.tsv b.tsv

The merge keeps, for every (dataset, split, algo, clients), the row with the
most repeats and then the latest ``recorded_at`` -- the same selection the
two-machine summary uses.

Usage note: this covers the image (CelebA / LFWA+ / UTKFace / FairFace) and
text (bios) modalities; the tabular cells come from ``hetero_raw.tsv`` and the
speech cells from ``aud_raw.tsv`` / ``audio_mnist_raw.tsv``.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys

FIELDS = ["dataset", "split", "algo", "hypo", "clients",
          "ACC", "ACC_sd", "DEO", "DEO_sd", "SPD", "SPD_sd",
          "HM", "HM_sd", "FR", "FR_sd",
          "n_rep", "commit", "dirty", "recorded_at", "host", "source"]

CLIENT_DIR = re.compile(r"^(\d+)Clients$")


def stat(metrics, key):
    entry = metrics.get(key, {})
    if not isinstance(entry, dict):
        return "", ""
    mean, sd = entry.get("mean"), entry.get("std")
    return ("%.10f" % mean if mean is not None else "",
            "%.10f" % sd if sd is not None else "")


def repeats(metrics):
    best = 0
    for entry in metrics.values():
        if isinstance(entry, dict):
            values = entry.get("values")
            if isinstance(values, list):
                best = max(best, len(values))
    return best


def scan(root, out):
    host = os.uname().nodename
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "*", "*", "*", "*", "*", "*.json"))):
        parts = path.split(os.sep)
        clients_dir = parts[-2]
        hit = CLIENT_DIR.match(clients_dir)
        if not hit:
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (ValueError, OSError):
            continue
        metrics = payload.get("aggregate", {}).get("metrics", {})
        if "ACC" not in metrics:
            continue
        row = {"dataset": parts[-6], "split": parts[-5], "algo": parts[-4],
               "hypo": parts[-3], "clients": hit.group(1)}
        for key in ("ACC", "DEO", "SPD", "HM", "FR"):
            mean, sd = stat(metrics, key)
            row[key] = mean
            row[key + "_sd"] = sd
        row["n_rep"] = repeats(metrics)
        version = payload.get("code_version", {})
        row["commit"] = version.get("git_commit", "")
        row["dirty"] = version.get("git_dirty", "")
        row["recorded_at"] = version.get("recorded_at", "")
        row["host"] = host
        row["source"] = os.path.basename(path)
        rows.append(row)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print("scanned %s -> %s (%d rows, host %s)" % (root, out, len(rows), host))


def merge(out, inputs):
    best = {}
    for name in inputs:
        with open(name, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                key = (row["dataset"], row["split"], row["algo"], row["clients"])
                rank = (int(row["n_rep"] or 0), row["recorded_at"], row["host"])
                if key not in best or rank > best[key][0]:
                    best[key] = (rank, row)
    rows = [row for _, row in best.values()]
    rows.sort(key=lambda r: (r["dataset"], r["split"], r["clients"], r["algo"]))
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print("merged %d inputs -> %s (%d cells)" % (len(inputs), out, len(rows)))


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    mode = sys.argv[1]
    if mode == "scan":
        scan(sys.argv[2], sys.argv[3])
    elif mode == "merge":
        merge(sys.argv[2], sys.argv[3:])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
