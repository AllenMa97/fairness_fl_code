#!/usr/bin/env python3
"""Image-CLF gap scheduler helper (CelebA / FairFace / LFWA+).

Subcommands
  scan            扫描本地指定数据集结果，输出 split|algo|K|complete|njson
  partition-init  按现有缺口与算法耗时生成两机静态二分清单 + 初始队列
  build-queues    运行时按本地/对端结果过滤，生成本机队列
  monitor         采样 60s GPU/CPU/内存，给出可用卡与槽位
  cleanup         停止后清理已完成格的 checkpoint（dry-run / apply）
  status          打印本地完成/缺口统计
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict

FL = "/root/fl"
DATASET = os.environ.get("IMG_DATASET", "LFWA+")
RESULT = os.path.join(FL, "result_path", DATASET)
SAVE = os.path.join(FL, "save_path", DATASET)
PARTITION = os.path.join(FL, os.environ.get("IMG_PARTITION", "lfwa_partition.json"))

SPLITS = ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"]
ALGOS = [
    "FairFed", "FedAvg", "FederatedAverageWithPo", "FedFair", "FedFB", "FedNova",
    "FedProto", "FedProx", "FedRenyi", "FedRep", "FL_FairBatch", "LoGoFair",
    "mFairFL", "PDFFed", "PDFFed_DP", "Scaffold",
]
KS = [20, 30, 40]

PY = "/root/miniconda3/envs/federated/bin/python"
MAIN = os.path.join(FL, "main_IMG_CLF.py")

EXTRA_ARGS = (
    "-task IMG_CLF -dataset %s -communication_round_I 50 -algorithm_epoch_T 2 "
    "-exp_repeat_times 3 -batch_size 256 -test_batch_size 256 "
    "-learning_rate 0.0003 -optimize_method sgd -test_eval_freq 5 "
    "-final_artifact_policy metrics_only -resume" % DATASET
)


def key_of(sp, al, k):
    return "%s|%s|%d" % (sp, al, k)


def parse_key(keystr):
    sp, al, k = keystr.split("|")
    return sp, al, int(k)


def universe():
    return [key_of(sp, al, k) for sp in SPLITS for al in ALGOS for k in KS]


def cell_dir(sp, al, k):
    return os.path.join(RESULT, sp, al, "BERTCLASSIFIER", "%dClients" % k)


def json_complete(path):
    """返回 (是否完整, gpu_seconds均值或0.0)；解析失败返回 (False, 0.0)。"""
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return False, 0.0
    try:
        vals = d["aggregate"]["metrics"]["ACC"]["values"]
        cfg = d.get("experiment_config") or {}
        rnd = cfg.get("communication_round_I", d.get("communication_round_I"))
        if isinstance(vals, list) and len(vals) >= 3 and rnd == 50:
            try:
                gs = d["aggregate"].get("gpu_seconds") or []
                gs = [float(x) for x in gs if isinstance(x, (int, float))]
                return True, (statistics.mean(gs) if gs else 0.0)
            except Exception:
                return True, 0.0
    except Exception:
        pass
    return False, 0.0


def scan_cells():
    """return {key: {"complete": bool, "njson": int, "gpu_seconds": float|None}}"""
    cells = {}
    for sp in SPLITS:
        for al in ALGOS:
            for k in KS:
                d = cell_dir(sp, al, k)
                files = []
                if os.path.isdir(d):
                    files = [f for f in os.listdir(d) if f.endswith(".json")]
                ok = False
                gs = None
                for f in files:
                    good, sec = json_complete(os.path.join(d, f))
                    if good:
                        ok = True
                        if not gs and sec:
                            gs = sec
                cells[key_of(sp, al, k)] = {
                    "complete": ok, "njson": len(files), "gpu_seconds": gs,
                }
    return cells


def load_scan(path):
    """Load scan output (split|algo|K|complete|njson)."""
    out = {}
    if not path or not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.count("|") < 4:
                continue
            p = ln.split("|")
            try:
                k = int(p[2])
                ok = p[3] == "1"
                n = int(p[4])
            except Exception:
                continue
            out[key_of(p[0], p[1], k)] = {"complete": ok, "njson": n, "gpu_seconds": None}
    return out


def cmd_scan(a):
    cells = scan_cells()
    lines = []
    for k in universe():
        c = cells[k]
        lines.append("%s|%d|%d" % (k, 1 if c["complete"] else 0, c["njson"]))
    text = "\n".join(lines) + "\n"
    if a.out == "-":
        sys.stdout.write(text)
    else:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)


def cmd_status(a):
    cells = scan_cells()
    gaps = [k for k in universe() if not cells[k]["complete"]]
    print("universe=%d complete=%d gap=%d" % (len(universe()), len(universe()) - len(gaps), len(gaps)))
    print("gap_by_K=%s" % dict(Counter(parse_key(k)[2] for k in gaps)))
    print("gap_by_split=%s" % dict(Counter(parse_key(k)[0] for k in gaps)))
    if a.verbose:
        for k in gaps:
            print("GAP %s json=%d" % (k, cells[k]["njson"]))


def load_partition(path):
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"cells": {}, "complete_at_init": []}


def algo_weights(cells):
    by_algo = defaultdict(list)
    for k, c in cells.items():
        if c["gpu_seconds"]:
            by_algo[parse_key(k)[1]].append(c["gpu_seconds"])
    med = {al: statistics.median(v) for al, v in by_algo.items() if v}
    fallback = statistics.median(list(med.values())) if med else 1800.0
    return med, fallback


def cell_weight(k, med, fallback):
    return med.get(parse_key(k)[1], fallback)


def queue_line(k):
    sp, al, kk = parse_key(k)
    return ("nice -n 10 %s %s %s -split_strategy %s -algorithm %s -num_clients_K %d"
            % (PY, MAIN, EXTRA_ARGS, sp, al, kk))


def cmd_partition_init(a):
    local = scan_cells()
    peer = load_scan(a.peer_scan)
    complete = {k for k in universe() if local.get(k, {}).get("complete") or peer.get(k, {}).get("complete")}
    gaps = [k for k in universe() if k not in complete]
    med, fallback = algo_weights(local)
    # LPT: 重格优先，贪心放入当前总权重较小的一侧
    gaps_sorted = sorted(gaps, key=lambda k: -cell_weight(k, med, fallback))
    bins = {"30": [], "31": []}
    load = {"30": 0.0, "31": 0.0}
    for k in gaps_sorted:
        h = "30" if load["30"] <= load["31"] else "31"
        bins[h].append(k)
        load[h] += cell_weight(k, med, fallback)
    assignment = {}
    for h, ks in bins.items():
        for k in ks:
            assignment[k] = h
    # 已完成格也登记（按哈希固定归属），便于日后出现意外缺口时仍有静态归属
    for k in universe():
        if k not in assignment:
            h = int(hashlib.sha1(k.encode("utf-8")).hexdigest(), 16) % 2
            assignment[k] = "30" if h == 0 else "31"
    manifest = {
        "version": 1,
        "generated_at": time.strftime("%F %T"),
        "cells": assignment,
        "complete_at_init": sorted(complete),
        "weights_median_gpu_seconds": med,
        "fallback_weight": fallback,
        "host_stats": {h: {"cells": len(bins[h]), "weight": round(load[h], 1)} for h in bins},
    }
    with open(a.partition_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1, sort_keys=True)
    for h, out in (("30", a.queue30_out), ("31", a.queue31_out)):
        ks = sorted(bins[h], key=lambda k: -cell_weight(k, med, fallback))
        with open(out, "w", encoding="utf-8") as f:
            for k in ks:
                f.write(queue_line(k) + "\n")
    print(json.dumps({
        "complete": len(complete), "gaps": len(gaps),
        "host30": len(bins["30"]), "host31": len(bins["31"]),
        "weight30": round(load["30"], 1), "weight31": round(load["31"], 1),
        "gap_by_K": dict(Counter(parse_key(k)[2] for k in gaps)),
    }, ensure_ascii=False))


def cmd_build_queues(a):
    local = scan_cells()
    peer = load_scan(a.peer_scan) if a.peer_scan else {}
    man = load_partition(a.partition)
    baked = set(man.get("complete_at_init") or [])
    cells = man.get("cells") or {}
    med = man.get("weights_median_gpu_seconds") or {}
    fallback = man.get("fallback_weight", 1800.0)
    def host_of(k):
        # 冻结分区表里已登记的格用登记归属；未登记的新格用哈希决定，
        # 保证两机对同一格得出同一归属（旧写法默认归本地，会让两机都认领同一格）。
        h = cells.get(k)
        if h:
            return h
        return "30" if int(hashlib.sha1(k.encode("utf-8")).hexdigest(), 16) % 2 == 0 else "31"

    assigned = [k for k in universe() if host_of(k) == a.host]
    q = []
    for k in assigned:
        if local.get(k, {}).get("complete"):
            continue
        if peer.get(k, {}).get("complete"):
            continue
        if k in baked:
            continue
        q.append(k)
    q.sort(key=lambda k: -cell_weight(k, med, fallback))
    with open(a.queue_out, "w", encoding="utf-8") as f:
        for k in q:
            f.write(queue_line(k) + "\n")
    union_gaps = [k for k in universe()
                  if not local.get(k, {}).get("complete")
                  and not peer.get(k, {}).get("complete")
                  and k not in baked]
    summary = {
        "host": a.host, "assigned": len(assigned), "queue": len(q),
        "local_complete": sum(1 for k in universe() if local.get(k, {}).get("complete")),
        "peer_scan": bool(peer), "baked_complete": len(baked),
        "union_gaps": len(union_gaps), "queue_by_K": dict(Counter(parse_key(k)[2] for k in q)),
    }
    with open(a.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False))


def _read_load():
    try:
        with open("/proc/loadavg") as f:
            return float(f.read().split()[0])
    except Exception:
        return 0.0


def _mem_available_mb():
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemAvailable:"):
                    return int(ln.split()[1]) // 1024
    except Exception:
        pass
    return 0


def cmd_monitor(a):
    cmd = ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
           "--format=csv,noheader,nounits", "-l", "1"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    per = defaultdict(lambda: {"mem": [], "util": [], "total": 0})
    loads, mems = [], []
    t0 = time.time()
    while time.time() - t0 < a.seconds:
        line = p.stdout.readline()
        if not line:
            time.sleep(0.2)
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx, mu, mt, ut = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        per[idx]["mem"].append(mu)
        per[idx]["util"].append(ut)
        per[idx]["total"] = mt
        loads.append(_read_load())
        mems.append(_mem_available_mb())
    p.terminate()
    try:
        p.wait(timeout=5)
    except Exception:
        p.kill()
    nproc = os.cpu_count() or 1
    load_mean = statistics.mean(loads) if loads else _read_load()
    load_max = max(loads) if loads else load_mean
    avail_min = min(mems) if mems else _mem_available_mb()
    gpus = []
    for idx in sorted(per):
        d = per[idx]
        used_mean = statistics.mean(d["mem"])
        used_peak = max(d["mem"])
        util_mean = statistics.mean(d["util"])
        util_peak = max(d["util"])
        free_min = d["total"] - used_peak
        idle = (used_peak <= a.max_used_mb and free_min >= a.min_free_mb
                and util_mean <= a.mean_util and util_peak <= a.peak_util)
        gpus.append({
            "index": idx, "mem_total": d["total"], "used_mean_mb": round(used_mean),
            "used_peak_mb": round(used_peak), "free_min_mb": round(free_min),
            "util_mean": round(util_mean, 1), "util_peak": round(util_peak, 1),
            "idle": idle,
        })
    # CPU/内存闸门：load1 均值不超过核数的 75%，可用内存 ≥ 64GiB
    cpu_ok = (load_mean <= a.max_load_ratio * nproc) and (avail_min >= a.min_avail_mb)
    idle_sorted = sorted([g for g in gpus if g["idle"]], key=lambda g: -g["free_min_mb"])
    slots = len(idle_sorted) if cpu_ok else 0
    if slots:
        headroom = max(0.0, 0.85 * nproc - load_mean)
        slots = min(slots, a.max_slots, max(1, int(headroom // a.omp_per_job)))
    chosen = [g["index"] for g in idle_sorted[:slots]]
    out = {
        "at": time.strftime("%F %T"), "seconds": a.seconds,
        "nproc": nproc, "load1_mean": round(load_mean, 2), "load1_max": round(load_max, 2),
        "mem_available_min_mb": avail_min, "cpu_ok": cpu_ok,
        "gpus": gpus, "chosen": chosen, "slots": slots,
        "thresholds": {
            "max_used_mb": a.max_used_mb, "min_free_mb": a.min_free_mb,
            "mean_util": a.mean_util, "peak_util": a.peak_util,
            "max_load_ratio": a.max_load_ratio, "min_avail_mb": a.min_avail_mb,
        },
    }
    text = json.dumps(out, ensure_ascii=False)
    if a.out and a.out != "-":
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)


REPEAT_RE = re.compile(
    r"^.*/save_path/%s/([^/]+)/([^/]+)/BERTCLASSIFIER/(\d+)Clients/experiment_state/[0-9a-f]{64}/repeat_\d+$"
    % re.escape(DATASET))


def _dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def cmd_cleanup(a):
    cells = scan_cells()
    gaps = {k for k in universe() if not cells[k]["complete"]}
    candidates = []
    for k in universe():
        if k not in gaps:
            sp, al, kk = parse_key(k)
            base = os.path.join(SAVE, sp, al, "BERTCLASSIFIER", "%dClients" % kk,
                                "experiment_state")
            if not os.path.isdir(base):
                continue
            for hashdir in sorted(os.listdir(base)):
                hd = os.path.join(base, hashdir)
                if not os.path.isdir(hd):
                    continue
                for rp in sorted(glob.glob(os.path.join(hd, "repeat_*"))):
                    if os.path.isdir(rp):
                        candidates.append((k, rp))
    # 安全校验：任何候选路径对应的格若在缺口集合里，整单作废
    bad = [c for c in candidates if c[0] in gaps]
    if bad:
        print("ABORT: %d candidate(s) belong to unfinished cells" % len(bad))
        for _, p in bad[:10]:
            print("BAD", p)
        return 2
    for _, p in candidates:
        if not REPEAT_RE.match(p):
            print("ABORT: unexpected path %s" % p)
            return 2
    tmps = []
    for base in (SAVE, os.path.join(FL, "result_path", DATASET)):
        for root, _, files in os.walk(base):
            for f in files:
                if f.endswith(".tmp"):
                    tmps.append(os.path.join(root, f))
    sizes = [(k, p, _dir_size(p)) for k, p in candidates]
    tmp_sizes = [(p, os.path.getsize(p)) for p in tmps if os.path.exists(p)]
    total = sum(s for _, _, s in sizes) + sum(s for _, s in tmp_sizes)
    lines = ["%s [cleanup:%s] cells_complete=%d gaps=%d repeat_dirs=%d tmp_files=%d bytes=%d" % (
        time.strftime("%F %T"), a.mode, len(universe()) - len(gaps), len(gaps),
        len(candidates), len(tmps), total)]
    for k, p, s in sizes:
        lines.append("  DEL %-12s %10d  %s" % (k, s, p))
    for p, s in tmp_sizes:
        lines.append("  TMP %10d  %s" % (s, p))
    text = "\n".join(lines)
    print(text)
    if a.log:
        os.makedirs(os.path.dirname(a.log), exist_ok=True)
        with open(a.log, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    if a.mode == "apply":
        for _, p in candidates:
            shutil.rmtree(p, ignore_errors=True)
        for p, _ in tmp_sizes:
            try:
                os.remove(p)
            except OSError:
                pass
        msg = "%s [cleanup:apply] freed=%d bytes repeat_dirs=%d tmp=%d" % (
            time.strftime("%F %T"), total, len(candidates), len(tmp_sizes))
        print(msg)
        if a.log:
            with open(a.log, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scan")
    p.add_argument("--out", default="-")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("status")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("partition-init")
    p.add_argument("--peer-scan", default="")
    p.add_argument("--partition-out", default=PARTITION)
    p.add_argument("--queue30-out", default="/tmp/imgq/lfwa_30.txt")
    p.add_argument("--queue31-out", default="/tmp/imgq/lfwa_31.txt")
    p.set_defaults(func=cmd_partition_init)

    p = sub.add_parser("build-queues")
    p.add_argument("--host", required=True)
    p.add_argument("--partition", default=PARTITION)
    p.add_argument("--peer-scan", default="")
    p.add_argument("--queue-out", required=True)
    p.add_argument("--summary-out", required=True)
    p.set_defaults(func=cmd_build_queues)

    p = sub.add_parser("monitor")
    p.add_argument("--seconds", type=int, default=60)
    p.add_argument("--max-slots", type=int, default=4)
    p.add_argument("--min-free-mb", type=int, default=18000)
    p.add_argument("--max-used-mb", type=int, default=10000)
    p.add_argument("--mean-util", type=float, default=10.0)
    p.add_argument("--peak-util", type=float, default=20.0)
    p.add_argument("--max-load-ratio", type=float, default=0.75)
    p.add_argument("--min-avail-mb", type=int, default=65536)
    p.add_argument("--omp-per-job", type=int, default=12)
    p.add_argument("--out", default="-")
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("cleanup")
    p.add_argument("--mode", choices=["dry", "apply"], default="dry")
    p.add_argument("--log", default="")
    p.set_defaults(func=cmd_cleanup)

    a = ap.parse_args()
    rc = a.func(a)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
