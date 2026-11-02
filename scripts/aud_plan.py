#!/usr/bin/env python3
"""Audio-CLF 缺口调度 helper（VCTK / CREMA-D / RAVDESS / AudioMNIST）。

结果目录约定 / Result layout
    result_path/<dataset>/<split>/<algo>/<model>/<K>Clients/*.txt.json

完成标准 / Completion rule
    - json 可解析；
    - aggregate.metrics.ACC.values 长度 >= 3；
    - experiment_config.communication_round_I == 200（VCTK 协议）。
    注意：这里用 >=3 判定「3 repeats 都有聚合结果」，不要求文件名是 1/2/3。

Subcommands
    scan            扫描本地结果，输出 split|algo|K|complete|njson
    build-queues    剔除本地已完成 + 正在跑（手工/其它 launcher）的格后生成本机队列
    status          打印完成/缺口统计
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

FL = "/root/fl"
PY = "/root/miniconda3/envs/federated/bin/python"
MAIN = os.path.join(FL, "main_AUD_CLF.py")

DATASET = os.environ.get("AUD_DATASET", "VCTK")
MODEL = os.environ.get("AUD_MODEL", "AUDIOCNNRNN")
KS = [int(x) for x in os.environ.get("AUD_KS", "20").split(",") if x.strip()]
ROUNDS = int(os.environ.get("AUD_ROUNDS", "200"))
MIN_VALUES = 3

# 与图像基准保持同一套算法清单，方便横向比较。
ALGOS = [
    "FairFed", "FedAvg", "FederatedAverageWithPo", "FedFair", "FedFB", "FedNova",
    "FedProto", "FedProx", "FedRenyi", "FedRep", "FL_FairBatch", "LoGoFair",
    "mFairFL", "PDFFed", "PDFFed_DP", "Scaffold",
]

# 各音频数据集实际使用的 split；新数据集可用 AUD_SPLITS 覆盖。
DATASET_SPLITS = {
    "VCTK": ["Speaker"],
    "CREMA-D": ["Dirichlet1", "Uniform", "Speaker"],
    "CREMA-D_ar": ["Speaker"],
    "CREMA-D_gender": ["Speaker"],
    "RAVDESS": ["Speaker"],
    "RAVDESS_ar": ["Speaker"],
    "AudioMNIST": ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"],
}

EXTRA_ARGS = (
    "-task AUD_CLF -dataset {ds} -communication_round_I %d -algorithm_epoch_T 2 "
    "-exp_repeat_times 3 -batch_size 64 -test_batch_size 64 "
    "-learning_rate 0.001 -optimize_method adam -test_eval_freq 50 "
    "-final_artifact_policy metrics_only -resume -dataloader_num_workers 2" % ROUNDS
)


def result_root(ds):
    return os.path.join(FL, "result_path", ds)


def splits_for(ds):
    env = os.environ.get("AUD_SPLITS", "").strip()
    if env:
        return [x.strip() for x in env.split(",") if x.strip()]
    if ds in DATASET_SPLITS:
        return list(DATASET_SPLITS[ds])
    root = result_root(ds)
    if os.path.isdir(root):
        got = [d for d in sorted(os.listdir(root))
               if os.path.isdir(os.path.join(root, d))]
        if got:
            return got
    return ["Speaker"]


def key_of(sp, al, k):
    return "%s|%s|%d" % (sp, al, k)


def parse_key(keystr):
    sp, al, k = keystr.split("|")
    return sp, al, int(k)


def universe(ds):
    return [key_of(sp, al, k) for sp in splits_for(ds) for al in ALGOS for k in KS]


def cell_dir(ds, sp, al, k):
    return os.path.join(result_root(ds), sp, al, MODEL, "%dClients" % k)


def json_complete(path):
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return False
    try:
        vals = d["aggregate"]["metrics"]["ACC"]["values"]
        cfg = d.get("experiment_config") or {}
        rnd = cfg.get("communication_round_I", d.get("communication_round_I"))
        if isinstance(vals, list) and len(vals) >= MIN_VALUES and rnd == ROUNDS:
            return True
    except Exception:
        pass
    return False


def scan_cells(ds):
    """return {key: {"complete": bool, "njson": int}}"""
    cells = {}
    for sp in splits_for(ds):
        for al in ALGOS:
            for k in KS:
                d = cell_dir(ds, sp, al, k)
                files = []
                if os.path.isdir(d):
                    files = [f for f in os.listdir(d) if f.endswith(".txt.json")]
                ok = any(json_complete(os.path.join(d, f)) for f in files)
                cells[key_of(sp, al, k)] = {"complete": ok, "njson": len(files)}
    return cells


def load_keys(path):
    out = set()
    if not path or not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#") or ln.count("|") != 2:
                continue
            out.add(ln)
    return out


def load_scan(path):
    """读取 scan 输出：split|algo|K|complete|njson。"""
    out = {}
    if not path or not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            p = ln.strip().split("|")
            if len(p) < 5:
                continue
            try:
                out[key_of(p[0], p[1], int(p[2]))] = p[3] == "1"
            except ValueError:
                continue
    return out


def running_keys(ds):
    """本机正在跑的 AUD_CLF 格（split|algo|K）。

    训练进程 argv 会被 tool/process_alias.py 改写成 VLLM::Worker_TPn，
    直接匹配 main_AUD_CLF.py 往往失败；因此优先读取正在运行的
    launch_parallel.sh 的队列文件，再补上未被改写的直跑进程。
    """
    keys = set()
    cmdlines = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open("/proc/%s/cmdline" % pid, "rb") as f:
                raw = f.read().decode("utf-8", "ignore")
        except OSError:
            continue
        argv = [x for x in raw.split("\0") if x]
        if not argv:
            continue
        joined = " ".join(argv)
        if "launch_parallel.sh" in joined:
            q = argv[-1]
            if q.startswith("/tmp/imgq/") and q.endswith(".txt") and os.path.isfile(q):
                try:
                    with open(q, encoding="utf-8", errors="ignore") as f:
                        cmdlines.extend(f.read().splitlines())
                except OSError:
                    pass
        if "main_AUD_CLF.py" in joined:
            cmdlines.append(joined)
    pat = re.compile(
        r"-dataset\s+(\S+).*?-split_strategy\s+(\S+)\s+"
        r"-algorithm\s+(\S+)\s+-num_clients_K\s+(\d+)")
    for c in cmdlines:
        m = pat.search(c)
        if m and m.group(1) == ds:
            keys.add("%s|%s|%s" % (m.group(2), m.group(3), m.group(4)))
    return keys


def queue_line(ds, k):
    sp, al, kk = parse_key(k)
    return ("nice -n 10 %s %s %s -split_strategy %s -algorithm %s -num_clients_K %d"
            % (PY, MAIN, EXTRA_ARGS.format(ds=ds), sp, al, kk))


def cmd_scan(a):
    cells = scan_cells(a.dataset)
    lines = ["%s|%d|%d" % (k, 1 if cells[k]["complete"] else 0, cells[k]["njson"])
             for k in universe(a.dataset)]
    text = "\n".join(lines) + "\n"
    if a.out == "-":
        sys.stdout.write(text)
    else:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)


def cmd_status(a):
    ds = a.dataset
    if not os.path.isdir(os.path.join(FL, "dataset", ds)):
        print("dataset dir missing on this host: %s/dataset/%s" % (FL, ds))
        return 0
    cells = scan_cells(ds)
    uni = universe(ds)
    gaps = [k for k in uni if not cells[k]["complete"]]
    print("dataset=%s universe=%d complete=%d gap=%d" % (ds, len(uni), len(uni) - len(gaps), len(gaps)))
    print("gap_by_split=%s" % dict(Counter(parse_key(k)[0] for k in gaps)))
    print("gap_by_algo=%s" % dict(Counter(parse_key(k)[1] for k in gaps)))
    if a.verbose:
        for k in gaps:
            print("GAP %s json=%d" % (k, cells[k]["njson"]))
    return 0


def cmd_build_queues(a):
    ds = a.dataset
    local = scan_cells(ds)
    peer = load_scan(a.peer_scan)
    excluded = load_keys(a.exclude_keys)
    uni = universe(ds)
    gaps = [k for k in uni if not local[k]["complete"] and not peer.get(k, False)]
    inflight = [k for k in gaps if k in excluded]
    q = [k for k in gaps if k not in excluded]
    peer_complete = sum(1 for k in uni if peer.get(k, False))

    # 两机静态归属（与 img_plan 同款）：有 partition 文件按登记，否则按 key 哈希。
    # 目的：两台都有同一数据集时各跑一半，避免重复消耗 GPU。
    cells = {}
    if a.partition and os.path.isfile(a.partition):
        with open(a.partition, encoding="utf-8") as f:
            cells = (json.load(f).get("cells") or {})

    def host_of(k):
        h = cells.get(k)
        if h:
            return h
        import hashlib
        return "30" if int(hashlib.sha1(k.encode("utf-8")).hexdigest(), 16) % 2 == 0 else "31"

    q = [k for k in q if host_of(k) == a.host]

    with open(a.queue_out, "w", encoding="utf-8") as f:
        for k in q:
            f.write(queue_line(ds, k) + "\n")
    summary = {
        "host": a.host, "dataset": ds, "universe": len(uni),
        "local_complete": sum(1 for k in uni if local[k]["complete"]),
        "peer_complete": peer_complete, "peer_scan": bool(peer),
        "gap": len(gaps), "excluded_running": len(inflight), "queue": len(q),
        "queue_by_split": dict(Counter(parse_key(k)[0] for k in q)),
    }
    with open(a.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def cmd_running_keys(a):
    for k in sorted(running_keys(a.dataset)):
        print(k)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scan")
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--out", default="-")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("status")
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("build-queues")
    p.add_argument("--host", required=True)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--exclude-keys", default="")
    p.add_argument("--peer-scan", default="")
    p.add_argument("--partition", default="")
    p.add_argument("--queue-out", required=True)
    p.add_argument("--summary-out", required=True)
    p.set_defaults(func=cmd_build_queues)

    p = sub.add_parser("running-keys")
    p.add_argument("--dataset", default=DATASET)
    p.set_defaults(func=cmd_running_keys)

    a = ap.parse_args()
    rc = a.func(a)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
