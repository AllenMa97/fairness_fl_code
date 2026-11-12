#!/usr/bin/env python3
"""恢复 GPUStack 上被置为 replicas=0 的 vLLM 模型服务。

周一回来后执行：
    python3 /root/fl/gs_restore_vllm.py            # 把 5 个模型全部拉回 replicas=1
    python3 /root/fl/gs_restore_vllm.py --dry-run  # 只打印，不改
    python3 /root/fl/gs_restore_vllm.py --wait 900 # 改完后最多等 900s 看是否 ready

设计说明：
- GPUStack 的自愈控制器会把被 docker stop 的容器重新拉起，所以停/启都必须在 GPUStack 层做
  （replicas 置 0 / 置回 1），只动容器是没有用的。
- 服务地址优先 127.0.0.1（本机即管理节点），失败再试 192.168.1.30。
- token 优先取环境变量 GPUSTACK_TOKEN，其次从 gpustack-worker 容器里自动读取。
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_MODELS = [
    "Qwen3.6-35B-A3B",
    "Qwen3.8-27B",
    "Qwen3-Embedding",
    "Qwen3-Reranker",
    "hunyuan-mt-chimera-7b-fp8",
]
CANDIDATE_BASES = ["http://127.0.0.1", "http://192.168.1.30"]


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return ""


def discover_token(explicit):
    if explicit:
        return explicit
    if os.environ.get("GPUSTACK_TOKEN"):
        return os.environ["GPUSTACK_TOKEN"]
    raw = sh("docker inspect gpustack-worker --format '{{json .Config.Env}}' 2>/dev/null")
    for item in raw.strip().strip("[]").split(","):
        item = item.strip().strip('"')
        if item.startswith("GPUSTACK_TOKEN="):
            return item.split("=", 1)[1]
    for c in ("gpustack", "gpustack-worker"):
        raw = sh("docker inspect %s --format '{{json .Config.Env}}' 2>/dev/null" % c)
        for item in raw.strip().strip("[]").split(","):
            item = item.strip().strip('"')
            if item.startswith("GPUSTACK_TOKEN="):
                return item.split("=", 1)[1]
    return ""


def call(base, token, method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        base + path, data=data, method=method,
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode()
    return json.loads(raw) if raw.strip() else {}


def pick_base(token):
    last = None
    for base in CANDIDATE_BASES:
        try:
            call(base, token, "GET", "/v2/models")
            return base
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise SystemExit("无法连接 GPUStack API: %s" % last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=None, help="GPUStack 地址，如 http://192.168.1.30")
    ap.add_argument("--token", default=None)
    ap.add_argument("--replicas", type=int, default=1)
    ap.add_argument("--models", nargs="*", default=None, help="只处理这些模型名（默认 5 个）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--wait", type=int, default=0, help="改完后等待 ready 的秒数")
    args = ap.parse_args()

    token = discover_token(args.token)
    if not token:
        raise SystemExit("找不到 GPUSTACK_TOKEN：请用 --token 传入，或确认本机有 gpustack-worker 容器")
    base = args.server or pick_base(token)

    wanted = args.models or DEFAULT_MODELS
    items = call(base, token, "GET", "/v2/models").get("items", [])
    targets = {}
    for m in items:
        for w in wanted:
            if m["name"] == w or m["name"].startswith(w):
                targets[w] = m
                break

    print("server : %s" % base)
    print("%-4s %-32s %-8s %-8s" % ("id", "name", "replicas", "ready"))
    missing = [w for w in wanted if w not in targets]
    for w in wanted:
        m = targets.get(w)
        if m:
            print("%-4s %-32s %-8s %-8s" % (m["id"], m["name"], m["replicas"], m["ready_replicas"]))
        else:
            print("%-4s %-32s %-8s %-8s" % ("-", w, "NOT FOUND", "-"))
    if missing:
        print("\n[警告] 以下模型在 GPUStack 中未找到：%s" % ", ".join(missing))
    if args.dry_run:
        print("\n--dry-run：不做修改")
        return

    changed = []
    for w in wanted:
        m = targets.get(w)
        if not m:
            continue
        if m["replicas"] == args.replicas:
            print("[skip] %-32s replicas 已是 %s" % (m["name"], args.replicas))
            continue
        body = dict(m)
        body.pop("ready_replicas", None)
        body.pop("owner_principal_id", None)
        body["replicas"] = args.replicas
        try:
            call(base, token, "PUT", "/v2/models/%d" % m["id"], body)
            changed.append(m["name"])
            print("[ ok ] %-32s replicas -> %s" % (m["name"], args.replicas))
        except urllib.error.HTTPError as exc:
            print("[FAIL] %-32s HTTP %s %s" % (m["name"], exc.code, exc.read().decode()[:200]))
        except Exception as exc:  # noqa: BLE001
            print("[FAIL] %-32s %s" % (m["name"], exc))

    if args.wait and changed:
        print("\n等待 ready（最多 %ds）..." % args.wait)
        deadline = time.time() + args.wait
        pending = set(changed)
        while time.time() < deadline and pending:
            time.sleep(20)
            cur = {m["name"]: m for m in call(base, token, "GET", "/v2/models").get("items", [])}
            for name in list(pending):
                m = cur.get(name)
                if m and m.get("ready_replicas", 0) >= args.replicas:
                    print("[ready] %-32s ready=%s" % (name, m["ready_replicas"]))
                    pending.discard(name)
            print("  ... still pending: %s" % (", ".join(sorted(pending)) or "none"))
        if pending:
            print("[警告] 仍未 ready：%s（去 GPUStack 看实例状态/显存是否够）" % ", ".join(sorted(pending)))

    print("\n完成，共修改 %d 个模型。可到 GPUStack 界面确认实例状态。" % len(changed))


if __name__ == "__main__":
    sys.exit(main())
