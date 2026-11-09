#!/usr/bin/env python3
"""周一 08:50 自动收尾：停掉当前所有实验（保留已落盘 checkpoint），再恢复 5 个 vLLM 服务。

用法（在两台机器各起一个常驻守护，互不依赖 ssh）：
    # .30（GPUStack 管理节点）
    setsid nohup python3 /root/fl/monday_0850_stop.py --role server --at "2026-11-09 08:50:00" \
        > /root/fl/parallel_logs/monday_0850.log 2>&1 < /dev/null &
    # .31
    setsid nohup python3 /root/fl/monday_0850_stop.py --role worker --at "2026-11-09 08:50:00" \
        > /root/fl/parallel_logs/monday_0850.log 2>&1 < /dev/null &

其它模式：
    --list     只列出将被停止的进程，不做任何操作
    --now      立即执行一遍收尾流程（测试用）

停止顺序：先停 launcher（不再起新格）-> 再 SIGTERM 训练进程树 -> 90s 后 SIGKILL 残留
         -> server 角色再等 120s 调 gs_restore_vllm.py 把 replicas 置回 1 并等 ready。
训练进程没有 SIGTERM 落盘钩子，所以"保存"= 已写入 checkpoint_latest.pt 的进度保留，
重启时带 -resume 会从最近一次落盘处继续。
"""
import argparse
import datetime as dt
import os
import re
import signal
import subprocess
import sys
import time

LAUNCH_PATTERNS = [
    "launch_img.sh",
    "launch_pad3.sh",
    "launch_ff.sh",
    "launch_aud",
    "launch_parallel.sh",
    "chain_ff.sh",
]
TRAIN_PATTERNS = [
    "main_IMG_CLF.py",
    "main_AUD_CLF.py",
    "main_Tabular_CLF.py",
]
ALIAS_RE = re.compile(r"^VLLM::Worker_TP\d+$")
EXCLUDE_PATTERNS = ["monday_0850_stop", "gs_restore_vllm", "monday_0850"]


def log(msg):
    print("[%s] %s" % (dt.datetime.now().strftime("%F %T"), msg), flush=True)


def cmdline(pid):
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as fh:
            return fh.read().replace(b"\x00", b" ").decode("utf-8", "ignore").strip()
    except Exception:
        return ""


def ppid_of(pid):
    try:
        with open("/proc/%d/stat" % pid, "r") as fh:
            return int(fh.read().split()[3])
    except Exception:
        return 0


def snapshot():
    procs = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            pid = int(d)
            procs[pid] = (ppid_of(pid), cmdline(pid))
    return procs


def match(procs, patterns, alias=False):
    hits = {}
    me = os.getpid()
    for pid, (_, cl) in procs.items():
        if pid == me or not cl:
            continue
        if any(x in cl for x in EXCLUDE_PATTERNS):
            continue
        if any(p in cl for p in patterns) or (alias and ALIAS_RE.match(cl)):
            hits[pid] = cl
    return hits


def tree_of(procs, roots):
    kids = {}
    for pid, (pp, _) in procs.items():
        kids.setdefault(pp, []).append(pid)
    out = set()
    stack = list(roots)
    while stack:
        cur = stack.pop()
        if cur in out:
            continue
        out.add(cur)
        stack.extend(kids.get(cur, []))
    return out


def kill(pid, sig):
    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", default="2026-11-09 08:50:00", help="本地时间 YYYY-MM-DD HH:MM:SS")
    ap.add_argument("--role", choices=["server", "worker"], default="server")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--now", action="store_true")
    ap.add_argument("--kill-wait", type=int, default=90, help="SIGTERM 后等多久再 SIGKILL")
    ap.add_argument("--restore-delay", type=int, default=120, help="停训练后等多久再恢复 vLLM")
    args = ap.parse_args()

    if args.list:
        procs = snapshot()
        launch = match(procs, LAUNCH_PATTERNS)
        train = match(procs, TRAIN_PATTERNS, alias=True)
        log("launchers=%d" % len(launch))
        for pid, cl in sorted(launch.items()):
            log("  L %-8d %s" % (pid, cl[:110]))
        log("trainers=%d" % len(train))
        for pid, cl in sorted(train.items()):
            log("  T %-8d %s" % (pid, cl[:110]))
        return 0

    if not args.now:
        target = dt.datetime.strptime(args.at, "%Y-%m-%d %H:%M:%S")
        log("armed: role=%s target=%s now=%s"
            % (args.role, target.strftime("%F %T"), dt.datetime.now().strftime("%F %T")))
        last_log = time.time()
        while True:
            left = (target - dt.datetime.now()).total_seconds()
            if left <= 0:
                break
            if time.time() - last_log >= 600:
                log("waiting %s (%.0fs left)" % (target.strftime("%F %T"), left))
                last_log = time.time()
            time.sleep(min(60, max(1, left)))
        log("target reached: %s" % target.strftime("%F %T"))

    # 1) 先停 launcher：不再起新格
    procs = snapshot()
    launch = match(procs, LAUNCH_PATTERNS)
    log("phase1 stop launchers: %d" % len(launch))
    for pid, cl in sorted(launch.items()):
        log("  TERM launcher %-8d %s" % (pid, cl[:100]))
        kill(pid, signal.SIGTERM)
    time.sleep(10)

    # 2) 再 SIGTERM 训练进程树（含别名 VLLM::Worker_TP*）
    procs = snapshot()
    train = match(procs, TRAIN_PATTERNS, alias=True)
    targets = tree_of(procs, train.keys())
    log("phase2 stop trainers: roots=%d tree=%d" % (len(train), len(targets)))
    for pid in sorted(targets, reverse=True):
        cl = procs.get(pid, (0, ""))[1]
        log("  TERM %-8d %s" % (pid, cl[:100]))
        kill(pid, signal.SIGTERM)

    deadline = time.time() + args.kill_wait
    while time.time() < deadline:
        alive = [p for p in targets if os.path.exists("/proc/%d" % p)]
        if not alive:
            break
        time.sleep(10)
    alive = [p for p in targets if os.path.exists("/proc/%d" % p)]
    if alive:
        log("phase2b SIGKILL survivors: %s" % sorted(alive))
        for pid in sorted(alive, reverse=True):
            kill(pid, signal.SIGKILL)
    log("training stopped; alive_after=%d" % len([p for p in targets if os.path.exists("/proc/%d" % p)]))

    out = subprocess.run("nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader",
                         shell=True, capture_output=True, text=True).stdout.strip()
    log("gpu after stop:\n%s" % out)

    # 3) server 角色：恢复 5 个 vLLM 模型
    if args.role != "server":
        log("worker role: skip vLLM restore")
        return 0

    log("waiting %ds before restoring vLLM ..." % args.restore_delay)
    time.sleep(args.restore_delay)
    rc = subprocess.run(
        [sys.executable, "/root/fl/gs_restore_vllm.py", "--wait", "900"],
        capture_output=True, text=True,
    )
    log("gs_restore_vllm rc=%d\n%s\n%s" % (rc.returncode, rc.stdout, rc.stderr))
    return 0


if __name__ == "__main__":
    sys.exit(main())
