#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FL 实验队列监控 + 邮件播报服务（服务器端常驻，由 cron 调用）。

用途：本机脱离网段后，仍可通过邮件掌握 .30/.31 上所有实验队列的进度、
失败、显存/磁盘与停滞告警。

配置文件（默认 /root/fl/scripts/monitor_conf.json，仅存于服务器，勿入库）:
{
  "smtp_host": "smtp.qq.com", "smtp_port": 465, "smtp_ssl": true,
  "username": "xxx@qq.com", "password": "<SMTP 授权码>",
  "sender": "xxx@qq.com", "recipients": ["you@example.com"],
  "digest_interval_min": 120, "root": "/root/fl",
  "active_window_min": 4, "disk_alert_pct": 88
}

运行模式：
  --mode dry    只打印报告，不发信（自检用）
  --mode once   立即发一封报告
  --mode test   发一封测试邮件，验证 SMTP 凭据
  --mode auto   cron 用：有告警立即发（去重），否则按 digest_interval_min 发摘要
"""
import argparse
import glob
import json
import os
import re
import smtplib
import socket
import subprocess
import time
from email.header import Header
from email.mime.text import MIMEText
from email.utils import formatdate

CONF_PATH = os.environ.get("FL_MONITOR_CONF", "/root/fl/scripts/monitor_conf.json")
RE_ROUND = re.compile(r"Communication Round:\s*(\d+)\s*/\s*(\d+)")
RE_FAIL = re.compile(r"worker_failures=(\d+)")
RE_BAD = re.compile(r"OutOfMemoryError|CUDA out of memory|Traceback \(most recent call last\)|invalid device ordinal")


def sh(cmd, timeout=90):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or "") + (p.stderr or "")
    except Exception as exc:  # noqa: BLE001
        return f"<cmd error: {exc}>"


def load_conf():
    if not os.path.exists(CONF_PATH):
        return {}
    with open(CONF_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def collect(conf):
    root = conf.get("root", "/root/fl")
    window = int(conf.get("active_window_min", 4))
    logs_dir = os.path.join(root, "parallel_logs")
    d = {"host": socket.gethostname(), "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "root": root}

    # 1) 队列 launcher 状态
    queues, fails_total, done_cnt = [], 0, 0
    for lp in sorted(glob.glob(os.path.join(logs_dir, "*launcher*.log"))):
        name = os.path.basename(lp).replace("_launcher", "").replace(".log", "")
        try:
            txt = open(lp, "r", errors="ignore").read()
        except OSError:
            continue
        dispatched = txt.count("cmd:")
        gates = txt.count("gate:")
        finished = "all done" in txt
        fm = RE_FAIL.findall(txt)
        fails = int(fm[-1]) if fm else 0
        age_min = round((time.time() - os.path.getmtime(lp)) / 60, 1)
        # 历史遗留 launcher（日志久无写入且未标 done）视为 stale，不计入活跃队列分母
        stale = (not finished) and age_min > float(conf.get("stale_min", 90))
        fails_total += fails
        if not stale:
            done_cnt += int(finished)
        queues.append({"name": name, "dispatched": dispatched, "gates": gates,
                       "fails": fails, "finished": finished, "stale": stale,
                       "age_min": age_min})
    d["queues"] = queues
    d["queues_live"] = [q for q in queues if not q["stale"]]
    d["fails_total"] = fails_total
    d["queues_done"] = done_cnt

    # 2) 活跃 worker 日志（近 window 分钟内有写入即视为在跑；进程标题被伪装，故用 mtime 判活）
    workers = []
    for wp in glob.glob(os.path.join(logs_dir, "**", "gpu*_seq*.log"), recursive=True):
        age = (time.time() - os.path.getmtime(wp)) / 60.0
        if age > window:
            continue
        tail = sh(f"tail -c 4000 '{wp}'")
        rounds = RE_ROUND.findall(tail)
        bad = len(RE_BAD.findall(tail))
        workers.append({"path": os.path.relpath(wp, logs_dir), "age_min": round(age, 1),
                        "round": f"{rounds[-1][0]}/{rounds[-1][1]}" if rounds else "-",
                        "err": bad})
    workers.sort(key=lambda x: x["path"])
    d["workers"] = workers
    d["active"] = len(workers)
    d["err_hits"] = sum(w["err"] for w in workers)

    # 3) 结果落盘计数
    counts = {}
    rp = os.path.join(root, "result_path")
    if os.path.isdir(rp):
        for ds in sorted(os.listdir(rp)):
            p = os.path.join(rp, ds)
            if os.path.isdir(p):
                counts[ds] = len(glob.glob(os.path.join(p, "**", "*.json"), recursive=True))
    d["results"] = counts
    d["results_total"] = sum(counts.values())

    # 4) GPU / CPU / 磁盘
    gpu = sh("nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu "
             "--format=csv,noheader,nounits").strip().splitlines()
    d["gpu"] = [g.strip() for g in gpu if g.strip()]
    d["load"] = sh("cat /proc/loadavg").strip().split()[0:3]
    d["ncpu"] = sh("nproc").strip()
    df = sh("df -h / | tail -1").split()
    d["disk"] = {"used": df[2] if len(df) > 2 else "?", "free": df[3] if len(df) > 3 else "?",
                 "pct": int(df[4].rstrip("%")) if len(df) > 4 and df[4].endswith("%") else 0}
    return d


def render(d, conf):
    L = []
    L.append(f"主机 {d['host']}   时间 {d['ts']}")
    L.append(f"活跃作业 {d['active']}   队列完成 {d['queues_done']}/{len(d['queues_live'])}(活跃)   "
             f"累计失败 {d['fails_total']}   结果 json {d['results_total']}")
    L.append(f"负载 {d['load']}  (核数 {d['ncpu']})   磁盘 {d['disk']['used']} 已用 / "
             f"{d['disk']['free']} 空闲 ({d['disk']['pct']}%)")
    L.append("")
    L.append("── GPU (idx, 已用MiB, 空闲MiB, 利用率%) ──")
    for g in d["gpu"]:
        L.append("  " + g)
    L.append("")
    L.append("── 队列 ──")
    for q in sorted(d["queues"], key=lambda x: (x["stale"], x["finished"], x["name"])):
        flag = ("DONE" if q["finished"] else
                "stale" if q["stale"] else
                "gate-wait" if q["gates"] and not q["dispatched"] else "run")
        L.append(f"  {q['name']:<26} {flag:<9} 已发 {q['dispatched']:>3}  "
                 f"闸门 {q['gates']:>3}  失败 {q['fails']}  日志龄 {q['age_min']}min")
    L.append("")
    L.append("── 活跃作业进度 ──")
    if not d["workers"]:
        L.append("  （无近 %s 分钟内有写入的 worker 日志）" % conf.get("active_window_min", 4))
    for w in d["workers"][:40]:
        mark = "  <== ERR" if w["err"] else ""
        L.append(f"  {w['path']:<34} Round {w['round']:<10} 龄 {w['age_min']}min{mark}")
    if len(d["workers"]) > 40:
        L.append(f"  ... 其余 {len(d['workers']) - 40} 个省略")
    L.append("")
    L.append("── 结果 json 按数据集 ──")
    for k, v in d["results"].items():
        L.append(f"  {k:<18} {v}")
    return "\n".join(L)


def detect_alerts(d, state, conf):
    a = []
    pending = [q for q in d.get("queues_live", d["queues"]) if not q["finished"]]
    if pending and d["active"] == 0:
        a.append(("STALL", f"仍有 {len(pending)} 个队列未完成，但近 "
                          f"{conf.get('active_window_min', 4)} 分钟无任何 worker 日志写入（可能全部卡死/OOM）"))
    if d["err_hits"]:
        a.append(("ERR", f"{d['err_hits']} 个活跃日志出现 OOM/Traceback"))
    if d["fails_total"] > state.get("fails_total", d["fails_total"]):
        a.append(("FAIL", f"累计 worker 失败数上升到 {d['fails_total']}"))
    if d["disk"]["pct"] >= int(conf.get("disk_alert_pct", 88)):
        a.append(("DISK", f"根分区使用率 {d['disk']['pct']}%（仅剩 {d['disk']['free']}）"))
    if pending == [] and d["queues"] and not state.get("all_done_sent"):
        a.append(("ALLDONE", "所有队列均已结束"))
    return a


def send_mail(conf, subject, body):
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = conf.get("sender", conf.get("username", ""))
    msg["To"] = ", ".join(conf.get("recipients", []))
    msg["Date"] = formatdate(localtime=True)
    host, port = conf["smtp_host"], int(conf.get("smtp_port", 465))
    if conf.get("smtp_ssl", True):
        srv = smtplib.SMTP_SSL(host, port, timeout=30)
    else:
        srv = smtplib.SMTP(host, port, timeout=30)
        srv.starttls()
    try:
        srv.login(conf["username"], conf["password"])
        srv.sendmail(msg["From"], conf["recipients"], msg.as_string())
    finally:
        srv.quit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="auto", choices=["auto", "once", "dry", "test"])
    args = ap.parse_args()
    conf = load_conf()

    if args.mode == "test":
        if not conf:
            print(f"[monitor] 缺少配置 {CONF_PATH}")
            return 2
        send_mail(conf, f"[FL-monitor] 测试邮件 from {socket.gethostname()}",
                  "如果你收到这封邮件，说明 SMTP 凭据与网络出口都正常。\n")
        print("[monitor] test mail sent")
        return 0

    d = collect(conf)
    body = render(d, conf)

    if args.mode == "dry":
        print(body)
        return 0

    if not conf:
        print(f"[monitor] 报告已生成但未发信：缺少配置 {CONF_PATH}")
        print(body[:400])
        return 3

    state_path = os.path.join(d["root"], "parallel_logs", "monitor_state.json")
    state = {}
    if os.path.exists(state_path):
        try:
            state = json.load(open(state_path, "r", encoding="utf-8"))
        except Exception:  # noqa: BLE001
            state = {}

    alerts = detect_alerts(d, state, conf)
    now = time.time()
    last_digest = state.get("last_digest_ts", 0)
    interval = int(conf.get("digest_interval_min", 120)) * 60
    sent_keys = state.get("sent_alerts", {})

    to_send, subject = False, ""
    fresh = [x for x in alerts if sent_keys.get(x[0]) != x[1]]
    if fresh and args.mode == "auto":
        to_send = True
        subject = f"[FL-alert:{','.join(k for k, _ in fresh)}] {d['host']} 活跃{d['active']} 失败{d['fails_total']}"
        for k, v in fresh:
            sent_keys[k] = v
        if any(k == "ALLDONE" for k, _ in fresh):
            state["all_done_sent"] = True
    elif args.mode == "once" or (now - last_digest) >= interval:
        to_send = True
        subject = (f"[FL-digest] {d['host']} 活跃{d['active']} 队列{d['queues_done']}/{len(d['queues'])} "
                   f"结果{d['results_total']} 磁盘{d['disk']['pct']}%")
    if alerts:
        body = "!! 告警 !!\n" + "\n".join(f"  [{k}] {v}" for k, v in alerts) + "\n\n" + body

    if to_send:
        try:
            send_mail(conf, subject, body)
            state["last_digest_ts"] = now
            state["sent_alerts"] = sent_keys
            state["fails_total"] = d["fails_total"]
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            json.dump(state, open(state_path, "w", encoding="utf-8"))
            print(f"[monitor] mail sent: {subject}")
        except Exception as exc:  # noqa: BLE001
            print(f"[monitor] MAIL FAILED: {exc}")
            return 4
    else:
        state["fails_total"] = d["fails_total"]
        json.dump(state, open(state_path, "w", encoding="utf-8"))
        print(f"[monitor] no send (next digest in "
              f"{max(0, int((interval - (now - last_digest)) / 60))} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
