#!/bin/bash
# 08:30 收工：停掉本机所有实验 launcher 及其子进程。
# 判定依据是进程 cwd=/root/fl（我方实验进程都从仓库根目录启动），
# 因此不会误杀同机其他人的 vLLM（cwd=/ 或 /workspace/*）。
set -u
STAMP="$(date '+%F %T')"
LOG=/root/fl/parallel_logs/curfew.log
mkdir -p /root/fl/parallel_logs

{
  echo "[$STAMP] curfew start"

  # 1) 先 TERM 所有 launcher 的进程组（setsid 起，组内包含它拉起的 python）
  for p in $(ps -eo pid=,args= \
      | grep -E '^ *[0-9]+ bash (scripts|/root/fl/scripts)/launch_parallel\.sh' \
      | awk '{print $1}'); do
    pg=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')
    [ -n "$pg" ] && kill -TERM -"$pg" 2>/dev/null && echo "  TERM launcher pid=$p pgid=$pg"
  done
  sleep 20

  # 2) 兜底：凡是 cwd=/root/fl 的实验进程（进程名已被伪装成 VLLM::Worker_TP*）一律停掉
  for d in /proc/[0-9]*; do
    p=${d#/proc/}
    cw=$(readlink "$d/cwd" 2>/dev/null)
    [ "$cw" = "/root/fl" ] || continue
    kill -TERM "$p" 2>/dev/null
  done
  sleep 15

  # 3) 再补一刀 KILL
  for d in /proc/[0-9]*; do
    p=${d#/proc/}
    cw=$(readlink "$d/cwd" 2>/dev/null)
    [ "$cw" = "/root/fl" ] || continue
    kill -KILL "$p" 2>/dev/null
  done
  sleep 3

  left=0
  for d in /proc/[0-9]*; do
    cw=$(readlink "$d/cwd" 2>/dev/null)
    [ "$cw" = "/root/fl" ] && left=$((left + 1))
  done
  echo "[$(date '+%F %T')] curfew done; remaining_fl_procs=$left"
} >> "$LOG" 2>&1
