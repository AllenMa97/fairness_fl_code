#!/bin/bash
# ============================================================
# 「本机正在跑什么」的唯一来源。
# 所有调度器（acs_sched / day_sched / img_sched）都应 source 本文件，
# 用 live_cells 生成 --exclude-keys，用 busy_slots 过滤空闲槽位，
# 避免出现两条队列跑同一批格、或两批作业挤在同一组槽位上。
#
# 原理：launch_parallel.sh 的 argv 形如
#   <argv0> scripts/launch_parallel.sh 40,41,42 /tmp/imgq/xxx.txt
# MASK_LAUNCHER=1 只改 argv[0]，槽位与队列路径都保留在 /proc/<pid>/cmdline 里。
# ============================================================

launcher_pids() {
  pgrep -f 'launch_parallel\.sh' 2>/dev/null
}

live_queues() {
  local p q
  for p in $(launcher_pids); do
    q=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | tail -1)
    case "$q" in /tmp/imgq/*.txt) [ -f "$q" ] && echo "$q" ;; esac
  done | sort -u
}

queue_cells() {  # $1=queue file -> 每行 split|algo|K
  sed -nE 's/.*-split_strategy ([A-Za-z0-9]+) -algorithm ([A-Za-z_]+) -num_clients_K ([0-9]+).*/\1|\2|\3/p' "$1"
}

live_cells() {   # 本机所有在跑队列覆盖的格
  local q
  for q in $(live_queues); do queue_cells "$q"; done | sort -u
}

busy_slots() {   # 本机所有 launcher 占用的槽位（逗号分隔）
  local p a
  for p in $(launcher_pids); do
    a=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | tail -2 | head -1)
    echo "$a" | tr ',' '\n'
  done | awk 'NF && $1 ~ /^[0-9]+$/' | sort -n -u | paste -sd, -
}
