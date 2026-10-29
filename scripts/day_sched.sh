#!/bin/bash
# ============================================================
# 白天 CPU 档调度 / Daytime CPU-slot scheduler
#
# 背景：夜间 18:55-08:40 由 lfwa_sched.sh / acs_sched.sh 在空闲 GPU 上跑；
# 工作日白天 GPU 被同机其他用户占满，但 192 核里仍有大量空闲核。
# 本脚本在工作日白天把 ACSIncome 缺口放到「CPU 槽位」（假卡号 40+）继续推进，
# 18:45 收工，把核与内存让给 18:55 启动的夜间 GPU 调度。
#
# 时间规则（Asia/Shanghai）：
#   周一~周五 08:45-18:45 运行；周末与夜间一律停止。
#   08:42 的 curfew.sh 会先清场，因此白天档从 08:45 起，避免刚起就被收工脚本杀掉。
#   cron 每 5 分钟唤醒，本脚本幂等判断「启动 / 停止 / 什么都不做」。
#
# 与夜间调度的衔接：
#   白天队列文件名固定为 /tmp/imgq/acs_cpu_<host>.txt —— acs_sched.sh 在夜间建队列时
#   会把这个文件里正在跑的格子加入 --exclude-keys，因此不会和夜间 GPU 队列撞格。
#   18:45 停掉后 pgrep 失效，夜间调度即可接管这些格子（checkpoint 支持 CPU↔GPU 续跑）。
#
# 用法：
#   day_sched.sh               正常调度（cron 调用）
#   day_sched.sh --dry-run     按当前时间只打印判定，不做动作
#   day_sched.sh --at "2026-10-30 10:00"  按指定时间只打印判定（边界测试）
#   day_sched.sh --status      打印当前状态
#   day_sched.sh --stop-now    立即停掉白天队列（手工用）
# ============================================================
set -u

FL=/root/fl
PY=/root/miniconda3/envs/federated/bin/python
PLAN=$FL/scripts/acs_plan.py
LAUNCH=$FL/scripts/launch_parallel.sh
PART=$FL/acs_partition.json
SCHED=$FL/parallel_logs/day_sched
STATE=$SCHED/state
LOG=$SCHED/$(date +%F).log

mkdir -p "$SCHED" "$STATE" /tmp/imgq

HOST=$(tr -d '[:space:]' < "$FL/.lfwa_host" 2>/dev/null)
case "$HOST" in
  30|31) ;;
  *) HOST=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^192\.168\.1\.(30|31)$' | head -1 | awk -F. '{print $4}') ;;
esac
case "$HOST" in
  30|31) ;;
  *) echo "FATAL: cannot determine host id (expected 30/31)" >&2; exit 1 ;;
esac
if [ "$HOST" = "30" ]; then PEER=192.168.1.31; else PEER=192.168.1.30; fi

QUEUE=/tmp/imgq/acs_cpu_${HOST}.txt          # 固定名：acs_sched.sh 靠这个名字做夜间去重
SUMMARY=/tmp/imgq/acs_cpu_${HOST}.summary.json
PEER_SCAN=/tmp/imgq/acs_cpu_peer_scan_${HOST}.txt

log() { printf '%s [%s] %s\n' "$(date '+%F %T')" "$HOST" "$*" | tee -a "$LOG"; }

# ---- 只读子命令：不加锁，随时可查 ----
case "${1:-}" in
  --dry-run|--at|--status) ;;
  *)
    # ---- 单实例锁（只给会改状态的主流程用） ----
    exec 9>"$STATE/lock"
    if ! flock -n 9; then
      echo "another day_sched instance is running; skip" >&2
      exit 0
    fi ;;
esac

# ---- 时间判定：工作日 08:45-18:45 ----
# $1=epoch -> 设置 RUN=1/0 与 REASON
decide() {
  local ts="$1" u hm
  u=$(date -d "@$ts" +%u)
  hm=$(date -d "@$ts" +%H%M); hm=$((10#$hm))
  if [ "$u" -ge 6 ]; then
    RUN=0; REASON="weekend: GPU scheduler owns the box 24h"
  elif [ "$hm" -ge 845 ] && [ "$hm" -lt 1845 ]; then
    RUN=1; REASON="weekday 08:45-18:45: CPU slots on"
  else
    RUN=0; REASON="outside weekday 08:45-18:45: stopped"
  fi
}

find_launchers() {
  local p cmd
  for p in $(pgrep -f "acs_cpu_${HOST}\.txt" 2>/dev/null); do
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
    case "$cmd" in
      *launch_parallel.sh*) echo "$p" ;;
    esac
  done
}

launcher_running() { [ -n "$(find_launchers)" ]; }

slots() {
  local n
  n=$(tr -d '[:space:]' < "$FL/.day_cpu_slots" 2>/dev/null)
  case "$n" in ''|*[!0-9]*) n=4 ;; esac
  echo "$n"
}

fetch_peer_scan() {
  rm -f "$PEER_SCAN"
  if timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
      "root@${PEER}" "$PY $PLAN scan --out -" > "$PEER_SCAN" 2>/dev/null \
      && [ -s "$PEER_SCAN" ]; then
    log "peer scan ok: $(wc -l < "$PEER_SCAN") cells from $PEER"
    return 0
  fi
  rm -f "$PEER_SCAN"
  log "peer scan skipped (ssh $PEER unavailable); rely on local scan + host partition"
  return 1
}

start_if_needed() {
  if launcher_running; then
    log "day launcher already running (pids: $(find_launchers | tr '\n' ' ')); skip start"
    return 0
  fi
  [ -f "$PART" ] || { log "no $PART; skip"; return 1; }

  fetch_peer_scan || true
  local peer_arg=""
  [ -s "$PEER_SCAN" ] && peer_arg="$PEER_SCAN"
  "$PY" "$PLAN" build-queues --host "$HOST" --partition "$PART" \
    --peer-scan "$peer_arg" --queue-out "$QUEUE" --summary-out "$SUMMARY" >> "$LOG" 2>&1

  local qn
  qn=$("$PY" -c "import json;print(json.load(open('$SUMMARY'))['queue'])" 2>/dev/null || echo -1)
  log "queue=$QUEUE lines=$qn summary=$(cat "$SUMMARY" 2>/dev/null)"
  if [ "$qn" = "0" ]; then
    log "no ACSIncome gaps left for host $HOST; nothing to do"
    return 0
  fi
  [ "$qn" -le 0 ] && { log "queue build failed"; return 1; }

  local n ids="" i
  n=$(slots)
  for i in $(seq 0 $((n - 1))); do
    ids="$ids$((40 + i)),"
  done
  ids=${ids%,}
  local runlog="$SCHED/run_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$runlog"
  log "launch CPU slots=$ids queue=$QUEUE logdir=$runlog"
  (
    cd "$FL" || exit 1
    # 9>&- ：必须把脚本的 flock 描述符关掉，否则 launcher 会一直持有锁，
    # 之后每次 cron tick 都判定「已有实例在跑」而永远走不到 18:45 的收工分支。
    LOG_DIR="$runlog" MIN_FREE_MB=0 PER_GPU_LIMIT=1 MAX_ATTEMPTS=2 OMP_PER_JOB=12 \
      setsid bash "$LAUNCH" "$ids" "$QUEUE" >> "$runlog/launcher.log" 2>&1 < /dev/null 9>&- &
  )
  sleep 4
  local pid pgid
  pid=$(find_launchers | head -1)
  if [ -z "$pid" ]; then
    log "launch failed: launcher not found"
    return 1
  fi
  pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
  printf 'PID=%s\nPGID=%s\nSLOTS=%s\nSTARTED=%s\nQUEUE=%s\n' \
    "$pid" "$pgid" "$ids" "$(date '+%F %T')" "$QUEUE" > "$STATE/launcher_${HOST}.env"
  log "day launcher started pid=$pid pgid=$pgid"
}

stop_launchers() {
  local pids pgids pgid p stopped=0
  pids=$(find_launchers)
  if [ -z "$pids" ]; then
    log "stop: no day launcher running"
    return 0
  fi
  pgids=$(for p in $pids; do ps -o pgid= -p "$p" 2>/dev/null | tr -d ' '; done | sort -u)
  for pgid in $pgids; do
    [ -z "$pgid" ] && continue
    if [ "$pgid" = "1" ]; then
      log "ABORT: refusing to kill pgid 1"
      continue
    fi
    log "stop: TERM pgid $pgid"
    sync
    kill -TERM -- "-$pgid" 2>/dev/null
    stopped=1
  done
  if [ "$stopped" = "1" ]; then
    local i alive
    for i in $(seq 1 30); do
      alive=$(find_launchers)
      [ -z "$alive" ] && break
      sleep 2
    done
    for pgid in $pgids; do
      [ -n "$pgid" ] && [ "$pgid" != "1" ] && kill -KILL -- "-$pgid" 2>/dev/null
    done
    rm -f "$STATE/launcher_${HOST}.env"
    log "stop: done"
  fi
}

case "${1:-}" in
  --dry-run|--at)
    if [ "${1}" = "--at" ]; then
      ts=$(date -d "${2:?need time}" +%s)
    else
      ts=$(date +%s)
    fi
    decide "$ts"
    printf 'at=%s RUN=%s reason=%s launcher=%s\n' \
      "$(date -d "@$ts" '+%F %T')" "$RUN" "$REASON" "$(launcher_running && echo yes || echo no)"
    exit 0 ;;
  --status)
    printf 'host=%s slots=%s queue=%s lines=%s launcher=%s\n' \
      "$HOST" "$(slots)" "$QUEUE" "$(grep -c . "$QUEUE" 2>/dev/null || echo 0)" \
      "$(launcher_running && echo yes || echo no)"
    [ -f "$STATE/launcher_${HOST}.env" ] && cat "$STATE/launcher_${HOST}.env"
    exit 0 ;;
  --stop-now)
    stop_launchers; exit 0 ;;
esac

decide "$(date +%s)"
log "tick: RUN=$RUN ($REASON)"
if [ "$RUN" = "1" ]; then
  start_if_needed
else
  stop_launchers
fi
