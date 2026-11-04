#!/bin/bash
# ============================================================
# 图像数据集（CelebA / FairFace）缺口补跑调度器
#
# 时间规则（Asia/Shanghai，与 acs_sched.sh 保持一致）：
#   周一~周五 20:00 启动；次日 08:40 停止。
#   周五晚启动后，周六/周日全天继续跑（周末不停），直到下周一 08:40。
#   cron 每 5 分钟唤醒，本脚本幂等：run 窗口内确保 launcher 在跑，窗口外停掉。
#
# 多个数据集按顺序排队：每次 tick 只跑一个数据集的队列，跑完（队列空 ->
# 写 done flag）后下一个 tick 自动切到下一个数据集。
#
# 用法：
#   IMG_DATASETS=CelebA,FairFace img_sched.sh    正常调度（cron 调用）
#   IMG_DATASETS=FairFace img_sched.sh --dry-run 只打印时间判定
#   img_sched.sh --status                        打印各数据集缺口
# ============================================================
set -u

FL=/root/fl
PY=/root/miniconda3/envs/federated/bin/python
PLAN=$FL/scripts/img_plan.py
LAUNCH=$FL/scripts/launch_parallel.sh
SCHED_ROOT=$FL/parallel_logs/img_sched
DATASETS="${IMG_DATASETS:-PAD-UFES-20,CelebA,FairFace}"
export IMG_PARTITION="${IMG_PARTITION:-img_partition.json}"

mkdir -p "$SCHED_ROOT" /tmp/imgq

HOST=$(tr -d '[:space:]' < $FL/.lfwa_host 2>/dev/null)
case "$HOST" in
  30|31) ;;
  *) HOST=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^192\.168\.1\.(30|31)$' | head -1 | awk -F. '{print $4}') ;;
esac
case "$HOST" in
  30|31) ;;
  *) echo "FATAL: cannot determine host id" >&2; exit 1 ;;
esac
if [ "$HOST" = "30" ]; then PEER=192.168.1.31; else PEER=192.168.1.30; fi

SCHED=$SCHED_ROOT
LOG=$SCHED/$(date +%F).log
mkdir -p "$SCHED"
log() { printf '%s [%s] %s\n' "$(date '+%F %T')" "$HOST" "$*" | tee -a "$LOG"; }

exec 9>"$SCHED/lock"
flock -n 9 || { echo "another img_sched instance is running; skip" >&2; exit 0; }

# ---- 时间判定：工作日 20:00-08:40 + 周末全天 ----
decide() {
  if [ "${FORCE_RUN:-0}" = "1" ]; then RUN=1; REASON="forced by FORCE_RUN"; return 0; fi
  local ts="$1" u hm
  u=$(date -d "@$ts" +%u); hm=$(date -d "@$ts" +%H%M); hm=$((10#$hm))
  if [ "$u" -ge 6 ]; then RUN=1; REASON="weekend: Friday session keeps running"
  elif [ "$hm" -ge 2000 ] || [ "$hm" -lt 840 ]; then RUN=1; REASON="weekday 20:00-08:40: run"
  else RUN=0; REASON="weekday 08:40-20:00: stopped"; fi
}

queue_of()  { echo "/tmp/imgq/img_${1}_${HOST}.txt"; }
summary_of(){ echo "/tmp/imgq/img_${1}_${HOST}.summary.json"; }
done_flag() { echo "$FL/img_${1}_done.flag"; }
state_env() { echo "$SCHED/state_${1}_${HOST}.env"; }

find_launchers() {  # $1=dataset
  # 1) 新机制：mask 的 launcher argv 已改成 VLLM::EngineCore，
  #    扫 $SCHED/run_${1}_* 下 .launcher_pid 文件（exec 不换 pid，永远有效）。
  # 2) 兜底：老机制——未 mask 的 launcher，扫 argv 匹配。
  local pidfile pid
  for pidfile in $(find "$SCHED" -maxdepth 3 -path "$SCHED/run_${1}_*" -name ".launcher_pid" -mmin -120 2>/dev/null); do
    pid=$(cat "$pidfile" 2>/dev/null)
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && echo "$pid"
  done
  local p cmd
  for p in $(pgrep -f "img_${1}_${HOST}\.txt" 2>/dev/null); do
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
    case "$cmd" in *launch_parallel.sh*) echo "$p" ;; esac
  done
}

any_launcher() {
  local ds p
  for ds in $(echo "$DATASETS" | tr ',' ' '); do
    p=$(find_launchers "$ds")
    [ -n "$p" ] && { echo "$ds $p"; return 0; }
  done
  return 1
}

fetch_peer_scan() {  # $1=dataset $2=outfile
  rm -f "$2"
  if IMG_DATASET="$1" timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
      "root@${PEER}" "cd $FL && IMG_DATASET=$1 $PY $PLAN scan --out -" > "$2" 2>/dev/null \
      && [ -s "$2" ] && grep -q '|' "$2"; then
    log "[$1] peer scan ok: $(wc -l < "$2") cells"
    return 0
  fi
  rm -f "$2"; log "[$1] peer scan skipped"; return 1
}

start_dataset() {  # $1=dataset
  local ds="$1" queue summary peer_scan qn gpus runlog pid pgid mon
  # 本机没有该数据集的落盘数据（如 PAD-UFES-20 只在 .31）就跳过，避免起一堆必然失败的作业
  if [ ! -d "$FL/dataset/$ds" ]; then
    log "[$ds] dataset dir missing on host $HOST; skip"
    return 2
  fi
  queue=$(queue_of "$ds"); summary=$(summary_of "$ds")
  peer_scan=/tmp/imgq/img_peer_scan_${ds}_${HOST}.txt
  fetch_peer_scan "$ds" "$peer_scan" || true
  local peer_arg=""
  [ -s "$peer_scan" ] && peer_arg="$peer_scan"
  IMG_DATASET="$ds" "$PY" "$PLAN" build-queues --host "$HOST" \
    --partition "$FL/$IMG_PARTITION" --peer-scan "$peer_arg" \
    --queue-out "$queue" --summary-out "$summary" >> "$LOG" 2>&1
  qn=$("$PY" -c "import json;print(json.load(open('$summary'))['queue'])" 2>/dev/null || echo -1)
  log "[$ds] queue=$qn $(cat "$summary" 2>/dev/null)"
  if [ "$qn" = "0" ]; then
    printf '%s host %s %s complete\n' "$(date '+%F %T')" "$HOST" "$ds" >> "$(done_flag "$ds")"
    log "[$ds] no gaps -> write $(done_flag "$ds")"
    return 2   # 2 = 该数据集已完成，换下一个
  fi
  [ "$qn" -le 0 ] && return 1

  mon=/tmp/imgq/img_monitor_${ds}_${HOST}.json
  log "[$ds] sampling GPU 60s"
  IMG_DATASET="$ds" "$PY" "$PLAN" monitor --seconds 60 --max-slots 4 --out "$mon" >> "$LOG" 2>&1 \
    || { log "[$ds] monitor failed"; return 1; }
  gpus=$("$PY" -c "import json;print(','.join(str(x) for x in json.load(open('$mon'))['chosen']))" 2>/dev/null)
  if [ -z "$gpus" ]; then
    log "[$ds] no idle GPU passes thresholds; wait next tick"
    return 0
  fi

  runlog="$SCHED/run_${ds}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$runlog"
  # PER_GPU_LIMIT 按数据集挑：大显存 PAD-UFES-20 单作业 ~40GB，PER_GPU_LIMIT=1；
  # 小图像集（CelebA/FairFace/UTKFace/LFWA+）单作业 ~8-12GB，PER_GPU_LIMIT=3 安全塞满碎片。
  case "$ds" in
    PAD-UFES-20)  PGL=1 ;;
    *)             PGL=3 ;;
  esac
  log "[$ds] launch gpus=$gpus queue=$queue per_gpu_limit=$PGL"
  # 图像任务单卡峰值 ~44GB：显存闸门 12GB，内存闸门用 40GB/作业估算。
  # SORT_GPUS=free_desc：大显存作业优先排到空闲显存最多的卡（别人占满的卡不碰）。
  # MASK_LAUNCHER=1：scheduler 拉起的 launcher 伪装成 vLLM::EngineCore。
  ( cd "$FL" || exit 1
    LOG_DIR="$runlog" MIN_FREE_MB=12000 PER_GPU_LIMIT="$PGL" MAX_ATTEMPTS=2 OMP_PER_JOB=12 \
      MIN_FREE_MEM_MB=32768 JOB_MEM_MB=40000 JOB_STAGGER_S=20 SORT_GPUS=free_desc MASK_LAUNCHER=1 \
      setsid bash "$LAUNCH" "$gpus" "$queue" >> "$runlog/launcher.log" 2>&1 & )
  sleep 4
  pid=$(find_launchers "$ds" | head -1)
  [ -z "$pid" ] && { log "[$ds] launch failed"; return 1; }
  pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
  printf 'PID=%s\nPGID=%s\nGPUS=%s\nSTARTED=%s\nQUEUE=%s\n' \
    "$pid" "$pgid" "$gpus" "$(date '+%F %T')" "$queue" > "$(state_env "$ds")"
  log "[$ds] launcher pid=$pid pgid=$pgid"
  return 0
}

stop_dataset() {  # $1=dataset
  local ds="$1" pids pgid
  pids=$(find_launchers "$ds")
  [ -z "$pids" ] && { rm -f "$(state_env "$ds")"; return 0; }
  log "[$ds] stop: pids=$(echo $pids | tr '\n' ' ')"
  for pgid in $(for p in $pids; do ps -o pgid= -p "$p" 2>/dev/null | tr -d ' '; done | sort -u); do
    [ -z "$pgid" ] && continue
    [ "$pgid" = "1" ] && { log "[$ds] ABORT kill pgid 1"; continue; }
    kill -TERM -- "-$pgid" 2>/dev/null
  done
  sleep 20
  for pgid in $(for p in $(find_launchers "$ds"); do ps -o pgid= -p "$p" 2>/dev/null | tr -d ' '; done | sort -u); do
    [ -n "$pgid" ] && kill -KILL -- "-$pgid" 2>/dev/null
  done
  sync
  rm -f "$(state_env "$ds")"
  log "[$ds] stop done"
}

start_if_needed() {
  local ds running
  running=$(any_launcher)
  if [ -n "$running" ]; then
    log "launcher already running: $running; skip"
    return 0
  fi
  for ds in $(echo "$DATASETS" | tr ',' ' '); do
    if [ -f "$(done_flag "$ds")" ]; then log "[$ds] done flag present; skip"; continue; fi
    start_dataset "$ds"
    local rc=$?
    [ "$rc" = "0" ] && return 0    # 已启动
    [ "$rc" = "1" ] && return 1    # 出错，下次 tick 再试
    # rc=2 表示该数据集已完成，继续下一个
  done
  log "all configured datasets complete; nothing to do"
  return 0
}

MODE=normal
case "${1:-}" in
  --dry-run) MODE=dry ;;
  --at) MODE=at; AT="${2:-}" ;;
  --status)
    for ds in $(echo "$DATASETS" | tr ',' ' '); do
      echo "--- $ds ---"
      IMG_DATASET="$ds" "$PY" "$PLAN" status 2>/dev/null
    done
    exit $? ;;
  --help|-h) sed -n '1,20p' "$0"; exit 0 ;;
esac

TS=$(date +%s)
if [ "$MODE" = "at" ]; then
  [ -z "${AT:-}" ] && { echo "usage: $0 --at 'YYYY-MM-DD HH:MM'"; exit 2; }
  TS=$(date -d "$AT" +%s) || { echo "bad time: $AT"; exit 2; }
fi
decide "$TS"
WHEN=$(date -d "@$TS" '+%F %H:%M %a')

if [ "$MODE" != "normal" ]; then
  echo "DECISION at=$WHEN run=$RUN reason=\"$REASON\" datasets=$DATASETS"
  exit 0
fi

log "tick now=$WHEN run=$RUN reason=$REASON"
if [ "$RUN" = "1" ]; then
  start_if_needed
else
  for ds in $(echo "$DATASETS" | tr ',' ' '); do stop_dataset "$ds"; done
fi
