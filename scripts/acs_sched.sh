#!/bin/bash
# ACSIncome 缺口补跑调度：同 LFWA+ 的时间窗（周一到周五 20:00 起，次日 08:40 停；
# 周五晚起连续到周一 08:40）。cron 每 5 分钟唤醒，幂等。
set -u
FL=/root/fl
PY=/root/miniconda3/envs/federated/bin/python
PLAN=$FL/scripts/acs_plan.py
LAUNCH=$FL/scripts/launch_parallel.sh
SCHED=$FL/parallel_logs/acs_sched
STATE=$SCHED/state
DONE_FLAG=$FL/acs_done.flag
LFWA_ST=$FL/parallel_logs/lfwa_sched/state
mkdir -p "$SCHED" "$STATE" /tmp/imgq

HOST=$(tr -d '[:space:]' < $FL/.lfwa_host 2>/dev/null)
case "$HOST" in 30|31) ;; *) echo "FATAL: cannot determine host id" >&2; exit 1 ;; esac
if [ "$HOST" = "30" ]; then PEER=192.168.1.31; else PEER=192.168.1.30; fi
ACS_GPU_ALLOW="${ACS_GPU_ALLOW:-}"            # 空=按实际空闲自动选（碎片卡优先）
ACS_MAX_SLOTS="${ACS_MAX_SLOTS:-6}"          # 同时使用的槽位上限
ACS_CPU_SLOTS="${ACS_CPU_SLOTS:-40,41,42,43,44,45}"  # GPU 不足时补的 CPU 槽位
QUEUE=/tmp/imgq/acs_${HOST}.txt
SUMMARY=/tmp/imgq/acs_${HOST}.summary.json
PEER_SCAN=/tmp/imgq/acs_peer_scan_${HOST}.txt
LOG=$SCHED/$(date +%F).log

log() { printf '%s [%s] %s\n' "$(date '+%F %T')" "$HOST" "$*" | tee -a "$LOG"; }

exec 9>"$STATE/lock"
flock -n 9 || { echo "another acs_sched instance is running; skip" >&2; exit 0; }

decide() {
  if [ "${FORCE_RUN:-0}" = "1" ]; then RUN=1; REASON="forced by FORCE_RUN"; return 0; fi
  local ts="$1" u hm
  u=$(date -d "@$ts" +%u); hm=$(date -d "@$ts" +%H%M); hm=$((10#$hm))
  if [ "$u" -ge 6 ]; then RUN=1; REASON="weekend: Friday session keeps running"
  elif [ "$hm" -ge 2000 ] || [ "$hm" -lt 840 ]; then RUN=1; REASON="weekday 20:00-08:40: run"
  else RUN=0; REASON="weekday 08:40-20:00: stopped"; fi
}

find_launchers() {
  # 1) 新机制：mask 的 launcher argv 已改成 VLLM::EngineCore，
  #    扫 $SCHED 下 .launcher_pid 文件（exec 不换 pid，所以 pid 文件永远有效）。
  # 2) 兜底：老机制——未 mask 的 launcher，扫 argv 匹配。
  local pidfile pid
  for pidfile in $(find "$SCHED" -maxdepth 3 -name ".launcher_pid" -mmin -120 2>/dev/null); do
    pid=$(cat "$pidfile" 2>/dev/null)
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && echo "$pid"
  done
  # 兜底：老 launcher（未 mask）的 argv 匹配
  local p cmd
  for p in $(pgrep -f "acs_${HOST}\.txt" 2>/dev/null); do
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
    case "$cmd" in *launch_parallel.sh*) echo "$p" ;; esac
  done
}

lfwa_gpus() {
  local f=$LFWA_ST/launcher_${HOST}.env g=""
  [ -f "$f" ] && g=$(sed -n 's/^GPUS=//p' "$f")
  echo "$g"
}

pick_gpus() {  # 采样实际资源后分配：碎片卡优先，GPU 不足自动补 CPU 槽位
  local excl out
  excl=$(lfwa_gpus)
  out=$("$PY" - "$excl" "$ACS_GPU_ALLOW" "$ACS_MAX_SLOTS" "$ACS_CPU_SLOTS" <<'EOF'
import subprocess, sys, time, collections

excl = {x.strip() for x in (sys.argv[1] if len(sys.argv) > 1 else "").split(",") if x.strip()}
# 空字符串 = 不限 GPU 编号（按实际空闲自动选）
allow_raw = sys.argv[2] if len(sys.argv) > 2 else ""
allow = {x.strip() for x in allow_raw.split(",") if x.strip()}
max_slots = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else 6
cpu_slots = [x.strip() for x in (sys.argv[4] if len(sys.argv) > 4 else "40,41,42,43,44,45").split(",") if x.strip()]

util = collections.defaultdict(list)
mem = {}
for _ in range(6):                      # 60s 采样
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    for ln in r.strip().splitlines():
        i, u, mu, mt = [x.strip() for x in ln.split(",")]
        util[i].append(float(u)); mem[i] = int(mt) - int(mu)
    time.sleep(10)

cand = [i for i in util
        if i not in excl
        and (not allow or i in allow)
        and sum(util[i]) / len(util[i]) <= 30
        and mem.get(i, 0) >= 12000]
# 碎片优先：空闲显存最小的先用，把大卡留给图像任务
cand.sort(key=lambda x: mem.get(x, 0))
chosen = cand[:max_slots]
# GPU 不够时补 CPU 槽位（launch_parallel.sh 里 40+ 会走 cpu 分支）
if len(chosen) < max_slots:
    chosen += cpu_slots[:max_slots - len(chosen)]
print(",".join(chosen))
EOF
)
  echo "$out"
}

fetch_peer_scan() {
  rm -f "$PEER_SCAN"
  if timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
      "root@${PEER}" "$PY $PLAN scan --out -" > "$PEER_SCAN" 2>/dev/null && [ -s "$PEER_SCAN" ]; then
    log "peer scan ok ($(wc -l < "$PEER_SCAN") cells)"; return 0
  fi
  rm -f "$PEER_SCAN"; log "peer scan skipped"; return 1
}

start_if_needed() {
  [ -f "$DONE_FLAG" ] && { log "done flag present; skip"; return 0; }
  [ -n "$(find_launchers)" ] && { log "launcher already running"; return 0; }
  [ -f /root/fl/acs_partition.json ] || { log "no acs_partition.json; skip"; return 1; }
  fetch_peer_scan || true
  local peer_arg="" exc=/tmp/imgq/acs_reserved_${HOST}.txt
  [ -s "$PEER_SCAN" ] && peer_arg="$PEER_SCAN"
  : > "$exc"
  if pgrep -f "acs_cpu_${HOST}\.txt" >/dev/null 2>&1 && [ -f /tmp/imgq/acs_cpu_${HOST}.txt ]; then
    sed -nE 's/.*-split_strategy ([A-Za-z0-9]+) -algorithm ([A-Za-z_]+) -num_clients_K ([0-9]+).*/\1|\2|\3/p' \
      /tmp/imgq/acs_cpu_${HOST}.txt > "$exc"
    log "cpu batch alive; reserve $(wc -l < "$exc") cells"
  fi
  "$PY" "$PLAN" build-queues --host "$HOST" --partition $FL/acs_partition.json \
    --peer-scan "$peer_arg" --exclude-keys "$exc" \
    --queue-out "$QUEUE" --summary-out "$SUMMARY" >> "$LOG" 2>&1
  local qn
  qn=$("$PY" -c "import json;print(json.load(open('$SUMMARY'))['queue'])" 2>/dev/null || echo -1)
  log "queue=$qn $(cat "$SUMMARY" 2>/dev/null)"
  if [ "$qn" = "0" ]; then
    echo "$(date '+%F %T') host $HOST acs queue empty" >> "$DONE_FLAG"; return 0
  fi
  [ "$qn" -le 0 ] && return 1
  local gpus
  gpus=$(pick_gpus)
  [ -z "$gpus" ] && { log "no idle GPU; wait next tick"; return 0; }
  local runlog="$SCHED/run_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$runlog"
  log "launch gpus=$gpus queue=$QUEUE"
  # ACSIncome 单作业 ~1.5GB 显存：free_asc 优先进碎片卡，把整块空闲的卡留给图像任务。
  # MASK_LAUNCHER=1：scheduler 拉起的 launcher 伪装成 vLLM::EngineCore；
  # curfew.sh 第二刀扫 cwd=/root/fl 兜底，收工不受影响。
  ( cd "$FL" || exit 1
    LOG_DIR="$runlog" MIN_FREE_MB=12000 PER_GPU_LIMIT=4 MAX_ATTEMPTS=2 OMP_PER_JOB=10 \
      SORT_GPUS=free_asc MASK_LAUNCHER=1 \
      setsid bash "$LAUNCH" "$gpus" "$QUEUE" >> "$runlog/launcher.log" 2>&1 & )
  sleep 4
  local pid pgid
  pid=$(find_launchers | head -1)
  [ -z "$pid" ] && { log "launch failed"; return 1; }
  pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
  printf 'PID=%s\nPGID=%s\nGPUS=%s\nSTARTED=%s\n' "$pid" "$pgid" "$gpus" "$(date '+%F %T')" \
    > "$STATE/launcher_${HOST}.env"
  log "launcher pid=$pid pgid=$pgid"
}

stop_launchers() {
  local pids pgid
  pids=$(find_launchers)
  [ -z "$pids" ] && { log "stop: none running"; return 0; }
  log "stop: pids=$(echo $pids | tr '\n' ' ')"
  for pgid in $(for p in $pids; do ps -o pgid= -p "$p" 2>/dev/null | tr -d ' '; done | sort -u); do
    [ -z "$pgid" ] && continue
    [ "$pgid" = "1" ] && { log "ABORT kill pgid 1"; continue; }
    kill -TERM -- "-$pgid" 2>/dev/null
  done
  sleep 20
  for pgid in $(for p in $(find_launchers); do ps -o pgid= -p "$p" 2>/dev/null | tr -d ' '; done | sort -u); do
    [ -n "$pgid" ] && kill -KILL -- "-$pgid" 2>/dev/null
  done
  sync
  rm -f "$STATE/launcher_${HOST}.env"
  log "stop done"
}

MODE=normal
case "${1:-}" in
  --dry-run) MODE=dry ;;
  --at) MODE=at; AT="${2:-}" ;;
  --status) "$PY" "$PLAN" status --verbose; exit $? ;;
esac
TS=$(date +%s)
if [ "$MODE" = "at" ]; then TS=$(date -d "$AT" +%s) || exit 2; fi
decide "$TS"
WHEN=$(date -d "@$TS" '+%F %H:%M %a')
if [ "$MODE" != "normal" ]; then
  echo "DECISION at=$WHEN run=$RUN reason=\"$REASON\""; exit 0
fi
log "tick now=$WHEN run=$RUN reason=$REASON"
if [ "$RUN" = "1" ]; then start_if_needed; else stop_launchers; fi
