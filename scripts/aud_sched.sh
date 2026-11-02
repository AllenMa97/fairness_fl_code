#!/bin/bash
# ============================================================
# 音频数据集（默认 VCTK）缺口补跑调度器
#
# 时间规则（Asia/Shanghai）：工作日 18:55 启动，次日 08:40 收工
# （08:42 curfew.sh 清场前主动收）；周五晚启动后周末全天继续，
# 直到下周一 08:40。cron 每 5 分钟唤醒，本脚本幂等：
# 运行窗口内确保 launcher 在跑，窗口外停掉。
#
# 队列文件 /tmp/imgq/aud_<ds>_<host>.txt，每个数据集一个 launcher；
# 队列里只保留「本地尚无完整结果、且本机当前没有在跑」的格，避免与
# 手工队列（如 /tmp/imgq/vctk31_{a,b,c,d}.txt）重复跑同一批实验。
#
# 用法：
#   AUD_DATASETS=VCTK aud_sched.sh            正常调度（cron 调用）
#   AUD_DATASETS=VCTK aud_sched.sh --dry-run  只打印时间判定
#   aud_sched.sh --status                     打印各数据集缺口
#   aud_sched.sh --stop-now                   立即停掉本调度器的 launcher
# ============================================================
set -u

FL=/root/fl
PY=/root/miniconda3/envs/federated/bin/python
PLAN=$FL/scripts/aud_plan.py
LAUNCH=$FL/scripts/launch_parallel.sh
SCHED_ROOT=$FL/parallel_logs/aud_sched
DATASETS="${AUD_DATASETS:-VCTK}"
AUD_GPU_PREF="${AUD_GPU_PREF:-4,5}"              # 优先使用的卡
AUD_MIN_FREE_MB="${AUD_MIN_FREE_MB:-8192}"       # 显存剩余闸门
AUD_CPU_SLOTS="${AUD_CPU_SLOTS:-46,47,48,49,50,51}"  # GPU 不满足时退回 CPU 槽
AUD_MAX_SLOTS="${AUD_MAX_SLOTS:-4}"              # 单次最多用几个槽位

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
log() { printf '%s [%s] %s\n' "$(date '+%F %T')" "$HOST" "$*" | tee -a "$LOG"; }

exec 9>"$SCHED/lock"
flock -n 9 || { echo "another aud_sched instance is running; skip" >&2; exit 0; }

# ---- 时间判定：工作日 18:55-08:40 + 周末全天 ----
decide() {
  if [ "${FORCE_RUN:-0}" = "1" ]; then RUN=1; REASON="forced by FORCE_RUN"; return 0; fi
  local ts="$1" u hm
  u=$(date -d "@$ts" +%u); hm=$(date -d "@$ts" +%H%M); hm=$((10#$hm))
  if [ "$u" -ge 6 ]; then RUN=1; REASON="weekend: Friday session keeps running"
  elif [ "$hm" -ge 1855 ] || [ "$hm" -lt 840 ]; then RUN=1; REASON="weekday 18:55-08:40: run"
  else RUN=0; REASON="weekday 08:40-18:55: stopped"; fi
}

queue_of()  { echo "/tmp/imgq/aud_${1}_${HOST}.txt"; }
summary_of(){ echo "/tmp/imgq/aud_${1}_${HOST}.summary.json"; }
done_flag() { echo "$FL/aud_${1}_done.flag"; }
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
  for p in $(pgrep -f "aud_${1}_${HOST}\.txt" 2>/dev/null); do
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

# 收集本机 + 对端正在跑的 AUD_CLF 任务格（split|algo|K）。
# 训练进程的 argv 会被 tool/process_alias.py 改写（显示为 VLLM::Worker_TPn），
# 所以 pgrep -af main_AUD_CLF.py 经常匹配不到；aud_plan.py 的 running-keys
# 以「正在运行的 launch_parallel.sh 的队列文件」为主来源、直跑进程为补充，
# 确保手工队列 vctk31_*.txt 里的格一定被剔除。
ssh_peer() {
  timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
    "root@${PEER}" "$@" 2>/dev/null
}

collect_running_keys() {  # $1=dataset $2=outfile
  : > "$2"
  AUD_DATASET="$1" "$PY" "$PLAN" running-keys --dataset "$1" >> "$2" 2>/dev/null || true
  ssh_peer "AUD_DATASET=$1 $PY $PLAN running-keys --dataset $1" >> "$2" 2>/dev/null || true
  sort -u "$2" -o "$2"
}

fetch_peer_scan() {  # $1=dataset $2=outfile
  rm -f "$2"
  if ssh_peer "AUD_DATASET=$1 $PY $PLAN scan --dataset $1" > "$2" \
      && [ -s "$2" ] && grep -q '|' "$2"; then
    log "[$1] peer scan ok: $(wc -l < "$2") cells"
    return 0
  fi
  rm -f "$2"; log "[$1] peer scan skipped"; return 1
}

peer_has_dataset() {  # $1=dataset
  [ "$(ssh_peer "test -d $FL/dataset/$1 && echo 1 || echo 0" | tr -d '[:space:]')" = "1" ]
}

# 本机已被其它 launcher 占用的卡（含手工队列与 acs/img 调度器）
busy_gpus() {
  local p q g out=""
  for p in $(pgrep -f 'launch_parallel\.sh' 2>/dev/null); do
    g=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | tail -2 | head -1)
    case "$g" in ''|*[!0-9,]*) continue ;; esac
    out="$out,$g"
  done
  echo "$out" | tr ',' '\n' | awk 'NF && $1 < 40' | sort -un | paste -sd, -
}

# 采样 30s 显存：优先 4,5 号卡，剩余显存 >= 8G 才用；否则退回 CPU 假卡号 46-51
pick_slots() {
  local busy
  busy=$(busy_gpus)
  "$PY" - "$busy" "$AUD_GPU_PREF" "$AUD_MIN_FREE_MB" "$AUD_MAX_SLOTS" "$AUD_CPU_SLOTS" <<'EOF'
import subprocess, sys, time
busy = {x for x in sys.argv[1].split(',') if x}
pref = [x for x in sys.argv[2].split(',') if x]
need = int(sys.argv[3]); maxs = int(sys.argv[4])
cpu = [x for x in sys.argv[5].split(',') if x]
free = {}
for _ in range(3):                      # 3 次 x 10s
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout
    cur = {}
    for ln in r.strip().splitlines():
        try:
            i, mu, mt = [x.strip() for x in ln.split(",")]
            cur[i] = int(mt) - int(mu)
        except ValueError:
            continue
    free = cur if not free else {k: min(free.get(k, 0), v) for k, v in cur.items()}
    time.sleep(10)
chosen = [g for g in pref if g not in busy and free.get(g, 0) >= need][:maxs]
print(",".join(chosen) if chosen else ",".join(cpu[:maxs]))
EOF
}

start_dataset() {  # $1=dataset
  local ds="$1" queue summary excl peer_scan peer_arg qn gpus runlog pid pgid inflight
  # 本机没有该数据集的落盘数据（如 VCTK 只在 .31）就跳过，避免起一堆必然失败的作业
  if [ ! -d "$FL/dataset/$ds" ]; then
    log "[$ds] dataset dir missing on host $HOST; skip"
    return 2
  fi
  queue=$(queue_of "$ds"); summary=$(summary_of "$ds")
  excl=/tmp/imgq/aud_excl_${ds}_${HOST}.txt
  collect_running_keys "$ds" "$excl"
  log "[$ds] running cells excluded=$(wc -l < "$excl")"
  peer_scan=/tmp/imgq/aud_peer_scan_${ds}_${HOST}.txt
  if ! fetch_peer_scan "$ds" "$peer_scan"; then
    # 对端有数据但扫描失败：宁可等下一轮，也不要在两机各跑一遍同样的格
    if peer_has_dataset "$ds"; then
      log "[$ds] peer unreachable but has dataset; wait next tick to avoid duplicate runs"
      return 1
    fi
  fi
  peer_arg=""
  [ -s "$peer_scan" ] && peer_arg="$peer_scan"
  AUD_DATASET="$ds" "$PY" "$PLAN" build-queues --host "$HOST" --dataset "$ds" \
    --peer-scan "$peer_arg" --exclude-keys "$excl" \
    --queue-out "$queue" --summary-out "$summary" >> "$LOG" 2>&1
  qn=$("$PY" -c "import json;print(json.load(open('$summary'))['queue'])" 2>/dev/null || echo -1)
  log "[$ds] queue=$qn $(cat "$summary" 2>/dev/null)"
  if [ "$qn" = "0" ]; then
    inflight=$("$PY" -c "import json;print(json.load(open('$summary'))['excluded_running'])" 2>/dev/null || echo 0)
    if [ "${inflight:-0}" -gt 0 ]; then
      # 缺口都被手工/其它队列占着：只是等待，不能写 done flag，否则那批跑完后不会再接手。
      log "[$ds] all remaining gaps in flight; wait next tick"
      return 0
    fi
    printf '%s host %s %s complete\n' "$(date '+%F %T')" "$HOST" "$ds" >> "$(done_flag "$ds")"
    log "[$ds] no gaps -> write $(done_flag "$ds")"
    return 2   # 2 = 该数据集已完成，换下一个
  fi
  [ "$qn" -le 0 ] && return 1

  gpus=$(pick_slots)
  if [ -z "$gpus" ]; then
    log "[$ds] no slot passes thresholds; wait next tick"
    return 0
  fi
  runlog="$SCHED/run_${ds}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$runlog"
  log "[$ds] launch slots=$gpus queue=$queue"
  # SORT_GPUS=free_asc：VCTK 单作业仅 ~0.7GB，优先塞进剩余显存最少的卡，把碎片用掉，
  # 把整块空闲的卡留给图像等大显存任务。
  # MASK_LAUNCHER=1：scheduler 拉起的 launcher 伪装成 vLLM::EngineCore。
  # VCTK 单作业 ~0.7GB 显存 + 数据已全部 decode 到 cache；PER_GPU_LIMIT=8 让 GPU 不闲
  # dataloader_num_workers 由 queue 生成脚本写入 per-cmdline（aud_plan.py），此处不动
  ( cd "$FL" || exit 1
    LOG_DIR="$runlog" MIN_FREE_MB=8000 PER_GPU_LIMIT=8 MAX_ATTEMPTS=2 OMP_PER_JOB=8 \
      MIN_FREE_MEM_MB=32768 JOB_MEM_MB=16000 JOB_STAGGER_S=15 SORT_GPUS=free_asc MASK_LAUNCHER=1 \
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
    [ "$rc" = "0" ] && return 0    # 已启动 / 在等在跑的任务
    [ "$rc" = "1" ] && return 1    # 出错，下次 tick 再试
    # rc=2 表示该数据集已完成或无数据，继续下一个
  done
  log "all configured datasets complete; nothing to do"
  return 0
}

MODE=normal
case "${1:-}" in
  --dry-run) MODE=dry ;;
  --at) MODE=at; AT="${2:-}" ;;
  --stop-now) MODE=stop ;;
  --status)
    for ds in $(echo "$DATASETS" | tr ',' ' '); do
      echo "--- $ds ---"
      if [ ! -d "$FL/dataset/$ds" ]; then
        echo "dataset dir missing on host $HOST; skipped by guard"
        continue
      fi
      AUD_DATASET="$ds" "$PY" "$PLAN" status --dataset "$ds" 2>/dev/null
    done
    exit $? ;;
  --help|-h) sed -n '1,22p' "$0"; exit 0 ;;
esac

if [ "$MODE" = "stop" ]; then
  for ds in $(echo "$DATASETS" | tr ',' ' '); do stop_dataset "$ds"; done
  exit 0
fi

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
