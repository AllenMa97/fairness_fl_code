#!/bin/bash
# ============================================================
# 多 GPU 并行实验启动器 / Multi-GPU parallel experiment launcher
#
# 场景 / Scenario:
#   租用多卡服务器时，一卡一实验并行推进（默认单卡跑单实验）。
#   One experiment per GPU, running in parallel on rented multi-GPU servers.
#
# 用法 / Usage:
#   bash scripts/launch_parallel.sh <gpu_ids_csv> <experiment_cmds_file>
#   bash scripts/launch_parallel.sh 0,1,2,3 experiments.txt
#
# experiments.txt 每行一条实验命令（不含 -cuda 参数，脚本自动注入）：
#   Each line is one experiment command WITHOUT -cuda (injected by this script):
#     main_Tabular_CLF.py -algorithm FedAvg -dataset ADULT -task Tabular_CLF ...
#     main_IMG_CLF.py -algorithm PDFFed -dataset UTKFace -task IMG_CLF ...
#
# 调度行为 / Scheduling:
#   - 任务由各卡 worker 动态领取（原子计数器），单卡同时最多 PER_GPU_LIMIT 个作业
#     （默认 1 = 严格串行）。任务数多于卡数时自动排队，绝不会把多余任务一次性
#     压到同一张卡上造成显存超配 OOM。
#     Tasks are claimed dynamically; each GPU runs at most PER_GPU_LIMIT jobs
#     (default 1 = sequential). Extra tasks queue up instead of oversubscribing.
#   - 每次起跑前可选等待显存闸门 MIN_FREE_MB（默认 0 = 不等待），用于与同机
#     其他用户的服务共存，避免抢显存。
#     Optional VRAM gate before each launch (MIN_FREE_MB, 0 = disabled).
#   - 每条实验日志写入 ${LOG_DIR}/gpu<gpu_id>_seq<cmd_idx>.log
#     Each experiment logs to ${LOG_DIR}/gpu<gpu_id>_seq<cmd_idx>.log
#   - 等待全部任务结束后统一退出
#     Waits for all tasks to finish before exiting
#
# 可用环境变量 / Environment variables:
#   PER_GPU_LIMIT  单卡并发上限，默认 1 / max concurrent jobs per GPU (default 1)
#   MIN_FREE_MB    起跑前要求的最小空闲显存(MB)，默认 0 / required free VRAM (default 0)
#   MAX_ATTEMPTS   单个作业失败重试上限，默认 1 / retry limit per job (default 1)
#   GATE_POLL_S    显存闸门轮询间隔秒，默认 20 / VRAM gate poll interval (default 20)
#   LOG_DIR        日志目录，默认 ./parallel_logs / log directory (default ./parallel_logs)
#   NUMACTL_NODES  绑定的 NUMA 节点列表（如 "0,1"），留空=自动探测 / NUMA nodes (auto if empty)
#   OMP_PER_JOB    每作业线程数，留空/0=按 逻辑核数/并发槽位 自动算 / per-job thread count
# ============================================================
set -u

GPU_IDS="${1:?usage: launch_parallel.sh <gpu_ids_csv> <cmds_file>}"
CMDS_FILE="${2:?usage: launch_parallel.sh <gpu_ids_csv> <cmds_file>}"

PER_GPU_LIMIT="${PER_GPU_LIMIT:-1}"
MIN_FREE_MB="${MIN_FREE_MB:-0}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"
GATE_POLL_S="${GATE_POLL_S:-20}"
LOG_DIR="${LOG_DIR:-./parallel_logs}"

# ---- CPU 绑核与线程预算 / CPU affinity & thread budget (P2) ----
# 同机并发多个训练进程时，若各自都用满整机线程，会在核心间互相抢占并跨 NUMA
# 节点迁移。这里按并发槽位数把整机逻辑核数均分给每个作业（OMP/MKL_NUM_THREADS），
# 并用 numactl --cpunodebind 把各槽位轮转绑定到不同 NUMA 节点；numactl 缺失或
# 单节点时自动跳过绑核。
#   NUMACTL_NODES  显式指定节点号列表（如 "0,1"）；留空 = 自动探测
#   OMP_PER_JOB    每作业线程数；留空或 0 = 按 逻辑核数/槽位数 自动计算
NUMACTL_NODES="${NUMACTL_NODES:-}"
OMP_PER_JOB="${OMP_PER_JOB:-0}"
AFFINITY_NODES=()

cpu_affinity_init() {  # $1=并发槽位总数
  local slots="${1:-1}" cores
  (( slots < 1 )) && slots=1
  cores="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
  if (( OMP_PER_JOB <= 0 )); then
    OMP_PER_JOB=$(( cores / slots ))
    (( OMP_PER_JOB < 1 )) && OMP_PER_JOB=1
  fi
  if [ -n "$NUMACTL_NODES" ]; then
    IFS=',' read -r -a AFFINITY_NODES <<< "$NUMACTL_NODES"
  elif command -v numactl >/dev/null 2>&1; then
    mapfile -t AFFINITY_NODES < <(numactl --hardware 2>/dev/null \
      | awk '/^node [0-9]+ cpus:/ {print $2}')
  fi
}

cpu_bind_prefix() {  # $1=槽位号 -> echo（可能为空的）numactl 前缀
  local slot="$1" n=${#AFFINITY_NODES[@]}
  if (( n > 1 )); then
    printf 'numactl --cpunodebind=%s ' "${AFFINITY_NODES[$(( slot % n ))]}"
  fi
}

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

# 按并发槽位总数均分线程并按槽位轮转绑 NUMA 节点（P2）。
cpu_affinity_init "$(( NUM_GPUS * PER_GPU_LIMIT ))"

mkdir -p "$LOG_DIR"

# 读取有效命令：剔除空行与整行注释 / read runnable commands (skip blank & comment lines)
mapfile -t CMDS < <(awk 'NF && $1 !~ /^#/' "$CMDS_FILE")
TOTAL=${#CMDS[@]}
if (( TOTAL == 0 )); then
  echo "[launch_parallel] no runnable commands in ${CMDS_FILE}" >&2
  exit 1
fi

COUNTER="$(mktemp)"
LOCK_FILE="${COUNTER}.lock"
echo 0 > "$COUNTER"
cleanup() { rm -f "$COUNTER" "$LOCK_FILE"; }
trap cleanup EXIT

claim() {  # 原子领取下一个任务下标 / atomically claim next task index
  local idx
  {
    flock 200
    idx=$(cat "$COUNTER")
    echo $((idx + 1)) > "$COUNTER"
  } 200>"$LOCK_FILE"
  echo "$idx"
}

gpu_free_mb() {  # $1=物理 GPU 序号
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
    | head -n 1 | tr -d ' '
}

wait_vram() {  # $1=gpu：等待该卡空闲显存达到闸门
  (( MIN_FREE_MB <= 0 )) && return 0
  local free
  while :; do
    free="$(gpu_free_mb "$1")"
    [[ -z "$free" ]] && free=0
    (( free >= MIN_FREE_MB )) && return 0
    echo "[launch_parallel] gate: gpu=$1 VRAM ${free}MB < ${MIN_FREE_MB}MB -> wait ${GATE_POLL_S}s"
    sleep "$GATE_POLL_S"
  done
}

run_one() {  # $1=slot $2=gpu $3=cmd_idx $4=cmd
  local slot="$1" gpu="$2" idx="$3" cmd="$4"
  local log_file="${LOG_DIR}/gpu${gpu}_seq${idx}.log"
  local attempt=0 rc=0
  while (( attempt < MAX_ATTEMPTS )); do
    attempt=$((attempt + 1))
    wait_vram "$gpu"
    echo "[launch_parallel] gpu=${gpu} seq=${idx} attempt=${attempt} omp=${OMP_PER_JOB} node=$(cpu_bind_prefix "$slot") log=${log_file}"
    echo "  cmd: ${cmd} -cuda ${gpu}"
    # env 承接 per-job 线程数；numactl 前缀（可能为空）把作业钉到某个 NUMA 节点。
    # CUDA_VISIBLE_DEVICES 双保险（部分框架直读设备号，部分读环境变量）
    $(cpu_bind_prefix "$slot") env \
      OMP_NUM_THREADS="$OMP_PER_JOB" MKL_NUM_THREADS="$OMP_PER_JOB" \
      CUDA_VISIBLE_DEVICES="$gpu" ${cmd} -cuda "$gpu" > "$log_file" 2>&1
    rc=$?
    (( rc == 0 )) && return 0
    echo "[launch_parallel] gpu=${gpu} seq=${idx} attempt=${attempt} exited rc=${rc}"
    sleep 10
  done
  echo "[launch_parallel] gpu=${gpu} seq=${idx} FAILED after ${MAX_ATTEMPTS} attempt(s)" >&2
  return 1
}

worker() {  # $1=slot：该并发槽位（用于绑核） $2=gpu：该卡的调度循环
  local slot="$1" gpu="$2" idx
  while :; do
    idx="$(claim)"
    (( idx >= TOTAL )) && break
    run_one "$slot" "$gpu" "$idx" "${CMDS[$idx]}" || true
  done
}

echo "[launch_parallel] tasks=${TOTAL} gpus=${GPU_IDS} per_gpu_limit=${PER_GPU_LIMIT} min_free_mb=${MIN_FREE_MB}"

pids=()
for (( gi=0; gi<NUM_GPUS; gi++ )); do
  for (( c=0; c<PER_GPU_LIMIT; c++ )); do
    worker "$(( gi * PER_GPU_LIMIT + c ))" "${GPUS[$gi]}" &
    pids+=($!)
  done
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=$((fail + 1))
done
echo "[launch_parallel] all done; worker_failures=${fail}"
exit "$fail"
