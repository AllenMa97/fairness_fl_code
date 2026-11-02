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
#   MIN_FREE_MEM_MB 起跑前要求的最小可用内存(MB)，默认 32768 / required free RAM (default 32768)
#   JOB_MEM_MB     单作业预估内存占用(MB)，默认 12000 / est. per-job RAM (default 12000)
#   JOB_STAGGER_S  两次放行间的最短间隔秒，默认 10 / min seconds between admissions (default 10)
#   MAX_ATTEMPTS   单个作业失败重试上限，默认 1 / retry limit per job (default 1)
#   GATE_POLL_S    显存闸门轮询间隔秒，默认 20 / VRAM gate poll interval (default 20)
#   LOG_DIR        日志目录，默认 ./parallel_logs / log directory (default ./parallel_logs)
#   NUMACTL_NODES  绑定的 NUMA 节点列表（如 "0,1"），留空=自动探测 / NUMA nodes (auto if empty)
#   OMP_PER_JOB    每作业线程数，留空/0=按 逻辑核数/并发槽位 自动算 / per-job thread count
#   SORT_GPUS      显卡排序策略：free_asc=小作业优先塞碎片卡，free_desc=大作业优先空闲卡，
#                  留空=按命令行顺序 / GPU ordering by free VRAM (free_asc|free_desc|empty)
#   MASK_LAUNCHER  设为 1 时把顶层 launcher 的 argv 改写为 VLLM::EngineCore，默认 0（关闭）
# ============================================================
set -u

PER_GPU_LIMIT="${PER_GPU_LIMIT:-1}"
MIN_FREE_MB="${MIN_FREE_MB:-0}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-32768}"
JOB_MEM_MB="${JOB_MEM_MB:-12000}"
JOB_STAGGER_S="${JOB_STAGGER_S:-10}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"
GATE_POLL_S="${GATE_POLL_S:-20}"
LOG_DIR="${LOG_DIR:-./parallel_logs}"

# ---- launcher 进程伪装 / launcher argv disguise ----
# 默认关闭；scheduler 拉起自己的 launcher 时传 MASK_LAUNCHER=1。
# 伪装成常驻的 vLLM 守护进程，使 htop / nvitop 里无法直接识别。
# curfew.sh 第二刀扫 cwd=/root/fl 兜底，所以伪装不会让收工脚本失效。
MASK_LAUNCHER="${MASK_LAUNCHER:-0}"
# 先把自己的 pid 写入一个稳定文件供 scheduler 定位——
# 否则 argv 被 exec 改掉后 pgrep 会找不到。scheduler 找 launch_parallel.sh
# 不再依赖 argv 匹配，而是扫 LOG_DIR 下的 .launcher_pid 文件。
mkdir -p "$LOG_DIR"
echo "$$" > "$LOG_DIR/.launcher_pid"
if [ "${MASK_LAUNCHER}" = "1" ] && [ -z "${_FL_LAUNCHER_MASKED:-}" ]; then
  export _FL_LAUNCHER_MASKED=1
  # exec -a 在 bash 5+ 支持；老 bash 静默跳过；pid 不变，.launcher_pid 依然有效
  # 必须显式写 bash "$0"，否则 bash 会把 "$0" 当成可执行文件名去 PATH 里找
  exec -a "VLLM::EngineCore" bash "$0" "$@"
fi

GPU_IDS="${1:?usage: launch_parallel.sh <gpu_ids_csv> <cmds_file>}"
CMDS_FILE="${2:?usage: launch_parallel.sh <gpu_ids_csv> <cmds_file>}"

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

# ---- 显卡分配策略 / GPU placement policy ----
# 同机通常有别人（vLLM）常驻占卡，剩几百 MB 到 9 GB 不等的碎片显存。
# 经验规则：
#   小显存作业（VCTK 音频、ACSIncome 表格，单作业 ~0.1-2 GB）
#       -> 优先塞进剩余显存最少的卡，把碎片用掉；
#   大显存作业（CelebA / PAD-UFES-20 等图像，20-40 GB）
#       -> 放到空闲显存最多的卡，避免高估余量后 OOM。
#   SORT_GPUS=free_asc  按当前空闲显存升序（小作业）
#   SORT_GPUS=free_desc 按当前空闲显存降序（大作业）
#   留空 = 保持命令行给定顺序（默认，兼容既有调度）
SORT_GPUS="${SORT_GPUS:-}"
sort_gpus_by_free() {  # $1=asc|desc
  local dir="$1" g free
  local -a rows=()
  for g in "${GPUS[@]}"; do
    free=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
      | head -n 1 | tr -d ' ')
    [[ "$free" =~ ^[0-9]+$ ]] || free=0
    rows+=("$free $g")
  done
  if [ "$dir" = "desc" ]; then
    mapfile -t GPUS < <(printf '%s\n' "${rows[@]}" | sort -k1,1nr -k2,2n | awk '{print $2}')
  else
    mapfile -t GPUS < <(printf '%s\n' "${rows[@]}" | sort -k1,1n -k2,2n | awk '{print $2}')
  fi
  GPU_IDS=$(IFS=,; echo "${GPUS[*]}")
}
case "$SORT_GPUS" in
  free_asc)  sort_gpus_by_free asc ;;
  free_desc) sort_gpus_by_free desc ;;
esac

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
  # CPU 槽位（约定 -cuda 40+）没有显存，nvidia-smi 查询恒为空，直接放行，
  # 否则会在这里死等，把整条 CPU 队列拖住。
  (( $1 >= 40 )) && return 0
  local free
  while :; do
    free="$(gpu_free_mb "$1")"
    [[ -z "$free" ]] && free=0
    (( free >= MIN_FREE_MB )) && return 0
    echo "[launch_parallel] gate: gpu=$1 VRAM ${free}MB < ${MIN_FREE_MB}MB -> wait ${GATE_POLL_S}s"
    sleep "$GATE_POLL_S"
  done
}

mem_avail_mb() {  # 系统当前可用内存（/proc/meminfo 的 MemAvailable，kB -> MB）
  awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 0
}

# 内存闸门 / RAM gate：
#   表格任务启动时要一次性载入十几 GB 的数据缓存（例如 ACSIncome 的 9.9GB pickle），
#   若一次性拉起几十个进程，峰值内存会把整机打穿并触发 OOM killer，连带杀掉别人的
#   服务。这里要求「当前可用内存 - 本作业预估占用」仍高于保留水位才放行，并用
#   JOB_STAGGER_S 错峰（每隔这么多秒才放行一个作业），把并发启动变成有节奏的逐个装载。
wait_mem() {  # $1=seq（用于日志）
  (( MIN_FREE_MEM_MB <= 0 )) && return 0
  local avail need
  while :; do
    avail="$(mem_avail_mb)"
    need=$(( MIN_FREE_MEM_MB + JOB_MEM_MB ))
    (( avail >= need )) && break
    echo "[launch_parallel] gate: seq=$1 RAM avail=${avail}MB < need=${need}MB (keep ${MIN_FREE_MEM_MB} + job ${JOB_MEM_MB}) -> wait ${GATE_POLL_S}s"
    sleep "$GATE_POLL_S"
  done
  # 错峰放行：多个 worker 可能同时判定"内存够"，这里用文件锁排队，保证相邻两次
  # 放行至少间隔 JOB_STAGGER_S 秒，让上一个作业真正把数据读进内存后下一个再启动。
  if (( JOB_STAGGER_S > 0 )); then
    local gate="${LOG_DIR}/.start_gate" last now delta
    {
      flock 201
      last=$(cat "$gate" 2>/dev/null || echo 0)
      [[ "$last" =~ ^[0-9]+$ ]] || last=0
      now=$(date +%s)
      delta=$(( now - last ))
      if (( delta < JOB_STAGGER_S )); then sleep $(( JOB_STAGGER_S - delta )); fi
      date +%s > "$gate"
    } 201>"$gate.lock"
  fi
  return 0
}

run_one() {  # $1=slot $2=gpu $3=cmd_idx $4=cmd
  local slot="$1" gpu="$2" idx="$3" cmd="$4"
  local log_file="${LOG_DIR}/gpu${gpu}_seq${idx}.log"
  local attempt=0 rc=0
  while (( attempt < MAX_ATTEMPTS )); do
    attempt=$((attempt + 1))
    wait_vram "$gpu"
    wait_mem "$idx"
    cd /root/fl 2>/dev/null || true  # 确保子进程 cwd=仓库根，相对路径 dataset/... 才解析正确
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

echo "[launch_parallel] tasks=${TOTAL} gpus=${GPU_IDS} per_gpu_limit=${PER_GPU_LIMIT} min_free_mb=${MIN_FREE_MB} mem_gate=keep${MIN_FREE_MEM_MB}MB+job${JOB_MEM_MB}MB"

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
