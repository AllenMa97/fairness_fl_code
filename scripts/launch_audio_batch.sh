#!/bin/bash
# =============================================================================
# 语音（AUD_CLF）实验批量并发启动器 / Audio-experiment batch launcher
#
# 背景 / Background:
#   跨模态评测的语音一档：CREMA-D（情感语音，7442 条、91 位演员）+ 自研
#   AUDIOCNNRNN（2 层 Conv1d + 双向 GRU，见 hypothesis/AUDIOCNNRNN.py）。
#   音频模型远小于图像/文本档，单作业显存与算力占用都很低，因此本脚本让 GPU 槽位
#   与 CPU 槽位「同时开工、共用一个任务队列」，但保证 GPU 优先：任务清单头部的
#   gpu_slots 个任务由 GPU 槽位按下标直接认领（共享计数器从 gpu_slots 起），CPU
#   槽位只能拿到其后的任务；同时用 numactl 把 GPU 作业钉在一半物理核、CPU 作业钉在
#   另一半物理核，两类作业互不抢核——既满足「先用 GPU，随后用 CPU」的优先级，又
#   做到「资源最大化利用」。
#
#   The audio model is far smaller than the image/text ones, so GPU slots and CPU
#   slots start together and share one task queue. GPU priority is kept by letting
#   GPU slots claim the head of the task list directly (the shared counter starts at
#   gpu_slots), and numactl pins GPU jobs to one half of the cores and CPU jobs to the
#   other half so the two kinds never contend.
#
# 与其它批次启动器的关系 / Relation to other batch launchers:
#   结构照搬 scripts/launch_sent_batch.sh（bash 数组逐元素传参，避免
#   -tb_monitor '{"...": false}' 这类 JSON 实参被词分割拆散）。
#
# 用法 / Usage:
#   bash scripts/launch_audio_batch.sh
#   GPU_IDS=0,1,3 SLOTS_PER_GPU=3 bash scripts/launch_audio_batch.sh
#   ALGORITHMS_OVERRIDE="FedAvg,LoGoFair,FedFair" SPLITS_OVERRIDE="Dirichlet05" \
#     COMM_ROUNDS=2 CPU_SLOTS=0 bash scripts/launch_audio_batch.sh
#   GPU_IDS="" CPU_SLOTS=12 bash scripts/launch_audio_batch.sh   # 纯 CPU 跑
#   CPU_SLOTS=-1 bash scripts/launch_audio_batch.sh              # 纯 GPU 跑
#
# 环境变量 / Env vars:
#   GPU_IDS / SLOTS_PER_GPU      可用卡与每卡槽位，默认 "0,1,3" / 3
#                                留空 GPU_IDS = 不用 GPU（全部走 CPU）
#   CPU_SLOTS                    CPU 槽位数；0 = 自动（分给 CPU 的核数 /
#                                CPU_OMP），负数 = 不启用 CPU 槽位（纯 GPU）
#   CPU_OMP                      每个 CPU 槽位的线程数（= 绑定的核数），
#                                0 = 自动（分给 CPU 的核数 / CPU_SLOTS）
#   HOST_FREE_MB / GPU_FREE_MB   投递门限（宿主可用内存 / 目标卡空闲显存），
#                                默认 8000 / 3000（音频作业峰值远低于图像档）
#   DATASET                      CREMA-D 或 AudioMNIST，默认 CREMA-D
#   SYSTEM_DATA_COUNT            训练与测试样本上限；留空 = 全量
#   NUM_CLIENTS_K / COMM_ROUNDS / EPOCH_T / REPEAT_TIMES   默认 20 / 10 / 2 / 1
#   BATCH_SIZE / TEST_BATCH_SIZE                           默认 64 / 64
#   DP_FEATURE_BOUND            PDFFed_DP 的特征逐维先验上界 B，默认 24
#                               （音频需按实测特征幅度回填，否则 DP 保证不成立）
#   ALGORITHMS_OVERRIDE / SPLITS_OVERRIDE   临时子集（逗号分隔），不改默认排期
#   NUMACTL_NODES               参与切分的主机 NUMA 节点列表（如 "0,1"），
#                                留空 = 自动探测；节点数 >= 2 时 GPU 作业钉在
#                                首个节点、CPU 作业钉在其余节点，否则按核数对半切
#   OMP_PER_JOB                 GPU 作业线程数，留空/0 = 按 分给 GPU 的核数/槽位 自动算
#   LOG_DIR                     日志与 counter 目录，默认 ./parallel_logs/audio_batch
#
# 产物 / Artifacts:
#   parallel_logs/audio_batch/gpu<id>_<algo>_<split>.log   （CPU 作业为 cpu<N>_...）
#   parallel_logs/audio_batch/done.tsv   # split<TAB>algo<TAB>OK|FAILED
#   parallel_logs/audio_batch/launcher.log
#
# 运维提示 / Ops note:
#   入口在启动时调用 apply_process_alias()，进程 COMMAND 被改写成
#   VLLM::Worker_TPn，无法用 pgrep -f 认出自己的作业，只能靠环境变量
#   VLLM_WORKER_RANK=<sha1("<算法>|<数据集>|<划分>") 前 12 位>：
#     for p in $(ls /proc | grep -E '^[0-9]+$'); do
#       grep -qa '^VLLM_WORKER_RANK=' /proc/$p/environ 2>/dev/null && \
#         echo "$p $(tr '\0' ' ' < /proc/$p/cmdline)"
#     done
# =============================================================================
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="${PY:-/root/miniconda3/envs/federated/bin/python}"
ENTRY="${ENTRY:-main_AUD_CLF.py}"

GPU_IDS="${GPU_IDS-0,1,3}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"

HOST_FREE_MB="${HOST_FREE_MB:-8000}"
GPU_FREE_MB="${GPU_FREE_MB:-3000}"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"
RETRY_SLEEP="${RETRY_SLEEP:-30}"
RAM_POLL_SLEEP="${RAM_POLL_SLEEP:-60}"

BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-64}"

# CREMA-D：7442 条、91 位演员（说话人分层划分，避免同一演员跨集泄漏）。
# 留空 SYSTEM_DATA_COUNT 表示用全量；冒烟时才给一个小值。
DATASET="${DATASET:-CREMA-D}"
SYSTEM_DATA_COUNT="${SYSTEM_DATA_COUNT:-}"
NUM_CLIENTS_K="${NUM_CLIENTS_K:-20}"
COMM_ROUNDS="${COMM_ROUNDS:-10}"
EPOCH_T="${EPOCH_T:-2}"
REPEAT_TIMES="${REPEAT_TIMES:-1}"

# PDFFed_DP 的特征逐维先验上界 B（只被 use_dp=True 的算法消费）。
# 音频特征幅度与文本档量级相近，先沿用文本档 24；正式投递 PDFFed_DP 前，
# 必须用训练日志里的实测 B_obs 回填，否则敏感度被低估、DP 保证不成立。
DP_FEATURE_BOUND="${DP_FEATURE_BOUND:-24}"

HYPOTHESIS="AUDIOCNNRNN"

# ---- 绑核与线程预算 / core affinity & thread budget ----
#   GPU 槽位与 CPU 槽位同时运行，因此主机物理核必须事先分家：默认把参与的主机
#   核按 NUMA 节点对半切——第一个节点（通常是 node0）归 GPU 作业，其余节点归
#   CPU 作业；只有一个节点时按核数对半切。每一侧内部再切成「槽位数」段连续核，
#   每个作业独占一段，既不跨节点迁移、也不与另一类作业抢核。
#   numactl 缺失时退回 taskset -c，两者都没有则不绑核（仅作降级）。
#
#   GPU and CPU slots run at the same time, so the host cores are split up front:
#   by NUMA node when there are >= 2 nodes (first node -> GPU, rest -> CPU),
#   otherwise split in half by core count. Each side is then cut into one
#   contiguous core range per slot, so no job shares cores with another.
NUMACTL_NODES="${NUMACTL_NODES:-}"
OMP_PER_JOB="${OMP_PER_JOB:-0}"
CPU_SLOTS="${CPU_SLOTS:-0}"        # 0=自动; 负数=不启用 CPU 槽位
CPU_OMP="${CPU_OMP:-0}"            # 0=自动（CPU 侧核数 / CPU_SLOTS）
GPU_CORE_RANGES=()
CPU_CORE_RANGES=()
RANGES_OUT=()

expand_cpu_list() {  # $1="0-47,96-143" -> "0 1 ... 47 96 ... 143"
  local spec="$1" part a b i
  local parts=()
  IFS=',' read -r -a parts <<< "$spec"
  for part in "${parts[@]}"; do
    if [[ "$part" == *-* ]]; then
      a="${part%-*}"; b="${part#*-}"
      for (( i = a; i <= b; i++ )); do printf '%s ' "$i"; done
    else
      printf '%s ' "$part"
    fi
  done
}

build_core_ranges() {  # $1=段数  $2..=核列表  -> 写入全局 RANGES_OUT（每段为逗号串）
  local slots="$1"; shift
  local cores_list=("$@")
  local total=${#cores_list[@]}
  RANGES_OUT=()
  (( slots < 1 || total == 0 )) && return 0
  (( slots > total )) && slots="$total"
  local per rem
  per=$(( total / slots ))
  rem=$(( total - per * slots ))
  local cursor=0 i j n seg
  for (( i = 0; i < slots; i++ )); do
    n="$per"
    (( i < rem )) && n=$(( n + 1 ))
    seg=""
    for (( j = cursor; j < cursor + n; j++ )); do seg+="${cores_list[$j]},"; done
    RANGES_OUT+=( "${seg%,}" )
    cursor=$(( cursor + n ))
  done
}

range_bind_prefix() {  # $1=核串（可能为空） -> 绑核前缀（可能为空）
  local core_range="$1"
  [ -z "$core_range" ] && return 0
  if command -v numactl >/dev/null 2>&1; then
    printf 'numactl --physcpubind=%s ' "$core_range"
  elif command -v taskset >/dev/null 2>&1; then
    printf 'taskset -c %s ' "$core_range"
  fi
}

cpu_affinity_init() {  # $1=GPU 并发槽位数  $2=1 表示禁用 CPU 槽位
  local gpu_slots="${1:-0}" cpu_disabled="${2:-0}" cores
  cores="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

  # 主机核：先探测每个 NUMA 节点的核集合（numactl 给出的是 "0-47,96-143" 这类
  # 紧凑区间串），再用 NUMACTL_NODES（节点 id 列表）过滤出本次要用的节点。
  local node_ids=() node_specs=() all_cores=() gpu_cores=() cpu_cores=() spec k
  if command -v numactl >/dev/null 2>&1; then
    while read -r nid nspec; do
      node_ids+=("$nid"); node_specs+=("$nspec")
    done < <(numactl --hardware 2>/dev/null \
      | awk '/^node [0-9]+ cpus:/ {nid=$2; sub(/^node [0-9]+ cpus: */, ""); print nid, $0}')
  fi

  if [ -n "$NUMACTL_NODES" ]; then
    local want=() keep_ids=() keep_specs=() idx w
    IFS=',' read -r -a want <<< "$NUMACTL_NODES"
    for (( idx = 0; idx < ${#node_ids[@]}; idx++ )); do
      for w in "${want[@]}"; do
        if [ "$w" = "${node_ids[$idx]}" ]; then
          keep_ids+=( "${node_ids[$idx]}" )
          keep_specs+=( "${node_specs[$idx]}" )
        fi
      done
    done
    node_ids=( ${keep_ids[@]+"${keep_ids[@]}"} )
    node_specs=( ${keep_specs[@]+"${keep_specs[@]}"} )
  fi

  for spec in ${node_specs[@]+"${node_specs[@]}"}; do
    # shellcheck disable=SC2207
    all_cores+=( $(expand_cpu_list "$spec") )
  done
  if (( ${#all_cores[@]} == 0 )); then
    all_cores=( $(seq 0 $(( cores - 1 ))) )
  fi

  if (( gpu_slots == 0 )); then
    # 纯 CPU：全部核归 CPU 侧
    cpu_cores=( ${all_cores[@]+"${all_cores[@]}"} )
  elif (( cpu_disabled == 1 )); then
    # 纯 GPU：全部核归 GPU 侧
    gpu_cores=( ${all_cores[@]+"${all_cores[@]}"} )
  elif (( ${#node_specs[@]} >= 2 )); then
    # 多 NUMA 节点：首个节点给 GPU，其余给 CPU
    gpu_cores=( $(expand_cpu_list "${node_specs[0]}") )
    for (( k = 1; k < ${#node_specs[@]}; k++ )); do
      # shellcheck disable=SC2207
      cpu_cores+=( $(expand_cpu_list "${node_specs[$k]}") )
    done
  else
    # 单节点：按核数对半切
    local half=$(( ${#all_cores[@]} / 2 ))
    gpu_cores=( "${all_cores[@]:0:half}" )
    cpu_cores=( "${all_cores[@]:half}" )
  fi

  # ---- 线程预算 / thread budget ----
  if (( OMP_PER_JOB <= 0 )); then
    if (( gpu_slots > 0 )); then
      OMP_PER_JOB=$(( ${#gpu_cores[@]} / gpu_slots ))
    else
      OMP_PER_JOB=1
    fi
  fi
  (( OMP_PER_JOB < 1 )) && OMP_PER_JOB=1

  if (( cpu_disabled == 1 || ${#cpu_cores[@]} == 0 )); then
    CPU_SLOTS=0
  elif (( CPU_SLOTS == 0 )); then
    (( CPU_OMP < 1 )) && CPU_OMP=16
    CPU_SLOTS=$(( ${#cpu_cores[@]} / CPU_OMP ))
    (( CPU_SLOTS < 1 )) && CPU_SLOTS=1
  elif (( CPU_OMP < 1 )); then
    CPU_OMP=$(( ${#cpu_cores[@]} / CPU_SLOTS ))
    (( CPU_OMP < 1 )) && CPU_OMP=1
  fi

  # ---- 每槽位一段连续核 / one contiguous core range per slot ----
  build_core_ranges "$gpu_slots" ${gpu_cores[@]+"${gpu_cores[@]}"}
  GPU_CORE_RANGES=( ${RANGES_OUT[@]+"${RANGES_OUT[@]}"} )
  build_core_ranges "$CPU_SLOTS" ${cpu_cores[@]+"${cpu_cores[@]}"}
  CPU_CORE_RANGES=( ${RANGES_OUT[@]+"${RANGES_OUT[@]}"} )
}

# 算法清单（顺序 = 排期优先级；任务按「算法优先」派发）
#   第 0 批：基线 + 与本轮改动耦合最紧的机制（分群组正则 / 群组互斥）
#   FedFACT / PraFFL 只支持 SENT_CLF，不在本清单内。
ALGORITHMS=(
  # 第 0 批
  FedAvg LoGoFair FedFair FairFed
  # 第 1 批
  FedProx FedNova FederatedAverageWithPo
  # 第 2 批
  Scaffold FedRep FedProto mFairFL
  # 第 3 批
  FedFB FL_FairBatch FedRenyi
  # 第 4 批
  PDFFed_DP
)
if [ -n "${ALGORITHMS_OVERRIDE:-}" ]; then
  IFS=', ' read -r -a ALGORITHMS <<< "$ALGORITHMS_OVERRIDE"
fi

# 划分协议：主设定 α=0.5，消融补 α∈{0.1,1.0} 与均匀划分
SPLITS=(Dirichlet01 Dirichlet05 Dirichlet1 Uniform)
if [ -n "${SPLITS_OVERRIDE:-}" ]; then
  IFS=', ' read -r -a SPLITS <<< "$SPLITS_OVERRIDE"
fi

# TensorBoard 观察参数分级（与 launch_sent_batch.sh 同一口径）：
# 只有基线三件套记全量逐轮探针，其余精简，避免探针开销盖过音频任务本身。
FULL_TB_ALGOS=" FedAvg FedProx FedNova "
FULL_TB_MONITOR='{"fisher": true, "sharpness": true, "embedding": true, "activation": true, "gradient": true, "neural_collapse": true, "client_divergence": true}'
LITE_TB_MONITOR='{"fisher": false, "sharpness": false, "embedding": false, "activation": false, "gradient": false, "neural_collapse": false, "client_divergence": false}'

COMMON_ARGS=(
  -task AUD_CLF
  -dataset "$DATASET"
  -num_clients_K "$NUM_CLIENTS_K"
  -communication_round_I "$COMM_ROUNDS"
  -algorithm_epoch_T "$EPOCH_T"
  -exp_repeat_times "$REPEAT_TIMES"
  -parallel_repeats 1
  -final_artifact_policy metrics_only
  -dataloader_num_workers 4
  -dp_feature_bound "$DP_FEATURE_BOUND"
  -resume
)
if [ -n "$SYSTEM_DATA_COUNT" ]; then
  COMMON_ARGS+=( -system_data_count "$SYSTEM_DATA_COUNT" )
fi

LOG_DIR="${LOG_DIR:-./parallel_logs/audio_batch}"
mkdir -p "$LOG_DIR"
DONE_TSV="$LOG_DIR/done.tsv"
touch "$DONE_TSV"

GPU_ARR=()
if [ -n "$GPU_IDS" ]; then
  IFS=',' read -r -a GPU_ARR <<< "$GPU_IDS"
fi
NUM_GPUS=${#GPU_ARR[@]}

log() { echo "[$(date '+%F %T')] $*"; }

# 按算法返回对应的 TensorBoard 观察参数 JSON
tb_monitor_for() {
  case "$FULL_TB_ALGOS" in
    *" $1 "*) printf '%s' "$FULL_TB_MONITOR" ;;
    *) printf '%s' "$LITE_TB_MONITOR" ;;
  esac
}

# 作业标记：进程 COMMAND 被伪装成 VLLM::Worker_TPn 后只能用这个短哈希识别
marker_for() {
  printf '%s' "$1" | sha1sum | cut -c1-12
}

mem_available_mb() {
  awk '/^MemAvailable:/ {printf "%d", $2/1024}' /proc/meminfo
}

gpu_free_mb() {  # $1=物理 GPU 序号
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
    | head -n 1 | tr -d ' '
}

wait_for_host_ram() {  # $1=host need MB
  local need="$1" avail
  while :; do
    avail="$(mem_available_mb)"
    (( avail >= need )) && return 0
    log "gate(host): RAM ${avail}MB/${need}MB -> wait ${RAM_POLL_SLEEP}s"
    sleep "$RAM_POLL_SLEEP"
  done
}

wait_for_resources() {  # $1=host need MB  $2=gpu id（空=CPU 作业）  $3=gpu need MB
  local need="$1" gpu="$2" gpu_need="$3" avail free
  while :; do
    avail="$(mem_available_mb)"
    if [ -z "$gpu" ]; then
      (( avail >= need )) && return 0
      log "gate(cpu): RAM ${avail}MB/${need}MB -> wait ${RAM_POLL_SLEEP}s"
      sleep "$RAM_POLL_SLEEP"
      continue
    fi
    free="$(gpu_free_mb "$gpu")"
    [[ -z "$free" ]] && free=0
    if (( avail >= need && free >= gpu_need )); then
      return 0
    fi
    log "gate(gpu=${gpu}): RAM ${avail}MB/${need}MB, VRAM ${free}MB/${gpu_need}MB -> wait ${RAM_POLL_SLEEP}s"
    sleep "$RAM_POLL_SLEEP"
  done
}

claim() {  # $1=counter file -> echo 下一个任务下标
  local counter="$1" idx
  {
    flock 200
    idx=$(cat "$counter" 2>/dev/null || echo 0)
    echo $((idx + 1)) > "$counter"
  } 200>"$counter.lock"
  echo "$idx"
}

run_one() {  # $1=slot $2=gpu（空字符串=CPU） $3=algo $4=split
  local slot="$1" gpu="$2" algo="$3" split="$4"
  local tag
  if [ -n "$gpu" ]; then tag="gpu${gpu}"; else tag="cpu${slot}"; fi
  local log_file="$LOG_DIR/${tag}_${algo}_${split}.log"
  local attempt=0 rc=0 log_mark=0 marker omp bind
  marker="$(marker_for "${algo}|${DATASET}|${split}")"
  if [ -n "$gpu" ]; then
    omp="$OMP_PER_JOB"
    bind="$(range_bind_prefix "${GPU_CORE_RANGES[$slot]:-}")"
  else
    omp="$CPU_OMP"
    bind="$(range_bind_prefix "${CPU_CORE_RANGES[$slot]:-}")"
  fi
  while (( attempt < MAX_ATTEMPTS )); do
    attempt=$((attempt + 1))
    wait_for_resources "$HOST_FREE_MB" "$gpu" "$GPU_FREE_MB"
    echo "[$(date '+%F %T')] START attempt=${attempt}/${MAX_ATTEMPTS} algo=${algo} split=${split} device=${tag} batch=${BATCH_SIZE} marker=${marker} omp=${omp} bind=${bind:-none}" >> "$log_file"
    log_mark=$(wc -l < "$log_file" 2>/dev/null || echo 0)
    # GPU 作业：CUDA_VISIBLE_DEVICES=<id> 且 -cuda <id>；
    # CPU 作业：清空 CUDA_VISIBLE_DEVICES 且 -cuda ""，让入口的
    # torch.cuda.is_available() 落到 False，走 CPU 路径。
    ${bind} env \
      OMP_NUM_THREADS="$omp" MKL_NUM_THREADS="$omp" \
      CUDA_VISIBLE_DEVICES="$gpu" VLLM_WORKER_RANK="$marker" \
      "$PY" "$ENTRY" \
      -algorithm "$algo" -split_strategy "$split" -cuda "$gpu" \
      -batch_size "$BATCH_SIZE" -test_batch_size "$TEST_BATCH_SIZE" \
      "${COMMON_ARGS[@]}" -tb_monitor "$(tb_monitor_for "$algo")" >> "$log_file" 2>&1
    rc=$?
    if (( rc == 0 )); then
      echo "[$(date '+%F %T')] DONE rc=0" >> "$log_file"
      rm -rf "save_path/${DATASET}/${split}/${algo}/${HYPOTHESIS}" 2>/dev/null || true
      printf '%s\t%s\t%s\n' "$split" "$algo" "OK" >> "$DONE_TSV"
      log "DONE ${algo}/${split} (${tag})"
      return 0
    fi
    echo "[$(date '+%F %T')] FAIL rc=${rc} attempt=${attempt}/${MAX_ATTEMPTS}" >> "$log_file"
    log "FAIL ${algo}/${split} rc=${rc} attempt=${attempt}/${MAX_ATTEMPTS} (${tag})"
    if tail -n "+$((log_mark + 1))" "$log_file" | grep -aq 'CheckpointCompatibilityError'; then
      rm -rf save_path/"${DATASET}"/"${split}"/"${algo}/${HYPOTHESIS}"/*/experiment_state 2>/dev/null || true
      log "CLEAN ${algo}/${split} 续跑状态与算法不兼容，已清空 experiment_state"
    fi
    sleep "$RETRY_SLEEP"
  done
  printf '%s\t%s\t%s\n' "$split" "$algo" "FAILED" >> "$DONE_TSV"
  return 1
}

worker() {  # $1=slot idx $2=gpu（空=CPU） $3=counter $4=首任务下标（-1=无） $5..=tasks
  local sid="$1" gpu="$2" counter="$3" first="$4"
  shift 4
  local tasks=("$@")
  local idx entry algo split
  # GPU 槽位用预留下标直接认领任务清单头部，保证「GPU 优先」；
  # 其余情况（CPU 槽位、或 GPU 槽位做完预留任务后）一律走共享计数器。
  if (( first >= 0 )); then
    idx="$first"
  else
    idx="$(claim "$counter")"
  fi
  while (( idx < ${#tasks[@]} )); do
    entry="${tasks[$idx]}"
    algo="${entry%%|*}"
    split="${entry##*|}"
    run_one "$sid" "$gpu" "$algo" "$split" || true
    idx="$(claim "$counter")"
  done
}

# ---- 组装任务清单 / build task list ----
tasks=()
for algo in "${ALGORITHMS[@]}"; do
  for split in "${SPLITS[@]}"; do tasks+=("${algo}|${split}"); done
done

gpu_slots=$(( NUM_GPUS * SLOTS_PER_GPU ))
cpu_disabled=0
if (( CPU_SLOTS < 0 )); then cpu_disabled=1; CPU_SLOTS=0; fi
cpu_affinity_init "$gpu_slots" "$cpu_disabled"

log "launcher start: GPUS=${GPU_IDS:-none} slots/gpu=${SLOTS_PER_GPU} gpu_slots=${gpu_slots} cpu_slots=${CPU_SLOTS} omp/gpu_job=${OMP_PER_JOB} omp/cpu_job=${CPU_OMP} tasks=${#tasks[@]} K=${NUM_CLIENTS_K} I=${COMM_ROUNDS} T=${EPOCH_T} repeat=${REPEAT_TIMES} data=${SYSTEM_DATA_COUNT:-all}"
log "cores: gpu_jobs -> ${GPU_CORE_RANGES[*]:-none} | cpu_jobs -> ${CPU_CORE_RANGES[*]:-none}"
log "logs -> ${LOG_DIR}"

# 共享计数器从 gpu_slots 起：前 gpu_slots 个任务由 GPU 槽位按下标直接认领，
# CPU 槽位只能从其后取，从而在不牺牲并发的前提下保证 GPU 优先。
counter="$LOG_DIR/.counter_all"
echo "$gpu_slots" > "$counter"

# ---- GPU 槽位与 CPU 槽位同时开工 / start GPU and CPU slots together ----
rc=0
pids=()
for (( i = 0; i < gpu_slots; i++ )); do
  gpu="${GPU_ARR[$(( i % NUM_GPUS ))]}"
  worker "$i" "$gpu" "$counter" "$i" "${tasks[@]}" &
  pids+=($!)
done
for (( i = 0; i < CPU_SLOTS; i++ )); do
  worker "$i" "" "$counter" -1 "${tasks[@]}" &
  pids+=($!)
done
for pid in ${pids[@]+"${pids[@]}"}; do wait "$pid" || rc=1; done

log "all tasks finished (rc=${rc}, gpu_slots=${gpu_slots}, cpu_slots=${CPU_SLOTS})"
log "summary from ${DONE_TSV}:"
sort "$DONE_TSV" | awk -F'\t' '{c[$3]++} END {for (k in c) printf "  %s: %d\n", k, c[k]}'
