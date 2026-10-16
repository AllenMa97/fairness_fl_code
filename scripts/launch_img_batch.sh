#!/bin/bash
# =============================================================================
# 图像实验批量并发启动器 / Concurrent image-experiment batch launcher
#
# 背景 / Background:
#   在多卡服务器上并行推进「算法 × 图像数据集」的实验矩阵（算法清单即排期，
#   见 ALGORITHMS；本轮 15 个算法，PDFFed 本身暂不跑、只跑其 DP 变体）。
#   训练统一使用 batch_size=256（不做梯度累积）；同一张 GPU 上允许并发多个
#   算法，并发的槽位数由实测显存决定（见下表）。每个 (算法, 数据集) 组合跑完
#   后清理其 save_path 权重，避免占用磁盘。
#   TensorBoard 观察参数按算法分级：仅 FedAvg/FedProx/FedNova
#   记录全量逐轮探针，其余算法（含本轮未跑的）一律精简为仅 test/system
#   （见 FULL_TB_ALGOS）。
#
# 显存实测 / Measured VRAM (LFWA+ 250×250, train batch 256, 单进程峰值):
#   test_batch_size=256 -> 34.0 GB      每卡只能跑 1 个
#   test_batch_size=64  -> 15.4 GB      每卡可跑 2 个
#   小分辨率 (CelebA/UTKFace 64×64)     ~2-3 GB，每卡可跑多个
#   => 高分辨率数据集下调 test_batch_size（仅影响评估吞吐，不影响任何指标），
#      从而在保留训练 batch 256 的前提下获得并发度。
#
# 用法 / Usage:
#   bash scripts/launch_img_batch.sh
#   GPUS=0,1,2,3 LARGE_SLOTS_PER_GPU=1 SMALL_SLOTS_PER_GPU=2 bash scripts/launch_img_batch.sh
#
# 环境变量 / Env vars:
#   GPUS                可用 GPU 列表（逗号分隔），默认 0,1,2,3
#   LARGE_SLOTS_PER_GPU 高分辨率池每卡槽位（FairFace/LFWA+），默认 1
#   SMALL_SLOTS_PER_GPU 小分辨率池每卡槽位（CelebA/UTKFace），默认 2
#   LARGE_FREE_MB       高分辨率池投递前的最小可用内存，默认 30000
#   SMALL_FREE_MB       小分辨率池投递前的最小可用内存，默认 20000
#   LARGE_GPU_FREE_MB   高分辨率池投递前目标卡的最小空闲显存，默认 5000
#   SMALL_GPU_FREE_MB   小分辨率池投递前目标卡的最小空闲显存，默认 2500
#   PY                  解释器路径，默认 /root/miniconda3/envs/federated/bin/python
#   BATCH_SIZE          训练 batch size，默认 256（小显存卡可调小，如 128）
#   REDUCED_BATCH_SIZE  REDUCED_ALGOS 的训练 batch，默认 64（须为 256 的约数）
#   REDUCED_ALGOS       显存峰值高、需降 batch 的算法名单（默认含 FedProto/
#                       FL_FairBatch/FedFair/FairFed/Scaffold/LoGoFair）
#   LOG_DIR             日志与 done.tsv/counter 目录，默认 ./parallel_logs/img_batch
#                       （多实例并发时必须各自指定，否则 counter 互相串号）
#   NUMACTL_NODES       绑定的 NUMA 节点列表（如 "0,1"），留空 = 自动探测
#   OMP_PER_JOB         每作业线程数，留空/0 = 按 逻辑核数/并发槽位 自动算
#
# 产物 / Artifacts:
#   parallel_logs/img_batch/<pool>_gpu<id>_<algo>_<dataset>.log
#   parallel_logs/img_batch/done.tsv   # dataset<TAB>algo<TAB>OK|FAILED
#   parallel_logs/img_batch/launcher.log
# =============================================================================
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="${PY:-/root/miniconda3/envs/federated/bin/python}"
ENTRY="${ENTRY:-main_IMG_CLF.py}"

GPUS="${GPUS:-0,1,2,3}"
LARGE_SLOTS_PER_GPU="${LARGE_SLOTS_PER_GPU:-1}"
SMALL_SLOTS_PER_GPU="${SMALL_SLOTS_PER_GPU:-2}"

LARGE_FREE_MB="${LARGE_FREE_MB:-30000}"
SMALL_FREE_MB="${SMALL_FREE_MB:-20000}"
# GPU 空闲显存门限：.31 已被 vLLM 推理服务长期占满（最大空闲仅约 11GB），
# 原 20000MB 门限永久不可达，作业会无限期卡在 wait_for_resources。
# 配合上面的 REDUCED_BATCH_SIZE（等效 batch 不变、峰值下降）下调门限，
# 使作业能"抢"现有空闲卡起跑；显存不足时仍由 -resume 断点重试兜底。
LARGE_GPU_FREE_MB="${LARGE_GPU_FREE_MB:-5000}"
SMALL_GPU_FREE_MB="${SMALL_GPU_FREE_MB:-2500}"

BATCH_SIZE="${BATCH_SIZE:-256}"
# 算法级 batch size 覆盖 / Per-algorithm batch-size override
#   .31 现被 gpustack 的 vLLM 推理服务长期占用（单卡 49GB 常剩 0.08~11GB），
#   下列算法显存峰值高、历史上反复 CUDA OOM。对它们调小训练 batch，靠各自内部的
#   梯度累积（accumulation_steps = max(1, int(256/batch_size))）保持等效 batch=256，
#   只降低单卡瞬时峰值、不改变优化动态。REDUCED_BATCH_SIZE 需为 256 的约数
#   （64 -> 累积 4 步，128 -> 累积 2 步）。
REDUCED_BATCH_SIZE="${REDUCED_BATCH_SIZE:-64}"
REDUCED_ALGOS="${REDUCED_ALGOS:- FedProto FL_FairBatch FedFair FairFed Scaffold LoGoFair }"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"
RETRY_SLEEP="${RETRY_SLEEP:-30}"
RAM_POLL_SLEEP="${RAM_POLL_SLEEP:-30}"

HYPOTHESIS="BERTCLASSIFIER"

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
# 强制急加载图像缓存：懒加载 + 训练集 shuffle 会造成 shard 抖动（见 module/dataset.py）
export IMAGE_CACHE_LAZY="${IMAGE_CACHE_LAZY:-0}"

# 算法清单（顺序 = 本轮排期优先级；任务按「算法优先」派发，见下方任务清单组装）
#   第 1 批 基线    : FedAvg / FedProx / FedNova / FederatedAverageWithPo
#   第 2 批         : Scaffold / FedRep / FedProto / mFairFL
#   第 3 批         : FedFair / FedFB / FL_FairBatch / FairFed / FedRenyi
#   第 4 批         : LoGoFair
#   第 5 批         : PDFFed_DP（PDFFed 本身暂不跑，只跑其 DP 变体）
# PraFFL 与 FedFACT 仅支持 SENT_CLF（依赖 BERT 编码器），图像入口会直接抛
# ValueError 退出，故不纳入图像实验矩阵。
ALGORITHMS=(
  # 第 1 批
  FedAvg FedProx FedNova FederatedAverageWithPo
  # 第 2 批
  Scaffold FedRep FedProto mFairFL
  # 第 3 批
  FedFair FedFB FL_FairBatch FairFed FedRenyi
  # 第 4 批
  LoGoFair
  # 第 5 批（PDFFed 本身暂不跑，只跑其 DP 变体）
  PDFFed_DP
)
# 算法清单覆盖 / Optional algorithm-list override
#   用于定向重跑（例如只重跑修复后的算法），不改动上面的默认排期。
#   ALGORITHMS_OVERRIDE="FedProto,FairFed,LoGoFair" bash scripts/launch_img_batch.sh
if [ -n "${ALGORITHMS_OVERRIDE:-}" ]; then
  IFS=', ' read -r -a ALGORITHMS <<< "$ALGORITHMS_OVERRIDE"
fi
# 数据池：高分辨率（单进程峰值 ~15GB）与小分辨率（~2-3GB）分别限流
LARGE_DATASETS=(FairFace "LFWA+")
SMALL_DATASETS=(CelebA UTKFace)
# 数据池覆盖 / Optional dataset-pool override
#   与 ALGORITHMS_OVERRIDE 同构：用于临时排除某个数据集而不改动上面的默认排期。
#   例：FairFace 标签口径返工期间只跑 LFWA+
#   LARGE_DATASETS_OVERRIDE="LFWA+" bash scripts/launch_img_batch.sh
if [ -n "${LARGE_DATASETS_OVERRIDE:-}" ]; then
  IFS=', ' read -r -a LARGE_DATASETS <<< "$LARGE_DATASETS_OVERRIDE"
fi
if [ -n "${SMALL_DATASETS_OVERRIDE:-}" ]; then
  IFS=', ' read -r -a SMALL_DATASETS <<< "$SMALL_DATASETS_OVERRIDE"
fi

# TensorBoard 观察参数分级 / Tiered TB monitoring
#   只有基线三件套 FedAvg/FedProx/FedNova 记全量逐轮探针；其余算法（含本轮
#   未跑的）一律精简为仅 test/system，关闭 fisher/sharpness/embedding/
#   activation/gradient/neural_collapse/client_divergence 这些高开销探针
#   （图像 batch 256 下每轮成本很高，且对最终指标无贡献）。
#   口径与 main_Tabular_CLF.py 的 DEFAULT_TB_MONITOR 一致。
FULL_TB_ALGOS=" FedAvg FedProx FedNova "
FULL_TB_MONITOR='{"fisher": true, "sharpness": true, "embedding": true, "activation": true, "gradient": true, "neural_collapse": true, "client_divergence": true}'
LITE_TB_MONITOR='{"fisher": false, "sharpness": false, "embedding": false, "activation": false, "gradient": false, "neural_collapse": false, "client_divergence": false}'

COMMON_ARGS=(
  -task IMG_CLF
  -communication_round_I 50
  -algorithm_epoch_T 2
  -exp_repeat_times 3
  -parallel_repeats 1
  -final_artifact_policy metrics_only
  -dataloader_num_workers 4
  -resume
  # 图像 CNN 特征实测逐维幅值上界 B_obs≈13.6（UTKFace 全量、K=20、d=512，
  # BN 后 gamma 可放大激活），远高于表格的 0.17；默认 B=1.0 会低估敏感度
  # 使 DP 声称失效。取 24 留约 76% 余量（DP 先验上界宁宽勿窄，一旦实测越界
  # 该组 DP 保证即作废，只能事后 warning 发现）。
  # 该参数仅 use_dp=True 的算法（PDFFed_DP）消费，其余算法自动忽略。
  -dp_feature_bound 24
)

LOG_DIR="${LOG_DIR:-./parallel_logs/img_batch}"
mkdir -p "$LOG_DIR"
DONE_TSV="$LOG_DIR/done.tsv"
touch "$DONE_TSV"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NUM_GPUS=${#GPU_ARR[@]}

log() { echo "[$(date '+%F %T')] $*"; }

# 高分辨率数据集下调 test_batch_size（仅影响评估吞吐，不影响指标）；
# 对 REDUCED_ALGOS 进一步下调，配合降 batch 把评估峰值压进共享卡的空闲区间。
test_batch_for() {  # $1=dataset  $2=algo
  case "$1" in
    FairFace|"LFWA+")
      case "$REDUCED_ALGOS" in *" $2 "*) echo 16 ;; *) echo 64 ;; esac ;;
    *)
      case "$REDUCED_ALGOS" in *" $2 "*) echo 128 ;; *) echo 256 ;; esac ;;
  esac
}

# 训练 batch size：REDUCED_ALGOS 用 REDUCED_BATCH_SIZE，其余用全局 BATCH_SIZE
batch_size_for() {  # $1=algo
  case "$REDUCED_ALGOS" in
    *" $1 "*) printf '%s' "$REDUCED_BATCH_SIZE" ;;
    *) printf '%s' "$BATCH_SIZE" ;;
  esac
}

# 按算法返回对应的 TensorBoard 观察参数 JSON
tb_monitor_for() {
  case "$FULL_TB_ALGOS" in
    *" $1 "*) printf '%s' "$FULL_TB_MONITOR" ;;
    *) printf '%s' "$LITE_TB_MONITOR" ;;
  esac
}

mem_available_mb() {
  awk '/^MemAvailable:/ {printf "%d", $2/1024}' /proc/meminfo
}

gpu_free_mb() {  # $1=物理 GPU 序号
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
    | head -n 1 | tr -d ' '
}

wait_for_resources() {  # $1=host need MB  $2=gpu id  $3=gpu need MB
  local need="$1" gpu="$2" gpu_need="$3" avail free
  while :; do
    avail="$(mem_available_mb)"
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

run_one() {  # $1=slot $2=gpu $3=algo $4=dataset $5=pool $6=host_need_mb $7=gpu_need_mb
  local slot="$1" gpu="$2" algo="$3" dataset="$4" pool="$5" need_mb="$6" gpu_need_mb="$7"
  local log_file="$LOG_DIR/${pool}_gpu${gpu}_${algo}_${dataset}.log"
  local bs tbs attempt=0 rc=0 log_mark=0
  bs="$(batch_size_for "$algo")"
  tbs="$(test_batch_for "$dataset" "$algo")"
  while (( attempt < MAX_ATTEMPTS )); do
    attempt=$((attempt + 1))
    wait_for_resources "$need_mb" "$gpu" "$gpu_need_mb"
    echo "[$(date '+%F %T')] START attempt=${attempt}/${MAX_ATTEMPTS} algo=${algo} dataset=${dataset} gpu=${gpu} batch=${bs} test_batch=${tbs} omp=${OMP_PER_JOB} node=$(cpu_bind_prefix "$slot")" >> "$log_file"
    # 记录本次尝试的日志起点，失败后只在本次新增区间内找失败原因
    log_mark=$(wc -l < "$log_file" 2>/dev/null || echo 0)
    # env 承接 per-job 线程数；numactl 前缀（可能为空）把作业钉到某个 NUMA 节点。
    $(cpu_bind_prefix "$slot") env \
      OMP_NUM_THREADS="$OMP_PER_JOB" MKL_NUM_THREADS="$OMP_PER_JOB" \
      CUDA_VISIBLE_DEVICES="$gpu" \
      "$PY" "$ENTRY" \
      -algorithm "$algo" -dataset "$dataset" -cuda "$gpu" \
      -batch_size "$bs" -test_batch_size "$tbs" \
      "${COMMON_ARGS[@]}" -tb_monitor "$(tb_monitor_for "$algo")" >> "$log_file" 2>&1
    rc=$?
    if (( rc == 0 )); then
      echo "[$(date '+%F %T')] DONE rc=0" >> "$log_file"
      # 清理该 (算法, 数据集) 在 save_path 下的权重目录；
      # 保留 save_path/<dataset>/global_model_4_IMG_CLF_init.pt
      rm -rf "save_path/${dataset}"/*/"${algo}/${HYPOTHESIS}" 2>/dev/null || true
      printf '%s\t%s\t%s\n' "$dataset" "$algo" "OK" >> "$DONE_TSV"
      log "DONE ${algo}/${dataset} (gpu=${gpu})"
      return 0
    fi
    echo "[$(date '+%F %T')] FAIL rc=${rc} attempt=${attempt}/${MAX_ATTEMPTS}" >> "$log_file"
    log "FAIL ${algo}/${dataset} rc=${rc} attempt=${attempt}/${MAX_ATTEMPTS} (gpu=${gpu})"
    # 只在「续跑状态与算法不兼容」时才清 experiment_state，让重试干净重启。
    # 崩溃时写入的 checkpoint_latest.pt 对部分算法（其 algorithm_state 不可序列化）会让
    # 重试直接抛 CheckpointCompatibilityError 而永久失败，故这种情况下必须清。
    # 但 metrics.json 与 checkpoint_latest.pt 都在 experiment_state 之内，无差别清空等于把
    # 已训练几十轮的进度推倒重来；CUDA OOM 恰是本批次的头号失败原因，代价极高。
    # 其余失败（OOM / 被外部 kill / 瞬时故障）保留断点，重试靠 -resume 从最近一轮续上。
    if tail -n "+$((log_mark + 1))" "$log_file" | grep -aq 'CheckpointCompatibilityError'; then
      rm -rf save_path/"${dataset}"/*/"${algo}/${HYPOTHESIS}"/*/experiment_state 2>/dev/null || true
      log "CLEAN ${algo}/${dataset} 续跑状态与算法不兼容，已清空 experiment_state"
    fi
    sleep "$RETRY_SLEEP"
  done
  printf '%s\t%s\t%s\n' "$dataset" "$algo" "FAILED" >> "$DONE_TSV"
  return 1
}

worker() {  # $1=slot idx $2=gpu $3=pool $4=counter $5=host_need $6=gpu_need $7..=tasks
  local sid="$1" gpu="$2" pool="$3" counter="$4" need_mb="$5" gpu_need_mb="$6"
  shift 6
  local tasks=("$@")
  local idx entry algo dataset
  while :; do
    idx="$(claim "$counter")"
    (( idx >= ${#tasks[@]} )) && break
    entry="${tasks[$idx]}"
    algo="${entry%%|*}"
    dataset="${entry##*|}"
    run_one "$sid" "$gpu" "$algo" "$dataset" "$pool" "$need_mb" "$gpu_need_mb" || true
  done
}

run_pool() {  # $1=pool $2=slots_per_gpu $3=host_need $4=gpu_need $5=slot_base $6..=tasks
  local pool="$1" slots_per_gpu="$2" need_mb="$3" gpu_need_mb="$4" slot_base="$5"
  shift 5
  local tasks=("$@")
  local counter="$LOG_DIR/.counter_${pool}"
  echo 0 > "$counter"
  local slots=$(( NUM_GPUS * slots_per_gpu ))
  local pids=() i gpu
  for (( i = 0; i < slots; i++ )); do
    gpu="${GPU_ARR[$(( i % NUM_GPUS ))]}"
    worker "$(( slot_base + i ))" "$gpu" "$pool" "$counter" "$need_mb" "$gpu_need_mb" "${tasks[@]}" &
    pids+=($!)
  done
  local rc=0
  # 槽位数为 0（例如只用小分辨率池、把 LARGE_SLOTS_PER_GPU=0）时 pids 为空，
  # 直接展开 "${pids[@]}" 在 bash 4.2/4.3 + set -u 下会报 unbound variable。
  if (( ${#pids[@]} > 0 )); then
    for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  fi
  return $rc
}

# ---- 组装任务清单 / build task lists ----
# 外层按算法、内层按数据集：使 ALGORITHMS 的分批顺序成为真实派发顺序
# （第 1 批在所有数据集上跑完 → 第 2 批 → 第 3 批），而不是按数据集各自分组。
large_tasks=()
small_tasks=()
for algo in "${ALGORITHMS[@]}"; do
  for dataset in "${LARGE_DATASETS[@]}"; do large_tasks+=("${algo}|${dataset}"); done
done
for algo in "${ALGORITHMS[@]}"; do
  for dataset in "${SMALL_DATASETS[@]}"; do small_tasks+=("${algo}|${dataset}"); done
done

log "launcher start: GPUS=${GPUS} large=${LARGE_SLOTS_PER_GPU}/gpu small=${SMALL_SLOTS_PER_GPU}/gpu IMAGE_CACHE_LAZY=${IMAGE_CACHE_LAZY}"
log "large tasks=${#large_tasks[@]} (${LARGE_DATASETS[*]}); small tasks=${#small_tasks[@]} (${SMALL_DATASETS[*]})"
log "logs -> ${LOG_DIR}"

large_slots=$(( NUM_GPUS * LARGE_SLOTS_PER_GPU ))
small_slots=$(( NUM_GPUS * SMALL_SLOTS_PER_GPU ))
# 两个池的槽位并发存在，按总槽位均分线程并按槽位轮转绑 NUMA 节点（P2）。
cpu_affinity_init "$(( large_slots + small_slots ))"

run_pool large "$LARGE_SLOTS_PER_GPU" "$LARGE_FREE_MB" "$LARGE_GPU_FREE_MB" 0 "${large_tasks[@]}" &
large_pid=$!
run_pool small "$SMALL_SLOTS_PER_GPU" "$SMALL_FREE_MB" "$SMALL_GPU_FREE_MB" "$large_slots" "${small_tasks[@]}" &
small_pid=$!

wait "$large_pid"; large_rc=$?
wait "$small_pid"; small_rc=$?

log "all pools finished (large_rc=${large_rc} small_rc=${small_rc})"
log "summary from ${DONE_TSV}:"
sort "$DONE_TSV" | awk -F'\t' '{c[$3]++} END {for (k in c) printf "  %s: %d\n", k, c[k]}'
