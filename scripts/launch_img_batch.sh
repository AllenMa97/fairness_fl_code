#!/bin/bash
# =============================================================================
# 图像实验批量并发启动器 / Concurrent image-experiment batch launcher
#
# 背景 / Background:
#   在多卡服务器上并行推进「算法 × 图像数据集」的实验矩阵（算法清单即排期，
#   见 ALGORITHMS；本轮 14 个算法，PDFFed 系列暂不跑）。
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
#   LARGE_GPU_FREE_MB   高分辨率池投递前目标卡的最小空闲显存，默认 20000
#   SMALL_GPU_FREE_MB   小分辨率池投递前目标卡的最小空闲显存，默认 8000
#   PY                  解释器路径，默认 /root/miniconda3/envs/federated/bin/python
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
LARGE_GPU_FREE_MB="${LARGE_GPU_FREE_MB:-20000}"
SMALL_GPU_FREE_MB="${SMALL_GPU_FREE_MB:-8000}"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"
RETRY_SLEEP="${RETRY_SLEEP:-30}"
RAM_POLL_SLEEP="${RAM_POLL_SLEEP:-30}"

HYPOTHESIS="BERTCLASSIFIER"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
# 强制急加载图像缓存：懒加载 + 训练集 shuffle 会造成 shard 抖动（见 module/dataset.py）
export IMAGE_CACHE_LAZY="${IMAGE_CACHE_LAZY:-0}"

# 算法清单（顺序 = 本轮排期优先级；任务按「算法优先」派发，见下方任务清单组装）
#   第 1 批 基线    : FedAvg / FedProx / FedNova / FederatedAverageWithPo
#   第 2 批         : Scaffold / FedRep / FedProto / mFairFL
#   第 3 批         : FedFair / FedFB / FL_FairBatch / FairFed / FedRenyi
#   第 4 批         : LoGoFair
#   本轮不跑         : PDFFed 系列（PDFFed / PDFFed_DP），需要时把末行放回即可
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
  # 本轮不跑
  # PDFFed PDFFed_DP
)
# 数据池：高分辨率（单进程峰值 ~15GB）与小分辨率（~2-3GB）分别限流
LARGE_DATASETS=(FairFace "LFWA+")
SMALL_DATASETS=(CelebA UTKFace)

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

LOG_DIR="./parallel_logs/img_batch"
mkdir -p "$LOG_DIR"
DONE_TSV="$LOG_DIR/done.tsv"
touch "$DONE_TSV"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NUM_GPUS=${#GPU_ARR[@]}

log() { echo "[$(date '+%F %T')] $*"; }

# 高分辨率数据集下调 test_batch_size（仅影响评估吞吐，不影响指标）
test_batch_for() {
  case "$1" in
    FairFace|"LFWA+") echo 64 ;;
    *) echo 256 ;;
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

run_one() {  # $1=gpu $2=algo $3=dataset $4=pool $5=host_need_mb $6=gpu_need_mb
  local gpu="$1" algo="$2" dataset="$3" pool="$4" need_mb="$5" gpu_need_mb="$6"
  local log_file="$LOG_DIR/${pool}_gpu${gpu}_${algo}_${dataset}.log"
  local tbs attempt=0 rc=0
  tbs="$(test_batch_for "$dataset")"
  while (( attempt < MAX_ATTEMPTS )); do
    attempt=$((attempt + 1))
    wait_for_resources "$need_mb" "$gpu" "$gpu_need_mb"
    echo "[$(date '+%F %T')] START attempt=${attempt}/${MAX_ATTEMPTS} algo=${algo} dataset=${dataset} gpu=${gpu} batch=256 test_batch=${tbs}" >> "$log_file"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ENTRY" \
      -algorithm "$algo" -dataset "$dataset" -cuda "$gpu" \
      -batch_size 256 -test_batch_size "$tbs" \
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
    # 清除中间检查点后再重试：崩溃时写入的 checkpoint_latest.pt 对部分算法
    # （其 algorithm_state 不可序列化）会导致重试直接抛 CheckpointCompatibilityError
    # 而永久失败；清掉后重试可干净重启。已完成的 repeat 指标存于 experiment_state
    # 之外，不受影响。
    rm -rf save_path/"${dataset}"/*/"${algo}/${HYPOTHESIS}"/*/experiment_state 2>/dev/null || true
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
    run_one "$gpu" "$algo" "$dataset" "$pool" "$need_mb" "$gpu_need_mb" || true
  done
}

run_pool() {  # $1=pool $2=slots_per_gpu $3=host_need $4=gpu_need $5..=tasks
  local pool="$1" slots_per_gpu="$2" need_mb="$3" gpu_need_mb="$4"
  shift 4
  local tasks=("$@")
  local counter="$LOG_DIR/.counter_${pool}"
  echo 0 > "$counter"
  local slots=$(( NUM_GPUS * slots_per_gpu ))
  local pids=() i gpu
  for (( i = 0; i < slots; i++ )); do
    gpu="${GPU_ARR[$(( i % NUM_GPUS ))]}"
    worker "$i" "$gpu" "$pool" "$counter" "$need_mb" "$gpu_need_mb" "${tasks[@]}" &
    pids+=($!)
  done
  local rc=0
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
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

run_pool large "$LARGE_SLOTS_PER_GPU" "$LARGE_FREE_MB" "$LARGE_GPU_FREE_MB" "${large_tasks[@]}" &
large_pid=$!
run_pool small "$SMALL_SLOTS_PER_GPU" "$SMALL_FREE_MB" "$SMALL_GPU_FREE_MB" "${small_tasks[@]}" &
small_pid=$!

wait "$large_pid"; large_rc=$?
wait "$small_pid"; small_rc=$?

log "all pools finished (large_rc=${large_rc} small_rc=${small_rc})"
log "summary from ${DONE_TSV}:"
sort "$DONE_TSV" | awk -F'\t' '{c[$3]++} END {for (k in c) printf "  %s: %d\n", k, c[k]}'
