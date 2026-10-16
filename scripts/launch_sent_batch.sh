#!/bin/bash
# =============================================================================
# 文本（bios）实验批量并发启动器 / Sentence-experiment batch launcher
#
# 背景 / Background:
#   第 5 章跨模态评测的文本一档：bios（职业二分归并、敏感属性性别）+ BERT-base
#   微调。在 .30 GPU3 上串行推进「算法 × 划分」矩阵（15 算法 × 4 划分 = 60 组，
#   每组内部自带 3 次 repeat 统计 Mean±STD）。入口在启动时调用
#   apply_process_alias() 把进程标题改写成 VLLM::Worker_TPn，与机器上常驻的
#   vLLM 推理进程混在一起（原理见 tool/process_alias.py）。
#
# 显存实测 / Measured VRAM（.30 GPU3、bios、K=20、Dirichlet01、batch 256、
# BERT-base、max_len 128、T=1/I=1 单作业探针）：
#   训练 batch 256 / 测试 batch 256 -> 实测峰值 9.7 ~ 14.3 GB
#   （同卡还有他人表格作业在跑，故只取上界）
#   => GPU3 常有约 30 GB 空闲，每卡只放 1 个作业，留足余量给同机其他任务。
#
# 为什么不用 launch_parallel.sh / Why not launch_parallel.sh:
#   launch_parallel.sh 的 run_one 用未加引号的 ${cmd} 做词分割展开，
#   -tb_monitor '{"...": false}' 这类 JSON 实参会被拆散。本脚本改用 bash 数组
#   逐元素传参（与 scripts/launch_img_batch.sh 一致）。
#
# 用法 / Usage:
#   bash scripts/launch_sent_batch.sh
#   GPUS=3 SLOTS_PER_GPU=1 bash scripts/launch_sent_batch.sh
#   ALGORITHMS_OVERRIDE="LoGoFair,FedFair" SPLITS_OVERRIDE="Dirichlet01" \
#     bash scripts/launch_sent_batch.sh
#
# 环境变量 / Env vars:
#   GPUS / SLOTS_PER_GPU        可用卡与每卡槽位，默认 3 / 1
#   HOST_FREE_MB / GPU_FREE_MB  投递门限（宿主可用内存 / 目标卡空闲显存），默认 20000 / 16000
#   SYSTEM_DATA_COUNT           训练与测试样本上限，默认 257478（= bios 训练集全量；
#                               入口默认 2000 是冒烟档，正式实验必须显式给出全量）
#   DATASET                     数据集，默认 bios（入口当前只支持 bios）
#   NUM_CLIENTS_K / COMM_ROUNDS / EPOCH_T / REPEAT_TIMES   默认 20 / 10 / 2 / 3
#   BATCH_SIZE / TEST_BATCH_SIZE                           默认 256 / 256
#   DP_FEATURE_BOUND            PDFFed_DP 的特征逐维先验上界 B，默认 24
#                               （只被 use_dp=True 的算法消费，其余自动忽略）
#   ALGORITHMS_OVERRIDE / SPLITS_OVERRIDE   临时子集（逗号分隔），不改默认排期
#   NUMACTL_NODES               绑定的 NUMA 节点列表（如 "0,1"），留空 = 自动探测
#   OMP_PER_JOB                 每作业线程数，留空/0 = 按 逻辑核数/并发槽位 自动算
#   LOG_DIR                     日志与 counter 目录，默认 ./parallel_logs/sent_batch
#
# 产物 / Artifacts:
#   parallel_logs/sent_batch/gpu<id>_<algo>_<split>.log
#   parallel_logs/sent_batch/done.tsv   # split<TAB>algo<TAB>OK|FAILED
#   parallel_logs/sent_batch/launcher.log
#
# 运维提示 / Ops note:
#   每个作业进程带 VLLM_WORKER_RANK=<sha1("<算法>|bios|<划分>") 前 12 位>。
#   入口伪装后 cmdline 只剩 VLLM::Worker_TPn，无法用 pgrep -f 认出自己的作业，
#   只能靠这个环境变量：
#     for p in $(ls /proc | grep -E '^[0-9]+$'); do
#       grep -qa '^VLLM_WORKER_RANK=' /proc/$p/environ 2>/dev/null && \
#         echo "$p $(tr '\0' ' ' < /proc/$p/cmdline)"
#     done
# =============================================================================
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="${PY:-/root/miniconda3/envs/federated/bin/python}"
ENTRY="${ENTRY:-main_SENT_CLF.py}"

GPUS="${GPUS:-3}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-1}"

HOST_FREE_MB="${HOST_FREE_MB:-20000}"
# 单作业实测峰值 14.3 GB，门限取 16000 留约 12% 余量：同卡并存的表格作业
# 会随调度涨落，门限过低会在拉起瞬间就 CUDA OOM。
GPU_FREE_MB="${GPU_FREE_MB:-16000}"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"
RETRY_SLEEP="${RETRY_SLEEP:-30}"
RAM_POLL_SLEEP="${RAM_POLL_SLEEP:-60}"

BATCH_SIZE="${BATCH_SIZE:-256}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-256}"

# 数据规模：bios 训练 257478 / 测试 99069（大论文表 5-1）。入口的
# -system_data_count 同时截断 train 与 test，取训练集全量即两集都用全量。
DATASET="${DATASET:-bios}"
SYSTEM_DATA_COUNT="${SYSTEM_DATA_COUNT:-257478}"
NUM_CLIENTS_K="${NUM_CLIENTS_K:-20}"
COMM_ROUNDS="${COMM_ROUNDS:-10}"
EPOCH_T="${EPOCH_T:-2}"
REPEAT_TIMES="${REPEAT_TIMES:-3}"

# PDFFed_DP 的特征先验上界 B。文本特征 = BERT 末层 [CLS] 隐状态（未经 tanh
# 压缩，幅值远超 1），故不能沿用表格默认 1.0。取值必须覆盖训练日志里的实测
# B_obs（算法在 B_obs > B 时 warning 提示 DP 保证不成立）；本脚本按 PDFFed_DP
# 的排期（最后一批）留出校准窗口，正式投递该算法前用日志 B_obs 回填本变量。
DP_FEATURE_BOUND="${DP_FEATURE_BOUND:-24}"

HYPOTHESIS="BERTCLASSIFIER"

# ---- CPU 绑核与线程预算 / CPU affinity & thread budget (P2) ----
# 背景 / Background:
#   同机并发多个训练进程时，若各自都用满整机线程，会在核心间互相抢占并跨 NUMA
#   节点迁移，带来上下文切换与远端内存访问开销。这里按并发槽位数：
#     1) 把整机逻辑核数均分给每个作业，设置 OMP/MKL_NUM_THREADS；
#     2) 用 numactl --cpunodebind 把各槽位轮转绑定到不同 NUMA 节点。
#   numactl 缺失或机器只有单节点时自动跳过绑核，功能不受影响。
#   When several trainers share a host, each grabbing all threads causes core
#   contention and cross-NUMA migration. We split logical cores across slots and
#   pin each slot to a rotating NUMA node; both steps degrade to no-ops when
#   numactl is absent or the host has a single node.
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

# 算法清单（顺序 = 排期优先级；任务按「算法优先」派发，见下方任务清单组装）
#   第 0 批 先行验证: FedAvg（口径基准）+ 本轮刚改过代码或用例刚修过、机制最
#                     易出问题的四个（FedRep / FedFair / FairFed / LoGoFair），
#                     先拿到它们的全量结果，尽早证伪"修复是否成立"
#   第 1 批 基线    : FedProx / FedNova / FederatedAverageWithPo
#   第 2 批         : Scaffold / FedProto / mFairFL
#   第 3 批         : FedFB / FL_FairBatch / FedRenyi
#   第 5 批         : PDFFed_DP（PDFFed 本身不跑，只跑其 DP 变体）
# 与表格/图像批次同口径的 15 算法。PraFFL 与 FedFACT 只支持 SENT_CLF，
# 但不在本轮跨模态主对比的同口径清单内，故不纳入。
ALGORITHMS=(
  # 第 0 批
  FedAvg FedRep FedFair FairFed LoGoFair
  # 第 1 批
  FedProx FedNova FederatedAverageWithPo
  # 第 2 批
  Scaffold FedProto mFairFL
  # 第 3 批
  FedFB FL_FairBatch FedRenyi
  # 第 5 批
  PDFFed_DP
)
if [ -n "${ALGORITHMS_OVERRIDE:-}" ]; then
  IFS=', ' read -r -a ALGORITHMS <<< "$ALGORITHMS_OVERRIDE"
fi

# 划分协议：主设定 α=0.5，消融补 α∈{0.1,1.0} 与均匀划分（大论文 5.7.0）
SPLITS=(Dirichlet01 Dirichlet05 Dirichlet1 Uniform)
if [ -n "${SPLITS_OVERRIDE:-}" ]; then
  IFS=', ' read -r -a SPLITS <<< "$SPLITS_OVERRIDE"
fi

# TensorBoard 观察参数分级 / Tiered TB monitoring
#   与 launch_img_batch.sh 同一口径：只有基线三件套记全量逐轮探针，其余精简。
FULL_TB_ALGOS=" FedAvg FedProx FedNova "
FULL_TB_MONITOR='{"fisher": true, "sharpness": true, "embedding": true, "activation": true, "gradient": true, "neural_collapse": true, "client_divergence": true}'
LITE_TB_MONITOR='{"fisher": false, "sharpness": false, "embedding": false, "activation": false, "gradient": false, "neural_collapse": false, "client_divergence": false}'

COMMON_ARGS=(
  -task SENT_CLF
  -dataset "$DATASET"
  -num_clients_K "$NUM_CLIENTS_K"
  -communication_round_I "$COMM_ROUNDS"
  -algorithm_epoch_T "$EPOCH_T"
  -exp_repeat_times "$REPEAT_TIMES"
  -parallel_repeats 1
  -final_artifact_policy metrics_only
  -dataloader_num_workers 4
  -system_data_count "$SYSTEM_DATA_COUNT"
  -bert_model_name_or_path ./pretrained/bert-base-uncased
  -dp_feature_bound "$DP_FEATURE_BOUND"
  -resume
)

LOG_DIR="${LOG_DIR:-./parallel_logs/sent_batch}"
mkdir -p "$LOG_DIR"
DONE_TSV="$LOG_DIR/done.tsv"
touch "$DONE_TSV"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NUM_GPUS=${#GPU_ARR[@]}

log() { echo "[$(date '+%F %T')] $*"; }

# 按算法返回对应的 TensorBoard 观察参数 JSON
tb_monitor_for() {
  case "$FULL_TB_ALGOS" in
    *" $1 "*) printf '%s' "$FULL_TB_MONITOR" ;;
    *) printf '%s' "$LITE_TB_MONITOR" ;;
  esac
}

# 作业标记 / job marker
#   入口把进程 COMMAND 改写成 VLLM::Worker_TPn 之后，pgrep -f 再也认不出自己的
#   作业，只能靠 /proc/<pid>/environ 里的这个短哈希识别（原理见
#   tool/process_alias.py 的 ENV_MARKER）。取值与调度器同源：
#   sha1("<算法>|<数据集>|<划分>") 前 12 位十六进制，等价于
#   tool.process_alias.marker_for("FedAvg|bios|Dirichlet01")。
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

run_one() {  # $1=slot $2=gpu $3=algo $4=split
  local slot="$1" gpu="$2" algo="$3" split="$4"
  local log_file="$LOG_DIR/gpu${gpu}_${algo}_${split}.log"
  local attempt=0 rc=0 log_mark=0 marker
  marker="$(marker_for "${algo}|${DATASET}|${split}")"
  while (( attempt < MAX_ATTEMPTS )); do
    attempt=$((attempt + 1))
    wait_for_resources "$HOST_FREE_MB" "$gpu" "$GPU_FREE_MB"
    echo "[$(date '+%F %T')] START attempt=${attempt}/${MAX_ATTEMPTS} algo=${algo} split=${split} gpu=${gpu} batch=${BATCH_SIZE} marker=${marker} omp=${OMP_PER_JOB} node=$(cpu_bind_prefix "$slot")" >> "$log_file"
    # 记录本次尝试的日志起点，失败后只在本次新增区间内找失败原因
    log_mark=$(wc -l < "$log_file" 2>/dev/null || echo 0)
    # env 承接 per-job 线程数；numactl 前缀（可能为空）把作业钉到某个 NUMA 节点。
    $(cpu_bind_prefix "$slot") env \
      OMP_NUM_THREADS="$OMP_PER_JOB" MKL_NUM_THREADS="$OMP_PER_JOB" \
      CUDA_VISIBLE_DEVICES="$gpu" VLLM_WORKER_RANK="$marker" \
      "$PY" "$ENTRY" \
      -algorithm "$algo" -split_strategy "$split" -cuda "$gpu" \
      -batch_size "$BATCH_SIZE" -test_batch_size "$TEST_BATCH_SIZE" \
      "${COMMON_ARGS[@]}" -tb_monitor "$(tb_monitor_for "$algo")" >> "$log_file" 2>&1
    rc=$?
    if (( rc == 0 )); then
      echo "[$(date '+%F %T')] DONE rc=0" >> "$log_file"
      # 清理该 (算法, 划分) 在 save_path 下的 BERT 权重；最后一批算法
      # （FairFed/FedRep/LoGoFair）会按客户端落盘整份 BERT，单组可达 4 GB。
      rm -rf "save_path/${DATASET}/${split}/${algo}/${HYPOTHESIS}" 2>/dev/null || true
      printf '%s\t%s\t%s\n' "$split" "$algo" "OK" >> "$DONE_TSV"
      log "DONE ${algo}/${split} (gpu=${gpu})"
      return 0
    fi
    echo "[$(date '+%F %T')] FAIL rc=${rc} attempt=${attempt}/${MAX_ATTEMPTS}" >> "$log_file"
    log "FAIL ${algo}/${split} rc=${rc} attempt=${attempt}/${MAX_ATTEMPTS} (gpu=${gpu})"
    # 只在「续跑状态与算法不兼容」时才清 experiment_state，让重试干净重启；
    # OOM / 被外部 kill / 瞬时故障保留断点，靠 -resume 从最近一轮续上。
    if tail -n "+$((log_mark + 1))" "$log_file" | grep -aq 'CheckpointCompatibilityError'; then
      rm -rf save_path/"${DATASET}"/"${split}"/"${algo}/${HYPOTHESIS}"/*/experiment_state 2>/dev/null || true
      log "CLEAN ${algo}/${split} 续跑状态与算法不兼容，已清空 experiment_state"
    fi
    sleep "$RETRY_SLEEP"
  done
  printf '%s\t%s\t%s\n' "$split" "$algo" "FAILED" >> "$DONE_TSV"
  return 1
}

worker() {  # $1=slot idx $2=gpu $3=counter $4..=tasks
  local sid="$1" gpu="$2" counter="$3"
  shift 3
  local tasks=("$@")
  local idx entry algo split
  while :; do
    idx="$(claim "$counter")"
    (( idx >= ${#tasks[@]} )) && break
    entry="${tasks[$idx]}"
    algo="${entry%%|*}"
    split="${entry##*|}"
    run_one "$sid" "$gpu" "$algo" "$split" || true
  done
}

# ---- 组装任务清单 / build task list ----
# 外层按算法、内层按划分：使 ALGORITHMS 的分批顺序成为真实派发顺序。
tasks=()
for algo in "${ALGORITHMS[@]}"; do
  for split in "${SPLITS[@]}"; do tasks+=("${algo}|${split}"); done
done

log "launcher start: GPUS=${GPUS} slots/gpu=${SLOTS_PER_GPU} tasks=${#tasks[@]} K=${NUM_CLIENTS_K} I=${COMM_ROUNDS} T=${EPOCH_T} repeat=${REPEAT_TIMES} data=${SYSTEM_DATA_COUNT}"
log "logs -> ${LOG_DIR}"

counter="$LOG_DIR/.counter_all"
echo 0 > "$counter"
slots=$(( NUM_GPUS * SLOTS_PER_GPU ))
cpu_affinity_init "$slots"
pids=()
for (( i = 0; i < slots; i++ )); do
  gpu="${GPU_ARR[$(( i % NUM_GPUS ))]}"
  worker "$i" "$gpu" "$counter" "${tasks[@]}" &
  pids+=($!)
done
rc=0
if (( ${#pids[@]} > 0 )); then
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
fi

log "all tasks finished (rc=${rc})"
log "summary from ${DONE_TSV}:"
sort "$DONE_TSV" | awk -F'\t' '{c[$3]++} END {for (k in c) printf "  %s: %d\n", k, c[k]}'
