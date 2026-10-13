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
#     main_SENT_CLF.py -algorithm FedAvg -dataset bios -task SENT_CLF ...
#
# 行为 / Behavior:
#   - 第 i 条命令绑定到第 i 张卡（命令数多于卡数时排队轮转）
#     Command i binds to GPU i%num_gpus (queued round-robin if more cmds than GPUs)
#   - 每条实验日志写入 ./parallel_logs/<gpu_id>_<seq>.log
#     Each experiment logs to ./parallel_logs/<gpu_id>_<seq>.log
#   - 等待全部实验结束后统一退出
#     Waits for all experiments to finish before exiting
# ============================================================
set -e

GPU_IDS="${1:?usage: launch_parallel.sh <gpu_ids_csv> <cmds_file>}"
CMDS_FILE="${2:?usage: launch_parallel.sh <gpu_ids_csv> <cmds_file>}"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

LOG_DIR="./parallel_logs"
mkdir -p "$LOG_DIR"

pids=()
seq=0
while IFS= read -r cmd; do
  # 跳过空行与注释行 / skip blank and comment lines
  [[ -z "$cmd" || "$cmd" =~ ^[[:space:]]*# ]] && continue
  gpu="${GPUS[$((seq % NUM_GPUS))]}"
  log_file="${LOG_DIR}/gpu${gpu}_seq${seq}.log"
  echo "[launch_parallel] gpu=${gpu} seq=${seq} log=${log_file}"
  echo "  cmd: ${cmd} -cuda ${gpu}"
  # 后台启动；CUDA_VISIBLE_DEVICES 双保险（部分框架直读设备号，部分读环境变量）
  CUDA_VISIBLE_DEVICES="$gpu" nohup ${cmd} -cuda "$gpu" > "$log_file" 2>&1 &
  pids+=($!)
  seq=$((seq + 1))
done < "$CMDS_FILE"

echo "[launch_parallel] launched ${#pids[@]} experiments on ${NUM_GPUS} GPUs; waiting..."
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=$((fail + 1))
done
echo "[launch_parallel] all done; failures=${fail}"
exit "$fail"
