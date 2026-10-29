#!/bin/bash
# 把一条队列按 GPU 拆成互不相交的分片，每张卡各起一个 launcher。
# 用法: pack_launch.sh <queue_file> <tag> <spec>
#   spec 形如 "0:5,1:5,2:6,3:6,4:3,5:5,30:2"（gpu:单卡并发槽位数）
#   不存在的卡号（30/31/40…）会被当作 CPU 槽位：MIN_FREE_MB 自动置 0，否则显存闸门
#   永远读不到空闲显存而卡死。
# 分片规则：有效命令行按行号 round-robin 分配给各分片，保证一个格子只出现在一个分片里。
set -u
Q="${1:?queue}"
TAG="${2:?tag}"
SPEC="${3:?spec}"
BASE=/root/fl
# 必须在仓库根目录启动：数据集就绪检查用的是相对路径 dataset/...，
# 从别处启动会让所有任务误判「缺少必要文件」而秒退。
cd "$BASE" || exit 1
MAP="$BASE/parallel_logs/$TAG"
mkdir -p "$MAP"
IFS=',' read -r -a ENTRIES <<< "$SPEC"
N=${#ENTRIES[@]}
tmpdir="$(mktemp -d)"
awk -v n="$N" -v d="$tmpdir" 'NF && $1 !~ /^#/ {print > (d "/part_" ((c++) % n) ".txt")}' "$Q"
for i in "${!ENTRIES[@]}"; do
  gpu="${ENTRIES[$i]%%:*}"
  slots="${ENTRIES[$i]##*:}"
  q="$tmpdir/part_$i.txt"
  cnt=$(grep -c . "$q" 2>/dev/null || true)
  [ -z "$cnt" ] && cnt=0
  [ "$cnt" -eq 0 ] && continue
  # 真卡（0-5）才用显存闸门；假卡号是 CPU 槽位，nvidia-smi 会打印
  # "No devices were found" 让闸门里的算术展开报 unbound variable 直接退出。
  if [ "$gpu" -lt 6 ] 2>/dev/null; then
    mf="${MIN_FREE_MB:-1000}"
  else
    mf=0
  fi
  d="$MAP/gpu$gpu"
  mkdir -p "$d"
  MIN_FREE_MB="$mf" PER_GPU_LIMIT="$slots" MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}" \
    GATE_POLL_S="${GATE_POLL_S:-15}" OMP_PER_JOB="${OMP_PER_JOB:-4}" \
    LOG_DIR="$d" nohup setsid bash "$BASE/scripts/launch_parallel.sh" "$gpu" "$q" \
    >> "$MAP/gpu$gpu.launcher.log" 2>&1 < /dev/null &
  echo "gpu=$gpu slots=$slots cells=$cnt min_free=${mf}MB logdir=$d"
done
exit 0
