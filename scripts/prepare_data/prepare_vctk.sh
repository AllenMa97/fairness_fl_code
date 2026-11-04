#!/bin/bash
# ============================================================
# VCTK 数据准备：下载 22 kHz 归档 + 说话人信息表，解压到 dataset/VCTK
#
# 来源 / Source
#   音频（110 位说话人、约 4.4 万条短语音，22 kHz WAV）：
#     https://huggingface.co/datasets/arampacha/vctk_22kHz  (raw_data.zip, ~10.7 GB)
#   说话人表 speaker-info.txt（性别 / 口音，用于标签与敏感属性）：
#     https://huggingface.co/datasets/CSTR-Edinburgh/vctk
#   VCTK 官方主页：https://datashare.ed.ac.uk/handle/10283/3443
#
# 目标目录 / Destination
#   <repo>/dataset/VCTK/{p225..p376,s5}/{speaker}_{utt}.wav
#   <repo>/dataset/VCTK/speaker-info.txt
#
# 用法 / Usage
#   bash scripts/prepare_data/prepare_vctk.sh
#   DEST=/data/dataset/VCTK bash scripts/prepare_data/prepare_vctk.sh
#   KEEP_ARCHIVE=1 ...   # 解压成功后保留 raw_data.zip
# ============================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$HERE/../.." && pwd)}"
DEST="${DEST:-$ROOT/dataset/VCTK}"
KEEP_ARCHIVE="${KEEP_ARCHIVE:-0}"

ARCHIVE_URLS=(
  "https://hf-mirror.com/datasets/arampacha/vctk_22kHz/resolve/main/raw_data.zip"
  "https://huggingface.co/datasets/arampacha/vctk_22kHz/resolve/main/raw_data.zip"
)
SPK_URLS=(
  "https://hf-mirror.com/datasets/CSTR-Edinburgh/vctk/resolve/main/speaker-info.txt"
  "https://cdn.jsdelivr.net/gh/kan-bayashi/ParallelWaveGAN@master/egs/vctk/vctk/speaker-info.txt"
  "https://cdn.jsdelivr.net/gh/jik876/hifi-gan@master/filelists/vctk/speaker-info.txt"
)

mkdir -p "$DEST"

fetch() {  # $1=输出文件 $2...=候选 URL（依次尝试，支持断点续传）
  local out="$1"; shift
  local url
  for url in "$@"; do
    echo ">>> curl $(basename "$out") <- $url"
    if curl -fL --retry 8 --retry-all-errors --retry-delay 5 --connect-timeout 15 \
        -A "Mozilla/5.0" -C - -o "$out" "$url" -s -w "http=%{http_code} bytes=%{size_download}\n"; then
      [ -s "$out" ] && return 0
    fi
    echo "!!! 失败，换下一个镜像"
  done
  return 1
}

# 1) 说话人信息表
if [ ! -s "$DEST/speaker-info.txt" ]; then
  fetch "$DEST/speaker-info.txt" "${SPK_URLS[@]}"
fi

# 2) 音频归档
ZIP="$DEST/raw_data.zip"
if [ ! -f "$ZIP" ]; then
  fetch "$ZIP" "${ARCHIVE_URLS[@]}"
fi
unzip -tq "$ZIP" >/dev/null
echo ">>> 解压 $ZIP -> $DEST"
unzip -q -o "$ZIP" -d "$DEST"

# 3) 校验 / Verify
WAVS=$(find "$DEST" -name '*.wav' | wc -l)
SPKS=$(find "$DEST" -maxdepth 1 -mindepth 1 -type d -name 'p*' | wc -l)
echo ">>> wav=$WAVS speakers=$SPKS speaker_info_lines=$(wc -l < "$DEST/speaker-info.txt")"
echo ">>> bytes=$(du -sb "$DEST" | cut -f1)"
if [ "$WAVS" -lt 36000 ] || [ "$SPKS" -lt 90 ]; then
  echo "!!! 校验失败：期望约 36940 个 wav / 94 个说话人目录" >&2
  exit 1
fi

if [ "$KEEP_ARCHIVE" != "1" ]; then
  rm -f "$ZIP"
  echo ">>> 已删除归档（KEEP_ARCHIVE=1 可保留）"
fi
echo "VCTK ready: $DEST"
