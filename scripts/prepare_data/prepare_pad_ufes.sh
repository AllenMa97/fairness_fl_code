#!/bin/bash
# ============================================================
# PAD-UFES-20 数据准备：下载官方 zip -> 解压 metadata.csv + images/imgs_part_*
#
# 来源 / Source
#   Mendeley Data, DOI 10.17632/zr7vgbcyr2.1（PAD-UFES-20）
#   官方下载 API：https://data.mendeley.com/public-api/zip/zr7vgbcyr2/download/1
#   文件：PAD-UFES-20.zip（约 3.6 GB，内含 metadata.csv 与
#         images/imgs_part_{1,2,3}.zip，解压后 2298 张 png）
#
# 目标目录 / Destination
#   <repo>/dataset/PAD-UFES-20/metadata.csv
#   <repo>/dataset/PAD-UFES-20/images/imgs_part_{1,2,3}/*.png
#
# 用法 / Usage
#   bash scripts/prepare_data/prepare_pad_ufes.sh
#   DEST=/data/dataset/PAD-UFES-20 bash scripts/prepare_data/prepare_pad_ufes.sh
#   KEEP_ARCHIVE=1 ...   # 解压成功后保留 zip
# ============================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$HERE/../.." && pwd)}"
DEST="${DEST:-$ROOT/dataset/PAD-UFES-20}"
KEEP_ARCHIVE="${KEEP_ARCHIVE:-0}"

ZIP_URL="https://data.mendeley.com/public-api/zip/zr7vgbcyr2/download/1"

mkdir -p "$DEST"
ZIP="$DEST/PAD-UFES-20.zip"

# 1) 下载（支持断点续传）
if [ ! -f "$ZIP" ]; then
  echo ">>> curl PAD-UFES-20.zip"
  curl -fL --retry 8 --retry-all-errors --retry-delay 5 --connect-timeout 15 \
    -A "Mozilla/5.0" -C - -o "$ZIP" "$ZIP_URL" -s -w "http=%{http_code} bytes=%{size_download}\n"
fi
unzip -tq "$ZIP" >/dev/null

# 2) 外层解压：metadata.csv + images/imgs_part_*.zip
echo ">>> 解压外层 zip"
unzip -q -o "$ZIP" -d "$DEST"

# 3) 嵌套图片分卷解压到 images/
for f in "$DEST"/images/imgs_part_*.zip; do
  [ -f "$f" ] || continue
  echo ">>> 解压 $(basename "$f")"
  unzip -q -o "$f" -d "$DEST/images/"
  rm -f "$f"
done

# 4) 校验 / Verify
PNGS=$(find "$DEST" -name '*.png' | wc -l)
META=$(stat -c %s "$DEST/metadata.csv" 2>/dev/null || echo 0)
echo ">>> png=$PNGS metadata_bytes=$META bytes=$(du -sb "$DEST" | cut -f1)"
if [ "$PNGS" -ne 2298 ] || [ "$META" -ne 316209 ]; then
  echo "!!! 校验失败：期望 2298 张 png、metadata.csv 316209 字节" >&2
  exit 1
fi

if [ "$KEEP_ARCHIVE" != "1" ]; then
  rm -f "$ZIP"
  echo ">>> 已删除归档（KEEP_ARCHIVE=1 可保留）"
fi
echo "PAD-UFES-20 ready: $DEST"
