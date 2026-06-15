#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集下载与打包工具

用法:
    python setup_data.py                          # 检查所有数据集状态 + 下载缺失的
    python setup_data.py --check                  # 只检查，不下载
    python setup_data.py --datasets celeba UTKFace # 只处理指定的数据集
    python setup_data.py --pack                   # 打包所有数据集（用于上传 GitHub Release）
    python setup_data.py --pack --datasets celeba # 只打包 celeba
    python setup_data.py --list                   # 列出所有可用数据集

说明:
    - 表格/文本数据集（ADULT, COMPAS, DRUG, DUTCH, bios, moji）已包含在仓库中，无需下载
    - 图像数据集（CelebA, UTKFace）需要通过本脚本从 GitHub Release 下载
    - FairFace 和 LFWA+ 的标注文件已在仓库中，图片需要额外获取
"""

import os
import sys
import argparse
import tarfile
import time

try:
    import requests
except ImportError:
    print("ERROR: 需要安装 requests 库: pip install requests")
    sys.exit(1)

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

# ============================================================
# 修改下面的 GITHUB_REPO 为你的实际仓库地址
# 例如: "your-username/fairness_fl_code"
# ============================================================
GITHUB_REPO = "AllenMa97/fairness_fl_code"
RELEASE_TAG = "v1.0-data"

# 私有仓库需要Token才能下载Release文件
_GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
_RELEASE_AUTH = f"{_GITHUB_TOKEN}@" if _GITHUB_TOKEN else ""
RELEASE_BASE_URL = f"https://{_RELEASE_AUTH}github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}"

# 需要从 Release 下载的大文件数据集
RELEASE_DATASETS = {
    "celeba": {
        "archive": "celeba_images.tar.gz",
        "expected_files": [
            "celeba/Img/img_align_celeba/",
        ],
        "description": "CelebA 人脸图片 (~1.4 GB, 202599 张)",
        "task": "IMG_CLF",
    },
    "UTKFace": {
        "archive": "UTKFace_images.tar.gz",
        "expected_files": [
            "UTKFace/img/",
        ],
        "description": "UTKFace 人脸图片 (~1.5 GB)",
        "task": "IMG_CLF",
    },
    "FairFace": {
        "archive": "FairFace_images.tar.gz",
        "expected_files": [
            "FairFace/fairface-img-margin025-trainval/",
        ],
        "description": "FairFace 公平性人脸图片 (~500 MB)",
        "task": "IMG_CLF",
    },
    "LFWA+": {
        "archive": "LFWAPlus_images.tar.gz",
        "expected_files": [
            "LFWA+/lfw/",
        ],
        "description": "LFWA+ 人脸属性图片 (~100 MB)",
        "task": "IMG_CLF",
    },
}

# 已包含在仓库中的数据集（仅用于状态检查）
BUNDLED_DATASETS = {
    "ADULT": {
        "expected_files": [
            "ADULT/adult.data",
            "ADULT/adult.test",
            "ADULT/AdultTrain.csv",
            "ADULT/AdultTest.csv",
        ],
        "description": "ADULT 收入预测数据集 (~48 MB, 含 pickle)",
        "task": "Tabular_CLF",
    },
    "COMPAS": {
        "expected_files": [
            "COMPAS/compas-scores-two-years.csv",
        ],
        "description": "COMPAS 司法预测数据集 (~5 MB, 含 pickle)",
        "task": "Tabular_CLF",
    },
    "DRUG": {
        "expected_files": [
            "DRUG/drug_consumption.data",
        ],
        "description": "DRUG 药物消费数据集 (<1 MB, 含 pickle)",
        "task": "Tabular_CLF",
    },
    "DUTCH": {
        "expected_files": [
            "DUTCH/dutch_census_2001.arff",
        ],
        "description": "DUTCH 荷兰人口普查数据集 (~56 MB, 含 pickle)",
        "task": "Tabular_CLF",
    },
    "bios": {
        "expected_files": [
            "bios/train.parquet",
            "bios/test.parquet",
        ],
        "description": "BIOS 职业偏见表征数据集 (~86 MB)",
        "task": "SENT_CLF",
    },
    "moji": {
        "expected_files": [
            "moji/train.parquet",
            "moji/test.parquet",
        ],
        "description": "Moji 情感分析数据集 (~82 MB)",
        "task": "SENT_CLF",
    },
    "celeba_anno": {
        "expected_files": [
            "celeba/Anno/list_attr_celeba.txt",
            "celeba/Eval/list_eval_partition.txt",
        ],
        "description": "CelebA 标注文件 (已包含在仓库中)",
        "task": "IMG_CLF",
    },
}

ALL_DATASETS = {**RELEASE_DATASETS, **BUNDLED_DATASETS}


def check_dataset_status(name):
    if name not in ALL_DATASETS:
        return False, [f"未知数据集: {name}"]
    info = ALL_DATASETS[name]
    missing = []
    for ef in info["expected_files"]:
        full_path = os.path.join(DATASET_DIR, ef)
        if ef.endswith("/"):
            if not os.path.isdir(full_path):
                missing.append(ef)
        else:
            if not os.path.isfile(full_path):
                missing.append(ef)
    return len(missing) == 0, missing


def download_file(url, dest_path, chunk_size=8192):
    if os.path.exists(dest_path):
        print(f"  [跳过] 文件已存在: {os.path.basename(dest_path)}")
        return True

    print(f"  [下载] {url}")
    tmp_path = None
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0
        start_time = time.time()

        tmp_path = dest_path + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        elapsed = time.time() - start_time
                        speed = downloaded / elapsed / 1024 / 1024
                        pct = downloaded / total_size * 100
                        print(f"\r  进度: {pct:.1f}% ({downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB, {speed:.1f} MB/s)", end="", flush=True)

        print()
        os.rename(tmp_path, dest_path)
        return True
    except requests.exceptions.RequestException as e:
        print(f"\n  [错误] 下载失败: {e}")
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def extract_archive(archive_path, dest_dir):
    print(f"  [解压] {os.path.basename(archive_path)} -> {dest_dir}")
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(dest_dir)
        print(f"  [完成] 解压成功")
        return True
    except (tarfile.TarError, EOFError) as e:
        print(f"  [错误] 解压失败: {e}")
        return False


def download_dataset(name):
    if name not in RELEASE_DATASETS:
        if name in BUNDLED_DATASETS:
            print(f"[跳过] {name} 已包含在仓库中，无需下载")
        else:
            print(f"[跳过] 未知数据集: {name}")
        return

    info = RELEASE_DATASETS[name]
    ok, missing = check_dataset_status(name)
    if ok:
        print(f"[跳过] {name} 已存在，无需下载")
        return

    print(f"\n{'='*60}")
    print(f"数据集: {name}")
    print(f"描述:   {info['description']}")
    print(f"任务:   {info['task']}")
    print(f"{'='*60}")

    archive_url = f"{RELEASE_BASE_URL}/{info['archive']}"
    archive_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), info["archive"])

    if not download_file(archive_url, archive_path):
        print(f"[失败] {name} 下载失败")
        print(f"  请确认 Release 已创建: {RELEASE_BASE_URL}")
        return

    if not extract_archive(archive_path, DATASET_DIR):
        print(f"[失败] {name} 解压失败")
        return

    os.remove(archive_path)
    print(f"  [清理] 已删除压缩包: {info['archive']}")

    ok, missing = check_dataset_status(name)
    if ok:
        print(f"[成功] {name} 下载并解压完成!")
    else:
        print(f"[警告] {name} 解压后仍缺少文件: {missing}")
        print(f"  请检查压缩包内容是否正确")


def pack_dataset(name):
    if name not in RELEASE_DATASETS:
        print(f"[跳过] {name} 不需要打包（已包含在仓库中或未知）")
        return

    info = RELEASE_DATASETS[name]
    ok, missing = check_dataset_status(name)
    if not ok:
        print(f"[错误] {name} 数据不完整，缺少: {missing}")
        return

    print(f"\n[打包] {name} -> {info['archive']}")
    archive_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), info["archive"])

    sources = []
    for ef in info["expected_files"]:
        full_path = os.path.join(DATASET_DIR, ef)
        sources.append(full_path)

    with tarfile.open(archive_path, "w:gz") as tar:
        for src in sources:
            arcname = os.path.relpath(src, DATASET_DIR)
            tar.add(src, arcname=arcname)

    size_mb = os.path.getsize(archive_path) / 1024 / 1024
    print(f"[完成] {info['archive']} ({size_mb:.1f} MB)")
    print(f"  上传到 GitHub Release: {RELEASE_BASE_URL}/{info['archive']}")


def list_datasets():
    print("\n可用数据集:\n")
    print(f"  {'名称':16s} {'任务':12s} {'来源':10s} {'描述'}")
    print(f"  {'-'*16} {'-'*12} {'-'*10} {'-'*40}")

    print("\n  --- 需要下载的图像数据集 ---")
    for name, info in RELEASE_DATASETS.items():
        ok, _ = check_dataset_status(name)
        status = "✓ 就绪" if ok else "✗ 缺失"
        print(f"  {status} {name:14s} {info['task']:12s} {'Release':10s} {info['description']}")

    print("\n  --- 已包含在仓库中的数据集 ---")
    for name, info in BUNDLED_DATASETS.items():
        ok, _ = check_dataset_status(name)
        status = "✓ 就绪" if ok else "✗ 缺失"
        print(f"  {status} {name:14s} {info['task']:12s} {'仓库':10s} {info['description']}")


def main():
    parser = argparse.ArgumentParser(description="数据集下载/打包工具")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="指定数据集名称 (默认: 全部)")
    parser.add_argument("--check", action="store_true",
                        help="只检查数据集状态，不下载")
    parser.add_argument("--pack", action="store_true",
                        help="本地打包数据集（用于上传 GitHub Release）")
    parser.add_argument("--list", action="store_true",
                        help="列出所有可用数据集")
    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    names = args.datasets or list(ALL_DATASETS.keys())

    if args.pack:
        print("=" * 60)
        print("  打包数据集 (用于上传 GitHub Release)")
        print("=" * 60)
        for name in names:
            pack_dataset(name)
        print(f"\n提示: 修改脚本中的 GITHUB_REPO 为你的实际仓库地址")
        print(f"当前: {GITHUB_REPO}")
        return

    print("=" * 60)
    print("  Fairness FL 数据集管理工具")
    print("=" * 60)

    print("\n--- 数据集状态检查 ---\n")
    for name in names:
        if name not in ALL_DATASETS:
            print(f"  {name:16s} 未知数据集")
            continue
        info = ALL_DATASETS[name]
        ok, missing = check_dataset_status(name)
        source = "Release" if name in RELEASE_DATASETS else "仓库"
        status = "✓ 就绪" if ok else f"✗ 缺少 {len(missing)} 项"
        print(f"  {name:16s} {status}  [{source}] ({info['description']})")
        if not ok and not args.check:
            for m in missing:
                print(f"                  - {m}")

    if args.check:
        print("\n--- 检查完毕 ---")
        return

    missing_names = [n for n in names if n in RELEASE_DATASETS and not check_dataset_status(n)[0]]
    if not missing_names:
        print("\n所有指定数据集已就绪，无需下载!")
        return

    print(f"\n--- 开始下载: {', '.join(missing_names)} ---")
    for name in missing_names:
        download_dataset(name)

    print("\n" + "=" * 60)
    print("  全部完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
