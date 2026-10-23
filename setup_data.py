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
    - 音频数据集（CREMA-D, AudioMNIST）需要通过本脚本从 GitHub Release 下载
    - RAVDESS 需要通过本脚本从上游 Zenodo 下载（非本仓库 Release）
    - FairFace 和 LFWA+ 的标注文件已在仓库中，图片需要额外获取
"""

import os
import sys
import argparse
import tarfile
import time
import zipfile

try:
    import requests
except ImportError:
    print("ERROR: 需要安装 requests 库: pip install requests")
    sys.exit(1)

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
# 预训练模型权重目录（离线加载 BERT 等，避免运行时在线下载受阻）
PRETRAINED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained")

# ============================================================
# 修改下面的 GITHUB_REPO 为你的实际仓库地址
# 例如: "your-username/fairness_fl_code"
# ============================================================
GITHUB_REPO = "AllenMa97/fairness_fl_code"
RELEASE_TAG = "v1.0-data"

# 私有仓库需要Token才能下载Release文件（通过HTTP Header认证）
_GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
RELEASE_BASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}"

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
    "CREMA-D": {
        "archive": "CREMAD_audio.tar.gz",
        "expected_files": [
            "CREMA-D/AudioWAV/",
        ],
        "description": "CREMA-D 情感语音 (~450 MB, 7442 条, 91 位演员)",
        "task": "AUD_CLF",
    },
    "AudioMNIST": {
        "archive": "AudioMNIST_audio.tar.gz",
        "expected_files": [
            "AudioMNIST/data/",
        ],
        "description": "AudioMNIST 数字语音 (~1 GB, 30000 条, 60 位说话人)",
        "task": "AUD_CLF",
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
    "BANK": {
        "expected_files": [
            "BANK/bank-full.csv",
            "BANK/Bank_data.csv",
        ],
        "description": "BANK 银行营销数据集 (~9 MB, UCI Bank Marketing + Renyi-Fair-Inference One-Hot 特征)",
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

# 需要从上游公开地址（非本仓库 Release）下载的数据集
# Datasets downloaded from upstream public URLs (not this repo's GitHub Release)
EXTERNAL_DATASETS = {
    "ACSIncome": {
        # Retiring Adult (NeurIPS 2021) 的 ACSIncome 任务，数据来自 2018 ACS 1-Year PUMS。
        # 直接取 Census 官方原始文件，不依赖 folktables 包的运行时下载。
        "url": "https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_pus.zip",
        "archive": "csv_pus.zip",
        "expected_files": [
            "ACSIncome/psam_pusa.csv",
            "ACSIncome/psam_pusb.csv",
        ],
        "extract_subdir": "ACSIncome",
        "description": "ACSIncome / 2018 ACS 1-Year PUMS 个人记录 (~562 MB 压缩包, 过滤后约 1.66M 样本)",
        "task": "Tabular_CLF",
    },
    "RAVDESS": {
        "url": "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1",
        # Zenodo 在部分网络环境不可达，备用 ModelScope 镜像（同一份 pure-audio 压缩包）
        "fallback_url": "https://www.modelscope.cn/api/v1/datasets/xiang13/RAVDESS-Speech/repo?Revision=master&FilePath=Audio_Speech_Actors_01-24.zip",
        "archive": "Audio_Speech_Actors_01-24.zip",
        "expected_files": [
            "RAVDESS/",
        ],
        "extract_subdir": "RAVDESS",
        "description": "RAVDESS 情感语音 (~209 MB, 1440 条语音, 24 位演员; CC BY-NC-SA 4.0)",
        "task": "AUD_CLF",
    },
}

ALL_DATASETS = {**RELEASE_DATASETS, **EXTERNAL_DATASETS, **BUNDLED_DATASETS}

# 需要从 Release 下载的预训练模型权重（解压到 pretrained/ 目录）
# Pretrained model weights downloaded from Release (extracted into pretrained/)
RELEASE_MODELS = {
    "bert-base-uncased": {
        "archive": "bert-base-uncased.tar.gz",
        "expected_files": [
            "bert-base-uncased/config.json",
            "bert-base-uncased/vocab.txt",
            "bert-base-uncased/tokenizer.json",
            "bert-base-uncased/tokenizer_config.json",
            "bert-base-uncased/model.safetensors",
        ],
        "description": "bert-base-uncased 离线权重 (~420 MB, SENT_CLF 必需)",
        "task": "SENT_CLF",
    },
}


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


def download_file(url, dest_path, chunk_size=8192, headers=None):
    if os.path.exists(dest_path):
        print(f"  [跳过] 文件已存在: {os.path.basename(dest_path)}")
        return True

    # 私有仓库Release下载需要Token认证（通过Header，不能嵌在URL里）
    if _GITHUB_TOKEN and not headers:
        headers = {"Authorization": f"token {_GITHUB_TOKEN}"}
    safe_url = url.split("@github.com/")[-1] if "@github.com/" in url else url
    print(f"  [下载] {safe_url}")
    tmp_path = None
    try:
        resp = requests.get(url, stream=True, timeout=30, headers=headers)
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
        if archive_path.lower().endswith(".zip"):
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(dest_dir)
        else:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(dest_dir)
        print(f"  [完成] 解压成功")
        return True
    except (tarfile.TarError, EOFError, zipfile.BadZipFile) as e:
        print(f"  [错误] 解压失败: {e}")
        return False


def download_dataset(name):
    if name in RELEASE_DATASETS:
        info = RELEASE_DATASETS[name]
        archive_urls = [f"{RELEASE_BASE_URL}/{info['archive']}"]
        extract_dir = DATASET_DIR
    elif name in EXTERNAL_DATASETS:
        info = EXTERNAL_DATASETS[name]
        # 上游地址（如 Zenodo）在部分网络环境下不可达，按顺序尝试备用镜像
        archive_urls = [info["url"]]
        if info.get("fallback_url"):
            archive_urls.append(info["fallback_url"])
        extract_dir = (os.path.join(DATASET_DIR, info["extract_subdir"])
                       if info.get("extract_subdir") else DATASET_DIR)
    else:
        if name in BUNDLED_DATASETS:
            print(f"[跳过] {name} 已包含在仓库中，无需下载")
        else:
            print(f"[跳过] 未知数据集: {name}")
        return

    ok, missing = check_dataset_status(name)
    if ok:
        print(f"[跳过] {name} 已存在，无需下载")
        return

    print(f"\n{'='*60}")
    print(f"数据集: {name}")
    print(f"描述:   {info['description']}")
    print(f"任务:   {info['task']}")
    print(f"{'='*60}")

    archive_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), info["archive"])

    downloaded = False
    for idx, archive_url in enumerate(archive_urls):
        if download_file(archive_url, archive_path):
            downloaded = True
            break
        if idx + 1 < len(archive_urls):
            print(f"  [重试] 尝试备用地址: {archive_urls[idx + 1]}")
    if not downloaded:
        print(f"[失败] {name} 下载失败")
        print(f"  下载地址: {', '.join(archive_urls)}")
        return

    os.makedirs(extract_dir, exist_ok=True)
    if not extract_archive(archive_path, extract_dir):
        print(f"[失败] {name} 解压失败")
        return

    # 保留压缩包，不再删除：便于离线重装、再次解压校验、以及中转 / 上传 Release 资产时复用。
    # 下载前 download_file 会跳过已存在的同名文件，因此保留下来的归档不会造成重复下载。
    # Keep the archive instead of deleting it: useful for offline re-install, re-extract
    # verification and re-uploading the asset. download_file() skips existing files, so a
    # retained archive never triggers a redundant download.
    print(f"  [保留] 压缩包已保存: {archive_path}")

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

    if EXTERNAL_DATASETS:
        print("\n  --- 需要从上游公开地址下载的数据集 ---")
        for name, info in EXTERNAL_DATASETS.items():
            ok, _ = check_dataset_status(name)
            status = "✓ 就绪" if ok else "✗ 缺失"
            print(f"  {status} {name:14s} {info['task']:12s} {'上游':10s} {info['description']}")

    print("\n  --- 已包含在仓库中的数据集 ---")
    for name, info in BUNDLED_DATASETS.items():
        ok, _ = check_dataset_status(name)
        status = "✓ 就绪" if ok else "✗ 缺失"
        print(f"  {status} {name:14s} {info['task']:12s} {'仓库':10s} {info['description']}")

    print("\n  --- 预训练模型权重（用 --models 下载）---")
    for name, info in RELEASE_MODELS.items():
        ok, _ = check_model_status(name)
        status = "✓ 就绪" if ok else "✗ 缺失"
        print(f"  {status} {name:14s} {info['task']:12s} {'Release':10s} {info['description']}")


def check_model_status(name):
    """检查预训练模型权重是否完整 / Check whether pretrained model weights are complete"""
    if name not in RELEASE_MODELS:
        return False, [f"未知模型: {name}"]
    info = RELEASE_MODELS[name]
    missing = []
    for ef in info["expected_files"]:
        full_path = os.path.join(PRETRAINED_DIR, ef)
        if not os.path.isfile(full_path):
            missing.append(ef)
    return len(missing) == 0, missing


def download_model(name):
    """从 GitHub Release 下载预训练权重并解压到 pretrained/ / Download model weights from Release"""
    if name not in RELEASE_MODELS:
        print(f"[跳过] 未知模型: {name}")
        return
    info = RELEASE_MODELS[name]
    ok, missing = check_model_status(name)
    if ok:
        print(f"[跳过] {name} 已存在，无需下载")
        return

    print(f"\n{'='*60}")
    print(f"模型:   {name}")
    print(f"描述:   {info['description']}")
    print(f"{'='*60}")

    archive_url = f"{RELEASE_BASE_URL}/{info['archive']}"
    archive_path = os.path.join(PRETRAINED_DIR, info["archive"])
    os.makedirs(PRETRAINED_DIR, exist_ok=True)

    if not download_file(archive_url, archive_path):
        print(f"[失败] {name} 下载失败")
        print(f"  请确认 Release 已上传资产: {archive_url}")
        return

    if not extract_archive(archive_path, PRETRAINED_DIR):
        print(f"[失败] {name} 解压失败")
        return

    os.remove(archive_path)
    print(f"  [清理] 已删除压缩包: {info['archive']}")

    ok, missing = check_model_status(name)
    if ok:
        print(f"[成功] {name} 下载并解压完成!")
    else:
        print(f"[警告] {name} 解压后仍缺少文件: {missing}")


def pack_model(name):
    """把 pretrained/<name> 打包为 tar.gz（用于上传 GitHub Release） / Pack model dir into tar.gz"""
    if name not in RELEASE_MODELS:
        print(f"[跳过] 未知模型: {name}")
        return
    info = RELEASE_MODELS[name]
    ok, missing = check_model_status(name)
    if not ok:
        print(f"[错误] {name} 权重不完整，缺少: {missing}")
        return

    print(f"\n[打包] {name} -> {info['archive']}")
    archive_path = os.path.join(PRETRAINED_DIR, info["archive"])
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    src_dir = os.path.join(PRETRAINED_DIR, name)

    with tarfile.open(archive_path, "w:gz") as tar:
        for fname in sorted(os.listdir(src_dir)):
            tar.add(os.path.join(src_dir, fname), arcname=os.path.join(name, fname))

    size_mb = os.path.getsize(archive_path) / 1024 / 1024
    print(f"[完成] {info['archive']} ({size_mb:.1f} MB)")
    print(f"  上传到 GitHub Release: {RELEASE_BASE_URL}/{info['archive']}")


def main():
    parser = argparse.ArgumentParser(description="数据集下载/打包工具")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="指定数据集名称 (默认: 全部)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="指定预训练模型权重名称 (如 bert-base-uncased)，从 Release 下载到 pretrained/")
    parser.add_argument("--check", action="store_true",
                        help="只检查数据集状态，不下载")
    parser.add_argument("--pack", action="store_true",
                        help="本地打包数据集（用于上传 GitHub Release）")
    parser.add_argument("--pack-models", action="store_true",
                        help="本地打包预训练模型权重（用于上传 GitHub Release）")
    parser.add_argument("--list", action="store_true",
                        help="列出所有可用数据集")
    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if args.pack_models:
        print("=" * 60)
        print("  打包预训练模型权重 (用于上传 GitHub Release)")
        print("=" * 60)
        model_names = args.models or list(RELEASE_MODELS.keys())
        for name in model_names:
            pack_model(name)
        return

    if args.models:
        print("=" * 60)
        print("  预训练模型权重下载")
        print("=" * 60)
        model_names = args.models or list(RELEASE_MODELS.keys())
        for name in model_names:
            if args.check:
                ok, missing = check_model_status(name)
                status = "✓ 就绪" if ok else f"✗ 缺少 {len(missing)} 项"
                print(f"  {name:18s} {status}")
            else:
                download_model(name)
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
        source = ("Release" if name in RELEASE_DATASETS
                  else "Zenodo" if name in EXTERNAL_DATASETS else "仓库")
        status = "✓ 就绪" if ok else f"✗ 缺少 {len(missing)} 项"
        print(f"  {name:16s} {status}  [{source}] ({info['description']})")
        if not ok and not args.check:
            for m in missing:
                print(f"                  - {m}")

    if args.check:
        print("\n--- 检查完毕 ---")
        return

    downloadable = {**RELEASE_DATASETS, **EXTERNAL_DATASETS}
    missing_names = [n for n in names if n in downloadable and not check_dataset_status(n)[0]]
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
