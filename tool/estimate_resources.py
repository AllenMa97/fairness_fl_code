#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验资源预估工具 - 在跑实验前预估 GPU 显存和 CPU 内存用量

用法:
    python -m tool.estimate_resources --task SENT_CLF --model BERTCLASSIFIER --batch_size 256 --num_clients 20
    python -m tool.estimate_resources --task IMG_CLF --model CNNCLASSIFIER --dataset celeba --batch_size 256 --num_clients 40
    python -m tool.estimate_resources --task Tabular_CLF --model ANN --dataset ADULT --batch_size 256 --num_clients 20
    python -m tool.estimate_resources --task SENT_CLF --model BERTCLASSIFIER --batch_size 128 --max_len 128 --num_clients 20 --gpus 2
"""

import argparse

MODEL_PROFILES = {
    "BERTCLASSIFIER": {
        "params_m": 110,
        "weight_gb": 0.44,
        "optimizer_gb": 0.88,
        "activation_per_sample_kb": 16,
        "description": "BERT-base-uncased + Linear classifier",
    },
    "CNNCLASSIFIER": {
        "params_m": 5.2,
        "weight_gb": 0.02,
        "optimizer_gb": 0.04,
        "activation_per_sample_kb": 0.8,
        "description": "9-layer VGG-style CNN",
    },
    "ANN": {
        "params_m": 0.03,
        "weight_gb": 0.0001,
        "optimizer_gb": 0.0002,
        "activation_per_sample_kb": 0.01,
        "description": "3-layer MLP (input_size dependent)",
    },
    "LogisticRegression": {
        "params_m": 0.001,
        "weight_gb": 0.00001,
        "optimizer_gb": 0.00002,
        "activation_per_sample_kb": 0.001,
        "description": "Logistic Regression",
    },
}

DATASET_IMAGE_SIZES = {
    "celeba": (64, 64),
    "UTKFace": (64, 64),
    "FairFace": (224, 224),
    "LFWA+": (250, 250),
}

DATASET_SIZES = {
    "moji": {"train": 1_610_000, "test": 440_000},
    "bios": {"train": 250_000, "test": 99_000},
    "celeba": {"train": 162_770, "test": 19_967},
    "UTKFace": {"train": 19_290, "test": 4_819},
    "FairFace": {"train": 86_000, "test": 10_000},
    "LFWA+": {"train": 10_560, "test": 2_640},
    "ADULT": {"train": 30_718, "test": 7_680},
    "COMPAS": {"train": 4_320, "test": 1_080},
    "DRUG": {"train": 1_440, "test": 360},
    "DUTCH": {"train": 4_800, "test": 1_200},
}


def estimate_gpu_memory(model_name, task, batch_size, max_len=128, dataset=None, gpus=1):
    if model_name not in MODEL_PROFILES:
        return None, None, None

    profile = MODEL_PROFILES[model_name]

    single_model_gb = profile["weight_gb"] + profile["optimizer_gb"]

    if model_name == "BERTCLASSIFIER":
        seq_len = max_len
        hidden = 768
        layers = 12
        activation_gb = (batch_size * seq_len * hidden * layers * 4) / (1024 ** 3) * 2.5
        input_gb = (batch_size * seq_len * 8) / (1024 ** 3)
        per_model_gb = single_model_gb + activation_gb + input_gb
        peak_models = 2
    elif model_name == "CNNCLASSIFIER":
        if dataset and dataset in DATASET_IMAGE_SIZES:
            h, w = DATASET_IMAGE_SIZES[dataset]
        else:
            h, w = 64, 64
        channels = 3
        input_gb = (batch_size * channels * h * w * 4) / (1024 ** 3)
        activation_gb = input_gb * 8
        per_model_gb = single_model_gb + activation_gb + input_gb
        peak_models = 1
    else:
        per_model_gb = single_model_gb + 0.01
        peak_models = 1

    peak_gpu_gb = per_model_gb * peak_models
    per_gpu_gb = peak_gpu_gb / gpus

    safe_per_gpu_gb = per_gpu_gb * 1.3

    return per_gpu_gb, safe_per_gpu_gb, peak_models


def estimate_cpu_ram(model_name, num_clients, dataset=None):
    if model_name not in MODEL_PROFILES:
        return None, None

    profile = MODEL_PROFILES[model_name]
    model_size_gb = profile["weight_gb"]

    client_models_gb = num_clients * model_size_gb
    global_model_gb = model_size_gb

    data_ram_gb = 2.0
    if dataset and dataset in DATASET_SIZES:
        total_samples = DATASET_SIZES[dataset]["train"] + DATASET_SIZES[dataset]["test"]
        if dataset in ["moji", "bios"]:
            data_ram_gb = total_samples * 0.5 / (1024 ** 2)
        elif dataset in ["celeba", "UTKFace", "FairFace", "LFWA+"]:
            data_ram_gb = total_samples * 0.05 / (1024 ** 2)
        else:
            data_ram_gb = total_samples * 0.01 / (1024 ** 2)
        data_ram_gb = max(data_ram_gb, 1.0)

    total_ram_gb = client_models_gb + global_model_gb + data_ram_gb + 2.0
    safe_ram_gb = total_ram_gb * 1.5

    return total_ram_gb, safe_ram_gb


def estimate_disk(model_name, num_clients, communication_rounds):
    if model_name not in MODEL_PROFILES:
        return None

    profile = MODEL_PROFILES[model_name]
    model_size_gb = profile["weight_gb"]

    per_round_gb = model_size_gb * (1 + num_clients)
    total_gb = per_round_gb * communication_rounds

    return total_gb


def format_gb(gb):
    if gb < 0.01:
        return f"{gb * 1024:.0f} MB"
    elif gb < 1:
        return f"{gb * 1024:.0f} MB"
    else:
        return f"{gb:.1f} GB"


def main():
    parser = argparse.ArgumentParser(description="实验资源预估工具")
    parser.add_argument("--task", type=str, required=True,
                        choices=["SENT_CLF", "IMG_CLF", "Tabular_CLF"])
    parser.add_argument("--model", type=str, default=None,
                        choices=["BERTCLASSIFIER", "CNNCLASSIFIER", "ANN", "LogisticRegression"])
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--comm_rounds", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    if args.model is None:
        if args.task == "SENT_CLF":
            args.model = "BERTCLASSIFIER"
        elif args.task == "IMG_CLF":
            args.model = "CNNCLASSIFIER"
        else:
            args.model = "ANN"

    print("=" * 60)
    print("  实验资源预估")
    print("=" * 60)
    print(f"\n  配置:")
    print(f"    任务类型:     {args.task}")
    print(f"    模型:         {args.model} ({MODEL_PROFILES[args.model]['description']})")
    print(f"    数据集:       {args.dataset or '未指定'}")
    print(f"    Batch Size:   {args.batch_size}")
    if args.task == "SENT_CLF":
        print(f"    Max Length:   {args.max_len}")
    print(f"    客户端数量:   {args.num_clients}")
    print(f"    通信轮次:     {args.comm_rounds}")
    print(f"    GPU 数量:     {args.gpus}")

    per_gpu, safe_gpu, peak_models = estimate_gpu_memory(
        args.model, args.task, args.batch_size, args.max_len, args.dataset, args.gpus
    )

    total_ram, safe_ram = estimate_cpu_ram(args.model, args.num_clients, args.dataset)

    disk_gb = estimate_disk(args.model, args.num_clients, args.comm_rounds)

    print(f"\n  {'资源':15s} {'预估用量':>12s} {'安全阈值(×1.3)':>18s}")
    print(f"  {'-'*15} {'-'*12} {'-'*18}")

    if per_gpu is not None:
        print(f"  {'GPU 显存/卡':15s} {format_gb(per_gpu):>12s} {format_gb(safe_gpu):>18s}")
        print(f"  {'GPU 峰值模型数':15s} {str(peak_models):>12s} {'':>18s}")

    if total_ram is not None:
        print(f"  {'CPU 内存':15s} {format_gb(total_ram):>12s} {format_gb(safe_ram):>18s}")

    if disk_gb is not None:
        print(f"  {'磁盘 (无清理)':15s} {format_gb(disk_gb):>12s} {'':>18s}")
        print(f"  {'磁盘 (有清理)':15s} {format_gb(MODEL_PROFILES[args.model]['weight_gb']):>12s} {'':>18s}")

    print(f"\n  {'建议':}")
    if per_gpu is not None:
        if safe_gpu > 24:
            print(f"    ⚠ GPU 显存需求 {format_gb(safe_gpu)} 超过单卡上限 (24 GB)!")
            print(f"      建议: 减少 batch_size 或增加 GPU 数量")
            min_gpus = int(safe_gpu / 20) + 1
            print(f"      最少需要 {min_gpus} 张 GPU (假设每卡 24 GB)")
        elif safe_gpu > 16:
            print(f"    ⚠ GPU 显存需求 {format_gb(safe_gpu)} 较高，建议使用 24 GB 显卡")
        elif safe_gpu > 8:
            print(f"    ✓ GPU 显存需求 {format_gb(safe_gpu)}，需要 12 GB+ 显卡")
        else:
            print(f"    ✓ GPU 显存需求 {format_gb(safe_gpu)}，8 GB 显卡即可")

    if total_ram is not None:
        if safe_ram > 64:
            print(f"    ⚠ CPU 内存需求 {format_gb(safe_ram)} 较高")
        else:
            print(f"    ✓ CPU 内存需求 {format_gb(safe_ram)}")

    if args.num_clients > 20 and args.model == "BERTCLASSIFIER":
        print(f"\n  💡 提示: {args.num_clients} 个 BERT 客户端模型将占用 ~{format_gb(args.num_clients * 0.44)} 磁盘空间")
        print(f"     已启用自动清理，实验结束后只保留最终全局模型")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
