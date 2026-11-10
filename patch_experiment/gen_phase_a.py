"""生成 Phase A 机制验证队列（COMPAS / DUTCH）。

协议对齐（全部取自历史 COMPAS/PDFFed 的 experiment_config，保证与既有结果同口径）：
    communication_round_I=500, algorithm_epoch_T=2, batch_size=256, test_batch_size=256,
    learning_rate=3e-4, optimize_method=sgd, test_eval_freq=10, model_type=ANN,
    final_artifact_policy=metrics_only, client_parallel=1, use_amp=false

本阶段 repeat 固定为 1：只用于机制方向（Δ_rep 中介、达成代价、落差），
不做终点排名；终点结论仍需 3 repeats + 配对检验。

变体通过 `-variant_tag` 落到独立目录，便于与本体结果并列比较：
    PDFFed_noPA      ->  -w_pa 0        （关掉表征对齐项，H1 的对照）
    PDFFed_noEO      ->  -lambda_eo 0   （关掉服务端校准，H4 的对照）
    PDFFed_wpa0p01 / PDFFed_wpa0p3      （λ 剂量-响应，H1）
"""

import json
import os
import sys

# 与 result_path 中的历史算法名严格一致，避免出现两套命名
ALGOS = [
    "PDFFed", "PDFFed_DP",
    "FedRenyi", "FedAvg", "FedProx", "Scaffold", "FedNova", "FedRep",
    "FedProto", "FairFed", "FedFair", "FedFB", "FL_FairBatch",
    "LoGoFair", "mFairFL", "FederatedAverageWithPo", "PraFFL", "FedFACT", "FedPost",
]

# (variant_tag, 追加参数)。variant_tag 会拼进目录名：<algorithm>_<tag>
VARIANTS = [
    ("noPA", ["-w_pa", "0"]),
    ("noEO", ["-lambda_eo", "0"]),
    ("wpa0p01", ["-w_pa", "0.01"]),
    ("wpa0p3", ["-w_pa", "0.3"]),
]

BASE = [
    "-task", "Tabular_CLF", "-model_type", "ANN",
    "-communication_round_I", "500",
    "-algorithm_epoch_T", "2",
    "-exp_repeat_times", "1",
    "-batch_size", "256", "-test_batch_size", "256",
    "-learning_rate", "0.0003", "-optimize_method", "sgd",
    "-test_eval_freq", "10",
    "-final_artifact_policy", "metrics_only",
    "-client_parallel", "1",
    "-cuda", "cpu",
]

DATASETS = {
    "COMPAS": ["Dirichlet01", "Uniform"],
    "DUTCH": ["Dirichlet01", "Uniform"],
}
K = 20
SHARDS = 8


def build_cells(dataset):
    cells = []
    for split in DATASETS[dataset]:
        for algo in ALGOS:
            cells.append({"dataset": dataset, "split": split, "algo": algo,
                          "tag": None, "extra": []})
        for tag, extra in VARIANTS:
            cells.append({"dataset": dataset, "split": split, "algo": "PDFFed",
                          "tag": tag, "extra": extra})
    return cells


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(root, "queues")
    os.makedirs(out_dir, exist_ok=True)
    total = 0
    for dataset in DATASETS:
        cells = build_cells(dataset)
        # 轮转分片：按 split/algo 交错，保证每片都同时含两种划分，进度可控
        shards = [[] for _ in range(SHARDS)]
        for i, cell in enumerate(cells):
            shards[i % SHARDS].append(cell)
        for si, shard in enumerate(shards):
            path = os.path.join(out_dir, "%s_q%02d.jsonl" % (dataset.lower(), si))
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                for cell in shard:
                    argv = ["-dataset", dataset, "-split_strategy", cell["split"],
                            "-algorithm", cell["algo"], "-num_clients_K", str(K)]
                    argv += BASE
                    if cell["tag"]:
                        argv += ["-variant_tag", cell["tag"]]
                    argv += cell["extra"]
                    rec = dict(cell)
                    rec["argv"] = argv
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += len(shard)
            print("%-24s %d 格" % (os.path.basename(path), len(shard)))
    print("合计 %d 格（每个数据集 %d 格：2 划分 × %d 配置）" % (
        total, len(build_cells("COMPAS")), len(ALGOS) + len(VARIANTS)))


if __name__ == "__main__":
    main()
