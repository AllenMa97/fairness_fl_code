"""生成 FedFACT / PraFFL 在表格、图像、音频三种模态上的补跑队列。

协议参数全部取自 result_path 中同类历史结果（FedAvg 为基准），保证同口径可比：
    Tabular_CLF : I=500, E=2, bs=256, lr=3e-4, sgd, eval_freq=10
    IMG_CLF     : I=50,  E=2, bs=256, lr=3e-4, sgd, eval_freq=5
    AUD_CLF     : AudioMNIST I=100 / 其余 I=200, E=2, bs=64, eval_freq 按数据集

exp_repeat_times 固定 3：gap 判据要求 ACC.values 长度 ≥ 3，1 次 repeat 的试点不计入完成。

FedFACT 的两个论文前置条件由 main_*.py 保证：FL_fraction 强制 1.0、FL_drop_rate 为 0。
Dirichlet 划分下若某参与方只有单一 protected 组，FedFACT 会抛 SupportError 并 rc=1 快速失败，
这是算法内在限制，不在此处预先剔除（失败即记录，便于文档标注「不可用」）。

用法：
    python patch_experiment/gen_multimodal.py          # 生成到 patch_experiment/queues/mm_*.jsonl
"""

import json
import os
import sys

ALGOS = ["FedFACT", "PraFFL"]
SPLITS4 = ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"]
REPEATS = 3
# FedFACT 要求每个参与方的测试分片同时含两个 protected 组（论文前置条件）。
# Dirichlet 标签偏斜会让部分参与方的分片只剩单一组，评估期抛 SupportError；
# 实测 COMPAS 的 3 个 Dirichlet 档 × 3 个 K 全部 rc=1，故只跑 Uniform/Speaker。
FEDFACT_SPLITS = {"Uniform", "Speaker"}
# FedFACT 的「部分支撑」补跑：开启 -fedfact_partial_support 后，缺 protected 组或
# 缺 (group,label) 格的客户端不再报错（对应行在 calibrated_loss 里本就不会被选中）。
# 只补原版跑不了的那部分格，避免重复已在 Uniform/Speaker 上跑通的格。
FEDFACT_PARTIAL_FLAG = "-fedfact_partial_support"

PINNED = [
    "-algorithm_epoch_T", "2",
    "-final_artifact_policy", "metrics_only",
    "-client_parallel", "1",
    "-cuda", "cpu",
]

# 模态 -> 公共协议
TABULAR = {
    "train": "main_Tabular_CLF.py",
    "task": "Tabular_CLF",
    "extra_common": ["-model_type", "ANN"],
    "rounds": 500, "batch": 256, "test_batch": 256,
    "lr": "0.0003", "opt": "sgd", "eval_freq": 10,
}
IMAGE = {
    "train": "main_IMG_CLF.py",
    "task": "IMG_CLF",
    "extra_common": [],
    "rounds": 50, "batch": 256, "test_batch": 256,
    "lr": "0.0003", "opt": "sgd", "eval_freq": 5,
}
AUDIO = {
    "train": "main_AUD_CLF.py",
    "task": "AUD_CLF",
    "extra_common": [],
    "rounds": 200, "batch": 64, "test_batch": 64,
    "lr": "0.001", "opt": "adam", "eval_freq": 50,
}

# (dataset, splits, ks, 覆盖项, 备注)；顺序即队列执行顺序，先便宜后昂贵
TABULAR_CELLS = [
    ("COMPAS", SPLITS4, [20, 30, 40], {}, "500 轮约 1 分钟/次"),
    ("ADULT", SPLITS4, [20], {}, ""),
    ("BANK", SPLITS4, [20, 30, 40], {}, ""),
    ("DRUG", SPLITS4, [20, 30, 40], {}, ""),
    ("DUTCH", SPLITS4, [20, 30, 40], {}, "K=20 单次约 90 分钟"),
    ("ACSIncome", SPLITS4, [20, 30, 40], {}, "数据量最大"),
]

IMAGE_CELLS = [
    ("UTKFace", SPLITS4, [20, 30, 40], {}, ""),
    ("PAD-UFES-20", SPLITS4, [20, 30, 40], {"test_batch": 32}, ""),
    ("LFWA+", SPLITS4, [20, 30, 40], {}, ""),
    ("CelebA", SPLITS4, [20, 30, 40], {}, ""),
    ("FairFace", SPLITS4, [20, 30, 40], {"test_batch": 64}, ""),
]

AUDIO_CELLS = [
    ("AudioMNIST", SPLITS4, [20], {"rounds": 100, "lr": "0.01", "opt": "sgd", "eval_freq": 100}, ""),
    ("RAVDESS", ["Speaker"], [20], {"eval_freq": 5, "balance_sampling": 1}, ""),
    ("CREMA-D", ["Speaker"], [20], {}, ""),
    ("CREMA-D", ["Speaker"], [20], {"sensitive_attribute": "gender"}, "结果落入 CREMA-D_gender"),
    ("CREMA-D_ar", ["Speaker"], [20], {}, ""),
    ("VCTK", ["Speaker"], [20], {}, ""),
]


def build_argv(profile, dataset, split, algo, k, overrides):
    argv = ["-dataset", dataset, "-split_strategy", split,
            "-algorithm", algo, "-num_clients_K", str(k),
            "-task", profile["task"]]
    argv += profile["extra_common"]
    if overrides.get("sensitive_attribute"):
        argv += ["-sensitive_attribute", overrides["sensitive_attribute"]]
    argv += ["-communication_round_I", str(overrides.get("rounds", profile["rounds"]))]
    argv += PINNED
    argv += ["-exp_repeat_times", str(REPEATS)]
    argv += ["-batch_size", str(overrides.get("batch", profile["batch"]))]
    argv += ["-test_batch_size", str(overrides.get("test_batch", profile["test_batch"]))]
    argv += ["-learning_rate", str(overrides.get("lr", profile["lr"]))]
    argv += ["-optimize_method", str(overrides.get("opt", profile["opt"]))]
    argv += ["-test_eval_freq", str(overrides.get("eval_freq", profile["eval_freq"]))]
    if overrides.get("balance_sampling"):
        argv += ["-balance_sampling", str(overrides["balance_sampling"])]
    return argv


def build_queue(profile, cells, algo, prefix, soft=False, soft_keep_all_splits=False):
    records = []
    for dataset, splits, ks, overrides, note in cells:
        for split in splits:
            if algo == "FedFACT":
                # 原版：只跑 support 齐全的划分；soft：只补原版跳过的划分
                if not soft and split not in FEDFACT_SPLITS:
                    continue
                if soft and not soft_keep_all_splits and split in FEDFACT_SPLITS:
                    continue
            for k in ks:
                argv = build_argv(profile, dataset, split, algo, k, overrides)
                if soft and algo == "FedFACT":
                    argv = argv + [FEDFACT_PARTIAL_FLAG]
                records.append({
                    "dataset": dataset,
                    "split": split,
                    "algo": algo,
                    "tag": None,
                    "extra": [],
                    "train": profile["train"],
                    "note": note,
                    "argv": argv,
                })
    name = "%s_%s%s.jsonl" % (prefix, algo.lower(), "_soft" if soft else "")
    return name, records


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(root, "queues")
    os.makedirs(out_dir, exist_ok=True)

    if "--fedfact-partial" in sys.argv[1:]:
        # 只补原版 FedFACT 跑不了的格：
        #   表格：Dirichlet01/05/1 × K（原版只跑 Uniform）
        #   音频：除 AudioMNIST 外的 Speaker 格（原版 AudioMNIST 已跑通，不重复）
        audio_soft = [c for c in AUDIO_CELLS if c[0] != "AudioMNIST"]
        groups = [("mm_tab", TABULAR, TABULAR_CELLS, False),
                  ("mm_aud", AUDIO, audio_soft, True)]
        for prefix, profile, cells, keep_all in groups:
            name, records = build_queue(profile, cells, "FedFACT", prefix,
                                        soft=True, soft_keep_all_splits=keep_all)
            path = os.path.join(out_dir, name)
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                for rec in records:
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print("%-26s %3d 格  合计 %d 次训练" % (name, len(records), len(records) * REPEATS))
        return

    groups = [("mm_tab", TABULAR, TABULAR_CELLS),
              ("mm_img", IMAGE, IMAGE_CELLS),
              ("mm_aud", AUDIO, AUDIO_CELLS)]
    for prefix, profile, cells in groups:
        for algo in ALGOS:
            name, records = build_queue(profile, cells, algo, prefix)
            path = os.path.join(out_dir, name)
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                for rec in records:
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print("%-22s %3d 格  合计 %d 次训练" % (name, len(records), len(records) * REPEATS))


if __name__ == "__main__":
    main()
