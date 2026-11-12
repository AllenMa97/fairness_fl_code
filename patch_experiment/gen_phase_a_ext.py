"""生成 H1/H2 复现队列（Phase A 扩展）。

目的：Phase A 已在 COMPAS/DUTCH × K=20 上跑出机制方向，但 H1 的"λ=0.3 显著低于 λ=0"
与 H2 的"同 DEO 下 Δ_rep 更低"都需要更多配对格才能做统计检验。
本队列用 COMPAS 的 K=30/40（同数据集、换客户端数，避免跨数据集量纲混淆）扩充配对，
并补齐 DUTCH/Uniform 缺失的 PDFFed_noPA 对照。

协议与 Phase A 完全一致（BASE 直接复用 gen_phase_a），每个数据集只跑机制相关配置：
    PDFFed_noPA(λ=0) / PDFFed_wpa0p3(λ=0.3) / PDFFed(λ=1) / FedRenyi(H2 对照)

用法：
    python patch_experiment/gen_phase_a_ext.py
"""

import json
import os

from gen_phase_a import BASE, VARIANTS

SPLITS = ["Dirichlet01", "Uniform"]
COMPAS_KS = [30, 40]
# H1 需要 λ 三档；H2 需要 FedRenyi
CONFIGS = [
    ("PDFFed", "noPA", ["-w_pa", "0"]),
    ("PDFFed", "wpa0p3", ["-w_pa", "0.3"]),
    ("PDFFed", None, []),
    ("FedRenyi", None, []),
]
SHARDS = 4


def build(dataset, split, algo, tag, extra, k):
    argv = ["-dataset", dataset, "-split_strategy", split,
            "-algorithm", algo, "-num_clients_K", str(k)]
    argv += BASE
    if tag:
        argv += ["-variant_tag", tag]
    argv += extra
    return {"dataset": dataset, "split": split, "algo": algo, "tag": tag,
            "extra": extra, "note": "", "argv": argv}


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(root, "queues")
    os.makedirs(out_dir, exist_ok=True)

    cells = []
    for k in COMPAS_KS:
        for split in SPLITS:
            for algo, tag, extra in CONFIGS:
                cells.append(build("COMPAS", split, algo, tag, extra, k))
    shards = [[] for _ in range(SHARDS)]
    for i, cell in enumerate(cells):
        shards[i % SHARDS].append(cell)
    for si, shard in enumerate(shards):
        path = os.path.join(out_dir, "compas_ext_q%02d.jsonl" % si)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for cell in shard:
                f.write(json.dumps(cell, ensure_ascii=False) + "\n")
        print("%-24s %d 格" % (os.path.basename(path), len(shard)))

    # DUTCH/Uniform 缺 noPA 对照，补跑一格
    fix = [build("DUTCH", "Uniform", "PDFFed", "noPA", ["-w_pa", "0"], 20)]
    path = os.path.join(out_dir, "dutch_fix_q00.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for cell in fix:
            f.write(json.dumps(cell, ensure_ascii=False) + "\n")
    print("%-24s %d 格" % (os.path.basename(path), len(fix)))
    print("合计 %d 格" % (len(cells) + len(fix)))


if __name__ == "__main__":
    main()
