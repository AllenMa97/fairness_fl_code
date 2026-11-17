"""汇总 result_path 下全部已完成格，生成一份自包含的汇总表。

输入：result_path/<dataset>/<split>/<algo>/<model>/<K>Clients/<N>.txt.json
      （同一格的多次 repeat 是 N=1..3 三个文件，这里对 repeat 取均值）
输出：reported_results/all_results/summary.md  人读报表（含覆盖缺口）
      reported_results/all_results/cells.csv    long-form，一格一行

只用标准库，避免依赖 pandas。
"""

import csv
import json
import os
import time
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT = os.path.join(REPO, "result_path")
OUT = os.path.join(REPO, "reported_results", "all_results")

# 报表主列；结果里若还有别的指标（EOD 等）会追加在后面
MAIN = ["ACC", "HM", "DEO", "FR", "SPD", "ACC_gap"]

# 论文口径的核心算法（18 个）。实验目录里大量名字是这几族的变体
# （PDFFed_comb11011 / FedRenyi_lam2 / PDFFed_DP_eps8 …），覆盖统计按核心名归并，
# 明细表仍按目录原名逐行列出，方便查消融。
CORE = ["FedAvg", "FedProx", "FedNova", "FedFB", "FedRep", "FedProto", "FedFACT",
        "PraFFL", "FedRenyi", "FL_FairBatch", "mFairFL", "Scaffold", "LoGoFair",
        "FairFed", "FederatedAverageWithPo", "PDFFed_DP", "FedFair", "PDFFed"]
_CORE_BY_LEN = sorted(CORE, key=len, reverse=True)


def canonical(algo):
    """把消融变体名归并到核心算法名；认不出就返回原名。"""
    for name in _CORE_BY_LEN:
        if algo == name or algo.startswith(name + "_"):
            return name
    return algo


def walk_jsons():
    for root, _dirs, files in os.walk(RESULT):
        for name in files:
            if name.endswith(".txt.json"):
                yield os.path.join(root, name)


def parse_path(path):
    """拆出 dataset/split/algo/model/K；结构不符则返回 None。

    bios/Dirichlet01/FedNova/<variant>/BERTCLASSIFIER/20Clients/1.txt.json 这类
    多一级的历史目录，把两级算法名用 / 连起来，保证结果不被丢掉。
    """
    rel = os.path.relpath(path, RESULT).replace("\\", "/").split("/")
    if len(rel) == 7:
        ds, split, algo, sub, model, kdir, _fname = rel
        algo = algo + "/" + sub
    elif len(rel) == 6:
        ds, split, algo, model, kdir, _fname = rel
    else:
        return None
    if not kdir.endswith("Clients"):
        return None
    try:
        k = int(kdir[:-len("Clients")])
    except ValueError:
        return None
    return ds, split, algo, model, k


def cell_batch(cfg):
    """有效批：优先 effective_batch_size，其次 batch_size；返回 int/None。"""
    for key in ("effective_batch_size", "batch_size"):
        value = cfg.get(key)
        if isinstance(value, (int, float)) and value:
            return int(value)
    return None


def load():
    cells = {}
    metric_names = set()
    bad = []
    for path in walk_jsons():
        key = parse_path(path)
        if key is None:
            bad.append(path)
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except Exception as exc:  # 坏文件单独记账，不让整轮汇总挂掉
            bad.append("%s (%s)" % (path, exc))
            continue
        agg = doc.get("aggregate") or {}
        metrics = agg.get("metrics") or {}
        cfg = doc.get("experiment_config") or {}
        entry = cells.setdefault(key, {"runs": [], "metrics": defaultdict(list),
                                       "batch": cell_batch(cfg), "cfg": cfg})
        for name, block in metrics.items():
            mean = (block or {}).get("mean")
            if isinstance(mean, (int, float)):
                entry["metrics"][name].append(float(mean))
                metric_names.add(name)
        entry["runs"].append(os.path.basename(path))
        if entry["batch"] is None:
            entry["batch"] = cell_batch(cfg)
    return cells, metric_names, bad


def avg(values):
    return sum(values) / len(values) if values else None


def fmt(value, digits=4):
    return "—" if value is None else ("%.*f" % (digits, value))


def main():
    cells, metric_names, bad = load()
    extra = sorted(n for n in metric_names if n not in MAIN)
    # md 只放主指标，避免 16 个埋点列把表撑宽（埋点指标照样进 cells.csv）
    columns = [n for n in MAIN if n in metric_names]
    csv_columns = columns + extra

    ds_algos = defaultdict(set)
    ds_core = defaultdict(set)
    ds_splits = defaultdict(set)
    global_variants = set()
    for ds, split, algo, model, k in cells:
        ds_algos[ds].add(algo)
        ds_core[ds].add(canonical(algo))
        ds_splits[ds].add((split, k))
        global_variants.add(algo)

    lines = []
    add = lines.append
    add("# 实验结果总汇总")
    add("")
    add("> 生成时间：%s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    add("> 数据源：`result_path/` 下全部 `*.txt.json`（同一格的多次 repeat 取均值）")
    add("> 生成脚本：`analysis/build_all_results_summary.py`（可重复运行）")
    add("")
    add("- 结果文件：**%d** 个" % sum(len(c["runs"]) for c in cells.values()))
    add("- 已完成格（dataset×split×algo×model×K）：**%d**" % len(cells))
    add("- 涉及数据集：%d 个（%s）" % (len(ds_algos), "、".join(sorted(ds_algos))))
    add("- 算法目录名：%d 个（归并为 %d 个核心算法 + %d 个变体/其它）"
        % (len(global_variants), len({canonical(a) for a in global_variants} & set(CORE)),
           len(global_variants) - len({canonical(a) for a in global_variants} & set(CORE))))
    if bad:
        add("- 无法解析/读取的文件：%d 个（已跳过，见文末）" % len(bad))
    add("")

    # ---- 1. 覆盖总览 ----
    add("## 1. 覆盖总览")
    add("")
    add("| 数据集 | (split,K) 列 | 核心算法 | 缺失核心算法 | 其它算法名 |")
    add("|---|---|---|---|---|")
    for ds in sorted(ds_algos):
        core_hit = sorted(ds_core[ds] & set(CORE))
        missing = sorted(set(CORE) - set(core_hit))
        extras = sorted(ds_core[ds] - set(CORE))
        add("| %s | %d | %d/%d | %s | %s |" % (
            ds, len(ds_splits[ds]), len(core_hit), len(CORE),
            "、".join(missing) if missing else "—",
            "、".join(extras) if extras else "—"))
    add("")
    add("> 「缺失核心算法」的参照是论文口径的 %d 个算法（见脚本 `CORE`）。"
        "音频类数据集（CREMA-D / RAVDESS / AudioMNIST / VCTK）走独立计划，"
        "其缺失项需对照 `实验缺口明细.md` 解读。" % len(CORE))
    add("")

    # ---- 2. 分数据集明细 ----
    add("## 2. 分数据集明细")
    add("")
    for ds in sorted(ds_algos):
        add("### %s" % ds)
        add("")
        for split, k in sorted(ds_splits[ds]):
            rows = [(model, algo, cell) for (d, s, algo, model, kk), cell in cells.items()
                    if d == ds and s == split and kk == k]
            if not rows:
                continue
            add("#### %s / K=%d" % (split, k))
            add("")
            add("| 算法 | 模型 | 批量 | " + " | ".join(columns) + " | repeats |")
            add("|---|---|---|" + "---|" * len(columns) + "---|")
            for model, algo, cell in sorted(rows, key=lambda r: (r[1], r[0])):
                values = " | ".join(fmt(avg(cell["metrics"].get(name, []))) for name in columns)
                add("| %s | %s | %s | %s | %d |" % (
                    algo, model, cell["batch"] if cell["batch"] else "—", values, len(cell["runs"])))
            add("")

    # ---- 3. long-form CSV ----
    os.makedirs(OUT, exist_ok=True)
    csv_path = os.path.join(OUT, "cells.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "split", "algorithm", "model", "K", "effective_batch",
                         "repeats"] + csv_columns)
        for (ds, split, algo, model, k), cell in sorted(cells.items()):
            writer.writerow([ds, split, algo, model, k,
                             cell["batch"] if cell["batch"] else "",
                             len(cell["runs"])] +
                            [("%.6f" % avg(cell["metrics"][name])) if cell["metrics"].get(name) else ""
                             for name in csv_columns])

    if bad:
        add("## 附：未能解析的文件")
        add("")
        for item in bad[:50]:
            add("- `%s`" % item)
        if len(bad) > 50:
            add("- …（另有 %d 个）" % (len(bad) - 50))
        add("")

    md_path = os.path.join(OUT, "summary.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print("cells=%d files=%d datasets=%d variants=%d extra=%s"
          % (len(cells), sum(len(c['runs']) for c in cells.values()),
             len(ds_algos), len(global_variants), ",".join(extra) or "-"))
    print("wrote", md_path)
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
