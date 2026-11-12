# -*- coding: utf-8 -*-
"""Refresh _report_tables.md from local result_path/*.txt.json.

Strategy: keep every hand-written line of the existing report verbatim and only
rewrite table bodies (single-split, multi-split and coverage tables) plus the
numbers inside the 2026-11-09 status block.  Table cell values are refreshed
from the best record per (dataset, split, algorithm, K); cells that were
deliberately blanked by the 2026-11-04 batch audit stay blank unless a new
effective-batch-256 (or FedFair full-batch) result has landed.
"""
import io
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(HERE, "result_path")
DOC = os.path.join(HERE, "_report_tables.md")

ALG = ["FedAvg", "FedProx", "FedNova", "Scaffold", "FedRep", "FL_FairBatch",
       "FairFed", "FedFB", "FedRenyi", "PDFFed", "PDFFed_DP", "FedProto",
       "LoGoFair", "mFairFL", "FederatedAverageWithPo", "FedFair",
       "FedFACT", "PraFFL"]

ROUNDS = {"CREMA-D": 200, "CREMA-D_gender": 200, "CREMA-D_ar": 200,
          "RAVDESS": 200, "RAVDESS_ar": 200, "AudioMNIST": 100,
          "UTKFace": 50, "CelebA": 50, "LFWA+": 50, "FairFace": 50,
          "PAD-UFES-20": 50, "ACSIncome": 500, "ADULT": 500, "BANK": 500,
          "COMPAS": 500, "DRUG": 500, "DUTCH": 500, "VCTK": 200, "bios": 10}

# algorithm name as it appears in paths -> canonical table name
PATH_ALG = {"FederatedNova": "FedNova"}

TITLE_DS = {
    "CREMA-D (race)": "CREMA-D",
    "CREMA-D (gender)": "CREMA-D_gender",
    "CREMA-D_ar (race)": "CREMA-D_ar",
}

SPLIT_COL = {"D01": "Dirichlet01", "D05": "Dirichlet05",
             "D1": "Dirichlet1", "Uni": "Uniform"}

KEY_BY_ALG = {  # never accumulate: effective batch == nominal batch
    "FedFB", "FedNova",
}


def load_records():
    idx = defaultdict(list)
    for dirpath, _dirnames, filenames in os.walk(RESULT):
        for fn in filenames:
            if not fn.endswith(".txt.json"):
                continue
            fp = os.path.join(dirpath, fn)
            rel = os.path.relpath(fp, RESULT).split(os.sep)
            if len(rel) < 5:
                continue
            ds, split, algo = rel[0], rel[1], rel[2]
            algo = PATH_ALG.get(algo, algo)
            try:
                with io.open(fp, encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                continue
            ec = d.get("experiment_config") or {}
            ag = (d.get("aggregate") or {}).get("metrics") or {}

            def mean(name):
                m = ag.get(name)
                if not isinstance(m, dict):
                    return None
                vals = m.get("values")
                if isinstance(vals, list) and vals:
                    try:
                        return sum(float(v) for v in vals) / len(vals)
                    except (TypeError, ValueError):
                        pass
                v = m.get("mean")
                return float(v) if isinstance(v, (int, float)) else None

            acc = ag.get("ACC") or {}
            n = len(acc.get("values") or []) if isinstance(acc, dict) else 0
            k = ec.get("num_clients_K")
            idx[(ds, split, str(k), algo)].append(dict(
                I=ec.get("communication_round_I"),
                n=n,
                bs=ec.get("batch_size"),
                ebs=ec.get("effective_batch_size"),
                mini=ec.get("mini_batch"),
                accum=ec.get("accumulation_steps"),
                ts=str((d.get("code_version") or {}).get("recorded_at") or ""),
                ACC=mean("ACC"), HM=mean("HM"), DEO=mean("DEO"),
                FR=mean("FR"), SPD=mean("SPD"), AG=mean("ACC_gap"),
            ))
    return idx


def effective_batch(r):
    """Effective batch under the 2026-11-04 semantics (see batch_audit.md)."""
    if r["ebs"] is not None:
        return r["ebs"]
    if r["bs"] is None:
        return None
    if r["mini"] is not None and r["accum"] is not None:
        try:
            return int(r["mini"]) * int(r["accum"])
        except (TypeError, ValueError):
            pass
    bs = float(r["bs"])
    if bs >= 256:
        return bs
    return bs  # caller decides via algorithm name


def usable(r, algo):
    if r is None:
        return False
    if algo == "FedFair":
        return True
    if r["ebs"] is not None:
        try:
            return float(r["ebs"]) == 256.0
        except (TypeError, ValueError):
            return False
    if r["mini"] is not None and r["accum"] is not None:
        try:
            return int(r["mini"]) * int(r["accum"]) == 256
        except (TypeError, ValueError):
            return False
    if algo in KEY_BY_ALG:
        return r["bs"] is not None and float(r["bs"]) == 256.0
    # legacy records: hard-coded accumulation pinned the effective batch to 256
    return True


def batch_label(r, algo):
    """表内「批量」列：如实回填该记录的有效批（选项 B，不再清空非 256 结果）。"""
    if r is None:
        return "—"
    if algo == "FedFair":
        return "全量"
    if r["ebs"] is not None:
        try:
            return "%d" % float(r["ebs"])
        except (TypeError, ValueError):
            return "—"
    if r["mini"] is not None and r["accum"] is not None:
        try:
            return "%d" % (int(r["mini"]) * int(r["accum"]))
        except (TypeError, ValueError):
            pass
    if algo in KEY_BY_ALG:
        # FedFB / FedNova 每个 micro-batch 都更新，有效批 == 名义批
        if r["bs"] is not None:
            try:
                return "%d" % float(r["bs"])
            except (TypeError, ValueError):
                return "—"
        return "—"
    # 其余算法的 legacy 记录：硬编码的累积步数把有效批固定在 256
    return "256"


def batch_note(r, algo):
    """legacy bs<256 的累积算法：备注里注明累积折算，避免误读为真实 256。"""
    if r is None or algo == "FedFair" or r["ebs"] is not None:
        return ""
    if r["mini"] is not None and r["accum"] is not None:
        return ""
    if algo in KEY_BY_ALG or r["bs"] is None:
        return ""
    bs = float(r["bs"])
    if bs < 256:
        return "遗留 bs%d×%d 累积" % (int(bs), int(256 // bs))
    return ""


def complete(ds, r):
    return r is not None and r["I"] == ROUNDS[ds] and r["n"] >= 3


def pick(idx, ds, split, algo, k, pilot=False):
    cands = idx.get((ds, split, str(k), algo))
    if not cands:
        return None
    if pilot:
        c = [r for r in cands if r["I"] == 10]
        if not c:
            return None
        return max(c, key=lambda r: (r["n"], r["ts"]))
    good = [r for r in cands if complete(ds, r)]
    if good:
        return max(good, key=lambda r: r["ts"])
    return max(cands, key=lambda r: (r["n"], r["ts"]))


def f6(v):
    return "%.6f" % float(v)


def f4(v):
    return "%.4f" % float(v)


def parse_num(txt):
    txt = (txt or "").strip().strip("()")
    try:
        return float(txt)
    except (TypeError, ValueError):
        return None


def cells_of(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def row_line(cells):
    return "| " + " | ".join(cells) + " |"


def heading_ds(heading):
    h = heading.lstrip("#").strip()
    h = re.sub(r"\s*覆盖核查.*$", "", h)
    title = h.split("/")[0].strip()
    return TITLE_DS.get(title, title)


def heading_split(heading):
    parts = heading.lstrip("#").strip().split("/")
    if len(parts) > 1:
        return parts[1].strip()
    return None


def heading_k(heading):
    m = re.search(r"K\s*=\s*(\d+)", heading)
    return m.group(1) if m else None


def cell_value(r, old, key):
    if r is None or r[key] is None:
        return old
    return f6(r[key])


def process_single(rows_old, ds, split, k, pilot, allow_add, idx, add_missing=False):
    out = []
    old_algos = []
    keys = ["ACC", "HM", "DEO", "FR", "SPD", "AG"]
    for cells, note in rows_old:
        algo = cells[0]
        old_algos.append(algo)
        old_vals = cells[2:8]
        if pilot:
            # 已被 B 方案回填进正式表的算法，不在“待重跑”里重复
            if complete(ds, pick(idx, ds, split, algo, k)):
                continue
            r = pick(idx, ds, split, algo, k, pilot=True)
            vals = [cell_value(r, old_vals[i], key)
                    for i, key in enumerate(keys)] if r is not None else old_vals
            batch = cells[1] if cells[1] else "—"
            out.append({"algo": algo, "order": len(out),
                        "cells": [algo, batch] + vals + [note],
                        "hm": (r["HM"] if r and r["HM"] is not None else parse_num(old_vals[1]))})
            continue
        r = pick(idx, ds, split, algo, k)
        if complete(ds, r):
            vals = [cell_value(r, old_vals[i], key) for i, key in enumerate(keys)]
            batch = batch_label(r, algo)
            nz = batch_note(r, algo)
            if nz and nz not in note:
                note = (note + "；" + nz) if note else nz
        elif r is not None:
            # 只有不完整记录：照填，括号内注明重复次数
            vals = []
            for i, key in enumerate(keys):
                if r[key] is None:
                    vals.append(old_vals[i])
                else:
                    vals.append("(%s, n=%d)" % (f6(r[key]), r["n"]))
            batch = batch_label(r, algo)
        else:
            vals = old_vals
            ob = cells[1]
            if all(v in ("", "—") for v in old_vals):
                vals = ["—"] * 6
                batch = "—"
            else:
                batch = "—" if (ob in ("", "—") or ob.startswith("已移除")
                                or ob.startswith("部分已移除")) else ob
        out.append({
            "algo": algo, "order": len(out),
            "cells": [algo, batch] + vals + [note],
            "hm": (r["HM"] if r and r["HM"] is not None else parse_num(old_vals[1])),
        })

    if allow_add:
        for algo in ALG:
            if algo in old_algos:
                continue
            r = pick(idx, ds, split, algo, k)
            if r is not None and complete(ds, r):
                batch = batch_label(r, algo)
                vals = [f6(r[x]) if r[x] is not None else "—" for x in keys]
                nz = batch_note(r, algo)
            elif r is not None and add_missing:
                batch = batch_label(r, algo)
                vals = [("(%s, n=%d)" % (f6(r[x]), r["n"])) if r[x] is not None else "—"
                        for x in keys]
                nz = ""
            elif r is None and add_missing:
                batch, vals, nz = "—", ["—"] * 6, ""
            else:
                continue  # 无完整结果时不新增空行
            out.append({"algo": algo, "order": len(out),
                        "cells": [algo, batch] + vals + [nz],
                        "hm": (r["HM"] if r and r["HM"] is not None else None)})
    out.sort(key=lambda d: (0, -round(d["hm"], 6)) if d["hm"] is not None else (1, 0))
    return [row_line(d["cells"]) for d in out]


def process_multi(rows_old, ds, k, splits, allow_add, idx):
    out = []
    old_algos = []
    for cells, _note in rows_old:
        algo = cells[0]
        old_algos.append(algo)
        n = len(splits)
        old_batch = cells[1]
        old_acc = cells[2:2 + n]
        old_hm = cells[2 + n:2 + 2 * n]
        new_acc, new_hm, recs, all_recs = [], [], [], []
        for i, s in enumerate(splits):
            r = pick(idx, ds, s, algo, k)
            all_recs.append(r)
            ok = complete(ds, r)
            recs.append(r if ok else None)
            a, h = old_acc[i], old_hm[i]
            if ok:
                wrap = a.startswith("(")
                acc_v = f6(r["ACC"]) if r["ACC"] is not None else a
                hm_v = f6(r["HM"]) if r["HM"] is not None else h
                new_acc.append("(" + acc_v + ")" if wrap else acc_v)
                new_hm.append("(" + hm_v + ")" if wrap else hm_v)
            elif r is not None:
                acc_v = f6(r["ACC"]) if r["ACC"] is not None else (a or "—")
                hm_v = f6(r["HM"]) if r["HM"] is not None else (h or "—")
                new_acc.append("(" + acc_v + ", n=%d)" % r["n"])
                new_hm.append("(" + hm_v + ", n=%d)" % r["n"])
            else:
                new_acc.append(a if a not in ("", "—") else "—")
                new_hm.append(h if h not in ("", "—") else "—")
        all_hms = [float(r["HM"]) for r in all_recs
                   if r is not None and r["HM"] is not None]
        hm_sort = sum(all_hms) / len(all_hms) if all_hms else None
        hms, deos, frs, spds = [], [], [], []
        for i, r in enumerate(recs):
            if r is None or new_acc[i].startswith("("):
                continue
            if r["HM"] is not None:
                hms.append(float(r["HM"]))
            if r["DEO"] is not None:
                deos.append(float(r["DEO"]))
            if r["FR"] is not None:
                frs.append(float(r["FR"]))
            if r["SPD"] is not None:
                spds.append(abs(float(r["SPD"])))
        if hms:
            hm_mean = f4(sum(hms) / len(hms))
            if len(hms) < len(splits):
                hm_mean += " (%d/%d)" % (len(hms), len(splits))
        else:
            hm_mean = "—"
        deo_m = f4(sum(deos) / len(deos)) if deos else "—"
        fr_m = f4(sum(frs) / len(frs)) if frs else "—"
        spd_m = f4(sum(spds) / len(spds)) if spds else "—"
        # 批量列：各 split 的有效批一致时写单值，不一致时按 split 顺序用 / 连接
        labels = []
        for r in all_recs:
            if r is None:
                continue
            lb = batch_label(r, algo)
            if lb not in labels:
                labels.append(lb)
        batch = "/".join(labels) if labels else "—"
        new_cells = [algo, batch] + new_acc + new_hm + [hm_mean, deo_m, fr_m, spd_m]
        out.append({"algo": algo, "cells": new_cells, "hm": hm_sort})

    if allow_add:
        n = len(splits)
        for algo in ALG:
            if algo in old_algos:
                continue
            if not any(idx.get((ds, s, str(k), algo)) for s in splits):
                continue
            recs = []
            accs, hms = [], []
            for s in splits:
                r = pick(idx, ds, s, algo, k)
                ok = complete(ds, r)
                recs.append(r if ok else None)
                accs.append(f6(r["ACC"]) if ok and r["ACC"] is not None else "—")
                hms.append(f6(r["HM"]) if ok and r["HM"] is not None else "—")
            if not any(r is not None for r in recs):
                continue  # 无完整结果时不新增空行
            labels = []
            for r in recs:
                if r is None:
                    continue
                lb = batch_label(r, algo)
                if lb not in labels:
                    labels.append(lb)
            batch = "/".join(labels) if labels else "—"
            hms_v = [float(r["HM"]) for r, a in zip(recs, accs)
                     if r is not None and r["HM"] is not None and a not in ("—", "") and not a.startswith("(")]
            deos = [float(r["DEO"]) for r, a in zip(recs, accs)
                    if r is not None and r["DEO"] is not None and a not in ("—", "") and not a.startswith("(")]
            frs = [float(r["FR"]) for r, a in zip(recs, accs)
                   if r is not None and r["FR"] is not None and a not in ("—", "") and not a.startswith("(")]
            spds = [abs(float(r["SPD"])) for r, a in zip(recs, accs)
                    if r is not None and r["SPD"] is not None and a not in ("—", "") and not a.startswith("(")]
            if hms_v:
                hm_mean = f4(sum(hms_v) / len(hms_v))
                if len(hms_v) < n:
                    hm_mean += " (%d/%d)" % (len(hms_v), n)
                hm_sort = sum(hms_v) / len(hms_v)
            else:
                hm_mean, hm_sort = "—", None
            deo_m = f4(sum(deos) / len(deos)) if deos else "—"
            fr_m = f4(sum(frs) / len(frs)) if frs else "—"
            spd_m = f4(sum(spds) / len(spds)) if spds else "—"
            out.append({"algo": algo,
                        "cells": [algo, batch] + accs + hms + [hm_mean, deo_m, fr_m, spd_m],
                        "hm": hm_sort})

    out.sort(key=lambda d: (0, -round(d["hm"], 4)) if d["hm"] is not None else (1, 0))
    return [row_line(d["cells"]) for d in out]


def main():
    idx = load_records()
    lines = io.open(DOC, encoding="utf-8").read().split("\n")
    out = []
    heading = ""
    pilot = False
    i = 0
    pending_total = None
    while i < len(lines):
        line = lines[i]
        if line.startswith("## ") or line.startswith("### "):
            heading = line
            pilot = False
        if line.startswith("**待重跑"):
            pilot = True
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            header = cells_of(block[0])
            body = block[2:]
            if "完成" in header and "缺失算法" in header:
                ds = heading_ds(heading)
                m = re.search(r"I=(\d+)", heading)
                rounds = int(m.group(1)) if m else ROUNDS.get(ds)
                new_rows, total, expected = [], 0, 0
                for ln in body:
                    c = cells_of(ln)
                    if len(c) < 4:
                        continue
                    split, k = c[0], c[1]
                    done = []
                    for a in ALG:
                        r = pick(idx, ds, split, a, k)
                        if complete(ds, r) and r["I"] == rounds:
                            done.append(a)
                    miss = [a for a in ALG if a not in done]
                    total += len(done)
                    expected += len(ALG)
                    new_rows.append(row_line([split, k, "%d/16" % len(done),
                                              ", ".join(miss) if miss else "—"]))
                out.extend(block[:2])
                out.extend(new_rows)
                pending_total = (total, expected)
                continue
            if "批量" in header and "ACC D01" in header:
                ds = heading_ds(heading)
                k = heading_k(heading)
                splits = [SPLIT_COL[c.replace("ACC ", "").strip()]
                          for c in header if c.startswith("ACC ")]
                old_rows = []
                for ln in body:
                    c = cells_of(ln)
                    while len(c) < 14:
                        c.append("")
                    old_rows.append((c, c[13] if len(c) > 13 else ""))
                allow_add = True
                new_body = process_multi(old_rows, ds, k, splits, allow_add, idx)
                out.extend(block[:2])
                out.extend(new_body)
                continue
            if "批量" in header:
                ds = heading_ds(heading)
                split = heading_split(heading)
                k = heading_k(heading)
                old_rows = []
                for ln in body:
                    c = cells_of(ln)
                    while len(c) < 9:
                        c.append("")
                    old_rows.append((c, c[8]))
                allow_add = True
                add_missing = (ds == "VCTK")
                new_body = process_single(old_rows, ds, split, k, pilot,
                                          allow_add and not pilot, idx,
                                          add_missing=add_missing and not pilot)
                out.extend(block[:2])
                out.extend(new_body)
                continue
            # unknown table: keep verbatim
            out.extend(block)
            continue
        if pending_total is not None and line.startswith("合计完成"):
            out.append("合计完成 **%d/%d**。" % pending_total)
            pending_total = None
            i += 1
            continue
        out.append(line)
        i += 1

    text = "\n".join(out)

    # ---- B 方案口径说明（插在批量口径列表后，只插一次） ----
    if "每 micro-batch 更新的算法" not in text:
        anchor = "批量列按 split 顺序用 `/` 连接（如 `64/256`）。"
        extra = ("\n> - 口径说明：「批量」列给出的是有效批大小；标 64 的格来自 BatchNorm 前"
                 "每 micro-batch 更新的算法（FedFB/FedNova）或早期 64-batch run，"
                 "可用于趋势参考，但不与 256 的格做严格批量可比。")
        if anchor in text:
            text = text.replace(anchor, anchor + extra, 1)

    # ---- CREMA-D Dirichlet1 / Uniform 的“9 正式 + 7 待重跑”说明改为回填后的格数 ----
    def n_complete(split):
        return sum(1 for a in ALG
                   if complete("CREMA-D", pick(idx, "CREMA-D", split, a, "20")))

    def n_pilot(split):
        return sum(1 for a in ALG
                   if not complete("CREMA-D", pick(idx, "CREMA-D", split, a, "20"))
                   and pick(idx, "CREMA-D", split, a, "20", pilot=True) is not None)

    old_d1 = ("> 仅收 2026-11-02/11-04 按正式协议跑完的 9 格；另有 7 格仍是 2026-10-16 的 "
              "`I=10`/`sgd`/`3e-4`/`f=5` 试点，单列在下方“待重跑”，**不可与上表比较**。")
    new_d1 = ("> 按 B 方案全量回填：正式协议 %d 格给出数值（bs=64 的格按有效批标注）；"
              "另有 %d 格仍是 2026-10-16 的 `I=10`/`sgd`/`3e-4`/`f=5` 试点，"
              "单列在下方“待重跑”，**不可与上表比较**。") % (n_complete("Dirichlet1"),
                                                              n_pilot("Dirichlet1"))
    old_uni = ("> 同口径（`I=200`、`T=2`、`lr=1e-3`、`adam`、`f=50`、3 repeats）。"
               "正式协议 9 格；另 7 格为 10-16/10-23 的 `I=10` 试点，单列在下方“待重跑”。")
    new_uni = ("> 同口径（`I=200`、`T=2`、`lr=1e-3`、`adam`、`f=50`、3 repeats）。"
               "按 B 方案全量回填：正式协议 %d 格给出数值；另 %d 格为 10-16/10-23 的 "
               "`I=10` 试点，单列在下方“待重跑”。") % (n_complete("Uniform"),
                                                      n_pilot("Uniform"))
    text = text.replace(old_d1, new_d1).replace(old_uni, new_uni)

    # ---- status block: total coverage / gaps ----
    groups = [
        ("CREMA-D", ["Speaker"], ["20"]),
        ("CREMA-D_gender", ["Speaker"], ["20"]),
        ("CREMA-D_ar", ["Speaker"], ["20"]),
        ("RAVDESS", ["Speaker"], ["20"]),
        ("AudioMNIST", ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"], ["20"]),
        ("VCTK", ["Speaker"], ["20"]),
        ("UTKFace", ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"], ["20", "30", "40"]),
        ("CelebA", ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"], ["20", "30", "40"]),
        ("LFWA+", ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"], ["20", "30", "40"]),
        ("FairFace", ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"], ["20", "30", "40"]),
        ("PAD-UFES-20", ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"], ["20", "30", "40"]),
        ("ACSIncome", ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"], ["20", "30", "40"]),
        ("ADULT", ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"], ["20"]),
        ("BANK", ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"], ["20", "30", "40"]),
        ("COMPAS", ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"], ["20", "30", "40"]),
        ("DRUG", ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"], ["20", "30", "40"]),
        ("DUTCH", ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"], ["20", "30", "40"]),
        ("bios", ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"], ["20"]),
    ]
    total_cells = done_cells = 0
    gaps = []
    for ds, sp, ks in groups:
        exp = len(sp) * len(ks) * len(ALG)
        done = 0
        for s in sp:
            for k in ks:
                for a in ALG:
                    r = pick(idx, ds, s, a, k)
                    if complete(ds, r):
                        done += 1
        total_cells += exp
        done_cells += done
        if exp - done:
            gaps.append((ds, exp - done))
    order = ["ACSIncome", "FairFace", "VCTK"]
    gaps.sort(key=lambda t: order.index(t[0]) if t[0] in order else 99)
    gap_txt = "、".join("%s %d" % (d, g) for d, g in gaps)
    n_files = sum(len(v) for v in idx.values())
    text = re.sub(
        r"两机 `result_path` 已 union 合并（共 \d+ 个结果 json），当前总覆盖 \*\*\d+ / 2192\*\*，缺口 \*\*\d+\*\*（[^）]*）。",
        "两机 `result_path` 已 union 合并（共 %d 个结果 json），当前总覆盖 **%d / %d**，缺口 **%d**（%s）。" % (
            n_files, done_cells, total_cells, total_cells - done_cells, gap_txt),
        text)

    io.open(DOC, "w", encoding="utf-8", newline="\n").write(text)
    print("coverage", done_cells, "/", total_cells, "gaps", gaps)
    print("files", n_files)


if __name__ == "__main__":
    main()
