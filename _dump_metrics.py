import json, os, glob

OUT = "/root/fl/_metrics_dump.tsv"
DSS = ["CREMA-D", "CREMA-D_gender", "CREMA-D_ar",
       "UTKFace", "CelebA", "LFWA+", "FairFace",
       "RAVDESS", "RAVDESS_ar", "AudioMNIST",
       # 表格类
       "ACSIncome", "ADULT", "BANK", "COMPAS", "DRUG", "DUTCH",
       # 文本类
       "bios", "moji"]
MAIN = {"FedAvg", "FedProx", "FedNova", "FederatedNova", "Scaffold", "FedRep",
        "FL_FairBatch", "FairFed", "FedFB", "FedRenyi", "FederatedRenyi",
        "PDFFed", "PDFFed_DP", "FedProto", "LoGoFair", "mFairFL",
        "Simple_mFairFL", "FederatedAverageWithPo", "FedFair"}
ALIAS = {"FederatedNova": "FedNova", "FederatedRenyi": "FedRenyi", "Simple_mFairFL": "mFairFL"}
MARGIN = 0.01   # 塌缩判据：ACC <= floor + 0.01（见 appendix_collapse_disclosure.md §2）

# ---------------------------------------------------------------- floor 表
# 每格真实测试集的多数类占比 = 该分区 metadata.test_stats 的 label_counts 汇总后最大值占比。
# 与 partition_cache 指纹一一对应，故 (dataset, fingerprint) 即可定位。
FLOOR = {}
for ds in DSS:
    for mp in glob.glob("/root/fl/partition_cache/v2/%s/*/metadata.json" % ds):
        try:
            md = json.load(open(mp))
        except Exception:
            continue
        fp = md.get("fingerprint") or mp.split("/")[-2]
        cnt = {}
        for v in (md.get("test_stats") or {}).values():
            if isinstance(v, dict):
                for lab, n in (v.get("label_counts") or {}).items():
                    cnt[lab] = cnt.get(lab, 0) + n
        tot = sum(cnt.values())
        if tot:
            FLOOR[(ds, fp)] = max(cnt.values()) / tot

rows = []
for ds in DSS:
    base = "/root/fl/result_path/" + ds
    if not os.path.isdir(base):
        continue
    for h in glob.glob(os.path.join(base, "**", "*.txt.json"), recursive=True):
        rel = h.split("result_path/%s/" % ds)[-1].split("/")
        if len(rel) < 5:
            continue
        split, algo, kdir = rel[0], ALIAS.get(rel[1], rel[1]), rel[3]
        if algo not in MAIN:
            continue
        try:
            d = json.load(open(h))
        except Exception:
            continue
        c = d.get("experiment_config", {})
        a = d.get("aggregate", {})
        m = a.get("metrics", {})

        def g(k):
            v = m.get(k, {}).get("mean")
            return "" if v is None else "%.6f" % v

        vals = (m.get("ACC", {}) or {}).get("values") or []
        fps = a.get("partition_fingerprints") or []
        floor = FLOOR.get((ds, fps[0])) if fps else None
        acc = m.get("ACC", {}).get("mean")
        if floor is None or acc is None:
            fl = exc = coll = ""
        else:
            fl = "%.6f" % floor
            exc = "%+.2f" % ((acc - floor) * 100.0)
            coll = "1" if acc <= floor + MARGIN else "0"
        rows.append([ds, split, algo, kdir.replace("Clients", ""),
                     str(c.get("communication_round_I")), str(len(vals)),
                     g("ACC"), g("ACC_gap"), g("DEO"), g("FR"), g("HM"), g("SPD"),
                     str(c.get("device")), fl, exc, coll,
                     "" if m.get("ACC", {}).get("std") is None else "%.6f" % m["ACC"]["std"]])

HDR = ("dataset\tsplit\talgo\tK\tI\tn\tACC\tACC_gap\tDEO\tFR\tHM\tSPD\tdevice"
       "\tfloor\texcess_pp\tcollapse\tACC_std")
with open(OUT, "w", encoding="utf-8") as f:
    f.write(HDR + "\n")
    for r in rows:
        f.write("\t".join(r) + "\n")
print("wrote", OUT, len(rows), "floor_entries", len(FLOOR))
