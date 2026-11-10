"""H1-H4 机制命题的离线分析。

从 Phase A 产物计算四个命题的判据：
  H1  Δ_rep 随 λ_pa 单调下降，且 Δ_rep 与 DEO 在控 ACC 后正相关（p<0.05）
  H2  同 DEO（±0.01）下 PDFFed 的 Δ_rep 低于 FedRenyi
  H3  达到 DEO≤0.05 的轮次/累计通信量；聚合落差 gap(t) 是否收敛
  H4  Δ_calib = DEO(pre-post) - DEO(post-post)，>0.01 且跨格同号

输入目录需包含 Phase A 的两棵树：
  <root>/result_path/<ds>/<split>/<algo>/<model>/<K>Clients/*.json
  <root>/save_path/<ds>/<split>/<algo>/<model>/<K>Clients/probe_trace.jsonl

用法：
  python scripts/analyze_h1_h4.py <root> [--datasets COMPAS,DUTCH] [--model ANN] [--k 20]
"""

import argparse
import glob
import json
import math
import os

import numpy as np
from scipy import stats

TAU = 0.05
LAMBDA_TAGS = {"PDFFed_noPA": 0.0, "PDFFed_wpa0p01": 0.01,
               "PDFFed_wpa0p3": 0.3, "PDFFed": 1.0}


class Artifacts:
    def __init__(self, root, model, k):
        self.root = root
        self.model = model
        self.k = k
        self.cells = self._load_cells()

    def _load_cells(self):
        cells = {}
        pattern = os.path.join(self.root, "result_path", "*", "*", "*", "*", "*Clients", "*.json")
        for p in glob.glob(pattern):
            parts = p.replace("\\", "/").split("/")
            ds, split, algo, model = parts[-6], parts[-5], parts[-4], parts[-3]
            if model != self.model:
                continue
            try:
                d = json.load(open(p, encoding="utf-8"))
                m = d["aggregate"]["metrics"]
                cfg = d.get("experiment_config", {})
            except Exception:
                continue
            cells[(ds, split, algo)] = {
                "ACC": m.get("ACC", {}).get("mean"),
                "DEO": m.get("DEO", {}).get("mean"),
                "REP": m.get("REP_GAP", {}).get("mean"),
                "AGGM": m.get("AGG_GAP_MEAN", {}).get("mean"),
                "comm": (d["aggregate"].get("communication_cost") or [None])[0],
                "ef": cfg.get("test_eval_freq", 10) or 10,
                "rounds": cfg.get("communication_round_I", 0),
            }
        # json 没合并埋点（老结果）时，用轨迹最后一行兜底
        for (ds, split, algo), c in cells.items():
            if c["REP"] is None:
                tr = self.trace(ds, split, algo)
                if tr:
                    c["REP"] = tr[-1].get("REP_GAP")
        return cells

    def trace(self, ds, split, algo):
        p = os.path.join(self.root, "save_path", ds, split, algo, self.model,
                         "%dClients" % self.k, "probe_trace.jsonl")
        if not os.path.exists(p):
            return None
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        return rows or None


def split_trace(rows):
    """PDFFed 每轮一次 pre 评测（含 AGG_GAP），每 ef 轮一次 post 评测。

    返回 (pre_by_round, post_by_round, is_pdffed)；其它算法只有一条轨迹。
    """
    if not any("AGG_GAP" in r for r in rows):
        return {}, {i: r for i, r in enumerate(rows, 1)}, False
    pre, post = {}, {}
    rnd = 0
    for r in rows:
        if "AGG_GAP" in r:
            rnd += 1
            pre[rnd] = r
        else:
            post[rnd] = r
    return pre, post, True


def t_cross(series, tau=TAU):
    """首次/稳定越过阈值的轮次。

    first  = 首次 DEO≤τ（早期近随机模型也可能触发）
    stable = 最后一次 DEO>τ 的轮次，此后一直 ≤τ；最终仍 >τ 则为 None（删失）
    """
    if not series:
        return None, None
    first, last_bad = None, 0
    for r, deo in series:
        if deo is None:
            continue
        if deo <= tau:
            first = r if first is None else first
        else:
            last_bad = r
    final = series[-1][1]
    stable = None if (final is None or final > tau) else (first if last_bad == 0 else last_bad)
    return first, stable


def partial_corr(cells, keys, zscore=False):
    xs, ys, zs, used = [], [], [], []
    for k in keys:
        c = cells.get(k) or {}
        if c.get("REP") is None or c.get("DEO") is None or c.get("ACC") is None:
            continue
        xs.append(c["REP"]); ys.append(c["DEO"]); zs.append(c["ACC"]); used.append(k)
    if len(xs) < 5:
        return None
    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
    if zscore:
        groups = {}
        for i, k in enumerate(used):
            groups.setdefault(k[0], []).append(i)
        for idx in groups.values():
            for arr in (xs, ys, zs):
                seg = arr[idx]
                sd = seg.std(ddof=0)
                if sd > 0:
                    arr[idx] = (seg - seg.mean()) / sd
    X = np.column_stack([np.ones(len(xs)), xs, zs])
    beta, *_ = np.linalg.lstsq(X, ys, rcond=None)
    resid = ys - X @ beta
    dof = len(ys) - X.shape[1]
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = math.sqrt(max(cov[1, 1], 1e-30))
    t = beta[1] / se
    return dict(n=len(ys), coef=float(beta[1]), t=float(t),
                p=float(2 * (1 - stats.t.cdf(abs(t), dof))), dof=dof)


def h1(art, datasets):
    print("\n=== H1 Δ_rep 随 λ_pa 单调下降 ===")
    for ds in datasets:
        for split in ("Dirichlet01", "Uniform"):
            row = []
            for algo, lam in LAMBDA_TAGS.items():
                c = art.cells.get((ds, split, algo)) or {}
                if c.get("REP") is not None:
                    row.append((lam, c["REP"], c["DEO"]))
            print("  %-7s %-12s %s" % (ds, split, "  ".join(
                "λ=%.2f REP=%.4f DEO=%.4f" % r for r in row)))
            if len(row) >= 3:
                rho = stats.spearmanr([r[0] for r in row], [r[1] for r in row])
                print("     monotone spearman rho=%.3f p=%.4f" % (rho.statistic, rho.pvalue))
    keys = [(ds, s, a) for ds in datasets for s in ("Dirichlet01", "Uniform")
            for a in LAMBDA_TAGS]
    print("  partial corr (control ACC):", partial_corr(art.cells, keys))
    print("  partial corr (z-score within dataset):", partial_corr(art.cells, keys, zscore=True))


def h2(art, datasets):
    print("\n=== H2 同 DEO(±0.01) 下 Δ_rep vs FedRenyi ===")
    for ds in datasets:
        for split in ("Dirichlet01", "Uniform"):
            q = art.cells.get((ds, split, "FedRenyi")) or {}
            if q.get("DEO") is None or q.get("REP") is None:
                print("  %-7s %-12s FedRenyi 缺 DEO/REP" % (ds, split))
                continue
            print("  %-7s %-12s FedRenyi DEO=%.4f REP=%.4f" % (
                ds, split, q["DEO"], q["REP"]))
            for algo, lam in LAMBDA_TAGS.items():
                c = art.cells.get((ds, split, algo)) or {}
                if c.get("DEO") is None or c.get("REP") is None:
                    continue
                dd = c["DEO"] - q["DEO"]
                print("     %s λ=%.2f DEO=%.4f (Δ=%+.4f) REP=%.4f (Δ=%+.4f)" % (
                    "MATCH" if abs(dd) <= 0.01 else "     ", lam,
                    c["DEO"], dd, c["REP"], c["REP"] - q["REP"]))


def h3(art, datasets):
    print("\n=== H3 达成代价 t0.05 / 通信量 ===")
    for ds in datasets:
        for split in ("Dirichlet01", "Uniform"):
            rows = []
            for (d, s, algo), c in art.cells.items():
                if d != ds or s != split:
                    continue
                tr = art.trace(ds, split, algo)
                if not tr:
                    continue
                pre, post, is_pdf = split_trace(tr)
                ef = c["ef"]
                if is_pdf:
                    series = [(r, v["DEO"]) for r, v in sorted(post.items())]
                else:
                    series = [(i * ef, v["DEO"]) for i, (_, v) in enumerate(sorted(post.items()), 1)]
                first, stable = t_cross(series)
                comm = None
                if stable and c["comm"] and c["rounds"]:
                    comm = c["comm"] * stable / float(c["rounds"])
                rows.append((stable if stable else 10 ** 9, algo, first, stable,
                             c["DEO"], c["ACC"], comm))
            rows.sort()
            print("  -- %s / %s" % (ds, split))
            for _, algo, first, stable, deo, acc, comm in rows:
                print("     %-24s first=%-5s stable=%-5s DEO=%.4f ACC=%.4f comm@t=%s" % (
                    algo, first, stable, deo, acc if acc is not None else -1,
                    "%.1fMB" % comm if comm else "NA"))

    print("\n=== H3b 聚合落差 |gap| 收敛（前100轮 vs 后100轮）===")
    for ds in datasets:
        for split in ("Dirichlet01", "Uniform"):
            for algo in ("PDFFed", "PDFFed_noPA", "PDFFed_wpa0p3"):
                tr = art.trace(ds, split, algo)
                if not tr:
                    continue
                pre, _, is_pdf = split_trace(tr)
                if not is_pdf or len(pre) < 200:
                    continue
                g = [abs(pre[r]["AGG_GAP"]) for r in sorted(pre)]
                print("  %-7s %-12s %-16s first100=%.5f last100=%.5f" % (
                    ds, split, algo, float(np.mean(g[:100])), float(np.mean(g[-100:]))))


def h4(art, datasets):
    print("\n=== H4 Δ_calib = DEO(pre-post) - DEO(post-post) ===")
    for ds in datasets:
        for split in ("Dirichlet01", "Uniform"):
            for algo in ("PDFFed", "PDFFed_noEO", "PDFFed_wpa0p3", "PDFFed_noPA"):
                tr = art.trace(ds, split, algo)
                if not tr:
                    continue
                pre, post, is_pdf = split_trace(tr)
                if not is_pdf:
                    continue
                deltas = [pre[r]["DEO"] - post[r]["DEO"] for r in sorted(post) if r in pre]
                if not deltas:
                    continue
                arr = np.array(deltas)
                print("  %-7s %-12s %-16s mean=%+.5f sd=%.5f n=%d positive=%.0f%%" % (
                    ds, split, algo, arr.mean(), arr.std(ddof=1), len(arr),
                    float((arr > 0).mean()) * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="包含 result_path/ 与 save_path/ 的目录")
    ap.add_argument("--datasets", default="COMPAS,DUTCH")
    ap.add_argument("--model", default="ANN")
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    art = Artifacts(args.root, args.model, args.k)
    print("cells:", len(art.cells))
    h1(art, datasets)
    h2(art, datasets)
    h3(art, datasets)
    h4(art, datasets)


if __name__ == "__main__":
    main()
