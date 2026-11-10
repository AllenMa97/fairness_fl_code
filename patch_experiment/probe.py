"""Phase A 机制埋点（非侵入式）。

目的：主表只有终点 ACC/DEO/HM，无法回答"为什么"。本模块只做**记录**，
不参与任何训练逻辑，默认关闭；只有环境变量 PDFFED_PROBE=1 时才生效。

采集内容
--------
1. Δ_rep（表征偏差）：同类异组特征质心距离
   Δ_rep = 1/2 * ( ||μ(s=0,y=0) - μ(s=1,y=0)|| + ||μ(s=0,y=1) - μ(s=1,y=1)|| )
   特征取分类器倒数第二层输出（ANNCLASSIFIER.forward 的第二个返回值），
   因此**与算法无关**：PDFFed / FedRenyi / FedAvg / LoGoFair 都能算同一口径，
   这是 H1/H2 可比较的前提。
2. 逐次评测轨迹（JSONL）：ACC/DEO/SPD/Δ_rep，用于画 DEO(t)、Δ_rep(t)、
   以及"公平达成代价"（首次达到 DEO≤0.05 / 0.10 的评测序号）。

落盘位置：<model_path>/probe_trace.jsonl（每个 repeat 一份，追加写）。

口径说明
--------
- 评测函数被算法按自己的节奏调用（PDFFed 每轮、多数算法每 test_eval_freq 轮），
  故轨迹里的序号是"评测序号"而不是"通信轮次"。要与日志里的 Communication Round
  对齐时，按 DEO 序列匹配即可（同一次评测产生的 DEO 在两个地方数值相同）。
- 归一化 Δ_rep（REP_GAP_NORM）= Δ_rep / 两组质心范数均值，用于跨数据集/跨模态比较，
  避免特征尺度不同导致不可比（这正是之前误判"维度"问题的地方）。
"""

import json
import os
import time

ENABLE_ENV = "PDFFED_PROBE"
TRACE_NAME = "probe_trace.jsonl"

# 最近一次评测的机制指标，供 experiment._evaluate_global_model 合并进结果 json。
# 之所以用模块级缓存，是为了避免在最终评测时再额外跑一遍前向。
_LAST = {}

# 各轮聚合落差的累计值，用于在结果 json 里给出均值
_GAPS = []

# 采集"聚合落差"时需要临时评测若干参与方模型，这些评测不应写进正式轨迹，
# 也不应重复累加 Δ_rep；置 True 期间 utils 里的埋点会跳过写盘。
SILENT = False


def silent():
    return SILENT


def enabled():
    """PDFFED_PROBE=1 时开启；其它值（含未设置/0/false）一律关闭。"""
    return os.environ.get(ENABLE_ENV, "").strip().lower() not in ("", "0", "false", "no")


class RepAccumulator:
    """在评测前向过程中累加四组（类别×敏感组）特征的和与计数。

    只存 4 个向量和 4 个整数，额外显存/内存开销可忽略；
    测试集再大也只多 4×d 个浮点数。
    """

    def __init__(self):
        self._sum = {}
        self._cnt = {}
        self._dim = 0

    def add(self, feature, labels, protected):
        """feature: [B, d]；labels/protected: [B]（0/1 编码）。"""
        if feature is None:
            return
        feat = feature.detach()
        if feat.dim() != 2:
            feat = feat.reshape(feat.shape[0], -1)
        self._dim = int(feat.shape[1])
        y = labels.reshape(-1).long()
        s = protected.reshape(-1).long()
        for yv in (0, 1):
            for sv in (0, 1):
                mask = (y == yv) & (s == sv)
                n = int(mask.sum().item())
                if n == 0:
                    continue
                vec = feat[mask].sum(dim=0).float().cpu()
                key = (yv, sv)
                if key in self._sum:
                    self._sum[key] += vec
                    self._cnt[key] += n
                else:
                    self._sum[key] = vec
                    self._cnt[key] = n

    def _centroid(self, y, s):
        key = (y, s)
        if key not in self._sum or self._cnt[key] == 0:
            return None
        return self._sum[key] / float(self._cnt[key])

    def result(self):
        """返回 Δ_rep 相关指标；任一组缺失时返回空 dict（避免把 0 当有效值）。"""
        out = {}
        gaps = {}
        norms = []
        for y in (0, 1):
            c0, c1 = self._centroid(y, 0), self._centroid(y, 1)
            if c0 is None or c1 is None:
                continue
            gap = float((c0 - c1).norm().item())
            gaps[y] = gap
            norms.extend([float(c0.norm().item()), float(c1.norm().item())])
        if not gaps:
            return out
        out["REP_GAP"] = sum(gaps.values()) / len(gaps)
        for y, gap in gaps.items():
            out["REP_GAP_Y%d" % y] = gap
        if norms:
            denom = sum(norms) / len(norms)
            if denom > 0:
                out["REP_GAP_NORM"] = out["REP_GAP"] / denom
        out["FEAT_DIM"] = float(self._dim)
        # 各组样本数，用于判断小样本组是否让 Δ_rep 失真
        for (y, s), n in self._cnt.items():
            out["REP_N_Y%dS%d" % (y, s)] = float(n)
        return out


def record(param_dict, record_dict):
    """把一次评测的机制指标写入 trace，并更新供 json 合并的缓存。"""
    if not enabled():
        return
    _LAST.clear()
    _LAST.update(record_dict)
    # 聚合落差只在"聚合后、后训练前"那一次评测里有；单独累计，
    # 因为最终 json 取的是最后一次评测（后训练之后），那里没有 AGG_GAP。
    if "AGG_GAP" in record_dict:
        _GAPS.append(float(record_dict["AGG_GAP"]))
    try:
        base = param_dict.get("model_path")
        if not base:
            return
        os.makedirs(base, exist_ok=True)
        payload = {"ts": time.time()}
        payload.update({k: (float(v) if isinstance(v, (int, float)) else v)
                        for k, v in record_dict.items()})
        with open(os.path.join(base, TRACE_NAME), "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        # 埋点失败绝不能影响训练
        pass


def metrics_for_json():
    """最终评测时合并进结果 json 的键值（会被 aggregate_repeat_results 汇总 mean/std）。"""
    if not enabled():
        return {}
    out = {k: float(v) for k, v in _LAST.items() if isinstance(v, (int, float))}
    if _GAPS:
        out["AGG_GAP_MEAN"] = sum(_GAPS) / len(_GAPS)
        out["AGG_GAP_ABS_MEAN"] = sum(abs(g) for g in _GAPS) / len(_GAPS)
    return out
