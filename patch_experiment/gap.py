"""局部—全局落差（aggregation gap）采集。

用途：H3 要回答"局部训练成果能不能变成全局增益"，只看终点 ACC/DEO 是看不出来的，
必须量出每一轮"参与方模型在本轮的 DEO 均值"与"聚合后全局模型的 DEO"之间的差：

    AGG_GAP = DEO_local_mean - DEO_global

- AGG_GAP > 0：聚合把公平性拉好（局部偏差被平均掉）
- AGG_GAP < 0：聚合反而变差（局部个别好模型被拖累）

口径（2026-11-10 修正）：默认**全部参与方 × 全测试集**。原先是 3 家 × 8 batch，
实测在 COMPAS 上 |AGG_GAP| 仅 0.0017 而轮间 sd 0.0041，噪声是信号的 2.4 倍，
且固定取 0/6/12 号客户端在 Dirichlet 划分下系统性只反映同 3 家，故废弃。
DEO 是比值定义，与样本总数无关；ACC 在子集上会有偏差，这里只取 DEO。

max_clients / max_batches 仍保留为可选降级开关（None = 全量）。
抽样降级时用固定 seed，保证同一格可复现。
"""

import copy
import itertools
import random

# 降级抽样时使用的固定随机种子（全量口径下不生效）
_SAMPLE_SEED = 20261110


def local_deo_mean(client_states, weights, global_model, param_dict,
                   testing_dataloader, max_clients=None, max_batches=None):
    """返回参与方模型在本轮测试集上的加权平均 DEO。

    client_states: 本轮参与方上传的 state_dict 列表（与 weights 一一对应）
    max_clients:   None = 用全部参与方；否则随机抽这么多个
    max_batches:   None = 用全测试集；否则只用前这么多个 batch
    """
    if not client_states:
        return None
    from tool.utils import FL_fairness_and_accuracy_test_4_Tabular_CLF

    device = next(global_model.parameters()).device
    batch_size = int(param_dict.get("test_batch_size", 256) or 256)

    total = len(client_states)
    if max_clients is None or max_clients >= total:
        picked = list(range(total))
    else:
        picked = sorted(random.Random(_SAMPLE_SEED).sample(range(total), int(max_clients)))

    if max_batches is None:
        try:
            eval_size = len(testing_dataloader.dataset)
        except AttributeError:
            eval_size = len(testing_dataloader) * batch_size
    else:
        eval_size = int(max_batches) * batch_size

    deos, ws = [], []
    # 只深拷贝一次，之后逐个 load_state_dict 覆盖；比每客户端 copy 一次省一半以上开销
    m = copy.deepcopy(global_model)
    m.eval()
    for i in picked:
        try:
            m.load_state_dict(client_states[i], strict=True)
        except Exception:
            continue
        # islice 是一次性迭代器，必须每次重建；全量时直接复用 DataLoader（每次 for 新建迭代器）
        if max_batches is None:
            loader = testing_dataloader
        else:
            loader = itertools.islice(testing_dataloader, int(max_batches))
        try:
            ev = FL_fairness_and_accuracy_test_4_Tabular_CLF(
                m, param_dict, loader, eval_size)
        except Exception:
            continue
        deos.append(float(ev[1]))
        ws.append(float(weights[i]))
    del m

    if not deos or sum(ws) == 0:
        return None
    return sum(d * w for d, w in zip(deos, ws)) / sum(ws)
