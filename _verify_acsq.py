"""独立核对 ACSIncome 队列：
  1. 每条命令的规范参数必须等于 metrics.json 反查出的口径
     （bs=256 / lr=3e-4 / sgd / rounds=500 / T=2 / repeats=3 / client_parallel=2 /
      dataloader_num_workers=0）
  2. 派生命令必须存在「同数据集+同划分+同算法、仅 K 不同」的源命令
  3. 不得出现 -variant_tag
不做与生成器相同的选优逻辑，避免自证。
"""
from __future__ import annotations

CANON = {
    "-batch_size": "256",
    "-test_batch_size": "256",
    "-learning_rate": "3e-4",
    "-optimize_method": "sgd",
    "-communication_round_I": "500",
    "-algorithm_epoch_T": "2",
    "-exp_repeat_times": "3",
    "-client_parallel": "2",
    "-dataloader_num_workers": "0",
    "-model_type": "ANN",
    "-task": "Tabular_CLF",
}


def parse(cmd):
    toks = cmd.split()
    d = {}
    i = 0
    while i < len(toks) - 1:
        if toks[i].startswith("-"):
            d[toks[i]] = toks[i + 1]
            i += 2
        else:
            i += 1
    return d


queue = [l.strip() for l in open("_acsq.txt", encoding="utf-8") if l.strip()]
problems = []
keys = set()
for line in queue:
    d = parse(line)
    for k, v in CANON.items():
        if d.get(k) != v:
            problems.append("参数偏离口径 %s=%s (应为 %s)" % (k, d.get(k), v))
    if "-variant_tag" in line:
        problems.append("出现 -variant_tag")
    if "-resume" not in line:
        problems.append("缺 -resume")
    if d.get("-dataset") != "ACSIncome":
        problems.append("数据集不是 ACSIncome")
    key = (d.get("-split_strategy"), d.get("-algorithm"), str(d.get("-num_clients_K")))
    if key in keys:
        problems.append("重复格 %s" % (key,))
    keys.add(key)

print("队列条数=%d" % len(queue))
print("问题数=%d" % len(problems))
for p in problems[:20]:
    print("  !", p)
