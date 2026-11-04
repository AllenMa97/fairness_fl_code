# 重跑清单（Rerun Plan）

生成时间：2026-11-04　｜　口径：一个「格子」= `result_path/<数据集>/<split>/<算法>/<模型>/<K>Clients/1.txt.json` 的一份落盘结果（3 个 seed 合并为 1 格）。
成本口径：从各格子 `aggregate.gpu_seconds` 逐 repeat 求和；删掉的格子取归档清单里的历史值（见 `batch_delete_manifest.md`）。

---

## 0. 先读这三条

1. **参数语义已固定**：`batch_size` = 目标有效批（默认 256）；`mini_batch` = 每次前向/反向的 micro-batch；`accumulation_steps = ceil(batch_size / mini_batch)`；真实有效批 = `mini_batch × accumulation_steps`。旧 CLI `-batch_size` 仍按 micro-batch 解释（等价 `-mini_batch`），旧脚本无需改。
2. **ADULT 整体排除**：该数据集上各算法均坍缩（本轮冒烟 FedFB 的 epoch loss 为 `3.51e-07`），重跑无信息增量；G1 中已删的 ADULT 结果**不恢复、不补跑**，保留"已删除"状态即可。
3. **只删过 G1**：G2（FedFair 144 格）与 G3（91 格单 seed）均未删除；下表 P0 是唯一"数据已不在盘上"的组。

---

## P0 —— 必须重跑（G1：FedFB / FedNova，非 ADULT，有效批 ≠ 256，结果已删）

每个格子跑 3 个 seed（42 / 1042 / 2042）；模型与 K 见「格子数」列。

- [ ] **VCTK**（AUDIOCNNRNN，K20/30/40，3 格/算法）— FedFB 3 + FedNova 3 = **6 格 / 18 次**，目标 `mini_batch=64, batch_size=256, steps=4`，预估 **74.2 GPU·h**
- [ ] **FairFace**（BERTCLASSIFIER，K20/40，FedFB 3 格 + FedNova 5 格 = **8 格 / 24 次**），`mini_batch=64/32`，预估 **296.5 GPU·h**
- [ ] **LFWA+**（BERTCLASSIFIER，4 split × 3 K，FedFB 6 + FedNova 6 = **12 格 / 36 次**），预估 **38.8 GPU·h**
- [ ] **AudioMNIST**（AUDIOCNNRNN，K20/30/40，4 格/算法 = **8 格 / 24 次**），预估 **48.9 GPU·h**
- [ ] **CREMA-D**（AUDIOCNNRNN，3 格/算法 = **6 格 / 18 次**），预估 **14.3 GPU·h**
- [ ] **CREMA-D_ar**（**2 格 / 6 次**），预估 **11.1 GPU·h**
- [ ] **CREMA-D_gender**（**2 格 / 6 次**），预估 **8.6 GPU·h**
- [ ] **RAVDESS**（**2 格 / 6 次**），预估 **12.8 GPU·h**
- [ ] **RAVDESS_ar**（**2 格 / 6 次**），预估 **1.4 GPU·h**

**P0 合计：48 格 / 144 次运行 / ≈ 506.6 GPU·h（≈ 21.1 GPU·天）**

> 每格命令模板（以 VCTK FedFB 为例，其余数据集替换 `-dataset/-num_clients_K/-algorithm`）：
> ```
> python main_AUD_CLF.py -algorithm FedFB -dataset VCTK -task AUD_CLF \
>   -split_strategy Dirichlet01 -num_clients_K 20 -communication_round_I <I> \
>   -exp_repeat_times 3 -parallel_repeats 3 -mini_batch 64 -effective_batch_size 256
> ```
> 校验：结果 json 的 `experiment_config` 应含 `batch_size:256, mini_batch:64, accumulation_steps:4, effective_batch_size:256`，`aggregate.gpu_seconds` 有 3 个值。

---

## P1 —— 名义 bs=256 的 FedFB / FedNova（**技术上等价，成本巨大，默认不跑**）

现状扫描（`result_path`，排除 ADULT）：**225 格，全部名义 bs=256**。

| 数据集 | FedFB 格数 | FedNova 格数 | 3-seed 机时 |
|---|---|---|---|
| ACSIncome | 0 | 12 | 11,522.0 h |
| DUTCH | 14 | 20 | 2,371.2 h |
| COMPAS | 12 | 20 | 908.9 h |
| DRUG | 12 | 12 | 902.7 h |
| BANK | 12 | 15 | 770.5 h |
| FairFace | 2 | 9 | 295.5 h |
| CelebA | 12 | 12 | 287.1 h |
| UTKFace | 12 | 23 | 56.1 h |
| LFWA+ | 6 | 6 | 31.7 h |
| bios | 4 | 4 | 33.9 h |
| PAD-UFES-20 | 3 | 3 | 2.9 h |
| **合计** | **89** | **136** | **≈ 17,182.6 GPU·h（≈ 716 GPU·天）** |

- [ ] **裁定 A（推荐）**：不重跑。理由：bs=256 时 `accumulation_steps = ceil(256/256) = 1`，新旧代码逐位等价；这些格子的有效批本来就是 256，不存在口径问题。
- [ ] **裁定 B**：全部重跑 225 格（17,182.6 GPU·h），换取"全部结果出自同一代码版本"。
- [ ] 若选 B，优先级：**BANK/COMPAS/DRUG/DUTCH（表格主表）> FairFace/CelebA/UTKFace/LFWA+ > bios/PAD-UFES-20 > ACSIncome**（ACSIncome 单独 11.5k h，建议最后或不做）。

> 注：这里按你"其余数据集 FedFB/FedNova 全部进清单"的要求列出；但它与 P0 不同——P0 是数据已删、口径确实错，P1 只是代码版本洁净度。

---

## P2 —— G3：单 seed 格子补到 3 seed（**不删，只补**）

- [ ] 现状：**91 格仅有 1 个 seed**（`batch_delete_manifest.md` G3 清单），补 2 个 seed = **182 次运行**，预估 **≈ 135 GPU·h**。
- [ ] 复核口径（回应 91 vs 96）：
  - 91 = G3 本体（非 FedFB/FedNova、非 FedFair 的单 seed 格）。
  - 96 = 全库 `rep<3` 的格子总数 = **91（G3）+ 2（G1 中被删的 FedFB 单 seed 格）+ 3（G2 中 FedFair 单 seed 格）**。
  - 删除 G1 之后当前盘上应为 **94 = 91 + 3**；本次扫描结果确为 94（其中 FedFair 3 格），与公式一致。
  - 结论：**91 是 G3，96 是"全部单 seed"，不是矛盾，是口径不同。**
- [ ] 注意：P0 重跑会顺带补齐 G1 里那 2 个单 seed 格；G2 的 3 个单 seed 格随 FedFair 加注方案处理（见下）。

---

## G2 —— FedFair 144 格：保留 + 加注（不删、不重跑）

- [x] 保留全部 144 格（训练逻辑未改，只把 mock 的 micro-batch 读取改为 `mini_batch` 来源）。
- [ ] 在每张含 FedFair 的表下加注：
  > FedFair 为**整轮全量单步**（每次 optimizer step 覆盖本轮参与客户的全部样本），其有效批 = 该轮参与客户样本量之和，**不与"有效批 256"直接比较**。依据：arXiv:2109.05662v2 Section IV-D Eq.(4)/Eq.(19)/Algorithm 1；实现见 `algorithm/FedFair.py:619-623`，每轮记录见 `_runtime_fedfair_round_samples`。
- [ ] 已知实现偏离（写进附录，不改变结论方向）：论文 Eq.(20)/(21) 的公平项系数用 `N`（全部客户），我们实现用 `K`（本轮参与客户数）；`FL_fraction=1` 时二者相同，抽样时才有差异（`FedFair.py:418`、`635`）。
- [ ] 3 个单 seed 的 FedFair 格：若最终保留未重跑，在表中标注 `n=1`。

---

## 验收与勾选（跑完后逐条打勾）

- [ ] 每个重跑格子的 `experiment_config` 含 4 个批量字段，且 `effective_batch_size == batch_size`（P0 目标 256）。
- [ ] 每个重跑格子的 `aggregate.repeat_batch_sizes` 为 `[256,256,256]`；FedFair 为各 repeat 的**轮样本量**（非 256）。
- [ ] `aggregate.repeat_indices == [0,1,2]`（3 个 seed）。
- [ ] 表格主表重新生成后，`_report_tables.md` 中每个结果行标注有效批类别（`256` / `非256` / `全量`）。
- [ ] 归档包仍保留：`results/_archive/batch_inconsistent_20261104_1640.tar.gz`（50 条）及 `.30/.31` 两份（各 49 条），不删除。
