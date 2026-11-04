# 批量一致性全量审计（只审计，未改代码、未重启作业）

审计时间：2026-11-04 16:00 (Asia/Shanghai)
审计方式：静态代码扫描 + `result_path/` 落盘 `experiment_config` 全量取证（2567 份结果 JSON，19 个数据集）
审计范围：`algorithm/`、`experiment.py`、`main_*.py`、`tool/worker.py`、`run_experiments.py`、`ablation/`

---

## 0. 结论速览

1. **硬编码常量确实把"有效批"钉死为 256**：现网 34 个算法文件里都能找到 `accumulation_steps = max(1, int(256 / param_dict['batch_size']))`（`FederatedRenyi.py:431` 只是其中一处）。任何 `batch_size < 256` 的运行都被这一步"补"回 256。
2. **但 3 个算法根本没走这条路径**：`FedFB`、`FederatedNova`（表内名 `FedNova`）在每个 micro-batch 直接 `step + zero_grad`，有效批 = 名义 `batch_size`；`FedFair` 更特殊——每个通信轮只对全局模型做 **1 次** `step`，覆盖当轮所有客户端整片数据，有效批 ≈ 客户端数据量之和，既不是 64 也不是 256。
3. **PDFFed_DP 并不是"未实现累积"**：它整体复用 `PDFFed.PDF_Fed`，累积逻辑在 `PDFFed.py:609-633` 生效。此前"4 个算法未实现累积"的说法里，PDFFed_DP 这一条需要撤回。
4. **VCTK 确实是"部分 256、部分 64"混在一张表里**：15 个已跑算法中 12 个有效批 256（80%），2 个有效批 64（13.3%：FedFB、FedNova），1 个（FedFair，6.7%）为"整片/轮"的第三种口径。
5. **同一数据集、同一张比较表内混用 64 与 256 的情况在全部音频数据集上成立**（VCTK、AudioMNIST、RAVDESS/RAVDESS_ar、CREMA-D/CREMA-D_ar/CREMA-D_gender），图像线 FairFace / LFWA+ / UTKFace / CelebA 也有 32/64/128 与 256 混排。
6. **未发现"某个文件用 128 作为累积常量"**。128 在本仓库只出现在默认维度（`emb_dim`/`nn_input_size`/`max_len`）、测试批或历史单跑参数，不参与累积步数计算。真正的不一致是 `max(1, ...)` 的有无（见 1.4）。
7. **当前无在跑作业**：`192.168.1.30`、`192.168.1.31` 上均无 `main_*_CLF.py` 进程；`parallel_logs/monitor_state.json` 记录 `all_done_sent: true`。

---

## 1. 代码层全扫描

### 1.1 累积常量定义表（现网可调用文件）

| 文件:行号 | 常量写法 | 用途 | 生效算法 |
|---|---|---|---|
| `algorithm/FederatedRenyi.py:431` | `max(1, int(256 / bs))` | 计算 `accumulation_steps` | FedRenyi |
| `algorithm/CoBoosting.py:140` | 同上 | 同上 | CoBoosting |
| `algorithm/FAFI.py:139` | 同上 | 同上 | FAFI |
| `algorithm/FairFed.py:86` | 同上 | 同上 | FairFed |
| `algorithm/FairFedMOE.py:382` | 同上 | 同上 | FairFedMOE |
| `algorithm/FedBE.py:157` | 同上 | 同上 | FedBE |
| `algorithm/FedBEns.py:159` | 同上 | 同上 | FedBEns |
| `algorithm/FedCOG.py:155` | 同上 | 同上 | FedCOG |
| `algorithm/FedCVAE.py:212` | 同上 | 同上 | FedCVAE |
| `algorithm/FedDF.py:140` | 同上 | 同上 | FedDF |
| `algorithm/FederatedAverage.py:153` | 同上 | 同上 | FedAvg |
| `algorithm/FederatedAverageWithPo.py:209` | 同上 | 同上 | FederatedAverageWithPo |
| `algorithm/FederatedProto.py:40` | 同上 | 同上 | FedProto |
| `algorithm/FederatedProximal.py:42` | 同上 | 同上 | FedProx |
| `algorithm/FederatedRep.py:52` | 同上 | 同上 | FedRep |
| `algorithm/FederatedSum.py:400` | 同上 | 同上 | FederatedSum |
| `algorithm/FedET.py:137` | 同上 | 同上 | FedET |
| `algorithm/FedF2DG.py:204` | 同上 | 同上 | FedF2DG |
| `algorithm/FedFed.py:160` | 同上 | 同上 | FedFed |
| `algorithm/FedFree.py:193` | 同上 | 同上 | FedFree |
| `algorithm/FedGen.py:168` | 同上 | 同上 | FedGen |
| `algorithm/FedLGD.py:125` | 同上 | 同上 | FedLGD |
| `algorithm/FedLMG.py:124` | 同上 | 同上 | FedLMG |
| `algorithm/FedOMG.py:97` | 同上 | 同上 | FedOMG |
| `algorithm/FedTMOS.py:142` | 同上 | 同上 | FedTMOS |
| `algorithm/FL_FairBatch.py:611` | 同上 | 同上 | FL_FairBatch |
| `algorithm/FOL.py:164` | 同上 | 同上 | FOL |
| `algorithm/LoGoFair.py:252` | 同上 | 同上 | LoGoFair |
| `algorithm/FedRevive.py:151` | 同上 | 同上 | FedRevive |
| `algorithm/GeFL.py:191` | 同上 | 同上 | GeFL |
| `algorithm/MAHyFL.py:151` | 同上 | 同上 | MAHyFL |
| `algorithm/Scaffold.py:196` | 同上 | 同上 | Scaffold |
| `algorithm/mFairFL.py:181` | **`int(256 / bs)`（无 `max`）** | 同上 | mFairFL |
| `algorithm/OSFL.py:140` | **`int(256 / bs)`（无 `max`）** | 同上 | OSFL |
| `algorithm/PDFFed.py:839` | **`int(256 / bs)`（无 `max`）** | 同上 | PDFFed、PDFFed_DP（复用） |
| `algorithm/Simple_mFairFL.py:44` | **`int(256 / bs)`（无 `max`）** | 同上 | Simple_mFairFL |

Legacy（不在当前实验网格，仅供追溯）：`algorithm/backup/*.py`（DENSE/FedCAV/FedDEO/FedELMY/FedFisher/FENS/FedKD，7 处，均为 `max(1, ...)`）、`algorithm/abandon/*.py`（AggregatedProgressive:351、EMAProto_Progressive:351、PDFFed - 旧版:88，均无 `max`）、`ablation/PDFFed_Abl.py`（15 处）、`ablation/PDFFed_V2_Abl.py`（5 处）、`ablation/PDFFed_Abl-V0.py`（6 处），全部为 `int(256 / bs)` 无 `max`。

### 1.2 其他硬编码批量（非累积用途，但会改变名义批）

| 文件:行号 | 常量 | 用途 | 影响 |
|---|---|---|---|
| `tool/worker.py:162-163` | `batch_size = 32 if small_gpu else 256` | 远程任务命令构建时**覆盖** CLI 默认 | 小显存机器整条队列被强制成 32；这是 32 的来源之一 |
| `run_experiments.py:14` | `BATCH_SIZE = 256` | 独立调度脚本名义批 | 表格/图像/文本统一 256 |
| `run_experiments.py:15` | `IMG_BATCH_SIZE = 32` | 图像实验名义批，注释"梯度累积等效256" | 依赖累积实现才成立 |
| `run_experiments.py:16` | `SENT_BATCH_SIZE = 8` | 文本实验名义批，注释同上 | 依赖累积实现才成立 |
| `main_AUD_CLF.py:126-127` | 默认 `64 / 64` | 音频线 CLI 默认 | 音频线名义批 64 |
| `main_IMG_CLF.py:110-111` | 默认 `256 / 256` | 图像线 CLI 默认 | — |
| `main_Tabular_CLF.py:158-159` | 默认 `256 / 256` | 表格线 CLI 默认 | — |
| `main_SENT_CLF.py:171-172` | 默认 `256 / 256` | 文本线 CLI 默认 | — |
| `DataScalability.py:56` | 默认 `64` | 数据规模扩展实验 | 名义批 64 |
| `FedAvg_multi_lr.py:30` | 默认 `256` | 多学习率实验 | — |
| `algorithm/FedFair.py:52` | `micro_batch_size = max(1, bs // 4)` | **组内 micro-batch**，不是累积 | 只影响峰值显存，不改变"整片一次 step"的量级 |
| `algorithm/FederatedRenyi.py:98` | `get_v_batch_size` 默认 128 | 正则项估计的分块大小 | 与训练批量无关 |

**结论：不存在某个算法把累积常量写成 128 的情况。** 128 只出现在 `emb_dim`/`nn_input_size`/`max_len` 的默认值、`test_batch_size`、以及 ADULT 的旧单跑参数。

### 1.3 三类实现路径（决定有效批）

| 类别 | 代码锚点 | optimizer 更新频率 | 有效批 |
|---|---|---|---|
| A. 真正累积（34 个文件） | 例 `FederatedAverage.py:109-114,127-129`；`FederatedRenyi.py:621-633`；`Scaffold.py:108-137`；`PDFFed.py:609-633` | 每 `accumulation_steps` 个 micro-batch 一次 + 末尾冲刷 | **256**（bs ≤ 256 时） |
| B. 无累积（2 个） | `FedFB.py:272-279`（`scale_backward` → `scaler_step` → `model.zero_grad()` 每个 batch）；`FederatedNova.py:213-224`（同） | 每个 micro-batch 一次 | **= 名义 batch_size**（音频 64 / 图像 32·64） |
| C. 整片单步（1 个） | `FedFair.py:588-592`（`global_loss = 0*tmp_loss.sum() + sum(client_loss_list)`，随后一次 `scaler_step` + `zero_grad`） | **每通信轮 1 次**（覆盖当轮所有客户端整片数据） | **≈ Σ 选中客户端样本数**（万级），既非 64 也非 256 |

`PDFFed_DP.py` 全文只做 `param_dict` 包装并调用 `PDF_Fed`，因此属于 A 类（累积生效），**不属于 B 类**。

### 1.4 常量差异与潜在风险

1. **`max(1, ...)` 缺失**（mFairFL / OSFL / PDFFed / Simple_mFairFL + 全部 legacy）：当 `batch_size > 256` 时 `accumulation_steps` 会取 0，随后 `(batch_id + 1) % 0` 抛 `ZeroDivisionError`。当前所有落盘配置的 `batch_size ≤ 256`，**该分支从未被触发**，属潜在缺陷而非已发生故障。
2. **梯度归一化口径不统一（次要发现）**：`FederatedRenyi.py:621` 明确 `scale_backward(loss / accumulation_steps, ...)`，而 `FederatedAverage.py:110`、`FederatedProximal.py:103` 等直接 `scale_backward(loss, ...)`（`loss` 已是 micro-batch 均值）。等长 micro-batch 下，后者累加 4 步得到约 4× 全批平均梯度。Adam 对梯度整体缩放近似不变，因此数值影响有限，但严格意义上"累积 4 步 = 单批 256"只在 FedRenyi 一处严格成立。**这一点建议在论文中不要写成"数学严格等价"**。
3. **有效批没有落盘**：`result_path` 的 `experiment_config` 只记录 `batch_size` 与 `test_batch_size`，`aggregate.repeat_batch_sizes` 字段在所有结果里都是 `null`（样例见 `result_path/VCTK/Speaker/FedAvg/AUDIOCNNRNN/20Clients/1.txt.json:74-78`）。这意味着**仅凭结果文件无法事后还原有效批**，只能靠"算法名 + 代码知识"推断——这正是本次审计必须写文档的原因。

---

## 2. 运行层取证

### 2.1 取证口径

- 数据源：`result_path/<DATASET>/<SPLIT>/<ALGO>/<MODEL>/<K>Clients/*.txt.json`，共 2567 份，覆盖 19 个数据集。
- 逐份读取 `experiment_config.{dataset, algorithm, batch_size, test_batch_size, num_clients_K, communication_round_I}`、`aggregate.metrics.ACC.values`（长度 = 完成的 repeat 数）、`aggregate.gpu_seconds`、`experiment_config_hash`。
- 判定"已跑完"：`ACC.values` 长度 = 3（seed 42/1042/2042）；长度 = 1 视为历史单次。
- 判定"在跑"：`192.168.1.30` / `192.168.1.31` 进程扫描 + `parallel_logs/monitor_state.json`（`all_done_sent: true`）。**两机当前均无训练进程，故矩阵中无"在跑"格子。**

### 2.2 有效批矩阵（行=算法，列=数据集/模态）

单元格格式：`名义 bs / 累积步数 / 实际有效批`；状态：✔ 已跑完(3 seeds)｜◐ 历史单次(1 seed)｜✘ 未跑｜⚠ 需重跑。
为压缩篇幅，同模态同批量的数据集合并为一列，列头注明该列包含的数据集。

| 算法 | VCTK (音频) | AudioMNIST (音频) | RAVDESS / RAVDESS_ar (音频) | CREMA-D / _ar / _gender (音频) | CelebA (图像 224) | UTKFace (图像) | FairFace (图像 224) | LFWA+ (图像 250) | PAD-UFES-20 (图像) | ADULT (表格) | COMPAS/DRUG/DUTCH/BANK/ACSIncome (表格) | bios (文本) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FedAvg | 64/4/256 ✔ | 64/4/256 ✔ | 64/4/256 ✔ | 64/4/256 ✔ | 256/1/256 ✔；个别 64/4/256 ✔ | 256/1/256 ✔；1 格 64/4/256 ◐ | 256·64 混排，均为 256 ✔ | 256·64·32 混排，均为 256 ✔ | 256/1/256 ✔；1 格 32/8/256 ◐ | 256/1/256 ✔；旧 128/2/256 ◐、64/4/256 ◐ | 256/1/256 ✔ | 256/1/256 ✔；1 格 16/16/256 ◐ |
| FedProx | 64/4/256 ✔ | 64/4/256 ✔ | 64/4/256 ✔ | 64/4/256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ | 256·64 混排 → 256 ✔ | 256·64·32 混排 → 256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ |
| Scaffold | 64/4/256 ✔ | 64/4/256 ✔ | 64/4/256 ✔ | 64/4/256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ | 只有 64/4/256 ✔ | 只有 64/4/256 ✔ | 256/1/256 ✔ | 256/1/256 ✔；旧 64/4/256 ◐ | 256/1/256 ✔ | 256/1/256 ✔ |
| FedRep | 64/4/256 ✔ | 64/4/256 ✔ | 64/4/256 ✔ | 64/4/256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ | 32·64·256 混排 → 256 ✔ | 32·64·256 混排 → 256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ |
| FedProto | 64/4/256 ✔ | 64/4/256 ✔ | 64/4/256 ✔ | 64/4/256 ✔ | 64/4/256 与 256/1/256 混排 ✔ | 混排 ✔ | 混排 ✔ | 只有 64/4/256 ✔ | 256/1/256 ✔ | 256/1/256 ✔；旧 64/4/256 ◐ | 256/1/256 ✔ | 256/1/256 ✔ |
| FedRenyi | ✘ 未跑（目录为空） | 64/4/256 ✔ | 64/4/256 ✔ | 64/4/256 ✔ | 1 格 128/2/256，其余 256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ | 32·64·256 混排 → 256 ✔ | 256/1/256 ✔ | 256/1/256 ✔；旧 64/4/256 ◐ | 256/1/256 ✔ | 256/1/256 ✔ |
| FairFed | 64/4/256 ✔ | 64/4/256 ✔ | 64/4/256 ✔ | 64/4/256 ✔ | 1 格 64/4/256，其余 256 ✔ | 256/1/256 ✔ | 混排 → 256 ✔ | 混排 → 256 ✔ | 256/1/256 ✔ | 256/1/256 ✔；旧 64/4/256 ◐ | 256/1/256 ✔ | 256/1/256 ✔ |
| FAWP | 64/4/256 ✔ | 64/4/256 ✔ | 64/4/256 ✔ | 64/4/256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ | 混排 → 256 ✔ | 混排 → 256 ✔ | 256/1/256 ✔ | 256/1/256 ✔；旧 128 ◐ | 256/1/256 ✔ | 256/1/256 ✔ |
| mFairFL | 64/4/256 ✔ | 64/4/256 ✔ | 64/4/256 ✔ | 64/4/256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ | 混排 → 256 ✔ | 混排 → 256 ✔ | 256/1/256 ✔ | 256/1/256 ✔；旧 128/64 ◐ | 256/1/256 ✔ | 256/1/256 ✔ |
| LoGoFair | 64/4/256 ✔ | 64/4/256 ✔ | 64/4/256 ✔ | 64/4/256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ | 混排 → 256 ✔ | 混排 → 256 ✔ | 256/1/256 ✔ | 256/1/256 ✔；旧 64 ◐ | 256/1/256 ✔ | 256/1/256 ✔ |
| FL_FairBatch | 64/4/256 ✔ | 64/4/256 ✔ | 64/4/256 ✔ | 64/4/256 ✔ | 1 格 64/4/256，其余 256 ✔ | 11 格 64/4/256 + 其余 256 ✔ | 混排 → 256 ✔ | 只有 64/4/256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ |
| PDFFed | 64/4/256 ✔ | 64/4/256 ✔ | 64/4/256 ✔ | 混排 → 256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ | 混排 → 256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ |
| PDFFed_DP | 64/4/256 ✔ | 64/4/256 ✔ | 64/4/256 ✔ | 64/4/256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ | 混排 → 256 ✔ | 混排 → 256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ |
| **FedFB** | **64/1/64 ⚠** | **64/1/64 ⚠** | **64/1/64 ⚠** | **64/1/64 ⚠** | **256/1/256 ✔** | **256/1/256 ✔** | **64/1/64 ⚠（3 格）+256 ✔** | **32/1/32 ⚠（3 格）+64/1/64 ⚠（3 格）+256 ✔** | 256/1/256 ✔ | 256/1/256 ✔；旧 64/1/64 ◐ | 256/1/256 ✔ | 256/1/256 ✔ |
| **FedNova** | **64/1/64 ⚠** | **64/1/64 ⚠** | **64/1/64 ⚠** | **64/1/64 ⚠** | **256/1/256 ✔** | **256/1/256 ✔** | **64/1/64 ⚠（5 格）+256 ✔** | **32/1/32 ⚠（2 格）+64/1/64 ⚠（4 格）+256 ✔** | 256/1/256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ | 256/1/256 ✔ |
| **FedFair** | **64/—/整片 ⚠** | **64/—/整片 ⚠** | **64/—/整片 ⚠** | **64/—/整片 ⚠** | 1 格 64 + 11 格 256，均"整片"⚠ | 整片 ⚠ | 整片 ⚠ | 整片 ⚠ | 整片 ⚠ | 整片 ⚠；旧 64 ◐ | 整片 ⚠ | 整片 ⚠ |
| OSFL / Simple_mFairFL / FOL / FedBE 等 | ✘ | ✘ | ✘ | ◐ 仅旧单跑 | ✘ | ✘ | ✘ | ✘ | ✘ | ◐ 旧 128 单跑 | ✘ | ✘ |

> 说明：`64/4/256` 表示名义 bs=64、累积 4 步、有效批 256；`256/1/256` 表示 bs 已是 256，累积步数 1。图像线"32·64·256 混排"指同一算法在同一数据集的**不同格子**分别用了 32、64、256 三种名义批，但因都走累积路径，有效批均被拉回 256。

### 2.3 运行状态汇总

| 项 | 值 | 证据 |
|---|---|---|
| 已完成（3 seeds）格子 | 2471 / 2567 份 JSON 完成 3 seeds；按数据集：COMPAS 398/398、DRUG 287/287、DUTCH 356/356、BANK 215/215、VCTK 33/33、AudioMNIST 64/64、RAVDESS 16/16、RAVDESS_ar 16/16、CREMA-D_ar 16/16、CREMA-D_gender 16/16、FairFace 104/104 | `result_path` ACC 长度 |
| 未跑满的线 | ACSIncome 75/111（在补 K30/K40）、ADULT 64/101（旧单跑多）、bios 77/88、CelebA 192/199、LFWA+ 192/193、UTKFace 267/268、PAD-UFES-20 35/36、CREMA-D 48/50、VCTK 中 FedRenyi 0 格 | 同上 |
| 在跑 | 无 | `.30`/`.31` 无训练进程；monitor `all_done_sent=true` |
| 需重跑（FedFB/FedNova 且 bs≠256，已完成 3 seeds） | **48 格 / 144 次独立运行** | 见 4.3 |

---

## 3. 直接回答三个问题

**a) VCTK 上哪些算法有效批 256、哪些 64，占比多少？**
有效批 256 的 12 个：FedAvg、FedProx、Scaffold、FedRep、FedProto、FairFed、FederatedAverageWithPo、mFairFL、LoGoFair、FL_FairBatch、PDFFed、PDFFed_DP（**12/15 = 80%**）；有效批 64 的 2 个：FedFB、FedNova（**2/15 = 13.3%**）；FedFair 是第三种口径（整片/轮，1/15 = 6.7%）；FedRenyi 未跑。

**b) 是否存在同一数据集、同一比较表内混用 64 与 256（结论表不可比）？**
存在，且覆盖全部音频数据集（VCTK、AudioMNIST、RAVDESS、RAVDESS_ar、CREMA-D、CREMA-D_ar、CREMA-D_gender）：表中 FedFB/FedNova 两行是 64，其余 12 行是 256，FedFair 一行是整片单步——一张表里同时存在三种优化口径。

**c) 其他数据集是否触发同一问题？**
触发，共 14 个数据集存在 `bs≠256` 的格子：VCTK、AudioMNIST、RAVDESS、RAVDESS_ar、CREMA-D（含 16）、CREMA-D_ar、CREMA-D_gender、CelebA（64/128）、UTKFace（64）、FairFace（32/64）、LFWA+（32/64）、PAD-UFES-20（32）、ADULT（128/64）、bios（16）。但触发 ≠ 不可比：走累积路径的算法即使名义 bs=32/64/128，有效批仍是 256，**真正破坏可比性的只有 `FedFB`、`FedNova`（有效批 = 名义批）和 `FedFair`（整片单步）**，具体格子见 2.2 表 ⚠ 项。

---

## 4. 影响面与最小修法（方案，不实施）

### 4.1 受影响的论文表格与结论

| 表格/结论 | 依赖的前提 | 现状 | 处理建议 |
|---|---|---|---|
| VCTK 主表（尚未写 `results_summary`，数据已全） | "15 个算法同批 256" | 12 行 256、2 行 64、1 行整片 | 先补 FedFB/FedNova 累积后重跑（6 格），或表注"FedFB/FedNova 有效批 64、FedFair 整片单步" |
| 音频表（AudioMNIST / RAVDESS / RAVDESS_ar / CREMA-D / CREMA-D_ar / CREMA-D_gender） | 同上 | 三种口径 | 至少加注；结论若涉及"FedFB/FedNova 收敛更快/更慢"需重跑 |
| 图像表 FairFace / LFWA+ / UTKFace / CelebA | "所有算法批量一致" | 累积类有效批 256 但**同一算法跨格名义批 32/64/256 混排**；FedFB/FedNova 有效批 32/64 | 补注 + 统一名义批（建议全图像线固定 bs=64，累积 4）；FedFB/FedNova 需重跑 |
| 表格线 ADULT（K4 旧跑） | 批量一致 | 128/64/256 历史混排，多数为 1 seed | 旧 K4 结果若已进附录需标注批次；K20 主表不受影响 |
| **所有"批量一致"表述**（论文/汇报/结果文档） | 全局 256 | 仅对 A 类算法成立 | 改为"除 FedFB/FedNova/FedFair 外，有效批统一为 256（累积实现）"，并单列这三者的口径 |

### 4.2 最小改动方案（二选一，推荐 A+B）

**改动 A：把有效批常量提升为显式配置（最小、全局）**
1. 在 `experiment.py` 组装 `param_dict` 处新增 `effective_batch_size = int(args.effective_batch_size or 256)`，写入 `experiment_config`（`experiment_config_hash` 随之变化，旧结果可凭 hash 区分）。
2. 把现网 34 处 `max(1, int(256 / param_dict['batch_size']))` 改为 `max(1, int(param_dict.get('effective_batch_size', 256) / param_dict['batch_size']))`；同时给 mFairFL / OSFL / PDFFed / Simple_mFairFL 补上 `max(1, ...)`，消除 `bs>256` 时的除零隐患。
3. 在各算法记录 `accumulation_steps` 的地方，把真实有效批写入 `aggregate.repeat_batch_sizes`（该字段已存在但全为 `null`），使"有效批"可事后复核。

**改动 B：给 2 个未累积算法补累积（改动最小、直接消除可比性问题）**
- `algorithm/FedFB.py:272-279`：把 `scaler_step + model.zero_grad()` 包进 `if (batch_id + 1) % accumulation_steps == 0`，函数末尾补一次"残余冲刷"；在 `FedFB(...)` 入口按 A 的口径计算 `accumulation_steps` 并透传。
- `algorithm/FederatedNova.py:213-224`：同上（Nova 的归一化因子需按累积后的步数重新计算，避免与 FedNova 原论文的 `τ_i` 口径冲突，建议保留原 `τ_i` 定义、只改更新频率）。
- 两处各约 10 行，可完全照抄 `FederatedAverage.py:109-129` 的写法。

**不建议改 FedFair**：它的"一轮一次 step"是算法定义（Eq.10 的全局更新），改成 per-batch step 等于换成另一个算法，需重新调参并重写方法描述。推荐**保留 + 表注**（并在附录说明其为"整片/轮"口径，不与 256 直接对比）。

### 4.3 修完后的重跑清单与机时（按落盘 `gpu_seconds` 求和，含 3 个 seed）

前提：只做改动 B（FedFB / FederatedNova 补累积），其余算法无需重跑。

| 数据集 | 算法 | 需重跑格子 | 3-seed GPU 机时 |
|---|---|---|---|
| VCTK | FedFB / FedNova | 3 / 3 = 6 | 15.0 + 59.2 = **74.2 h** |
| FairFace | FedFB / FedNova | 3 / 5 = 8 | 44.0 + 252.5 = **296.5 h** |
| LFWA+ | FedFB / FedNova | 6 / 6 = 12 | 8.3 + 30.5 = **38.8 h** |
| AudioMNIST | FedFB / FedNova | 4 / 4 = 8 | 4.6 + 44.3 = **48.9 h** |
| CREMA-D | FedFB / FedNova | 3 / 3 = 6 | 2.2 + 12.1 = **14.3 h** |
| CREMA-D_ar | FedFB / FedNova | 1 / 1 = 2 | 2.2 + 8.9 = **11.1 h** |
| CREMA-D_gender | FedFB / FedNova | 1 / 1 = 2 | 5.9 + 2.7 = **8.6 h** |
| RAVDESS | FedFB / FedNova | 1 / 1 = 2 | 0.7 + 12.1 = **12.8 h** |
| RAVDESS_ar | FedFB / FedNova | 1 / 1 = 2 | 0.7 + 0.7 = **1.4 h** |
| ADULT | FedFB（旧 K4 单跑） | 1 | 0.1 h（可选） |
| **合计** | | **48 格 / 144 次运行** | **≈ 506.6 GPU·小时**（≈ 21 GPU·天） |

优先级建议：VCTK（主表，74 h）→ AudioMNIST（48.9 h）→ LFWA+（38.8 h）→ CREMA-D 系列（34 h）→ FairFace（296.5 h，最大头，可最后做或只补 FedNova K20/K30 关键格）。若机器可并行 8 卡，墙钟约 3 天。

**若只加注不重跑**：需在 7 个音频表 + FairFace/LFWA+/UTKFace/CelebA 表注中写明"FedFB/FedNova 为每 micro-batch 更新（有效批 = 名义批 64/32），其余算法累积至 256；FedFair 为每轮整片一次更新"，并撤下任何"所有算法批量一致"的表述。

---

## 5. FedFair 原论文核验（证据级，2026-11-04 补）

### 5.1 匹配到哪一篇

| 项 | 我们的实现 | 论文 | 是否匹配 |
|---|---|---|---|
| 类/函数名 | `FedFair`（`algorithm/FedFair.py:440`，内部 `D_hat_θ`） | FedFair | ✔ |
| 出处 | 文件头 `https://arxiv.org/pdf/2109.05662` | X. Che, J. Hu, Z. Zhou, Y. Zhang, L. Chu, *Training Fair Models in Federated Learning without Data Privacy Infringement*, arXiv:2109.05662v2（IEEE Big Data 2024） | ✔ |
| 超参名 | `λ_a`、`λ_b`、`β`、`γ`、`ϵ`（`FedFair.py:514`、`635-646`） | Eq.(17) 的 λa、λb；Eq.(21)(22) 的 β、γ、ϵ | ✔ |
| 目标函数 | `client_i_loss = (n_i/N)·L + (λ_a−λ_b)/K·D`（`FedFair.py:158-164`）；λ 更新 `max((1−γβ)λ + β/K·ΣD̂ − βϵ, 0)`（`632-646`） | Eq.(17) 拉格朗日 + Eq.(20) `(m_i/m_total)∇L̂ + (λa−λb)/N·∇D̂` + Eq.(21)(22) | ✔（系数 K vs N 见 5.3） |
| 数据分组 | `a, b = 1, 0`，按 (s,c) 四组统计 `D̂`（`FedFair.py:103-116`） | `D̂_i = L̂_{a,c} − L̂_{b,c}`（Section IV-A） | ✔ |

### 5.2 (a) 论文算法定义：每轮是否切 mini-batch？是否要求全量？

**否，论文不切 mini-batch；是，论文每轮用客户端全体数据算一次梯度、每轮只更新一次 θ。**

原文证据（arXiv:2109.05662v2，Section IV-D *Solving the FedFair and LCO Problems*）：

- Eq.(4)：`L̂_i(θ) = 1/m_i Σ_{(x,s,y)∈B_i} ℓ(f_θ(x), y)` —— 客户端私有集 B_i 的**全量平均损失**。
- Algorithm 1 第 4 行："Each client U_i uses B_i to compute D̂_i(θ^k), ∇D̂_i(θ^k) and ∇L̂_i(θ^k), and send them to Q."
- Eq.(19)：`θ^{k+1} = θ^k − α Σ_{i=1}^{N} ∇_i(θ^k)` —— 每个通信轮 k **只做一次** θ 更新。
- 全文检索 `mini-batch` / `minibatch` / `SGD` / `stochastic gradient`：**0 命中**（PDF 全文文本检索）。

结论：FedFair 的 optimizer step 次数 = 通信轮数，与 batch_size 无关；每次 step 覆盖本轮全部参与客户的全部样本。

### 5.3 (b) 我们的实现与论文是否一致

**核心一致；偏离 3 点（不含训练目标本身）。**

| 项 | 论文 | 实现 | 判定 |
|---|---|---|---|
| 每轮 step 次数 | 1 次（Eq.19） | 1 次：`global_loss=0*tmp_loss + sum(client_loss_list)` → `scale_backward` → `scaler_step`（`FedFair.py:619-623`） | ✔ 一致 |
| 全量梯度 | `∇L̂_i` 在 B_i 上精确计算 | 逐 client、逐 micro-batch `scale_backward`（`FedFair.py:162-164`），整轮累积后一次 step；因梯度线性可加，与全量梯度数学等价 | ✔ 等价 |
| D̂ 系数 | `(λ_a−λ_b)/N` | `(λ_a−λ_b)/K`（`FedFair.py:158-164` 的 `fair_coef`，定义在 `418`） | ✘ 偏离：K 为**本轮参与**客户数；FL_fraction=1 时 K=N 才一致 |
| λ 更新 | `β/N·Σ_i D̂_i`（Eq.21/22） | `β/len(idxs_users)·Σ D̂`（`FedFair.py:635`） | ✘ 同上，抽样时才不同 |
| θ 更新器 | 固定步长 α 的梯度下降 | 配置的 optimizer（sgd/adam）+ AMP + 梯度裁剪 | ✘ 实验设置偏离（非算法语义） |
| 数据读取 | 未定义批大小 | `mini_batch` 只切成 micro-batch 控制显存（`FedFair.py:72-78`），**不产生额外 step** | ✔ 等价；只是工程实现 |

**因此确认：FedFair 属于"算法固有全量单步"**——它不参与"有效批 = 256"的可比性；其有效批应记为**该轮参与客户样本量之和**，实现已在每轮写入 `param_dict['_runtime_fedfair_round_samples']`（`FedFair.py:531-535`），并由 `experiment.py::_recorded_batch_size` 优先读取（该键以 `_runtime_` 前缀命名，不进入 `experiment_config` 与 hash）。


## 附录：复现方式

- `result_path` 全量扫描：遍历 `**/*.json`，读 `experiment_config` 与 `aggregate.metrics.ACC.values`，按 `(dataset, algo, batch_size, clients)` 聚合。
- VCTK 明细：`result_path/VCTK/Speaker/<ALGO>/AUDIOCNNRNN/<K>Clients/1.txt.json`，15 个算法 / 33 格 / 全部 3 seeds。
- 代码锚点核对：`algorithm/FederatedAverage.py:109-129`（A 类模板）、`FedFB.py:272-279`、`FederatedNova.py:213-224`（B 类）、`FedFair.py:588-592`（C 类）、`PDFFed.py:839,609-633`（PDFFed_DP 复用）。
- 本次审计未修改任何训练代码，未启动/终止任何作业；临时扫描脚本位于工作区根目录（`_batch_scan*.py`），审计结束后删除。
