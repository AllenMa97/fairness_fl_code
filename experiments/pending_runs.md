# 待跑任务清单（pending runs，修订版 v3，只读统计）

生成时间：2026-11-04；**v3 更新：2026-11-05**（同步 `experiments/progress_snapshot_20261105.md` 的停机位置与断点轮次）。只读扫描 `result_path/`、`results/_archive/`、`experiments/batch_delete_manifest.md`、`logs/`、`algorithm/`、git 历史；**未改代码、未启动训练、未删除文件**。

## 修订记录（v2 → v3）

| 项目 | v2 | v3 | 变化 | 说明 |
|---|---|---|---|---|
| 待跑格数 | 49 | **20** | −29 | 2026-11-03～11-05 已完成 29 格并写入 `result_path`，已移出清单 |
| 其中"已停止可续跑" | — | **6** | +6 | 6 格 FairFace 因 2026-11-05 停机中断，已落 `checkpoint_latest.pt`，可 `-resume` |
| 其中"未启动" | — | **14** | — | RAVDESS_ar / CREMA-D_ar / CREMA-D_gender / LFWA+ FedFB / PAD-UFES-20 |

## 修订记录（v1 → v2）

| 项目 | v1 | v2 | 变化 | 说明 |
|---|---|---|---|---|
| 总待跑格数 | 100 | **49** | −51 | 复核“实际有效批已为 256”的格子后剔除 |
| 运行次数 | 300 | **147** | −153 | 剔除 51 格 × 3 seeds |
| 可量化 GPU·h | 506.5 | **506.5** | ≈0 | 被剔的 51 格在 v1 全部标为“待实测”，本就不在合计里 |
| P0 格数 | 44 | **44** | 0 | P0 全是 FedFB/FedNova（无累积，有效批 64/32），复核后不受影响 |

**修订原因**：质疑 1 成立。v1 把 51 个“单 seed / 协议提前终止”的格子列进了 P2，但它们的**实际有效批本来就是 256**（不是 v1 假设的“名义批≠256 待补”）：其中 49 格所属算法为累积类（`accumulation_steps = max(1, int(256 / batch_size))`），另 2 格是 `ACSIncome/Uniform/FedFB（FedFB_acsmoke_fedfb2/fedfb3）`——FedFB 虽属无累积的直通类，但其名义 bs 本身就是 256，故步数 1、有效批仍为 256。这些格子**不需要为批量重跑**；又因为它们都是 pilot/旧协议、不进主表，所以整格剔除。另有 5 个残留桩同目录已有完整 3-seed 正本，也一并移出。

## 质疑答复（直接结论）

### 质疑 1：是。清单里确实混进了“有效批已为 256”的格子——共 52 格

| 分类 | 格数 | 是否误列 | 处理 |
|---|---|---|---|
| P2 单 seed 组（bios 9、ACSIncome 36、CREMA-D K4 2、CelebA K2 3、UTKFace K4 1） | 51 | **是** | 全部移入 §3“已复核无需重跑” |
| P1：PAD-UFES-20/Uniform/FedAvg/K20（bs=32×8 步=256） | 1 | **否（保留）** | 有效批无缺陷，但协议只跑 I=2/50、主表因此缺一整格 → 仍需重跑 |

> 判定口径：只有 FedFB / FedNova 是“直通类”（无累积，有效批 = 名义 bs，见 `git show c9b96ada5^:algorithm/FedFB.py`、`FederatedNova.py` 中无任何 `accum` 代码）；FedFair 是“整片单步”（`FedFair.py:72-77` 明确不使用 accumulation_steps，`FedFair.py:574-623` 整轮累加后只做一次 `optimizer.step()`）；其余全部是累积类，历史代码 `accumulation_steps = max(1, int(256 / batch_size))`（`git show c9b96ada5^:algorithm/FederatedAverage.py:153`），故 bs=16/32/64 都被补回 256。

**bios 的 9 格：确实全部是有效批 256**（用户记忆正确）：

| 格坐标 | 算法 | 类 | 名义 bs | 累积步数 | 有效批 | 原始证据 |
|---|---|---|---|---|---|---|
| `bios/Dirichlet01/FairFed/BERTCLASSIFIER/10Clients/1.txt.json` | FairFed | A | 256 | 1 | 256 | log `batch_size : 256` |
| `bios/Dirichlet01/FedAvg/BERTCLASSIFIER/10Clients/1.txt.json` | FedAvg | A | 256 | 1 | 256 | log `batch_size : 256` |
| `bios/Dirichlet01/FedRep/BERTCLASSIFIER/10Clients/1.txt.json` | FedRep | A | 256 | 1 | 256 | log `batch_size : 256` |
| `bios/Dirichlet01/LoGoFair/BERTCLASSIFIER/10Clients/1.txt.json` | LoGoFair | A | 256 | 1 | 256 | log `batch_size : 256` |
| `bios/Dirichlet05/FedAvg/BERTCLASSIFIER/4Clients/1.txt.json` | FedAvg | A | 16 | 16 | 256 | log `batch_size : 16` + 16 步累积 |
| `bios/Uniform/FairFed/BERTCLASSIFIER/10Clients/1.txt.json` | FairFed | A | 256 | 1 | 256 | log `batch_size : 256` |
| `bios/Uniform/FedAvg/BERTCLASSIFIER/10Clients/1.txt.json` | FedAvg | A | 256 | 1 | 256 | log `batch_size : 256` |
| `bios/Uniform/FedRep/BERTCLASSIFIER/10Clients/1.txt.json` | FedRep | A | 256 | 1 | 256 | log `batch_size : 256` |
| `bios/Uniform/LoGoFair/BERTCLASSIFIER/10Clients/1.txt.json` | LoGoFair | A | 256 | 1 | 256 | log `batch_size : 256` |

- 8 格名义 bs=256 → 累积步数 1 → 有效批 256（日志第 9 行 `batch_size : 256`，如 `log_path/bios/Dirichlet01/FedAvg/BERTCLASSIFIER/10Clients/1.txt`）。
- 1 格 `bios/Dirichlet05/FedAvg/K4` 名义 bs=16 → 累积 16 步 → 有效批 256（日志 `batch_size : 16`，`FederatedAverage.py` 旧版 `:153` 按 `max(1, 256/16)=16` 计算）。
- 结果 JSON 里**没有** `mini_batch` / `accumulation_steps` / `effective_batch_size` 字段（旧语义只写 `batch_size`），日志也只打印 `batch_size`，**不打印 step 比例**；因此“有效批 256”是**按运行时代码路径推导**，不是结果文件里的直接字段。除上述两条外无法从 json/log 直接读到累积步数（如实说明）。

### 质疑 2：AudioMNIST **全部算法都用 64，没有混用**

已落盘 56 格（14 个算法 × 4 个 split：Dirichlet01/Dirichlet05/Dirichlet1/Uniform），**名义 bs=64 占 56/56 = 100%**；256、32、其它各 0 格。再加上已删的 FedFB/FedNova 8 格（归档 JSON 也都是 64），AudioMNIST 计划网格 64 格中 **64 格全部 bs=64**。

| 算法 | 类别 | 格数 | 名义 bs | 累积步数 | 实际有效批 |
|---|---|---|---|---|---|
| FedAvg | A | 4 | 64 | 4 | 256 |
| FedProto | A | 4 | 64 | 4 | 256 |
| FedProx | A | 4 | 64 | 4 | 256 |
| FedRenyi | A | 4 | 64 | 4 | 256 |
| FedRep | A | 4 | 64 | 4 | 256 |
| FL_FairBatch | A | 4 | 64 | 4 | 256 |
| FairFed | A | 4 | 64 | 4 | 256 |
| FederatedAverageWithPo | A | 4 | 64 | 4 | 256 |
| LoGoFair | A | 4 | 64 | 4 | 256 |
| PDFFed | A | 4 | 64 | 4 | 256 |
| PDFFed_DP | A | 4 | 64 | 4 | 256 |
| Scaffold | A | 4 | 64 | 4 | 256 |
| mFairFL | A | 4 | 64 | 4 | 256 |
| FedFair | C | 4 | 64 | 4 | 整片（非 256） |
| FedFB | B | 4 | 64 | 1 | 64 |
| FedNova | B | 4 | 64 | 1 | 64 |
| **合计** | — | **64** | **64（100%）** | — | A 类 52 格=256；C 类 4 格=整片；B 类 8 格=64 |

> 依据：`main_AUD_CLF.py:132` 的 `add_batch_size_arguments(parser, legacy_default=64)`（音频线历史默认为 64）；结果 JSON 与日志（如 `log_path/AudioMNIST/Uniform/FedAvg/AUDIO.../1.txt` 第 5 行 `batch_size : 64`）一致。故“AudioMNIST 混用 32/256”不成立。

## 0. 前提核对（保留 v1 结论）

| # | 核对结论 | 证据 |
|---|---|---|
| G1 并集 | **50 格**：本地 50、`.30` 49、`.31` 49；`.30`/`.31` 清单逐行相同 | 本地归档 50 条目，远端归档各 49 条目，`_g1_30.txt`/`_g1_31.txt` diff=0 |
| 50 vs 49 差异 | 本地多出 `ADULT/Uniform/FedFB/ANN/4Clients/1.txt.json`，仅本地存在 | 归档条目差集 |
| ADULT 排除 | G1 2 格、G3 34 格、G2 1 格 | 三组清单 |
| FedFair | 144 格保留，3 格 n=1 仅加注 | G2 清单 |
| 补累积是否改变 bs=256 | 不改变：`steps=ceil(256/256)=1` | `tool/amp_utils.py:203` |

### 0.1 “91 格”在 v2 下的分布

| 类别 | 格数 | 处理 |
|---|---|---|
| ADULT | 34 | 不跑（前提 2） |
| 有效批已为 256 的 pilot/旧协议（ACSIncome 36、CREMA-D K4 2、CelebA K2 3、UTKFace K4 1、bios 9） | 51 | 移入 §3，无需跑 |
| 同目录已有完整 3-seed 正本的残留桩 | 5 | 移入 §3，无需跑 |
| 主表真实缺口（PAD-UFES-20/Uniform/FedAvg/K20，I=2 桩） | 1 | **P1，保留** |
| **合计** | **91** | — |

## 1. 优先级规则

- **P0**：G1 已删 + 主表有对应空行/部分空行 → 直接影响主表；实际有效批 64/32（FedFB/FedNova 无累积）。
- **P1**：主表真实缺口（协议未完成导致该 row×split 无完整结果）。
- **P2**：已删但不在主表（CREMA-D Dirichlet1/Uniform）。
- **不跑**：ADULT、FedFair、§3 的 56 格。

## 2. 总表（剩余 20 行）

> 2026-11-05 更新：原 49 行里 **29 格已完成**（结果已写入 `result_path` 并同步三端），已从清单移除；剩下 **20 格**见 §2.2。

### 2.1 本轮已完成并移出（29 格）

| 数据集 | split | 算法 | K | 格数 |
|---|---|---|---|---|
| VCTK | Speaker | FedFB | 20/30/40 | 3 |
| VCTK | Speaker | FedNova | 20/30/40 | 3 |
| AudioMNIST | Dirichlet01/Dirichlet05/Dirichlet1/Uniform | FedFB | 20 | 4（**退化/不可用**，见 `_report_tables.md`） |
| AudioMNIST | Dirichlet01/Dirichlet05/Dirichlet1/Uniform | FedNova | 20 | 4 |
| RAVDESS | Speaker | FedFB / FedNova | 20 | 2 |
| CREMA-D | Speaker + Dirichlet1 + Uniform | FedFB / FedNova | 20 | 6 |
| FairFace | Dirichlet01 | FedFB | 30 | 1 |
| LFWA+ | Dirichlet01/Dirichlet05/Dirichlet1/Uniform | FedNova | 30/40 | 6 |
| **合计** | | | | **29** |

### 2.2 剩余待跑（20 格）

| # | 数据集 | split | 算法 | 模型 | K | 名义 bs | 现有 seeds | 待补 seeds | 优先级 | 状态（2026-11-05） |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | RAVDESS_ar | Speaker | FedFB | AUDIOCNNRNN | 20 | 64 | 0 | 3 | P0 | 未启动 |
| 2 | RAVDESS_ar | Speaker | FedNova | AUDIOCNNRNN | 20 | 64 | 0 | 3 | P0 | 未启动 |
| 3 | CREMA-D_ar | Speaker | FedFB | AUDIOCNNRNN | 20 | 64 | 0 | 3 | P0 | 未启动 |
| 4 | CREMA-D_ar | Speaker | FedNova | AUDIOCNNRNN | 20 | 64 | 0 | 3 | P0 | 未启动 |
| 5 | CREMA-D_gender | Speaker | FedFB | AUDIOCNNRNN | 20 | 64 | 0 | 3 | P0 | 未启动 |
| 6 | CREMA-D_gender | Speaker | FedNova | AUDIOCNNRNN | 20 | 64 | 0 | 3 | P0 | 未启动 |
| 7 | FairFace | Dirichlet01 | FedNova | BERTCLASSIFIER | 30 | 64 | 2（D01/D05） | 补 D1/Uni | P0 | 部分完成，未在停机名单内 |
| 8 | FairFace | Dirichlet05 | FedFB | BERTCLASSIFIER | 30 | 64 | — | 3 | P0 | **已停止**（`.31` 36/50，断点 31） |
| 9 | FairFace | Dirichlet05 | FedNova | BERTCLASSIFIER | 20 | 64 | — | 3 | P0 | **已停止**（`.30` 50/50，断点 41，收尾缺 metrics） |
| 10 | FairFace | Dirichlet05 | FedNova | BERTCLASSIFIER | 30 | 64 | — | 3 | P0 | **已停止**（`.30` 43/50，断点 41） |
| 11 | FairFace | Dirichlet1 | FedNova | BERTCLASSIFIER | 30 | 64 | — | 3 | P0 | **已停止**（`.30` 45/50，断点 41） |
| 12 | FairFace | Uniform | FedFB | BERTCLASSIFIER | 30 | 64 | — | 3 | P0 | **已停止**（`.30` 15/50，断点 11） |
| 13 | FairFace | Uniform | FedNova | BERTCLASSIFIER | 30 | 64 | — | 3 | P0 | **已停止**（`.30` 35/50，断点 31） |
| 14 | LFWA+ | Dirichlet01 | FedFB | BERTCLASSIFIER | 40 | 64 | 0 | 3 | P0 | 未启动 |
| 15 | LFWA+ | Dirichlet05 | FedFB | BERTCLASSIFIER | 20 | 32 | 0 | 3 | P0 | 未启动 |
| 16 | LFWA+ | Dirichlet05 | FedFB | BERTCLASSIFIER | 40 | 64 | 0 | 3 | P0 | 未启动 |
| 17 | LFWA+ | Dirichlet1 | FedFB | BERTCLASSIFIER | 30 | 32 | 0 | 3 | P0 | 未启动 |
| 18 | LFWA+ | Dirichlet1 | FedFB | BERTCLASSIFIER | 40 | 64 | 0 | 3 | P0 | 未启动 |
| 19 | LFWA+ | Uniform | FedFB | BERTCLASSIFIER | 20 | 32 | 0 | 3 | P0 | 未启动 |
| 20 | PAD-UFES-20 | Uniform | FedAvg | BERTCLASSIFIER | 20 | 32 | 0（I=2 桩） | 3 | P1 | 未启动 |

### 2.3 本轮被停机中断的全部实验作业（30 格，已停止、可 `-resume` 续跑）

> 断点文件均存在且 hash 与日志末条 boundary 一致（详见 `experiments/progress_snapshot_20261105.md` §3/§7）。`需补轮数 = 总轮次 − 断点轮次`，续跑会从断点重放该区间。

| # | 主机 | 数据集 | split | 算法 | K | 停时进度 | repeat | 断点轮次 | 需补轮数 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | .30 | ACSIncome | Dirichlet01 | FL_FairBatch | 20 | 112/500 | 2/3 | 111 | 389 |
| 2 | .30 | ACSIncome | Dirichlet01 | FL_FairBatch | 30 | 89/500 | 2/3 | 81 | 419 |
| 3 | .30 | ACSIncome | Dirichlet01 | FL_FairBatch | 40 | 89/500 | 2/3 | 81 | 419 |
| 4 | .30 | ACSIncome | Dirichlet01 | FairFed | 40 | 254/500 | 3/3 | 251 | 249 |
| 5 | .30 | ACSIncome | Dirichlet01 | FedFB | 20 | 190/500 | 1/3 | 181 | 319 |
| 6 | .30 | ACSIncome | Dirichlet01 | FedFB | 30 | 457/500 | 1/3 | 451 | 49 |
| 7 | .30 | ACSIncome | Dirichlet01 | FedProx | 40 | 179/500 | 1/3 | 171 | 329 |
| 8 | .30 | ACSIncome | Dirichlet01 | FedRep | 20 | 318/500 | 1/3 | 311 | 189 |
| 9 | .30 | ACSIncome | Dirichlet01 | FedRep | 30 | 296/500 | 1/3 | 291 | 209 |
| 10 | .30 | ACSIncome | Dirichlet01 | FedRep | 40 | 381/500 | 2/3 | 371 | 129 |
| 11 | .30 | ACSIncome | Dirichlet05 | FedProx | 30 | 355/500 | 2/3 | 351 | 149 |
| 12 | .30 | ACSIncome | Dirichlet05 | FederatedAverageWithPo | 20 | 484/500 | 2/3 | 481 | 19 |
| 13 | .30 | ACSIncome | Dirichlet1 | FL_FairBatch | 30 | 94/500 | 2/3 | 91 | 409 |
| 14 | .30 | ACSIncome | Dirichlet1 | FairFed | 20 | 231/500 | 3/3 | 221 | 279 |
| 15 | .30 | ACSIncome | Dirichlet1 | FairFed | 40 | 245/500 | 3/3 | 241 | 259 |
| 16 | .30 | ACSIncome | Dirichlet1 | FedRep | 20 | 478/500 | 2/3 | 471 | 29 |
| 17 | .30 | ACSIncome | Uniform | FedProx | 30 | 240/500 | 3/3 | 231 | 269 |
| 18 | .30 | ACSIncome | Uniform | FedRep | 30 | 258/500 | 2/3 | 251 | 249 |
| 19 | .30 | FairFace | Dirichlet05 | FedNova | 20 | 50/50 | 1/3 | 41 | 9（收尾未写 metrics） |
| 20 | .30 | FairFace | Dirichlet05 | FedNova | 30 | 43/50 | 1/3 | 41 | 9 |
| 21 | .30 | FairFace | Dirichlet1 | FedNova | 30 | 45/50 | 1/3 | 41 | 9 |
| 22 | .30 | FairFace | Uniform | FedFB | 30 | 15/50 | 1/3 | 11 | 39 |
| 23 | .30 | FairFace | Uniform | FedNova | 30 | 35/50 | 1/3 | 31 | 19 |
| 24 | .31 | VCTK | Speaker | FedRenyi | 20 | 159/200 | 2/3 | 151 | 49 |
| 25 | .31 | ACSIncome | Dirichlet01 | FedRenyi | 20 | 68/500 | 1/3 | 61 | 439 |
| 26 | .31 | ACSIncome | Dirichlet01 | FedRenyi | 40 | 70/500 | 1/3 | 61 | 439 |
| 27 | .31 | ACSIncome | Dirichlet01 | LoGoFair | 30 | 227/500 | 2/3 | 221 | 279 |
| 28 | .31 | ACSIncome | Dirichlet05 | FL_FairBatch | 20 | 230/500 | 2/3 | 221 | 279 |
| 29 | .31 | ACSIncome | Dirichlet05 | FedRep | 30 | 402/500 | 1/3 | 401 | 99 |
| 30 | .31 | FairFace | Dirichlet05 | FedFB | 30 | 36/50 | 3/3 | 31 | 19 |

> 注：上述 30 格中只有第 19–23、30 共 **6 格**属于本清单（§2.2 的 #8–#13）；其余 24 格是主表已完成/在跑的补充格，被停机一起打断。
> 说明：用户口径提到的"5 格 FairFace 在 `.30` 起跑、进度 4–15/50"与实测不符 —— 实测 `.30` 的 5 格 FairFace 停时进度为 **50/50、43/50、45/50、15/50、35/50**（只有 Uniform/FedFB/K30 处于 15/50），此处以实测为准。
>
> **2026-11-05 22:20 更新（第 24 格 VCTK）**：曾续跑至 Round 194（断点推进到 `next_round=190`），但 `.31` GPU0–2 被 FairFace/PAD 占满后无 3 张各留 ≥6 GB 的卡，OOM 中止。**按用户要求现已搁置**：不再尝试、不设自动触发、不占任何卡，待 PAD/FairFace 全部收尾后再议。届时若重启仍需 3 卡同时空闲（断点 `rng_state.cuda_device_count=3`，单卡续跑抛 `CheckpointCompatibilityError`）。代码仍在 `/root/fl_vctk31`（VCTK 专用，未删）。

### 2.4 分组小计（v3）

| 分组 | 格数 | 运行次数 | 状态 |
|---|---|---|---|
| 本轮已完成（移出） | 29 | 87 | 结果已落盘并同步 |
| 已停止可续跑 | 6 | 18 | 有断点，`-resume` |
| 未启动 | 14 | 42 | 待投递 |
| **合计（未完成）** | **20** | **60** | — |
按优先级：**P0 = 44 格 / 132 次 / 506.2 GPU·h**；**P1 = 1 格 / 3 次 / 0.01 GPU·h**；**P2 = 4 格 / 12 次 / 0.33 GPU·h**。

### 2.2 维度汇总

#### 按数据集

| 数据集 | P0 | P1 | P2 | 合计 | 运行次数 | 可量化 GPU·h |
|---|---|---|---|---|---|---|
| VCTK | 6 | 0 | 0 | 6 | 18 | 74.2 |
| AudioMNIST | 8 | 0 | 0 | 8 | 24 | 48.9 |
| RAVDESS | 2 | 0 | 0 | 2 | 6 | 12.8 |
| RAVDESS_ar | 2 | 0 | 0 | 2 | 6 | 1.4 |
| CREMA-D | 2 | 0 | 4 | 6 | 18 | 14.3 |
| CREMA-D_ar | 2 | 0 | 0 | 2 | 6 | 11.1 |
| CREMA-D_gender | 2 | 0 | 0 | 2 | 6 | 8.5 |
| FairFace | 8 | 0 | 0 | 8 | 24 | 296.6 |
| LFWA+ | 12 | 0 | 0 | 12 | 36 | 38.8 |
| PAD-UFES-20 | 0 | 1 | 0 | 1 | 3 | — |

#### 按算法

| 算法 | P0 | P1 | P2 | 合计 |
|---|---|---|---|---|
| FedNova | 23 | 0 | 2 | 25 |
| FedFB | 21 | 0 | 2 | 23 |
| FedAvg | 0 | 1 | 0 | 1 |

## 3. 已复核无需重跑（56 格，含本轮剔除的 51 格误列）

### 3.1 误列清单：有效批已为 256 的 pilot/旧协议（51 格）

| # | 数据集 | split | 算法（结果目录） | K | 算法类 | 名义 bs | 累积步数 | 有效批 | 协议轮数 | 为什么不该跑 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | CREMA-D | Uniform | FederatedAverage | 4 | A | 16 | 16 | 256 | 2/200 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 2 | CREMA-D | Uniform | PDFFed | 4 | A | 16 | 16 | 256 | 2/200 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 3 | CelebA | Dirichlet01 | FL_FairBatch | 2 | A | 256 | 1 | 256 | 1/50 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 4 | CelebA | Dirichlet05 | FL_FairBatch | 2 | A | 256 | 1 | 256 | 1/50 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 5 | CelebA | Dirichlet1 | FL_FairBatch | 2 | A | 256 | 1 | 256 | 1/50 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 6 | UTKFace | Dirichlet05 | FedAvg | 4 | A | 64 | 4 | 256 | 2/50 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 7 | bios | Dirichlet01 | FairFed | 10 | A | 256 | 1 | 256 | 6/10 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 8 | bios | Dirichlet01 | FedAvg | 10 | A | 256 | 1 | 256 | 6/10 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 9 | bios | Dirichlet01 | FedRep | 10 | A | 256 | 1 | 256 | 6/10 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 10 | bios | Dirichlet01 | LoGoFair | 10 | A | 256 | 1 | 256 | 6/10 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 11 | bios | Dirichlet05 | FedAvg | 4 | A | 16 | 16 | 256 | 2/10 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 12 | bios | Uniform | FairFed | 10 | A | 256 | 1 | 256 | 6/10 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 13 | bios | Uniform | FedAvg | 10 | A | 256 | 1 | 256 | 6/10 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 14 | bios | Uniform | FedRep | 10 | A | 256 | 1 | 256 | 6/10 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 15 | bios | Uniform | LoGoFair | 10 | A | 256 | 1 | 256 | 6/10 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 16 | ACSIncome | Dirichlet01 | FL_FairBatch（FL_FairBatch_acspilot） | 20 | A | 256 | 1 | 256 | 20/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 17 | ACSIncome | Dirichlet01 | FairFed（FairFed_acspilot） | 20 | A | 256 | 1 | 256 | 20/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 18 | ACSIncome | Dirichlet01 | FedAvg（FedAvg_acsbench2） | 20 | A | 256 | 1 | 256 | 6/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 19 | ACSIncome | Dirichlet01 | FedAvg（FedAvg_acspilot） | 20 | A | 256 | 1 | 256 | 20/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 20 | ACSIncome | Dirichlet01 | FedAvg（FedAvg_acssmoke） | 20 | A | 256 | 1 | 256 | 3/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 21 | ACSIncome | Dirichlet01 | FedProto（FedProto_acspilot） | 20 | A | 256 | 1 | 256 | 20/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 22 | ACSIncome | Dirichlet01 | FedProx（FedProx_acspilot） | 20 | A | 256 | 1 | 256 | 20/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 23 | ACSIncome | Dirichlet01 | FedRenyi（FedRenyi_acsrenyi） | 20 | A | 256 | 1 | 256 | 2/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 24 | ACSIncome | Dirichlet01 | PDFFed_DP（PDFFed_DP_acssmoke_pdffed_dp） | 20 | A | 256 | 1 | 256 | 2/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 25 | ACSIncome | Dirichlet01 | mFairFL（mFairFL_acssmoke_mfairfl） | 20 | A | 256 | 1 | 256 | 2/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 26 | ACSIncome | Uniform | FedAvg（FedAvg_nwbench_cp1_nw0） | 20 | A | 256 | 1 | 256 | 3/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 27 | ACSIncome | Uniform | FedAvg（FedAvg_nwbench_cp1_nw2） | 20 | A | 256 | 1 | 256 | 3/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 28 | ACSIncome | Uniform | FedAvg（FedAvg_nwbench_cp2_nw0） | 20 | A | 256 | 1 | 256 | 3/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 29 | ACSIncome | Uniform | FedAvg（FedAvg_nwbench_cp2_nw2） | 20 | A | 256 | 1 | 256 | 3/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 30 | ACSIncome | Uniform | FedAvg（FedAvg_nwlong_cp1_nw0） | 20 | A | 256 | 1 | 256 | 9/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 31 | ACSIncome | Uniform | FedAvg（FedAvg_nwlong_cp1_nw2） | 20 | A | 256 | 1 | 256 | 9/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 32 | ACSIncome | Uniform | FedAvg（FedAvg_nwlong_cp2_nw0） | 20 | A | 256 | 1 | 256 | 9/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 33 | ACSIncome | Uniform | FedAvg（FedAvg_nwlong_cp2_nw2） | 20 | A | 256 | 1 | 256 | 9/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 34 | ACSIncome | Uniform | FedFB（FedFB_acsmoke_fedfb2） | 20 | B | 256 | 1 | 256 | 2/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 35 | ACSIncome | Uniform | FedFB（FedFB_acssmoke_fedfb3） | 20 | B | 256 | 1 | 256 | 2/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 36 | ACSIncome | Uniform | FedProx（FedProx_acsbench4n0） | 20 | A | 256 | 1 | 256 | 1/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 37 | ACSIncome | Uniform | FedProx（FedProx_acsbench4n2） | 20 | A | 256 | 1 | 256 | 1/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 38 | ACSIncome | Uniform | PDFFed（PDFFed_acsbench3） | 20 | A | 256 | 1 | 256 | 1/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 39 | ACSIncome | Uniform | PDFFed（PDFFed_finnw_a1_cp2_nw0） | 20 | A | 256 | 1 | 256 | 6/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 40 | ACSIncome | Uniform | PDFFed（PDFFed_finnw_a1_cp2_nw2） | 20 | A | 256 | 1 | 256 | 6/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 41 | ACSIncome | Uniform | PDFFed（PDFFed_finnw_b1_cp2_nw0） | 20 | A | 256 | 1 | 256 | 6/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 42 | ACSIncome | Uniform | PDFFed（PDFFed_finnw_b1_cp2_nw2） | 20 | A | 256 | 1 | 256 | 6/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 43 | ACSIncome | Uniform | PDFFed（PDFFed_pdfabv2_a_cp2_nw0） | 20 | A | 256 | 1 | 256 | 4/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 44 | ACSIncome | Uniform | PDFFed（PDFFed_pdfabv2_a_cp2_nw2） | 20 | A | 256 | 1 | 256 | 4/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 45 | ACSIncome | Uniform | PDFFed（PDFFed_pdfabv2_b_cp2_nw0） | 20 | A | 256 | 1 | 256 | 4/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 46 | ACSIncome | Uniform | PDFFed（PDFFed_pdfabv2_b_cp2_nw2） | 20 | A | 256 | 1 | 256 | 4/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 47 | ACSIncome | Uniform | PDFFed（PDFFed_pdfla_cp2_nw0） | 20 | A | 256 | 1 | 256 | 12/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 48 | ACSIncome | Uniform | PDFFed（PDFFed_pdfla_cp2_nw2） | 20 | A | 256 | 1 | 256 | 12/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 49 | ACSIncome | Uniform | PDFFed（PDFFed_pdfnw_cp1_nw0） | 20 | A | 256 | 1 | 256 | 4/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 50 | ACSIncome | Uniform | PDFFed（PDFFed_pdfnw_cp2_nw0） | 20 | A | 256 | 1 | 256 | 4/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |
| 51 | ACSIncome | Uniform | PDFFed（PDFFed_pdfnw_cp2_nw2） | 20 | A | 256 | 1 | 256 | 4/500 | 批量无缺陷（已 256）+ 非主表 pilot/旧协议 + 协议未完成 |

**逐格判定依据（分组）**：
- ACSIncome 36 格：全部 `bs=256 → 步数 1 → 有效批 256`；其中 2 格是 FedFB（B 类，但 bs=256 仍 1 步）；且都是带后缀的 pilot/调参目录（`*_acsbench*`/`*_acssmoke`/`*_nwbench*`/`*_pdfnw*`），主表用无后缀目录。
- bios 9 格：见上文，8 格 256/1 + 1 格 16/16，全部 256；K=10 为早期试点、K=4 为旧协议。
- CREMA-D K4 2 格：bs=16×16 步=256，旧 4-client 协议，主表只有 Speaker/K20。
- CelebA K2 3 格：bs=256×1 步，旧 2-client 协议。
- UTKFace K4 1 格：bs=64×4 步=256，旧 4-client 协议。

### 3.2 残留桩：同目录已有完整 3-seed 正本（5 格）

| 残留桩 | 正本 | 类 | 名义 bs | 累积步数 | 有效批 |
|---|---|---|---|---|---|
| `LFWA+/Dirichlet05/FedAvg/BERTCLASSIFIER/20Clients/1.txt.json` | `4.txt.json` | A | 256 | 1 | 256 |
| `CelebA/Dirichlet05/FedAvg/BERTCLASSIFIER/20Clients/1.txt.json` | `4.txt.json` | A | 256 | 1 | 256 |
| `CelebA/Uniform/FedProto/BERTCLASSIFIER/20Clients/10.txt.json` | `1.txt.json` | A | 64 | 4 | 256 |
| `CelebA/Uniform/FedProto/BERTCLASSIFIER/30Clients/11.txt.json` | `1.txt.json` | A | 64 | 4 | 256 |
| `CelebA/Uniform/FedProto/BERTCLASSIFIER/40Clients/12.txt.json` | `1.txt.json` | A | 64 | 4 | 256 |

> 这 5 格的主表数值来自正本（3 seeds），残留桩不影响主表，也不需要补跑。

## 4. 结论（v3 修订后）

**剩余待跑 20 格 / 60 次：其中 6 格已停止可 `-resume`（18 次，有断点），14 格未启动（42 次）。v2 的 49 格中 29 格已完成并移出。**

- 相比 v2：格数 49 → 20（−29），运行次数 147 → 60（−87）。
- 剩余缺口按数据集：FairFace 6 格（其中 6 格已停止/部分完成）+ LFWA+ 6 格 + 音频补充 split 6 格（RAVDESS_ar 2、CREMA-D_ar 2、CREMA-D_gender 2）+ PAD-UFES-20 1 格。
- 已完成的 29 格中，`AudioMNIST/*/FedFB/K20` 4 格为平凡解（ACC≈0.5、ACC_gap≈0），标"退化/不可用"保留数据。

## 5. 估算依据与不确定性

1. P0/P2 的 GPU·h = 归档 JSON `aggregate.gpu_seconds` 三 seed 求和（实测落盘值），与 `batch_audit.md` §4.3 的 506.6 对齐。
2. P1 的 0.01 h 取自同配置完整 3-seed 作业（PAD FedAvg K20 实测 14.9/12.0/14.5 s）。
3. 被剔除的 51 格在 v1 就标“待实测”（协议提前终止的记录无法外推），因此本次剔除不改变可量化的 GPU·h。
4. `gpu_seconds` 是共享 GPU 墙钟累计，P0 的 506 GPU·h 是偏保守上限。

## 6. 范围边界

- 不含“从未跑过”的协议空格（ACSIncome 69/192、PAD 仅 Dirichlet01、VCTK FedRenyi 等，见 `_report_tables.md`）。
- 不含 FedFair（144 格保留，3 格 n=1 加注）。
- 不含 ADULT。
