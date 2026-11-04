# 待跑任务清单（pending runs，只读统计）

生成时间：2026-11-04。只读扫描 `result_path/`、`results/_archive/`、`experiments/batch_delete_manifest.md`、`experiments/batch_audit.md`、`_report_tables.md`；**未改代码、未启动训练、未删除任何文件**。

> 口径：一个“格子” = `result_path/<数据集>/<split>/<算法>/<模型>/<K>Clients/` 下的一份结果。`名义 bs` 取自 `experiment_config.batch_size`，`目标有效批` 统一 256。

## 0. 前提复述与核对结果

| # | 前提 | 核对结论 | 证据 |
|---|---|---|---|
| 1 | G1 已删除并存档（50 格） | **并集 50 格**：本地 50、`.30` 49、`.31` 49 | 本地归档 50 条目；`batch_inconsistent_..._30.tar.gz` / `_31.tar.gz` 各 49 条目；`_g1_30.txt` 与 `_g1_31.txt` 各 49 行且 `diff=0` |
| 1a | 50 vs 49×2 的差异原因 | 本地多出的唯一一格 = `ADULT/Uniform/FedFB/ANN/4Clients/1.txt.json`，**仅本地存在**，两台远端从未落盘 | 归档条目差集 `local(50) - remote(49)` |
| 2 | ADULT 一律不重跑 | 已排除：G1 2 格、G3 34 格、G2 1 格 | 三组清单 |
| 3 | FedFair 144 格保留、不重跑 | 已保留：144 格全在盘，仅 3 格 n=1（`ADULT/Uniform` K4、`bios/Dirichlet01` K10、`bios/Uniform` K10），仅加注 | 同上 |
| 4 | 其余 FedFB/FedNova 只重跑 bs≠256 | **非 ADULT 的 G1 共 48 格，全部 bs≠256**（43 格 bs=64、5 格 bs=32），没有 bs=256 的 FedFB/FedNova 格 | 归档 JSON 的 `batch_size` |
| 4a | 补累积是否改变 bs=256 行为 | **不会，无需扩大清单**。`accumulation_steps = max(1, 256/bs)`，bs=256 → 1，前向/反向/优化器步数与现网一致 | `batch_audit.md` §2.2：A 类算法均为 `256/1/256 ✔` |
| 5 | 单 seed 格 ≈91–96 | **准确 96 = 91（G3）+ 2（G1，均 ADULT）+ 3（G2 FedFair）**；91 再拆：34 ADULT + 57 可动 | G3 清单 91 行；G1 seeds=1 仅 2 行；G2 seeds=1 共 3 行 |

### 0.1 “91 格”逐类核对（本清单最重要的修正）

| G3 的 91 格细分 | 格数 | 处理 |
|---|---|---|
| ADULT（前提 2） | 34 | 不跑 |
| 同目录已有完整 3-seed 结果的重复残留 | 5 | **无需补跑**，见 §2.3 |
| 主表真实缺口（PAD-UFES-20/Uniform/FedAvg/K20，I=2 桩） | 1 | P1 |
| 非主表 pilot/旧协议（ACSIncome 36、CREMA-D K4 2、CelebA K2 3、UTKFace K4 1、bios K10/K4 9） | 51 | P2 |
| **合计** | **91** | — |

**另一个必须指出的问题**：可动的 57 格**全部不是“只差 2 个 seed”，而是协议提前终止 + 只有 1 个 seed**（`CelebA/Uniform/FedProto/K20` 只跑到 I=2/50、`bios` K10 只跑到 I=6/10、`ACSIncome` pilot 只跑到 I=1~20/500）。所以“补齐 3 seeds”实际是**整格按完整协议重跑 3 次**，不能按 +2 seed 计价。

## 1. 优先级规则（本清单采用）

- **P0**：G1 已删，且该格在 `_report_tables.md` 主比较表中有对应空行/部分空行（`已移除（有效批≠256，待重跑）` / `部分已移除`）→ 直接影响主表对比结论。
- **P1**：主表真实缺口——该 row×split 无完整协议结果（当前括号值不计入均值），补上会改变主表均值覆盖。
- **P2**：不在主表或属旧协议/pilot（CREMA-D 的 Dirichlet1/Uniform split、ACSIncome pilot 目录、K2/K4/K10 旧协议、bios 试点）。
- **不跑**：ADULT（前提 2）；FedFair（前提 3）；5 个重复残留桩（已有 3-seed 正本）。

## 2. 总表（100 行）

| # | 数据集 | split | 算法（结果目录） | 模型 | K | 名义 bs | 目标有效批 | 现有 seeds | 待补 seeds | 优先级 | 预估 GPU·h | 依据 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | VCTK | Speaker | FedFB | AUDIOCNNRNN | 20 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 6.40 | 已删待重跑；主表有对应空行 |
| 2 | VCTK | Speaker | FedFB | AUDIOCNNRNN | 30 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 3.95 | 已删待重跑；主表有对应空行 |
| 3 | VCTK | Speaker | FedFB | AUDIOCNNRNN | 40 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 4.63 | 已删待重跑；主表有对应空行 |
| 4 | VCTK | Speaker | FedNova | AUDIOCNNRNN | 20 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 30.9 | 已删待重跑；主表有对应空行 |
| 5 | VCTK | Speaker | FedNova | AUDIOCNNRNN | 30 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 14.4 | 已删待重跑；主表有对应空行 |
| 6 | VCTK | Speaker | FedNova | AUDIOCNNRNN | 40 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 13.9 | 已删待重跑；主表有对应空行 |
| 7 | AudioMNIST | Dirichlet01 | FedFB | AUDIOCNNRNN | 20 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 1.18 | 已删待重跑；主表有对应空行 |
| 8 | AudioMNIST | Dirichlet01 | FedNova | AUDIOCNNRNN | 20 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 11.5 | 已删待重跑；主表有对应空行 |
| 9 | AudioMNIST | Dirichlet05 | FedFB | AUDIOCNNRNN | 20 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 1.13 | 已删待重跑；主表有对应空行 |
| 10 | AudioMNIST | Dirichlet05 | FedNova | AUDIOCNNRNN | 20 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 10.9 | 已删待重跑；主表有对应空行 |
| 11 | AudioMNIST | Dirichlet1 | FedFB | AUDIOCNNRNN | 20 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 1.16 | 已删待重跑；主表有对应空行 |
| 12 | AudioMNIST | Dirichlet1 | FedNova | AUDIOCNNRNN | 20 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 11.2 | 已删待重跑；主表有对应空行 |
| 13 | AudioMNIST | Uniform | FedFB | AUDIOCNNRNN | 20 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 1.15 | 已删待重跑；主表有对应空行 |
| 14 | AudioMNIST | Uniform | FedNova | AUDIOCNNRNN | 20 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 10.8 | 已删待重跑；主表有对应空行 |
| 15 | RAVDESS | Speaker | FedFB | AUDIOCNNRNN | 20 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 0.72 | 已删待重跑；主表有对应空行 |
| 16 | RAVDESS | Speaker | FedNova | AUDIOCNNRNN | 20 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 12.1 | 已删待重跑；主表有对应空行 |
| 17 | RAVDESS_ar | Speaker | FedFB | AUDIOCNNRNN | 20 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 0.66 | 已删待重跑；主表有对应空行 |
| 18 | RAVDESS_ar | Speaker | FedNova | AUDIOCNNRNN | 20 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 0.74 | 已删待重跑；主表有对应空行 |
| 19 | CREMA-D | Speaker | FedFB | AUDIOCNNRNN | 20 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 2.08 | 已删待重跑；主表有对应空行 |
| 20 | CREMA-D | Speaker | FedNova | AUDIOCNNRNN | 20 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 11.9 | 已删待重跑；主表有对应空行 |
| 21 | CREMA-D_ar | Speaker | FedFB | AUDIOCNNRNN | 20 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 2.17 | 已删待重跑；主表有对应空行 |
| 22 | CREMA-D_ar | Speaker | FedNova | AUDIOCNNRNN | 20 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 8.89 | 已删待重跑；主表有对应空行 |
| 23 | CREMA-D_gender | Speaker | FedFB | AUDIOCNNRNN | 20 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 5.88 | 已删待重跑；主表有对应空行 |
| 24 | CREMA-D_gender | Speaker | FedNova | AUDIOCNNRNN | 20 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 2.66 | 已删待重跑；主表有对应空行 |
| 25 | FairFace | Dirichlet01 | FedFB | BERTCLASSIFIER | 30 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 16.5 | 已删待重跑；主表有对应空行 |
| 26 | FairFace | Dirichlet01 | FedNova | BERTCLASSIFIER | 30 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 58.4 | 已删待重跑；主表有对应空行 |
| 27 | FairFace | Dirichlet05 | FedFB | BERTCLASSIFIER | 30 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 14.3 | 已删待重跑；主表有对应空行 |
| 28 | FairFace | Dirichlet05 | FedNova | BERTCLASSIFIER | 20 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 38.1 | 已删待重跑；主表有对应空行 |
| 29 | FairFace | Dirichlet05 | FedNova | BERTCLASSIFIER | 30 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 50.5 | 已删待重跑；主表有对应空行 |
| 30 | FairFace | Dirichlet1 | FedNova | BERTCLASSIFIER | 30 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 44.3 | 已删待重跑；主表有对应空行 |
| 31 | FairFace | Uniform | FedFB | BERTCLASSIFIER | 30 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 13.2 | 已删待重跑；主表有对应空行 |
| 32 | FairFace | Uniform | FedNova | BERTCLASSIFIER | 30 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 61.2 | 已删待重跑；主表有对应空行 |
| 33 | LFWA+ | Dirichlet01 | FedFB | BERTCLASSIFIER | 40 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 1.63 | 已删待重跑；主表有对应空行 |
| 34 | LFWA+ | Dirichlet01 | FedNova | BERTCLASSIFIER | 30 | 32 | 256 | 0（已删，归档3） | 3 | P0 | 3.38 | 已删待重跑；主表有对应空行 |
| 35 | LFWA+ | Dirichlet01 | FedNova | BERTCLASSIFIER | 40 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 4.97 | 已删待重跑；主表有对应空行 |
| 36 | LFWA+ | Dirichlet05 | FedFB | BERTCLASSIFIER | 20 | 32 | 256 | 0（已删，归档3） | 3 | P0 | 1.84 | 已删待重跑；主表有对应空行 |
| 37 | LFWA+ | Dirichlet05 | FedFB | BERTCLASSIFIER | 40 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 1.41 | 已删待重跑；主表有对应空行 |
| 38 | LFWA+ | Dirichlet05 | FedNova | BERTCLASSIFIER | 40 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 4.81 | 已删待重跑；主表有对应空行 |
| 39 | LFWA+ | Dirichlet1 | FedFB | BERTCLASSIFIER | 30 | 32 | 256 | 0（已删，归档3） | 3 | P0 | 0.64 | 已删待重跑；主表有对应空行 |
| 40 | LFWA+ | Dirichlet1 | FedFB | BERTCLASSIFIER | 40 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 1.02 | 已删待重跑；主表有对应空行 |
| 41 | LFWA+ | Dirichlet1 | FedNova | BERTCLASSIFIER | 30 | 32 | 256 | 0（已删，归档3） | 3 | P0 | 3.07 | 已删待重跑；主表有对应空行 |
| 42 | LFWA+ | Dirichlet1 | FedNova | BERTCLASSIFIER | 40 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 5.82 | 已删待重跑；主表有对应空行 |
| 43 | LFWA+ | Uniform | FedFB | BERTCLASSIFIER | 20 | 32 | 256 | 0（已删，归档3） | 3 | P0 | 1.71 | 已删待重跑；主表有对应空行 |
| 44 | LFWA+ | Uniform | FedNova | BERTCLASSIFIER | 40 | 64 | 256 | 0（已删，归档3） | 3 | P0 | 8.49 | 已删待重跑；主表有对应空行 |
| 45 | PAD-UFES-20 | Uniform | FedAvg | BERTCLASSIFIER | 20 | 32 | 256 | 0（I=2 桩，n=1） | 3 | P1 | 0.01 | 主表真实缺口：主表该 split 无完整协议结果（括号值不计入均值），I=2 桩 |
| 46 | CREMA-D | Dirichlet1 | FedFB | AUDIOCNNRNN | 20 | 64 | 256 | 0（已删，归档3） | 3 | P2 | 0.07 | 已删待重跑；该 split 不在主表（补充实验） |
| 47 | CREMA-D | Dirichlet1 | FedNova | AUDIOCNNRNN | 20 | 64 | 256 | 0（已删，归档3） | 3 | P2 | 0.09 | 已删待重跑；该 split 不在主表（补充实验） |
| 48 | CREMA-D | Uniform | FedFB | AUDIOCNNRNN | 20 | 64 | 256 | 0（已删，归档3） | 3 | P2 | 0.07 | 已删待重跑；该 split 不在主表（补充实验） |
| 49 | CREMA-D | Uniform | FedNova | AUDIOCNNRNN | 20 | 64 | 256 | 0（已删，归档3） | 3 | P2 | 0.10 | 已删待重跑；该 split 不在主表（补充实验） |
| 50 | CREMA-D | Uniform | FederatedAverage | AUDIOCNNRNN | 4 | 16 | 256 | 1（I=2/200，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=2/200）；旧协议/pilot，不进主表 |
| 51 | CREMA-D | Uniform | PDFFed | AUDIOCNNRNN | 4 | 16 | 256 | 1（I=2/200，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=2/200）；旧协议/pilot，不进主表 |
| 52 | CelebA | Dirichlet01 | FL_FairBatch | BERTCLASSIFIER | 2 | 256 | 256 | 1（I=1/50，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=1/50）；旧协议/pilot，不进主表 |
| 53 | CelebA | Dirichlet05 | FL_FairBatch | BERTCLASSIFIER | 2 | 256 | 256 | 1（I=1/50，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=1/50）；旧协议/pilot，不进主表 |
| 54 | CelebA | Dirichlet1 | FL_FairBatch | BERTCLASSIFIER | 2 | 256 | 256 | 1（I=1/50，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=1/50）；旧协议/pilot，不进主表 |
| 55 | UTKFace | Dirichlet05 | FedAvg | BERTCLASSIFIER | 4 | 64 | 256 | 1（I=2/50，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=2/50）；旧协议/pilot，不进主表 |
| 56 | bios | Dirichlet01 | FairFed | BERTCLASSIFIER | 10 | 256 | 256 | 1（I=6/10，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=6/10）；旧协议/pilot，不进主表 |
| 57 | bios | Dirichlet01 | FedAvg | BERTCLASSIFIER | 10 | 256 | 256 | 1（I=6/10，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=6/10）；旧协议/pilot，不进主表 |
| 58 | bios | Dirichlet01 | FedRep | BERTCLASSIFIER | 10 | 256 | 256 | 1（I=6/10，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=6/10）；旧协议/pilot，不进主表 |
| 59 | bios | Dirichlet01 | LoGoFair | BERTCLASSIFIER | 10 | 256 | 256 | 1（I=6/10，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=6/10）；旧协议/pilot，不进主表 |
| 60 | bios | Dirichlet05 | FedAvg | BERTCLASSIFIER | 4 | 16 | 256 | 1（I=2/10，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=2/10）；旧协议/pilot，不进主表 |
| 61 | bios | Uniform | FairFed | BERTCLASSIFIER | 10 | 256 | 256 | 1（I=6/10，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=6/10）；旧协议/pilot，不进主表 |
| 62 | bios | Uniform | FedAvg | BERTCLASSIFIER | 10 | 256 | 256 | 1（I=6/10，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=6/10）；旧协议/pilot，不进主表 |
| 63 | bios | Uniform | FedRep | BERTCLASSIFIER | 10 | 256 | 256 | 1（I=6/10，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=6/10）；旧协议/pilot，不进主表 |
| 64 | bios | Uniform | LoGoFair | BERTCLASSIFIER | 10 | 256 | 256 | 1（I=6/10，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=6/10）；旧协议/pilot，不进主表 |
| 65 | ACSIncome | Dirichlet01 | FL_FairBatch（FL_FairBatch_acspilot） | ANN | 20 | 256 | 256 | 1（I=20/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=20/500）；pilot/调参留档，不进主表 |
| 66 | ACSIncome | Dirichlet01 | FairFed（FairFed_acspilot） | ANN | 20 | 256 | 256 | 1（I=20/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=20/500）；pilot/调参留档，不进主表 |
| 67 | ACSIncome | Dirichlet01 | FedAvg（FedAvg_acsbench2） | ANN | 20 | 256 | 256 | 1（I=6/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=6/500）；pilot/调参留档，不进主表 |
| 68 | ACSIncome | Dirichlet01 | FedAvg（FedAvg_acspilot） | ANN | 20 | 256 | 256 | 1（I=20/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=20/500）；pilot/调参留档，不进主表 |
| 69 | ACSIncome | Dirichlet01 | FedAvg（FedAvg_acssmoke） | ANN | 20 | 256 | 256 | 1（I=3/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=3/500）；pilot/调参留档，不进主表 |
| 70 | ACSIncome | Dirichlet01 | FedProto（FedProto_acspilot） | ANN | 20 | 256 | 256 | 1（I=20/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=20/500）；pilot/调参留档，不进主表 |
| 71 | ACSIncome | Dirichlet01 | FedProx（FedProx_acspilot） | ANN | 20 | 256 | 256 | 1（I=20/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=20/500）；pilot/调参留档，不进主表 |
| 72 | ACSIncome | Dirichlet01 | FedRenyi（FedRenyi_acsrenyi） | ANN | 20 | 256 | 256 | 1（I=2/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=2/500）；pilot/调参留档，不进主表 |
| 73 | ACSIncome | Dirichlet01 | PDFFed_DP（PDFFed_DP_acssmoke_pdffed_dp） | ANN | 20 | 256 | 256 | 1（I=2/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=2/500）；pilot/调参留档，不进主表 |
| 74 | ACSIncome | Dirichlet01 | mFairFL（mFairFL_acssmoke_mfairfl） | ANN | 20 | 256 | 256 | 1（I=2/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=2/500）；pilot/调参留档，不进主表 |
| 75 | ACSIncome | Uniform | FedAvg（FedAvg_nwbench_cp1_nw0） | ANN | 20 | 256 | 256 | 1（I=3/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=3/500）；pilot/调参留档，不进主表 |
| 76 | ACSIncome | Uniform | FedAvg（FedAvg_nwbench_cp1_nw2） | ANN | 20 | 256 | 256 | 1（I=3/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=3/500）；pilot/调参留档，不进主表 |
| 77 | ACSIncome | Uniform | FedAvg（FedAvg_nwbench_cp2_nw0） | ANN | 20 | 256 | 256 | 1（I=3/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=3/500）；pilot/调参留档，不进主表 |
| 78 | ACSIncome | Uniform | FedAvg（FedAvg_nwbench_cp2_nw2） | ANN | 20 | 256 | 256 | 1（I=3/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=3/500）；pilot/调参留档，不进主表 |
| 79 | ACSIncome | Uniform | FedAvg（FedAvg_nwlong_cp1_nw0） | ANN | 20 | 256 | 256 | 1（I=9/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=9/500）；pilot/调参留档，不进主表 |
| 80 | ACSIncome | Uniform | FedAvg（FedAvg_nwlong_cp1_nw2） | ANN | 20 | 256 | 256 | 1（I=9/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=9/500）；pilot/调参留档，不进主表 |
| 81 | ACSIncome | Uniform | FedAvg（FedAvg_nwlong_cp2_nw0） | ANN | 20 | 256 | 256 | 1（I=9/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=9/500）；pilot/调参留档，不进主表 |
| 82 | ACSIncome | Uniform | FedAvg（FedAvg_nwlong_cp2_nw2） | ANN | 20 | 256 | 256 | 1（I=9/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=9/500）；pilot/调参留档，不进主表 |
| 83 | ACSIncome | Uniform | FedFB（FedFB_acsmoke_fedfb2） | ANN | 20 | 256 | 256 | 1（I=2/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=2/500）；pilot/调参留档，不进主表 |
| 84 | ACSIncome | Uniform | FedFB（FedFB_acssmoke_fedfb3） | ANN | 20 | 256 | 256 | 1（I=2/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=2/500）；pilot/调参留档，不进主表 |
| 85 | ACSIncome | Uniform | FedProx（FedProx_acsbench4n0） | ANN | 20 | 256 | 256 | 1（I=1/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=1/500）；pilot/调参留档，不进主表 |
| 86 | ACSIncome | Uniform | FedProx（FedProx_acsbench4n2） | ANN | 20 | 256 | 256 | 1（I=1/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=1/500）；pilot/调参留档，不进主表 |
| 87 | ACSIncome | Uniform | PDFFed（PDFFed_acsbench3） | ANN | 20 | 256 | 256 | 1（I=1/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=1/500）；pilot/调参留档，不进主表 |
| 88 | ACSIncome | Uniform | PDFFed（PDFFed_finnw_a1_cp2_nw0） | ANN | 20 | 256 | 256 | 1（I=6/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=6/500）；pilot/调参留档，不进主表 |
| 89 | ACSIncome | Uniform | PDFFed（PDFFed_finnw_a1_cp2_nw2） | ANN | 20 | 256 | 256 | 1（I=6/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=6/500）；pilot/调参留档，不进主表 |
| 90 | ACSIncome | Uniform | PDFFed（PDFFed_finnw_b1_cp2_nw0） | ANN | 20 | 256 | 256 | 1（I=6/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=6/500）；pilot/调参留档，不进主表 |
| 91 | ACSIncome | Uniform | PDFFed（PDFFed_finnw_b1_cp2_nw2） | ANN | 20 | 256 | 256 | 1（I=6/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=6/500）；pilot/调参留档，不进主表 |
| 92 | ACSIncome | Uniform | PDFFed（PDFFed_pdfabv2_a_cp2_nw0） | ANN | 20 | 256 | 256 | 1（I=4/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=4/500）；pilot/调参留档，不进主表 |
| 93 | ACSIncome | Uniform | PDFFed（PDFFed_pdfabv2_a_cp2_nw2） | ANN | 20 | 256 | 256 | 1（I=4/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=4/500）；pilot/调参留档，不进主表 |
| 94 | ACSIncome | Uniform | PDFFed（PDFFed_pdfabv2_b_cp2_nw0） | ANN | 20 | 256 | 256 | 1（I=4/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=4/500）；pilot/调参留档，不进主表 |
| 95 | ACSIncome | Uniform | PDFFed（PDFFed_pdfabv2_b_cp2_nw2） | ANN | 20 | 256 | 256 | 1（I=4/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=4/500）；pilot/调参留档，不进主表 |
| 96 | ACSIncome | Uniform | PDFFed（PDFFed_pdfla_cp2_nw0） | ANN | 20 | 256 | 256 | 1（I=12/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=12/500）；pilot/调参留档，不进主表 |
| 97 | ACSIncome | Uniform | PDFFed（PDFFed_pdfla_cp2_nw2） | ANN | 20 | 256 | 256 | 1（I=12/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=12/500）；pilot/调参留档，不进主表 |
| 98 | ACSIncome | Uniform | PDFFed（PDFFed_pdfnw_cp1_nw0） | ANN | 20 | 256 | 256 | 1（I=4/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=4/500）；pilot/调参留档，不进主表 |
| 99 | ACSIncome | Uniform | PDFFed（PDFFed_pdfnw_cp2_nw0） | ANN | 20 | 256 | 256 | 1（I=4/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=4/500）；pilot/调参留档，不进主表 |
| 100 | ACSIncome | Uniform | PDFFed（PDFFed_pdfnw_cp2_nw2） | ANN | 20 | 256 | 256 | 1（I=4/500，未完成） | 3 | P2 | 待实测 | 单 seed 且协议提前终止（I=4/500）；pilot/调参留档，不进主表 |

### 2.1 分组小计

| 分组 | 格数 | 运行次数（全部按 3 seeds） | 可量化 GPU·h | 待实测格数 |
|---|---|---|---|---|
| 已删待重跑（G1 非 ADULT） | 48 | 144 | 506.5 | 0 |
| 单 seed / 协议未完成需整格重跑 | 52 | 156 | 0.01 | 51 |
| **合计** | **100** | **300** | **506.5** | **51** |

按优先级：**P0 = 44 格 / 132 次 / 506.2 GPU·h**；**P1 = 1 格 / 3 次 / 0.01 GPU·h**；**P2 = 55 格 / 165 次 / 可量化 0.3 GPU·h + 51 格待实测**。

（51 格待实测是因为原记录本身就是“提前终止的单 seed”，其 `gpu_seconds` 对完整协议无参考价值；PAD 那 1 格用同配置完整 3-seed 作业实测 14.9/12.0/14.5 s 推算 ≈0.01 h。）

### 2.2 维度汇总

#### 按数据集

| 数据集 | P0 | P1 | P2 | 合计格数 | 运行次数 | 可量化 GPU·h |
|---|---|---|---|---|---|---|
| VCTK | 6 | 0 | 0 | 6 | 18 | 74.2 |
| AudioMNIST | 8 | 0 | 0 | 8 | 24 | 48.9 |
| RAVDESS | 2 | 0 | 0 | 2 | 6 | 12.8 |
| RAVDESS_ar | 2 | 0 | 0 | 2 | 6 | 1.4 |
| CREMA-D | 2 | 0 | 6 | 8 | 24 | 14.3 |
| CREMA-D_ar | 2 | 0 | 0 | 2 | 6 | 11.1 |
| CREMA-D_gender | 2 | 0 | 0 | 2 | 6 | 8.5 |
| FairFace | 8 | 0 | 0 | 8 | 24 | 296.6 |
| LFWA+ | 12 | 0 | 0 | 12 | 36 | 38.8 |
| CelebA | 0 | 0 | 3 | 3 | 9 | — |
| UTKFace | 0 | 0 | 1 | 1 | 3 | — |
| PAD-UFES-20 | 0 | 1 | 0 | 1 | 3 | — |
| bios | 0 | 0 | 9 | 9 | 27 | — |
| ACSIncome | 0 | 0 | 36 | 36 | 108 | — |

#### 按算法

| 算法 | P0 | P1 | P2 | 合计 |
|---|---|---|---|---|
| FedNova | 23 | 0 | 2 | 25 |
| FedFB | 21 | 0 | 4 | 25 |
| FedAvg | 0 | 1 | 15 | 16 |
| PDFFed | 0 | 0 | 15 | 15 |
| FL_FairBatch | 0 | 0 | 4 | 4 |
| FairFed | 0 | 0 | 3 | 3 |
| FedProx | 0 | 0 | 3 | 3 |
| LoGoFair | 0 | 0 | 2 | 2 |
| FedRep | 0 | 0 | 2 | 2 |
| mFairFL | 0 | 0 | 1 | 1 |
| FederatedAverage | 0 | 0 | 1 | 1 |
| PDFFed_DP | 0 | 0 | 1 | 1 |
| FedRenyi | 0 | 0 | 1 | 1 |
| FedProto | 0 | 0 | 1 | 1 |

### 2.3 无需跑：5 个重复残留（仅登记，用于清理判断）

| 单 seed 残留桩 | 同目录已有完整协议 3-seed 正本 | 正本每 seed 实测 gpu_seconds |
|---|---|---|
| `CelebA/Dirichlet05/FedAvg/BERTCLASSIFIER/20Clients/1.txt.json` | `4.txt.json` | 253/310/283 |
| `CelebA/Uniform/FedProto/BERTCLASSIFIER/20Clients/10.txt.json` | `1.txt.json` | 35950/36422/36289 |
| `CelebA/Uniform/FedProto/BERTCLASSIFIER/30Clients/11.txt.json` | `1.txt.json` | 99037/101917/102620 |
| `CelebA/Uniform/FedProto/BERTCLASSIFIER/40Clients/12.txt.json` | `1.txt.json` | 36097/37140/36675 |
| `LFWA+/Dirichlet05/FedAvg/BERTCLASSIFIER/20Clients/1.txt.json` | `4.txt.json` | 143/139/141 |

> 这 5 格的主表数值来自 3-seed 正本，`n<3` 只是残留桩，不影响主表，也不需要补跑。

## 3. 结论（一句话）

**总待跑 100 格 / 300 次运行；可量化 507 GPU·h，另有 51 格（ACSIncome pilot 36 + 旧协议 15）因原记录是提前终止的单 seed 只能“待实测”。P0 为 44 格 / 132 次 / 506 GPU·h（占可量化工时 99.9%），必须最先做；P1 仅 1 格 0.01 GPU·h；P2 是补充/留档项。**

缺口最大：**FairFace（296.5 GPU·h，8 格）> VCTK（74.2，6 格）> AudioMNIST（48.9，8 格）> LFWA+（38.8，12 格）**。只保主表对比：P0+P1 = 45 格 / 135 次 / 约 506 GPU·h。

### P0 按数据集工时（引用归档实测；与 `batch_audit.md` §4.3 对齐，逐格四舍五入差 ≤0.1）

| 数据集 | 格数（FedFB/FedNova） | 3-seed GPU·h |
|---|---|---|
| VCTK | 6 | 74.2 |
| AudioMNIST | 8 | 48.9 |
| RAVDESS | 2 | 12.8 |
| RAVDESS_ar | 2 | 1.4 |
| CREMA-D | 2 | 14.0 |
| CREMA-D_ar | 2 | 11.1 |
| CREMA-D_gender | 2 | 8.5 |
| FairFace | 8 | 296.6 |
| LFWA+ | 12 | 38.8 |
| **合计** | **44** | **506.2** |

## 4. 估算依据与不确定性

1. **P0 的 GPU·h = 归档 JSON 中 `aggregate.gpu_seconds` 三个 seed 求和**，是真实落盘的实测值，不是公式推算。逐格求和与 `batch_audit.md` §4.3 的 506.6 GPU·h 对齐（VCTK 74.2、FairFace 296.6、LFWA+ 38.8、AudioMNIST 48.9、RAVDESS 12.8、RAVDESS_ar 1.4、CREMA-D_ar 11.1、CREMA-D_gender 8.5；CREMA-D 在 audit 里是 14.3=6 格全量，本清单 P0 只取 Speaker 2 格 = 14.0，另 0.3 归入 P2）。补累积后前向/反向次数不变、仅减少优化器步数，故按“≈原值或略低”计。
2. **P1 的 0.01 h** 取自同数据集/同算法/同 K 的完整 3-seed 作业（PAD FedAvg K20 三个 seed 实测 14.9/12.0/14.5 s）。
3. **“待实测”51 格**：原记录本身就是 `I` 远小于协议轮数的单 seed 冒烟/调参跑（ACSIncome pilot 仅 I=1~20/500、bios K10 仅 I=6/10、CelebA K2 仅 I=1/50），其 `gpu_seconds`（0.8 s~44.7 h/seed）不能外推到完整协议；同配置完整作业的每 seed 实测跨度也达 0.3~240 h（共享卡争用导致 2~3 个数量级抖动），故**不给数字，标“待实测”**。
4. `gpu_seconds` 是共享 GPU 上的墙钟累计（多作业并行会显著偏高），因此 P0 的 506 GPU·h 是**偏保守的上限**；独占一卡会更快。按 8 卡并行，墙钟约 3 天量级（沿用 `batch_audit.md` 的换算）。

## 5. 范围边界（本清单不含）

- 不含“从未跑过”的协议空格（如 ACSIncome 69/192、PAD 仅 Dirichlet01 有结果、VCTK FedRenyi 缺失等），见 `_report_tables.md` 各节“覆盖核查”。
- 不含 FedFair（144 格保留，仅需表注“整轮整片单步/全量”，其中 3 格 n=1 一并加注）。
- 不含 ADULT（前提 2）。
