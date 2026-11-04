# 批量不一致结果删除清单

生成时间：2026-11-04 16:36　｜　口径：一个"格子" = `result_path/<数据集>/<split>/<算法>/<模型>/<K>Clients/1.txt.json` 的一份落盘结果（3 个 seed 合并为 1 格）。
本清单依据 [batch_audit.md](batch_audit.md) 的审计结论生成，**本步只生成清单，未删除任何文件**。

## 组汇总

| 组 | 定义 | 格数 | 文件数 | 总字节 | 折合 | 处理动作 |
|---|---|---|---|---|---|---|
| **G1** | FedFB / FedNova 且名义 bs≠256（有效批 = 名义批） | 50 | 50 | 187,750 | 183.35 KiB | 归档后删除（第 1 步） |
| **G2** | FedFair 全部（整轮整片单步） | 144 | 144 | 563,154 | 549.96 KiB | **暂不删除**，待用户确认 |
| **G3** | 仅 1 个 seed 且有效批 = 256 | 91 | 91 | 264,429 | 258.23 KiB | 不删除，仅标注"待补齐 seed" |
| 合计 | — | 285 | 285 | 1,015,333 | 991.54 KiB | — |

> 关于"96 个单 seed 格子"的口径修正：全库 `ACC.values` 长度 < 3 的格子共 **96** 个，其中 **2** 个属 G1（`ADULT/Uniform/FedFB`、`ADULT/Uniform/FederatedNova`）、**3** 个属 G2（FedFair 旧 K4 单跑），其余 **91** 个才是 G3。三者互斥，不会重复删除。

## G1：FedFB / FedNova 且名义 bs≠256（50 格，待归档并删除）

| # | 数据集 | split | 算法 | 模型 | K | 名义 bs | seeds | 字节 | 本地 | .30 | .31 | 相对路径 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ADULT | Dirichlet05 | FederatedNova | ANN | 4Clients | 128 | 1 | 2,491 | ✔ | ✔ | ✔ | `ADULT/Dirichlet05/FederatedNova/ANN/4Clients/1.txt.json` |
| 2 | ADULT | Uniform | FedFB | ANN | 4Clients | 64 | 1 | 2,948 | ✔ | ✘ | ✘ | `ADULT/Uniform/FedFB/ANN/4Clients/1.txt.json` |
| 3 | AudioMNIST | Dirichlet01 | FedFB | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,679 | ✔ | ✔ | ✔ | `AudioMNIST/Dirichlet01/FedFB/AUDIOCNNRNN/20Clients/1.txt.json` |
| 4 | AudioMNIST | Dirichlet01 | FedNova | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,880 | ✔ | ✔ | ✔ | `AudioMNIST/Dirichlet01/FedNova/AUDIOCNNRNN/20Clients/1.txt.json` |
| 5 | AudioMNIST | Dirichlet05 | FedFB | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,771 | ✔ | ✔ | ✔ | `AudioMNIST/Dirichlet05/FedFB/AUDIOCNNRNN/20Clients/1.txt.json` |
| 6 | AudioMNIST | Dirichlet05 | FedNova | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,921 | ✔ | ✔ | ✔ | `AudioMNIST/Dirichlet05/FedNova/AUDIOCNNRNN/20Clients/1.txt.json` |
| 7 | AudioMNIST | Dirichlet1 | FedFB | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,770 | ✔ | ✔ | ✔ | `AudioMNIST/Dirichlet1/FedFB/AUDIOCNNRNN/20Clients/1.txt.json` |
| 8 | AudioMNIST | Dirichlet1 | FedNova | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,916 | ✔ | ✔ | ✔ | `AudioMNIST/Dirichlet1/FedNova/AUDIOCNNRNN/20Clients/1.txt.json` |
| 9 | AudioMNIST | Uniform | FedFB | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,782 | ✔ | ✔ | ✔ | `AudioMNIST/Uniform/FedFB/AUDIOCNNRNN/20Clients/1.txt.json` |
| 10 | AudioMNIST | Uniform | FedNova | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,930 | ✔ | ✔ | ✔ | `AudioMNIST/Uniform/FedNova/AUDIOCNNRNN/20Clients/1.txt.json` |
| 11 | CREMA-D | Dirichlet1 | FedFB | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,681 | ✔ | ✔ | ✔ | `CREMA-D/Dirichlet1/FedFB/AUDIOCNNRNN/20Clients/1.txt.json` |
| 12 | CREMA-D | Dirichlet1 | FedNova | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,783 | ✔ | ✔ | ✔ | `CREMA-D/Dirichlet1/FedNova/AUDIOCNNRNN/20Clients/1.txt.json` |
| 13 | CREMA-D | Speaker | FedFB | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,738 | ✔ | ✔ | ✔ | `CREMA-D/Speaker/FedFB/AUDIOCNNRNN/20Clients/1.txt.json` |
| 14 | CREMA-D | Speaker | FedNova | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,747 | ✔ | ✔ | ✔ | `CREMA-D/Speaker/FedNova/AUDIOCNNRNN/20Clients/1.txt.json` |
| 15 | CREMA-D | Uniform | FedFB | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,679 | ✔ | ✔ | ✔ | `CREMA-D/Uniform/FedFB/AUDIOCNNRNN/20Clients/1.txt.json` |
| 16 | CREMA-D | Uniform | FedNova | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,890 | ✔ | ✔ | ✔ | `CREMA-D/Uniform/FedNova/AUDIOCNNRNN/20Clients/1.txt.json` |
| 17 | CREMA-D_ar | Speaker | FedFB | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,887 | ✔ | ✔ | ✔ | `CREMA-D_ar/Speaker/FedFB/AUDIOCNNRNN/20Clients/1.txt.json` |
| 18 | CREMA-D_ar | Speaker | FedNova | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,893 | ✔ | ✔ | ✔ | `CREMA-D_ar/Speaker/FedNova/AUDIOCNNRNN/20Clients/1.txt.json` |
| 19 | CREMA-D_gender | Speaker | FedFB | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,808 | ✔ | ✔ | ✔ | `CREMA-D_gender/Speaker/FedFB/AUDIOCNNRNN/20Clients/1.txt.json` |
| 20 | CREMA-D_gender | Speaker | FedNova | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,789 | ✔ | ✔ | ✔ | `CREMA-D_gender/Speaker/FedNova/AUDIOCNNRNN/20Clients/1.txt.json` |
| 21 | FairFace | Dirichlet01 | FedFB | BERTCLASSIFIER | 30Clients | 64 | 3 | 3,788 | ✔ | ✔ | ✔ | `FairFace/Dirichlet01/FedFB/BERTCLASSIFIER/30Clients/1.txt.json` |
| 22 | FairFace | Dirichlet01 | FedNova | BERTCLASSIFIER | 30Clients | 64 | 3 | 3,740 | ✔ | ✔ | ✔ | `FairFace/Dirichlet01/FedNova/BERTCLASSIFIER/30Clients/1.txt.json` |
| 23 | FairFace | Dirichlet05 | FedFB | BERTCLASSIFIER | 30Clients | 64 | 3 | 3,790 | ✔ | ✔ | ✔ | `FairFace/Dirichlet05/FedFB/BERTCLASSIFIER/30Clients/1.txt.json` |
| 24 | FairFace | Dirichlet05 | FedNova | BERTCLASSIFIER | 20Clients | 64 | 3 | 3,854 | ✔ | ✔ | ✔ | `FairFace/Dirichlet05/FedNova/BERTCLASSIFIER/20Clients/1.txt.json` |
| 25 | FairFace | Dirichlet05 | FedNova | BERTCLASSIFIER | 30Clients | 64 | 3 | 3,783 | ✔ | ✔ | ✔ | `FairFace/Dirichlet05/FedNova/BERTCLASSIFIER/30Clients/1.txt.json` |
| 26 | FairFace | Dirichlet1 | FedNova | BERTCLASSIFIER | 30Clients | 64 | 3 | 3,842 | ✔ | ✔ | ✔ | `FairFace/Dirichlet1/FedNova/BERTCLASSIFIER/30Clients/1.txt.json` |
| 27 | FairFace | Uniform | FedFB | BERTCLASSIFIER | 30Clients | 64 | 3 | 3,785 | ✔ | ✔ | ✔ | `FairFace/Uniform/FedFB/BERTCLASSIFIER/30Clients/1.txt.json` |
| 28 | FairFace | Uniform | FedNova | BERTCLASSIFIER | 30Clients | 64 | 3 | 3,847 | ✔ | ✔ | ✔ | `FairFace/Uniform/FedNova/BERTCLASSIFIER/30Clients/1.txt.json` |
| 29 | LFWA+ | Dirichlet01 | FedFB | BERTCLASSIFIER | 40Clients | 64 | 3 | 3,790 | ✔ | ✔ | ✔ | `LFWA+/Dirichlet01/FedFB/BERTCLASSIFIER/40Clients/1.txt.json` |
| 30 | LFWA+ | Dirichlet01 | FedNova | BERTCLASSIFIER | 30Clients | 32 | 3 | 3,724 | ✔ | ✔ | ✔ | `LFWA+/Dirichlet01/FedNova/BERTCLASSIFIER/30Clients/1.txt.json` |
| 31 | LFWA+ | Dirichlet01 | FedNova | BERTCLASSIFIER | 40Clients | 64 | 3 | 3,544 | ✔ | ✔ | ✔ | `LFWA+/Dirichlet01/FedNova/BERTCLASSIFIER/40Clients/1.txt.json` |
| 32 | LFWA+ | Dirichlet05 | FedFB | BERTCLASSIFIER | 20Clients | 32 | 3 | 3,785 | ✔ | ✔ | ✔ | `LFWA+/Dirichlet05/FedFB/BERTCLASSIFIER/20Clients/1.txt.json` |
| 33 | LFWA+ | Dirichlet05 | FedFB | BERTCLASSIFIER | 40Clients | 64 | 3 | 3,786 | ✔ | ✔ | ✔ | `LFWA+/Dirichlet05/FedFB/BERTCLASSIFIER/40Clients/1.txt.json` |
| 34 | LFWA+ | Dirichlet05 | FedNova | BERTCLASSIFIER | 40Clients | 64 | 3 | 3,769 | ✔ | ✔ | ✔ | `LFWA+/Dirichlet05/FedNova/BERTCLASSIFIER/40Clients/1.txt.json` |
| 35 | LFWA+ | Dirichlet1 | FedFB | BERTCLASSIFIER | 30Clients | 32 | 3 | 3,773 | ✔ | ✔ | ✔ | `LFWA+/Dirichlet1/FedFB/BERTCLASSIFIER/30Clients/1.txt.json` |
| 36 | LFWA+ | Dirichlet1 | FedFB | BERTCLASSIFIER | 40Clients | 64 | 3 | 3,784 | ✔ | ✔ | ✔ | `LFWA+/Dirichlet1/FedFB/BERTCLASSIFIER/40Clients/1.txt.json` |
| 37 | LFWA+ | Dirichlet1 | FedNova | BERTCLASSIFIER | 30Clients | 32 | 3 | 3,762 | ✔ | ✔ | ✔ | `LFWA+/Dirichlet1/FedNova/BERTCLASSIFIER/30Clients/1.txt.json` |
| 38 | LFWA+ | Dirichlet1 | FedNova | BERTCLASSIFIER | 40Clients | 64 | 3 | 3,776 | ✔ | ✔ | ✔ | `LFWA+/Dirichlet1/FedNova/BERTCLASSIFIER/40Clients/1.txt.json` |
| 39 | LFWA+ | Uniform | FedFB | BERTCLASSIFIER | 20Clients | 32 | 3 | 3,780 | ✔ | ✔ | ✔ | `LFWA+/Uniform/FedFB/BERTCLASSIFIER/20Clients/1.txt.json` |
| 40 | LFWA+ | Uniform | FedNova | BERTCLASSIFIER | 40Clients | 64 | 3 | 3,833 | ✔ | ✔ | ✔ | `LFWA+/Uniform/FedNova/BERTCLASSIFIER/40Clients/1.txt.json` |
| 41 | RAVDESS | Speaker | FedFB | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,736 | ✔ | ✔ | ✔ | `RAVDESS/Speaker/FedFB/AUDIOCNNRNN/20Clients/1.txt.json` |
| 42 | RAVDESS | Speaker | FedNova | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,719 | ✔ | ✔ | ✔ | `RAVDESS/Speaker/FedNova/AUDIOCNNRNN/20Clients/1.txt.json` |
| 43 | RAVDESS_ar | Speaker | FedFB | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,818 | ✔ | ✔ | ✔ | `RAVDESS_ar/Speaker/FedFB/AUDIOCNNRNN/20Clients/1.txt.json` |
| 44 | RAVDESS_ar | Speaker | FedNova | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,831 | ✔ | ✔ | ✔ | `RAVDESS_ar/Speaker/FedNova/AUDIOCNNRNN/20Clients/1.txt.json` |
| 45 | VCTK | Speaker | FedFB | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,863 | ✔ | ✔ | ✔ | `VCTK/Speaker/FedFB/AUDIOCNNRNN/20Clients/1.txt.json` |
| 46 | VCTK | Speaker | FedFB | AUDIOCNNRNN | 30Clients | 64 | 3 | 3,854 | ✔ | ✔ | ✔ | `VCTK/Speaker/FedFB/AUDIOCNNRNN/30Clients/1.txt.json` |
| 47 | VCTK | Speaker | FedFB | AUDIOCNNRNN | 40Clients | 64 | 3 | 3,856 | ✔ | ✔ | ✔ | `VCTK/Speaker/FedFB/AUDIOCNNRNN/40Clients/1.txt.json` |
| 48 | VCTK | Speaker | FedNova | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,868 | ✔ | ✔ | ✔ | `VCTK/Speaker/FedNova/AUDIOCNNRNN/20Clients/1.txt.json` |
| 49 | VCTK | Speaker | FedNova | AUDIOCNNRNN | 30Clients | 64 | 3 | 3,828 | ✔ | ✔ | ✔ | `VCTK/Speaker/FedNova/AUDIOCNNRNN/30Clients/1.txt.json` |
| 50 | VCTK | Speaker | FedNova | AUDIOCNNRNN | 40Clients | 64 | 3 | 3,859 | ✔ | ✔ | ✔ | `VCTK/Speaker/FedNova/AUDIOCNNRNN/40Clients/1.txt.json` |

**G1 主机分布**：本地 50 / .30 49 / .31 49；两台远端共同缺失 `ADULT/Uniform/FedFB/ANN/4Clients/1.txt.json`（仅本地存有）。
**G1 构成**：bs=64 共 44 格、bs=32 共 5 格、bs=128 共 1 格；按数据集 FairFace 8、LFWA+ 12、AudioMNIST 8、CREMA-D 6、VCTK 6、CREMA-D_ar 2、CREMA-D_gender 2、RAVDESS 2、RAVDESS_ar 2、ADULT 2。

## G2：FedFair 全部（144 格，本次不删除）

| # | 数据集 | split | 算法 | 模型 | K | 名义 bs | seeds | 字节 | 相对路径 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ACSIncome | Dirichlet01 | FedFair | ANN | 20Clients | 256 | 3 | 4,032 | `ACSIncome/Dirichlet01/FedFair/ANN/20Clients/1.txt.json` |
| 2 | ACSIncome | Dirichlet01 | FedFair | ANN | 30Clients | 256 | 3 | 4,028 | `ACSIncome/Dirichlet01/FedFair/ANN/30Clients/1.txt.json` |
| 3 | ACSIncome | Dirichlet01 | FedFair | ANN | 40Clients | 256 | 3 | 4,025 | `ACSIncome/Dirichlet01/FedFair/ANN/40Clients/1.txt.json` |
| 4 | ACSIncome | Dirichlet05 | FedFair | ANN | 20Clients | 256 | 3 | 4,018 | `ACSIncome/Dirichlet05/FedFair/ANN/20Clients/1.txt.json` |
| 5 | ACSIncome | Dirichlet05 | FedFair | ANN | 30Clients | 256 | 3 | 4,007 | `ACSIncome/Dirichlet05/FedFair/ANN/30Clients/1.txt.json` |
| 6 | ACSIncome | Dirichlet1 | FedFair | ANN | 20Clients | 256 | 3 | 3,996 | `ACSIncome/Dirichlet1/FedFair/ANN/20Clients/1.txt.json` |
| 7 | ACSIncome | Dirichlet1 | FedFair | ANN | 30Clients | 256 | 3 | 4,069 | `ACSIncome/Dirichlet1/FedFair/ANN/30Clients/1.txt.json` |
| 8 | ACSIncome | Dirichlet1 | FedFair | ANN | 40Clients | 256 | 3 | 4,114 | `ACSIncome/Dirichlet1/FedFair/ANN/40Clients/1.txt.json` |
| 9 | ACSIncome | Uniform | FedFair | ANN | 20Clients | 256 | 3 | 4,020 | `ACSIncome/Uniform/FedFair/ANN/20Clients/1.txt.json` |
| 10 | ACSIncome | Uniform | FedFair | ANN | 30Clients | 256 | 3 | 4,022 | `ACSIncome/Uniform/FedFair/ANN/30Clients/1.txt.json` |
| 11 | ADULT | Dirichlet01 | FedFair | ANN | 20Clients | 256 | 3 | 3,715 | `ADULT/Dirichlet01/FedFair/ANN/20Clients/1.txt.json` |
| 12 | ADULT | Dirichlet05 | FedFair | ANN | 20Clients | 256 | 3 | 3,716 | `ADULT/Dirichlet05/FedFair/ANN/20Clients/1.txt.json` |
| 13 | ADULT | Dirichlet1 | FedFair | ANN | 20Clients | 256 | 3 | 3,715 | `ADULT/Dirichlet1/FedFair/ANN/20Clients/1.txt.json` |
| 14 | ADULT | Uniform | FedFair | ANN | 20Clients | 256 | 3 | 3,714 | `ADULT/Uniform/FedFair/ANN/20Clients/1.txt.json` |
| 15 | ADULT | Uniform | FedFair | ANN | 4Clients | 64 | 1 | 2,949 | `ADULT/Uniform/FedFair/ANN/4Clients/1.txt.json` |
| 16 | AudioMNIST | Dirichlet01 | FedFair | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,511 | `AudioMNIST/Dirichlet01/FedFair/AUDIOCNNRNN/20Clients/1.txt.json` |
| 17 | AudioMNIST | Dirichlet05 | FedFair | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,812 | `AudioMNIST/Dirichlet05/FedFair/AUDIOCNNRNN/20Clients/1.txt.json` |
| 18 | AudioMNIST | Dirichlet1 | FedFair | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,676 | `AudioMNIST/Dirichlet1/FedFair/AUDIOCNNRNN/20Clients/1.txt.json` |
| 19 | AudioMNIST | Uniform | FedFair | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,816 | `AudioMNIST/Uniform/FedFair/AUDIOCNNRNN/20Clients/1.txt.json` |
| 20 | BANK | Dirichlet01 | FedFair | ANN | 20Clients | 256 | 3 | 3,960 | `BANK/Dirichlet01/FedFair/ANN/20Clients/1.txt.json` |
| 21 | BANK | Dirichlet01 | FedFair | ANN | 30Clients | 256 | 3 | 3,997 | `BANK/Dirichlet01/FedFair/ANN/30Clients/1.txt.json` |
| 22 | BANK | Dirichlet01 | FedFair | ANN | 40Clients | 256 | 3 | 3,990 | `BANK/Dirichlet01/FedFair/ANN/40Clients/1.txt.json` |
| 23 | BANK | Dirichlet05 | FedFair | ANN | 20Clients | 256 | 3 | 3,965 | `BANK/Dirichlet05/FedFair/ANN/20Clients/1.txt.json` |
| 24 | BANK | Dirichlet05 | FedFair | ANN | 30Clients | 256 | 3 | 3,974 | `BANK/Dirichlet05/FedFair/ANN/30Clients/1.txt.json` |
| 25 | BANK | Dirichlet05 | FedFair | ANN | 30Clients | 256 | 3 | 3,966 | `BANK/Dirichlet05/FedFair/ANN/30Clients/5.txt.json` |
| 26 | BANK | Dirichlet05 | FedFair | ANN | 40Clients | 256 | 3 | 3,993 | `BANK/Dirichlet05/FedFair/ANN/40Clients/1.txt.json` |
| 27 | BANK | Dirichlet1 | FedFair | ANN | 20Clients | 256 | 3 | 3,994 | `BANK/Dirichlet1/FedFair/ANN/20Clients/1.txt.json` |
| 28 | BANK | Dirichlet1 | FedFair | ANN | 30Clients | 256 | 3 | 3,988 | `BANK/Dirichlet1/FedFair/ANN/30Clients/1.txt.json` |
| 29 | BANK | Dirichlet1 | FedFair | ANN | 40Clients | 256 | 3 | 3,989 | `BANK/Dirichlet1/FedFair/ANN/40Clients/1.txt.json` |
| 30 | BANK | Uniform | FedFair | ANN | 20Clients | 256 | 3 | 4,018 | `BANK/Uniform/FedFair/ANN/20Clients/1.txt.json` |
| 31 | BANK | Uniform | FedFair | ANN | 30Clients | 256 | 3 | 3,967 | `BANK/Uniform/FedFair/ANN/30Clients/1.txt.json` |
| 32 | BANK | Uniform | FedFair | ANN | 30Clients | 256 | 3 | 3,961 | `BANK/Uniform/FedFair/ANN/30Clients/11.txt.json` |
| 33 | BANK | Uniform | FedFair | ANN | 40Clients | 256 | 3 | 3,991 | `BANK/Uniform/FedFair/ANN/40Clients/1.txt.json` |
| 34 | COMPAS | Dirichlet01 | FedFair | ANN | 20Clients | 256 | 3 | 3,977 | `COMPAS/Dirichlet01/FedFair/ANN/20Clients/1.txt.json` |
| 35 | COMPAS | Dirichlet01 | FedFair | ANN | 30Clients | 256 | 3 | 3,999 | `COMPAS/Dirichlet01/FedFair/ANN/30Clients/1.txt.json` |
| 36 | COMPAS | Dirichlet01 | FedFair | ANN | 30Clients | 256 | 3 | 3,966 | `COMPAS/Dirichlet01/FedFair/ANN/30Clients/2.txt.json` |
| 37 | COMPAS | Dirichlet01 | FedFair | ANN | 40Clients | 256 | 3 | 3,994 | `COMPAS/Dirichlet01/FedFair/ANN/40Clients/1.txt.json` |
| 38 | COMPAS | Dirichlet01 | FedFair | ANN | 40Clients | 256 | 3 | 3,964 | `COMPAS/Dirichlet01/FedFair/ANN/40Clients/3.txt.json` |
| 39 | COMPAS | Dirichlet05 | FedFair | ANN | 20Clients | 256 | 3 | 3,975 | `COMPAS/Dirichlet05/FedFair/ANN/20Clients/1.txt.json` |
| 40 | COMPAS | Dirichlet05 | FedFair | ANN | 30Clients | 256 | 3 | 3,989 | `COMPAS/Dirichlet05/FedFair/ANN/30Clients/1.txt.json` |
| 41 | COMPAS | Dirichlet05 | FedFair | ANN | 30Clients | 256 | 3 | 3,957 | `COMPAS/Dirichlet05/FedFair/ANN/30Clients/5.txt.json` |
| 42 | COMPAS | Dirichlet05 | FedFair | ANN | 40Clients | 256 | 3 | 3,995 | `COMPAS/Dirichlet05/FedFair/ANN/40Clients/1.txt.json` |
| 43 | COMPAS | Dirichlet05 | FedFair | ANN | 40Clients | 256 | 3 | 3,963 | `COMPAS/Dirichlet05/FedFair/ANN/40Clients/6.txt.json` |
| 44 | COMPAS | Dirichlet1 | FedFair | ANN | 20Clients | 256 | 3 | 3,973 | `COMPAS/Dirichlet1/FedFair/ANN/20Clients/1.txt.json` |
| 45 | COMPAS | Dirichlet1 | FedFair | ANN | 30Clients | 256 | 3 | 3,999 | `COMPAS/Dirichlet1/FedFair/ANN/30Clients/1.txt.json` |
| 46 | COMPAS | Dirichlet1 | FedFair | ANN | 30Clients | 256 | 3 | 3,968 | `COMPAS/Dirichlet1/FedFair/ANN/30Clients/8.txt.json` |
| 47 | COMPAS | Dirichlet1 | FedFair | ANN | 40Clients | 256 | 3 | 3,989 | `COMPAS/Dirichlet1/FedFair/ANN/40Clients/1.txt.json` |
| 48 | COMPAS | Dirichlet1 | FedFair | ANN | 40Clients | 256 | 3 | 3,958 | `COMPAS/Dirichlet1/FedFair/ANN/40Clients/9.txt.json` |
| 49 | COMPAS | Uniform | FedFair | ANN | 20Clients | 256 | 3 | 3,965 | `COMPAS/Uniform/FedFair/ANN/20Clients/1.txt.json` |
| 50 | COMPAS | Uniform | FedFair | ANN | 30Clients | 256 | 3 | 3,992 | `COMPAS/Uniform/FedFair/ANN/30Clients/1.txt.json` |
| 51 | COMPAS | Uniform | FedFair | ANN | 30Clients | 256 | 3 | 3,959 | `COMPAS/Uniform/FedFair/ANN/30Clients/11.txt.json` |
| 52 | COMPAS | Uniform | FedFair | ANN | 40Clients | 256 | 3 | 3,988 | `COMPAS/Uniform/FedFair/ANN/40Clients/1.txt.json` |
| 53 | COMPAS | Uniform | FedFair | ANN | 40Clients | 256 | 3 | 3,957 | `COMPAS/Uniform/FedFair/ANN/40Clients/12.txt.json` |
| 54 | CREMA-D | Dirichlet1 | FedFair | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,877 | `CREMA-D/Dirichlet1/FedFair/AUDIOCNNRNN/20Clients/1.txt.json` |
| 55 | CREMA-D | Speaker | FedFair | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,748 | `CREMA-D/Speaker/FedFair/AUDIOCNNRNN/20Clients/1.txt.json` |
| 56 | CREMA-D | Uniform | FedFair | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,681 | `CREMA-D/Uniform/FedFair/AUDIOCNNRNN/20Clients/1.txt.json` |
| 57 | CREMA-D_ar | Speaker | FedFair | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,892 | `CREMA-D_ar/Speaker/FedFair/AUDIOCNNRNN/20Clients/1.txt.json` |
| 58 | CREMA-D_gender | Speaker | FedFair | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,795 | `CREMA-D_gender/Speaker/FedFair/AUDIOCNNRNN/20Clients/1.txt.json` |
| 59 | CelebA | Dirichlet01 | FedFair | BERTCLASSIFIER | 20Clients | 64 | 3 | 3,981 | `CelebA/Dirichlet01/FedFair/BERTCLASSIFIER/20Clients/1.txt.json` |
| 60 | CelebA | Dirichlet01 | FedFair | BERTCLASSIFIER | 30Clients | 256 | 3 | 3,784 | `CelebA/Dirichlet01/FedFair/BERTCLASSIFIER/30Clients/1.txt.json` |
| 61 | CelebA | Dirichlet01 | FedFair | BERTCLASSIFIER | 40Clients | 256 | 3 | 3,790 | `CelebA/Dirichlet01/FedFair/BERTCLASSIFIER/40Clients/1.txt.json` |
| 62 | CelebA | Dirichlet05 | FedFair | BERTCLASSIFIER | 20Clients | 256 | 3 | 3,792 | `CelebA/Dirichlet05/FedFair/BERTCLASSIFIER/20Clients/1.txt.json` |
| 63 | CelebA | Dirichlet05 | FedFair | BERTCLASSIFIER | 30Clients | 256 | 3 | 3,790 | `CelebA/Dirichlet05/FedFair/BERTCLASSIFIER/30Clients/1.txt.json` |
| 64 | CelebA | Dirichlet05 | FedFair | BERTCLASSIFIER | 40Clients | 256 | 3 | 3,802 | `CelebA/Dirichlet05/FedFair/BERTCLASSIFIER/40Clients/1.txt.json` |
| 65 | CelebA | Dirichlet1 | FedFair | BERTCLASSIFIER | 20Clients | 256 | 3 | 3,792 | `CelebA/Dirichlet1/FedFair/BERTCLASSIFIER/20Clients/1.txt.json` |
| 66 | CelebA | Dirichlet1 | FedFair | BERTCLASSIFIER | 30Clients | 256 | 3 | 3,785 | `CelebA/Dirichlet1/FedFair/BERTCLASSIFIER/30Clients/1.txt.json` |
| 67 | CelebA | Dirichlet1 | FedFair | BERTCLASSIFIER | 40Clients | 256 | 3 | 3,787 | `CelebA/Dirichlet1/FedFair/BERTCLASSIFIER/40Clients/1.txt.json` |
| 68 | CelebA | Uniform | FedFair | BERTCLASSIFIER | 20Clients | 256 | 3 | 3,857 | `CelebA/Uniform/FedFair/BERTCLASSIFIER/20Clients/1.txt.json` |
| 69 | CelebA | Uniform | FedFair | BERTCLASSIFIER | 30Clients | 256 | 3 | 3,784 | `CelebA/Uniform/FedFair/BERTCLASSIFIER/30Clients/1.txt.json` |
| 70 | CelebA | Uniform | FedFair | BERTCLASSIFIER | 40Clients | 256 | 3 | 3,787 | `CelebA/Uniform/FedFair/BERTCLASSIFIER/40Clients/1.txt.json` |
| 71 | DRUG | Dirichlet01 | FedFair | ANN | 20Clients | 256 | 3 | 3,961 | `DRUG/Dirichlet01/FedFair/ANN/20Clients/1.txt.json` |
| 72 | DRUG | Dirichlet01 | FedFair | ANN | 30Clients | 256 | 3 | 3,985 | `DRUG/Dirichlet01/FedFair/ANN/30Clients/1.txt.json` |
| 73 | DRUG | Dirichlet01 | FedFair | ANN | 30Clients | 256 | 3 | 3,953 | `DRUG/Dirichlet01/FedFair/ANN/30Clients/2.txt.json` |
| 74 | DRUG | Dirichlet01 | FedFair | ANN | 40Clients | 256 | 3 | 3,948 | `DRUG/Dirichlet01/FedFair/ANN/40Clients/3.txt.json` |
| 75 | DRUG | Dirichlet05 | FedFair | ANN | 20Clients | 256 | 3 | 3,934 | `DRUG/Dirichlet05/FedFair/ANN/20Clients/1.txt.json` |
| 76 | DRUG | Dirichlet05 | FedFair | ANN | 30Clients | 256 | 3 | 3,952 | `DRUG/Dirichlet05/FedFair/ANN/30Clients/5.txt.json` |
| 77 | DRUG | Dirichlet05 | FedFair | ANN | 40Clients | 256 | 3 | 3,950 | `DRUG/Dirichlet05/FedFair/ANN/40Clients/6.txt.json` |
| 78 | DRUG | Dirichlet1 | FedFair | ANN | 20Clients | 256 | 3 | 3,984 | `DRUG/Dirichlet1/FedFair/ANN/20Clients/1.txt.json` |
| 79 | DRUG | Dirichlet1 | FedFair | ANN | 30Clients | 256 | 3 | 3,955 | `DRUG/Dirichlet1/FedFair/ANN/30Clients/8.txt.json` |
| 80 | DRUG | Dirichlet1 | FedFair | ANN | 40Clients | 256 | 3 | 3,947 | `DRUG/Dirichlet1/FedFair/ANN/40Clients/9.txt.json` |
| 81 | DRUG | Uniform | FedFair | ANN | 20Clients | 256 | 3 | 3,974 | `DRUG/Uniform/FedFair/ANN/20Clients/1.txt.json` |
| 82 | DRUG | Uniform | FedFair | ANN | 30Clients | 256 | 3 | 3,952 | `DRUG/Uniform/FedFair/ANN/30Clients/11.txt.json` |
| 83 | DRUG | Uniform | FedFair | ANN | 40Clients | 256 | 3 | 3,947 | `DRUG/Uniform/FedFair/ANN/40Clients/12.txt.json` |
| 84 | DUTCH | Dirichlet01 | FedFair | ANN | 20Clients | 256 | 3 | 3,965 | `DUTCH/Dirichlet01/FedFair/ANN/20Clients/1.txt.json` |
| 85 | DUTCH | Dirichlet01 | FedFair | ANN | 30Clients | 256 | 3 | 3,994 | `DUTCH/Dirichlet01/FedFair/ANN/30Clients/1.txt.json` |
| 86 | DUTCH | Dirichlet01 | FedFair | ANN | 30Clients | 256 | 3 | 3,952 | `DUTCH/Dirichlet01/FedFair/ANN/30Clients/2.txt.json` |
| 87 | DUTCH | Dirichlet01 | FedFair | ANN | 40Clients | 256 | 3 | 3,991 | `DUTCH/Dirichlet01/FedFair/ANN/40Clients/1.txt.json` |
| 88 | DUTCH | Dirichlet05 | FedFair | ANN | 20Clients | 256 | 3 | 3,968 | `DUTCH/Dirichlet05/FedFair/ANN/20Clients/1.txt.json` |
| 89 | DUTCH | Dirichlet05 | FedFair | ANN | 30Clients | 256 | 3 | 3,990 | `DUTCH/Dirichlet05/FedFair/ANN/30Clients/1.txt.json` |
| 90 | DUTCH | Dirichlet05 | FedFair | ANN | 30Clients | 256 | 3 | 3,960 | `DUTCH/Dirichlet05/FedFair/ANN/30Clients/5.txt.json` |
| 91 | DUTCH | Dirichlet05 | FedFair | ANN | 40Clients | 256 | 3 | 3,987 | `DUTCH/Dirichlet05/FedFair/ANN/40Clients/1.txt.json` |
| 92 | DUTCH | Dirichlet1 | FedFair | ANN | 20Clients | 256 | 3 | 3,985 | `DUTCH/Dirichlet1/FedFair/ANN/20Clients/1.txt.json` |
| 93 | DUTCH | Dirichlet1 | FedFair | ANN | 30Clients | 256 | 3 | 3,994 | `DUTCH/Dirichlet1/FedFair/ANN/30Clients/1.txt.json` |
| 94 | DUTCH | Dirichlet1 | FedFair | ANN | 30Clients | 256 | 3 | 3,954 | `DUTCH/Dirichlet1/FedFair/ANN/30Clients/8.txt.json` |
| 95 | DUTCH | Dirichlet1 | FedFair | ANN | 40Clients | 256 | 3 | 3,982 | `DUTCH/Dirichlet1/FedFair/ANN/40Clients/1.txt.json` |
| 96 | DUTCH | Uniform | FedFair | ANN | 20Clients | 256 | 3 | 3,982 | `DUTCH/Uniform/FedFair/ANN/20Clients/1.txt.json` |
| 97 | DUTCH | Uniform | FedFair | ANN | 30Clients | 256 | 3 | 3,986 | `DUTCH/Uniform/FedFair/ANN/30Clients/1.txt.json` |
| 98 | DUTCH | Uniform | FedFair | ANN | 30Clients | 256 | 3 | 3,951 | `DUTCH/Uniform/FedFair/ANN/30Clients/11.txt.json` |
| 99 | DUTCH | Uniform | FedFair | ANN | 40Clients | 256 | 3 | 3,985 | `DUTCH/Uniform/FedFair/ANN/40Clients/1.txt.json` |
| 100 | FairFace | Dirichlet01 | FedFair | BERTCLASSIFIER | 20Clients | 64 | 3 | 3,862 | `FairFace/Dirichlet01/FedFair/BERTCLASSIFIER/20Clients/1.txt.json` |
| 101 | FairFace | Dirichlet01 | FedFair | BERTCLASSIFIER | 30Clients | 64 | 3 | 3,792 | `FairFace/Dirichlet01/FedFair/BERTCLASSIFIER/30Clients/1.txt.json` |
| 102 | FairFace | Dirichlet01 | FedFair | BERTCLASSIFIER | 40Clients | 256 | 3 | 3,876 | `FairFace/Dirichlet01/FedFair/BERTCLASSIFIER/40Clients/1.txt.json` |
| 103 | FairFace | Dirichlet05 | FedFair | BERTCLASSIFIER | 20Clients | 256 | 3 | 3,869 | `FairFace/Dirichlet05/FedFair/BERTCLASSIFIER/20Clients/1.txt.json` |
| 104 | FairFace | Dirichlet05 | FedFair | BERTCLASSIFIER | 30Clients | 64 | 3 | 3,791 | `FairFace/Dirichlet05/FedFair/BERTCLASSIFIER/30Clients/1.txt.json` |
| 105 | FairFace | Uniform | FedFair | BERTCLASSIFIER | 30Clients | 64 | 3 | 3,864 | `FairFace/Uniform/FedFair/BERTCLASSIFIER/30Clients/1.txt.json` |
| 106 | LFWA+ | Dirichlet01 | FedFair | BERTCLASSIFIER | 20Clients | 64 | 3 | 3,980 | `LFWA+/Dirichlet01/FedFair/BERTCLASSIFIER/20Clients/1.txt.json` |
| 107 | LFWA+ | Dirichlet01 | FedFair | BERTCLASSIFIER | 30Clients | 64 | 3 | 3,978 | `LFWA+/Dirichlet01/FedFair/BERTCLASSIFIER/30Clients/2.txt.json` |
| 108 | LFWA+ | Dirichlet01 | FedFair | BERTCLASSIFIER | 40Clients | 64 | 3 | 3,784 | `LFWA+/Dirichlet01/FedFair/BERTCLASSIFIER/40Clients/1.txt.json` |
| 109 | LFWA+ | Dirichlet05 | FedFair | BERTCLASSIFIER | 20Clients | 256 | 3 | 3,789 | `LFWA+/Dirichlet05/FedFair/BERTCLASSIFIER/20Clients/1.txt.json` |
| 110 | LFWA+ | Dirichlet05 | FedFair | BERTCLASSIFIER | 30Clients | 32 | 3 | 3,780 | `LFWA+/Dirichlet05/FedFair/BERTCLASSIFIER/30Clients/1.txt.json` |
| 111 | LFWA+ | Dirichlet05 | FedFair | BERTCLASSIFIER | 40Clients | 64 | 3 | 3,785 | `LFWA+/Dirichlet05/FedFair/BERTCLASSIFIER/40Clients/1.txt.json` |
| 112 | LFWA+ | Dirichlet1 | FedFair | BERTCLASSIFIER | 20Clients | 256 | 3 | 3,789 | `LFWA+/Dirichlet1/FedFair/BERTCLASSIFIER/20Clients/1.txt.json` |
| 113 | LFWA+ | Dirichlet1 | FedFair | BERTCLASSIFIER | 30Clients | 256 | 3 | 3,853 | `LFWA+/Dirichlet1/FedFair/BERTCLASSIFIER/30Clients/1.txt.json` |
| 114 | LFWA+ | Dirichlet1 | FedFair | BERTCLASSIFIER | 40Clients | 64 | 3 | 3,782 | `LFWA+/Dirichlet1/FedFair/BERTCLASSIFIER/40Clients/1.txt.json` |
| 115 | LFWA+ | Uniform | FedFair | BERTCLASSIFIER | 20Clients | 256 | 3 | 3,783 | `LFWA+/Uniform/FedFair/BERTCLASSIFIER/20Clients/1.txt.json` |
| 116 | LFWA+ | Uniform | FedFair | BERTCLASSIFIER | 30Clients | 32 | 3 | 3,779 | `LFWA+/Uniform/FedFair/BERTCLASSIFIER/30Clients/1.txt.json` |
| 117 | LFWA+ | Uniform | FedFair | BERTCLASSIFIER | 40Clients | 256 | 3 | 3,860 | `LFWA+/Uniform/FedFair/BERTCLASSIFIER/40Clients/1.txt.json` |
| 118 | PAD-UFES-20 | Dirichlet01 | FedFair | BERTCLASSIFIER | 20Clients | 256 | 3 | 3,876 | `PAD-UFES-20/Dirichlet01/FedFair/BERTCLASSIFIER/20Clients/1.txt.json` |
| 119 | PAD-UFES-20 | Dirichlet01 | FedFair | BERTCLASSIFIER | 30Clients | 256 | 3 | 3,877 | `PAD-UFES-20/Dirichlet01/FedFair/BERTCLASSIFIER/30Clients/1.txt.json` |
| 120 | PAD-UFES-20 | Dirichlet01 | FedFair | BERTCLASSIFIER | 40Clients | 256 | 3 | 3,871 | `PAD-UFES-20/Dirichlet01/FedFair/BERTCLASSIFIER/40Clients/1.txt.json` |
| 121 | RAVDESS | Speaker | FedFair | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,744 | `RAVDESS/Speaker/FedFair/AUDIOCNNRNN/20Clients/1.txt.json` |
| 122 | RAVDESS_ar | Speaker | FedFair | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,849 | `RAVDESS_ar/Speaker/FedFair/AUDIOCNNRNN/20Clients/1.txt.json` |
| 123 | UTKFace | Dirichlet01 | FedFair | BERTCLASSIFIER | 20Clients | 256 | 3 | 3,868 | `UTKFace/Dirichlet01/FedFair/BERTCLASSIFIER/20Clients/1.txt.json` |
| 124 | UTKFace | Dirichlet01 | FedFair | BERTCLASSIFIER | 30Clients | 256 | 3 | 3,792 | `UTKFace/Dirichlet01/FedFair/BERTCLASSIFIER/30Clients/1.txt.json` |
| 125 | UTKFace | Dirichlet01 | FedFair | BERTCLASSIFIER | 40Clients | 256 | 3 | 3,865 | `UTKFace/Dirichlet01/FedFair/BERTCLASSIFIER/40Clients/1.txt.json` |
| 126 | UTKFace | Dirichlet05 | FedFair | BERTCLASSIFIER | 20Clients | 256 | 3 | 3,801 | `UTKFace/Dirichlet05/FedFair/BERTCLASSIFIER/20Clients/1.txt.json` |
| 127 | UTKFace | Dirichlet05 | FedFair | BERTCLASSIFIER | 30Clients | 256 | 3 | 3,793 | `UTKFace/Dirichlet05/FedFair/BERTCLASSIFIER/30Clients/1.txt.json` |
| 128 | UTKFace | Dirichlet05 | FedFair | BERTCLASSIFIER | 40Clients | 256 | 3 | 3,867 | `UTKFace/Dirichlet05/FedFair/BERTCLASSIFIER/40Clients/1.txt.json` |
| 129 | UTKFace | Dirichlet1 | FedFair | BERTCLASSIFIER | 20Clients | 256 | 3 | 3,794 | `UTKFace/Dirichlet1/FedFair/BERTCLASSIFIER/20Clients/1.txt.json` |
| 130 | UTKFace | Dirichlet1 | FedFair | BERTCLASSIFIER | 30Clients | 256 | 3 | 3,795 | `UTKFace/Dirichlet1/FedFair/BERTCLASSIFIER/30Clients/1.txt.json` |
| 131 | UTKFace | Dirichlet1 | FedFair | BERTCLASSIFIER | 40Clients | 256 | 3 | 3,790 | `UTKFace/Dirichlet1/FedFair/BERTCLASSIFIER/40Clients/1.txt.json` |
| 132 | UTKFace | Uniform | FedFair | BERTCLASSIFIER | 20Clients | 256 | 3 | 3,834 | `UTKFace/Uniform/FedFair/BERTCLASSIFIER/20Clients/1.txt.json` |
| 133 | UTKFace | Uniform | FedFair | BERTCLASSIFIER | 30Clients | 256 | 3 | 3,790 | `UTKFace/Uniform/FedFair/BERTCLASSIFIER/30Clients/1.txt.json` |
| 134 | UTKFace | Uniform | FedFair | BERTCLASSIFIER | 40Clients | 256 | 3 | 3,788 | `UTKFace/Uniform/FedFair/BERTCLASSIFIER/40Clients/1.txt.json` |
| 135 | VCTK | Speaker | FedFair | AUDIOCNNRNN | 20Clients | 64 | 3 | 3,835 | `VCTK/Speaker/FedFair/AUDIOCNNRNN/20Clients/1.txt.json` |
| 136 | VCTK | Speaker | FedFair | AUDIOCNNRNN | 30Clients | 64 | 3 | 3,871 | `VCTK/Speaker/FedFair/AUDIOCNNRNN/30Clients/1.txt.json` |
| 137 | VCTK | Speaker | FedFair | AUDIOCNNRNN | 40Clients | 64 | 3 | 3,860 | `VCTK/Speaker/FedFair/AUDIOCNNRNN/40Clients/1.txt.json` |
| 138 | bios | Dirichlet01 | FedFair | BERTCLASSIFIER | 10Clients | 256 | 1 | 3,658 | `bios/Dirichlet01/FedFair/BERTCLASSIFIER/10Clients/1.txt.json` |
| 139 | bios | Dirichlet01 | FedFair | BERTCLASSIFIER | 20Clients | 256 | 3 | 4,448 | `bios/Dirichlet01/FedFair/BERTCLASSIFIER/20Clients/1.txt.json` |
| 140 | bios | Dirichlet01 | FedFair | FedFair | BERTCLASSIFIER | 256 | 3 | 4,449 | `bios/Dirichlet01/FedNova/FedFair/BERTCLASSIFIER/20Clients/1.txt.json` |
| 141 | bios | Dirichlet05 | FedFair | BERTCLASSIFIER | 20Clients | 256 | 3 | 4,449 | `bios/Dirichlet05/FedFair/BERTCLASSIFIER/20Clients/1.txt.json` |
| 142 | bios | Dirichlet1 | FedFair | BERTCLASSIFIER | 20Clients | 256 | 3 | 4,475 | `bios/Dirichlet1/FedFair/BERTCLASSIFIER/20Clients/1.txt.json` |
| 143 | bios | Uniform | FedFair | BERTCLASSIFIER | 10Clients | 256 | 1 | 3,645 | `bios/Uniform/FedFair/BERTCLASSIFIER/10Clients/1.txt.json` |
| 144 | bios | Uniform | FedFair | BERTCLASSIFIER | 20Clients | 256 | 3 | 4,447 | `bios/Uniform/FedFair/BERTCLASSIFIER/20Clients/1.txt.json` |

## G3：仅 1 seed（91 格，不删除，待补齐 seed）

| # | 数据集 | split | 算法 | 模型 | K | 名义 bs | 有效批 | 字节 | 相对路径 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ACSIncome | Dirichlet01 | FL_FairBatch | ANN | 20Clients | 256 | 256 | 2,934 | `ACSIncome/Dirichlet01/FL_FairBatch_acspilot/ANN/20Clients/1.txt.json` |
| 2 | ACSIncome | Dirichlet01 | FairFed | ANN | 20Clients | 256 | 256 | 3,030 | `ACSIncome/Dirichlet01/FairFed_acspilot/ANN/20Clients/1.txt.json` |
| 3 | ACSIncome | Dirichlet01 | FedAvg | ANN | 20Clients | 256 | 256 | 2,936 | `ACSIncome/Dirichlet01/FedAvg_acsbench2/ANN/20Clients/1.txt.json` |
| 4 | ACSIncome | Dirichlet01 | FedAvg | ANN | 20Clients | 256 | 256 | 3,039 | `ACSIncome/Dirichlet01/FedAvg_acspilot/ANN/20Clients/1.txt.json` |
| 5 | ACSIncome | Dirichlet01 | FedAvg | ANN | 20Clients | 256 | 256 | 2,934 | `ACSIncome/Dirichlet01/FedAvg_acssmoke/ANN/20Clients/1.txt.json` |
| 6 | ACSIncome | Dirichlet01 | FedProto | ANN | 20Clients | 256 | 256 | 3,008 | `ACSIncome/Dirichlet01/FedProto_acspilot/ANN/20Clients/1.txt.json` |
| 7 | ACSIncome | Dirichlet01 | FedProx | ANN | 20Clients | 256 | 256 | 3,037 | `ACSIncome/Dirichlet01/FedProx_acspilot/ANN/20Clients/1.txt.json` |
| 8 | ACSIncome | Dirichlet01 | FedRenyi | ANN | 20Clients | 256 | 256 | 3,046 | `ACSIncome/Dirichlet01/FedRenyi_acsrenyi/ANN/20Clients/1.txt.json` |
| 9 | ACSIncome | Dirichlet01 | PDFFed_DP | ANN | 20Clients | 256 | 256 | 2,949 | `ACSIncome/Dirichlet01/PDFFed_DP_acssmoke_pdffed_dp/ANN/20Clients/1.txt.json` |
| 10 | ACSIncome | Dirichlet01 | mFairFL | ANN | 20Clients | 256 | 256 | 2,943 | `ACSIncome/Dirichlet01/mFairFL_acssmoke_mfairfl/ANN/20Clients/1.txt.json` |
| 11 | ACSIncome | Uniform | FedAvg | ANN | 20Clients | 256 | 256 | 3,043 | `ACSIncome/Uniform/FedAvg_nwbench_cp1_nw0/ANN/20Clients/1.txt.json` |
| 12 | ACSIncome | Uniform | FedAvg | ANN | 20Clients | 256 | 256 | 3,043 | `ACSIncome/Uniform/FedAvg_nwbench_cp1_nw2/ANN/20Clients/1.txt.json` |
| 13 | ACSIncome | Uniform | FedAvg | ANN | 20Clients | 256 | 256 | 3,044 | `ACSIncome/Uniform/FedAvg_nwbench_cp2_nw0/ANN/20Clients/1.txt.json` |
| 14 | ACSIncome | Uniform | FedAvg | ANN | 20Clients | 256 | 256 | 3,043 | `ACSIncome/Uniform/FedAvg_nwbench_cp2_nw2/ANN/20Clients/1.txt.json` |
| 15 | ACSIncome | Uniform | FedAvg | ANN | 20Clients | 256 | 256 | 3,040 | `ACSIncome/Uniform/FedAvg_nwlong_cp1_nw0/ANN/20Clients/1.txt.json` |
| 16 | ACSIncome | Uniform | FedAvg | ANN | 20Clients | 256 | 256 | 3,039 | `ACSIncome/Uniform/FedAvg_nwlong_cp1_nw2/ANN/20Clients/1.txt.json` |
| 17 | ACSIncome | Uniform | FedAvg | ANN | 20Clients | 256 | 256 | 3,041 | `ACSIncome/Uniform/FedAvg_nwlong_cp2_nw0/ANN/20Clients/1.txt.json` |
| 18 | ACSIncome | Uniform | FedAvg | ANN | 20Clients | 256 | 256 | 3,040 | `ACSIncome/Uniform/FedAvg_nwlong_cp2_nw2/ANN/20Clients/1.txt.json` |
| 19 | ACSIncome | Uniform | FedFB | ANN | 20Clients | 256 | 256 | 3,039 | `ACSIncome/Uniform/FedFB_acsmoke_fedfb2/ANN/20Clients/1.txt.json` |
| 20 | ACSIncome | Uniform | FedFB | ANN | 20Clients | 256 | 256 | 3,042 | `ACSIncome/Uniform/FedFB_acssmoke_fedfb3/ANN/20Clients/1.txt.json` |
| 21 | ACSIncome | Uniform | FedProx | ANN | 20Clients | 256 | 256 | 3,035 | `ACSIncome/Uniform/FedProx_acsbench4n0/ANN/20Clients/1.txt.json` |
| 22 | ACSIncome | Uniform | FedProx | ANN | 20Clients | 256 | 256 | 3,036 | `ACSIncome/Uniform/FedProx_acsbench4n2/ANN/20Clients/1.txt.json` |
| 23 | ACSIncome | Uniform | PDFFed | ANN | 20Clients | 256 | 256 | 3,030 | `ACSIncome/Uniform/PDFFed_acsbench3/ANN/20Clients/1.txt.json` |
| 24 | ACSIncome | Uniform | PDFFed | ANN | 20Clients | 256 | 256 | 3,042 | `ACSIncome/Uniform/PDFFed_finnw_a1_cp2_nw0/ANN/20Clients/1.txt.json` |
| 25 | ACSIncome | Uniform | PDFFed | ANN | 20Clients | 256 | 256 | 3,042 | `ACSIncome/Uniform/PDFFed_finnw_a1_cp2_nw2/ANN/20Clients/1.txt.json` |
| 26 | ACSIncome | Uniform | PDFFed | ANN | 20Clients | 256 | 256 | 3,041 | `ACSIncome/Uniform/PDFFed_finnw_b1_cp2_nw0/ANN/20Clients/1.txt.json` |
| 27 | ACSIncome | Uniform | PDFFed | ANN | 20Clients | 256 | 256 | 3,041 | `ACSIncome/Uniform/PDFFed_finnw_b1_cp2_nw2/ANN/20Clients/1.txt.json` |
| 28 | ACSIncome | Uniform | PDFFed | ANN | 20Clients | 256 | 256 | 3,041 | `ACSIncome/Uniform/PDFFed_pdfabv2_a_cp2_nw0/ANN/20Clients/1.txt.json` |
| 29 | ACSIncome | Uniform | PDFFed | ANN | 20Clients | 256 | 256 | 3,041 | `ACSIncome/Uniform/PDFFed_pdfabv2_a_cp2_nw2/ANN/20Clients/1.txt.json` |
| 30 | ACSIncome | Uniform | PDFFed | ANN | 20Clients | 256 | 256 | 3,041 | `ACSIncome/Uniform/PDFFed_pdfabv2_b_cp2_nw0/ANN/20Clients/1.txt.json` |
| 31 | ACSIncome | Uniform | PDFFed | ANN | 20Clients | 256 | 256 | 3,040 | `ACSIncome/Uniform/PDFFed_pdfabv2_b_cp2_nw2/ANN/20Clients/1.txt.json` |
| 32 | ACSIncome | Uniform | PDFFed | ANN | 20Clients | 256 | 256 | 3,039 | `ACSIncome/Uniform/PDFFed_pdfla_cp2_nw0/ANN/20Clients/1.txt.json` |
| 33 | ACSIncome | Uniform | PDFFed | ANN | 20Clients | 256 | 256 | 3,039 | `ACSIncome/Uniform/PDFFed_pdfla_cp2_nw2/ANN/20Clients/1.txt.json` |
| 34 | ACSIncome | Uniform | PDFFed | ANN | 20Clients | 256 | 256 | 3,035 | `ACSIncome/Uniform/PDFFed_pdfnw_cp1_nw0/ANN/20Clients/1.txt.json` |
| 35 | ACSIncome | Uniform | PDFFed | ANN | 20Clients | 256 | 256 | 3,034 | `ACSIncome/Uniform/PDFFed_pdfnw_cp2_nw0/ANN/20Clients/1.txt.json` |
| 36 | ACSIncome | Uniform | PDFFed | ANN | 20Clients | 256 | 256 | 3,035 | `ACSIncome/Uniform/PDFFed_pdfnw_cp2_nw2/ANN/20Clients/1.txt.json` |
| 37 | ADULT | Dirichlet05 | FAFI | ANN | 4Clients | 128 | 256 | 2,878 | `ADULT/Dirichlet05/FAFI/ANN/4Clients/1.txt.json` |
| 38 | ADULT | Dirichlet05 | FOL | ANN | 4Clients | 128 | 256 | 2,876 | `ADULT/Dirichlet05/FOL/ANN/4Clients/1.txt.json` |
| 39 | ADULT | Dirichlet05 | FairFedMOE | ANN | 4Clients | 128 | 256 | 2,488 | `ADULT/Dirichlet05/FairFedMOE/ANN/4Clients/1.txt.json` |
| 40 | ADULT | Dirichlet05 | FedAvg | ANN | 4Clients | 128 | 256 | 2,485 | `ADULT/Dirichlet05/FedAvg/ANN/4Clients/1.txt.json` |
| 41 | ADULT | Dirichlet05 | FedBE | ANN | 4Clients | 128 | 256 | 2,788 | `ADULT/Dirichlet05/FedBE/ANN/4Clients/1.txt.json` |
| 42 | ADULT | Dirichlet05 | FedBEns | ANN | 4Clients | 128 | 256 | 2,898 | `ADULT/Dirichlet05/FedBEns/ANN/4Clients/1.txt.json` |
| 43 | ADULT | Dirichlet05 | FedCOG | ANN | 4Clients | 128 | 256 | 2,483 | `ADULT/Dirichlet05/FedCOG/ANN/4Clients/1.txt.json` |
| 44 | ADULT | Dirichlet05 | FedCVAE | ANN | 4Clients | 128 | 256 | 2,485 | `ADULT/Dirichlet05/FedCVAE/ANN/4Clients/1.txt.json` |
| 45 | ADULT | Dirichlet05 | FedDF | ANN | 4Clients | 128 | 256 | 2,483 | `ADULT/Dirichlet05/FedDF/ANN/4Clients/1.txt.json` |
| 46 | ADULT | Dirichlet05 | FedET | ANN | 4Clients | 128 | 256 | 2,483 | `ADULT/Dirichlet05/FedET/ANN/4Clients/1.txt.json` |
| 47 | ADULT | Dirichlet05 | FedF2DG | ANN | 4Clients | 128 | 256 | 2,485 | `ADULT/Dirichlet05/FedF2DG/ANN/4Clients/1.txt.json` |
| 48 | ADULT | Dirichlet05 | FedFed | ANN | 4Clients | 128 | 256 | 2,485 | `ADULT/Dirichlet05/FedFed/ANN/4Clients/1.txt.json` |
| 49 | ADULT | Dirichlet05 | FedFree | ANN | 4Clients | 128 | 256 | 2,485 | `ADULT/Dirichlet05/FedFree/ANN/4Clients/1.txt.json` |
| 50 | ADULT | Dirichlet05 | FedGen | ANN | 4Clients | 128 | 256 | 2,483 | `ADULT/Dirichlet05/FedGen/ANN/4Clients/1.txt.json` |
| 51 | ADULT | Dirichlet05 | FedLGD | ANN | 4Clients | 128 | 256 | 2,484 | `ADULT/Dirichlet05/FedLGD/ANN/4Clients/1.txt.json` |
| 52 | ADULT | Dirichlet05 | FedLMG | ANN | 4Clients | 128 | 256 | 2,886 | `ADULT/Dirichlet05/FedLMG/ANN/4Clients/1.txt.json` |
| 53 | ADULT | Dirichlet05 | FedOMG | ANN | 4Clients | 128 | 256 | 2,484 | `ADULT/Dirichlet05/FedOMG/ANN/4Clients/1.txt.json` |
| 54 | ADULT | Dirichlet05 | FedRevive | ANN | 4Clients | 128 | 256 | 2,486 | `ADULT/Dirichlet05/FedRevive/ANN/4Clients/1.txt.json` |
| 55 | ADULT | Dirichlet05 | FedSum | ANN | 4Clients | 128 | 256 | 2,481 | `ADULT/Dirichlet05/FedSum/ANN/4Clients/1.txt.json` |
| 56 | ADULT | Dirichlet05 | FedTMOS | ANN | 4Clients | 128 | 256 | 2,883 | `ADULT/Dirichlet05/FedTMOS/ANN/4Clients/1.txt.json` |
| 57 | ADULT | Dirichlet05 | FederatedAverageWithPo | ANN | 4Clients | 128 | 256 | 2,501 | `ADULT/Dirichlet05/FederatedAverageWithPo/ANN/4Clients/1.txt.json` |
| 58 | ADULT | Dirichlet05 | FederatedAverage | ANN | 4Clients | 128 | 256 | 2,495 | `ADULT/Dirichlet05/FederatedAverage/ANN/4Clients/1.txt.json` |
| 59 | ADULT | Dirichlet05 | FederatedProximal | ANN | 4Clients | 128 | 256 | 2,495 | `ADULT/Dirichlet05/FederatedProximal/ANN/4Clients/1.txt.json` |
| 60 | ADULT | Dirichlet05 | GeFL | ANN | 4Clients | 128 | 256 | 2,883 | `ADULT/Dirichlet05/GeFL/ANN/4Clients/1.txt.json` |
| 61 | ADULT | Dirichlet05 | MAHyFL | ANN | 4Clients | 128 | 256 | 2,484 | `ADULT/Dirichlet05/MAHyFL/ANN/4Clients/1.txt.json` |
| 62 | ADULT | Dirichlet05 | Simple_mFairFL | ANN | 4Clients | 128 | 256 | 2,494 | `ADULT/Dirichlet05/Simple_mFairFL/ANN/4Clients/1.txt.json` |
| 63 | ADULT | Dirichlet05 | mFairFL | ANN | 4Clients | 128 | 256 | 2,487 | `ADULT/Dirichlet05/mFairFL/ANN/4Clients/1.txt.json` |
| 64 | ADULT | Uniform | FairFed | ANN | 4Clients | 64 | 256 | 2,947 | `ADULT/Uniform/FairFed/ANN/4Clients/1.txt.json` |
| 65 | ADULT | Uniform | FedProto | ANN | 4Clients | 64 | 256 | 2,942 | `ADULT/Uniform/FedProto/ANN/4Clients/1.txt.json` |
| 66 | ADULT | Uniform | FedRenyi | ANN | 4Clients | 64 | 256 | 2,949 | `ADULT/Uniform/FedRenyi/ANN/4Clients/1.txt.json` |
| 67 | ADULT | Uniform | FederatedAverage | ANN | 4Clients | 64 | 256 | 2,752 | `ADULT/Uniform/FederatedAverage/ANN/4Clients/1.txt.json` |
| 68 | ADULT | Uniform | LoGoFair | ANN | 4Clients | 64 | 256 | 2,950 | `ADULT/Uniform/LoGoFair/ANN/4Clients/1.txt.json` |
| 69 | ADULT | Uniform | Scaffold | ANN | 4Clients | 64 | 256 | 2,950 | `ADULT/Uniform/Scaffold/ANN/4Clients/1.txt.json` |
| 70 | ADULT | Uniform | mFairFL | ANN | 4Clients | 64 | 256 | 2,951 | `ADULT/Uniform/mFairFL/ANN/4Clients/1.txt.json` |
| 71 | CREMA-D | Uniform | FederatedAverage | AUDIOCNNRNN | 4Clients | 16 | 256 | 2,778 | `CREMA-D/Uniform/FederatedAverage/AUDIOCNNRNN/4Clients/1.txt.json` |
| 72 | CREMA-D | Uniform | PDFFed | AUDIOCNNRNN | 4Clients | 16 | 256 | 2,768 | `CREMA-D/Uniform/PDFFed/AUDIOCNNRNN/4Clients/1.txt.json` |
| 73 | CelebA | Dirichlet01 | FL_FairBatch | BERTCLASSIFIER | 2Clients | 256 | 256 | 2,942 | `CelebA/Dirichlet01/FL_FairBatch/BERTCLASSIFIER/2Clients/1.txt.json` |
| 74 | CelebA | Dirichlet05 | FL_FairBatch | BERTCLASSIFIER | 2Clients | 256 | 256 | 2,928 | `CelebA/Dirichlet05/FL_FairBatch/BERTCLASSIFIER/2Clients/2.txt.json` |
| 75 | CelebA | Dirichlet05 | FedAvg | BERTCLASSIFIER | 20Clients | 256 | 256 | 2,564 | `CelebA/Dirichlet05/FedAvg/BERTCLASSIFIER/20Clients/1.txt.json` |
| 76 | CelebA | Dirichlet1 | FL_FairBatch | BERTCLASSIFIER | 2Clients | 256 | 256 | 2,931 | `CelebA/Dirichlet1/FL_FairBatch/BERTCLASSIFIER/2Clients/3.txt.json` |
| 77 | CelebA | Uniform | FedProto | BERTCLASSIFIER | 20Clients | 64 | 256 | 2,713 | `CelebA/Uniform/FedProto/BERTCLASSIFIER/20Clients/10.txt.json` |
| 78 | CelebA | Uniform | FedProto | BERTCLASSIFIER | 30Clients | 64 | 256 | 2,718 | `CelebA/Uniform/FedProto/BERTCLASSIFIER/30Clients/11.txt.json` |
| 79 | CelebA | Uniform | FedProto | BERTCLASSIFIER | 40Clients | 64 | 256 | 2,715 | `CelebA/Uniform/FedProto/BERTCLASSIFIER/40Clients/12.txt.json` |
| 80 | LFWA+ | Dirichlet05 | FedAvg | BERTCLASSIFIER | 20Clients | 256 | 256 | 2,473 | `LFWA+/Dirichlet05/FedAvg/BERTCLASSIFIER/20Clients/1.txt.json` |
| 81 | PAD-UFES-20 | Uniform | FedAvg | BERTCLASSIFIER | 20Clients | 32 | 256 | 2,717 | `PAD-UFES-20/Uniform/FedAvg/BERTCLASSIFIER/20Clients/1.txt.json` |
| 82 | UTKFace | Dirichlet05 | FedAvg | BERTCLASSIFIER | 4Clients | 64 | 256 | 2,571 | `UTKFace/Dirichlet05/FedAvg/BERTCLASSIFIER/4Clients/1.txt.json` |
| 83 | bios | Dirichlet01 | FairFed | BERTCLASSIFIER | 10Clients | 256 | 256 | 3,661 | `bios/Dirichlet01/FairFed/BERTCLASSIFIER/10Clients/1.txt.json` |
| 84 | bios | Dirichlet01 | FedAvg | BERTCLASSIFIER | 10Clients | 256 | 256 | 3,660 | `bios/Dirichlet01/FedAvg/BERTCLASSIFIER/10Clients/1.txt.json` |
| 85 | bios | Dirichlet01 | FedRep | BERTCLASSIFIER | 10Clients | 256 | 256 | 3,660 | `bios/Dirichlet01/FedRep/BERTCLASSIFIER/10Clients/1.txt.json` |
| 86 | bios | Dirichlet01 | LoGoFair | BERTCLASSIFIER | 10Clients | 256 | 256 | 3,654 | `bios/Dirichlet01/LoGoFair/BERTCLASSIFIER/10Clients/1.txt.json` |
| 87 | bios | Dirichlet05 | FedAvg | BERTCLASSIFIER | 4Clients | 16 | 256 | 3,223 | `bios/Dirichlet05/FedAvg/BERTCLASSIFIER/4Clients/1.txt.json` |
| 88 | bios | Uniform | FairFed | BERTCLASSIFIER | 10Clients | 256 | 256 | 3,651 | `bios/Uniform/FairFed/BERTCLASSIFIER/10Clients/1.txt.json` |
| 89 | bios | Uniform | FedAvg | BERTCLASSIFIER | 10Clients | 256 | 256 | 3,650 | `bios/Uniform/FedAvg/BERTCLASSIFIER/10Clients/1.txt.json` |
| 90 | bios | Uniform | FedRep | BERTCLASSIFIER | 10Clients | 256 | 256 | 3,651 | `bios/Uniform/FedRep/BERTCLASSIFIER/10Clients/1.txt.json` |
| 91 | bios | Uniform | LoGoFair | BERTCLASSIFIER | 10Clients | 256 | 256 | 3,650 | `bios/Uniform/LoGoFair/BERTCLASSIFIER/10Clients/1.txt.json` |

## G3 按数据集汇总

| 数据集 | 单 seed 格数 |
|---|---|
| ACSIncome | 36 |
| ADULT | 34 |
| CREMA-D | 2 |
| CelebA | 7 |
| LFWA+ | 1 |
| PAD-UFES-20 | 1 |
| UTKFace | 1 |
| bios | 9 |

## 删除前置条件（第 1 步执行时填写）

- [ ] 归档文件：`results/_archive/batch_inconsistent_<timestamp>.tar.gz`
- [ ] 归档 sha256：`________________`
- [ ] 归档可读性验证（`tar -tzf` 条目数 = 50）
- [ ] 删除前本地文件数：`2567`；删除后应为 `2517`
- [ ] .30 删除 49 条、.31 删除 49 条，残留校验 0
