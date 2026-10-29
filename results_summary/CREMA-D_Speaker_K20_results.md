# CREMA-D / Speaker / K=20 实验结果

## 实验配置

| 项 | 值 |
|---|---|
| 任务 | `AUD_CLF`（CREMA-D，HAP-vs-rest 二分类） |
| 数据 | CREMA-D，7,442 条音频、91 位演员，按说话人分层划分（test_ratio=0.2, seed=42） |
| 敏感属性 | 种族（Caucasian=1，非 Caucasian=0；`Race == Unknown` 的演员整体丢弃） |
| 模型 | `AUDIOCNNRNN` |
| 客户端划分 | `Speaker`（按说话人划分，91 位说话人，K=20） |
| 客户端数 | K = 20 |
| 通信轮数 | I = 200 |
| 本地 epoch | T = 2 |
| batch size | 64（有效 batch 256，梯度累积 4） |
| 优化器 | Adam，lr = 1e-3 |
| repeat 次数 | 3（Mean±STD 为 3 次 repeat 统计） |
| 结果文件 | `192.168.1.31:/root/fl/result_path/CREMA-D/Speaker/<algo>/AUDIOCNNRNN/20Clients/1.txt` |
| 记录日期 | 2026-10-19 |

指标定义：

- **ACC**：整体准确率
- **ACC_gap**：各受保护组准确率的极差（max − min）
- **DEO**：|TPR(A=0) − TPR(A=1)|，即均等机会差
- **SPD**：P(ŷ=1|A=0) − P(ŷ=1|A=1)，即统计均等差

## 结果（按 ACC 降序）

| 算法 | ACC | ACC_gap | DEO | SPD | HM |
|---|---|---|---|---|---|
| FedNova | 0.859±0.003 | 0.014 | 0.081 | 0.042 | 0.888±0.018 |
| FedAvg | 0.848±0.003 | 0.016 | 0.199 | 0.055 | 0.823±0.009 |
| FedFB | 0.839±0.006 | 0.017 | 0.089 | 0.036 | 0.873±0.012 |
| FedRenyi | 0.838±0.001 | 0.007 | 0.091 | 0.024 | 0.872±0.008 |
| FedProx | 0.835±0.003 | 0.017 | 0.181 | 0.079 | 0.827±0.014 |
| LoGoFair | 0.833±0.005 | 0.005 | 0.053 | 0.014 | 0.884±0.031 |
| FairFed | 0.832±0.002 | 0.004 | 0.028 | 0.005 | 0.897±0.007 |
| mFairFL | 0.830±0.001 | 0.004 | 0.038 | 0.009 | 0.890±0.023 |
| FL_FairBatch | 0.829±0.000 | 0.000 | 0.000 | 0.000 | 0.907±0.000 |
| FedProto | 0.829±0.000 | 0.000 | 0.000 | 0.000 | 0.907±0.000 |
| Scaffold | 0.829±0.000 | 0.000 | 0.000 | 0.000 | 0.907±0.000 |
| FederatedAverageWithPo | 0.829±0.003 | 0.010 | 0.073 | 0.018 | 0.873±0.043 |
| PDFFed_DP | 0.828±0.006 | 0.008 | 0.027 | 0.008 | 0.894±0.002 |
| FedRep | 0.825±0.012 | 0.009 | 0.039 | 0.018 | 0.887±0.025 |
| **PDFFed** | **0.809±0.022** | 0.016 | **0.046** | 0.032 | 0.875±0.011 |
| FedFair | 0.658±0.023 | 0.043 | 0.040 | 0.056 | 0.781±0.013 |

HM 指标说明：`HM = harmonic_mean(ACC, FR)`，其中 `FR = 1 − DEO`，与 `algorithm/*.py` 中
`FR = 1-DEO; HM = get_HM_by_two_value(accuracy, FR)` 的定义一致。HM 由每次 repeat 的
(ACC, DEO) 逐对计算后再取 Mean±STD，数据来源为各格子的 `1.txt.json`。

**注意**：退化算法（FL_FairBatch / FedProto / Scaffold）的 HM = 0.907 是虚高——全负类预测
使 DEO = 0（FR = 1），HM 退化为 harmonic_mean(0.829, 1.0)。读表时不能把这三个 HM 当作
"最公平"的证据。

## 结果（按 HM 降序）

| 算法 | HM | ACC | ACC_gap | DEO | SPD |
|---|---|---|---|---|---|
| FL_FairBatch | 0.907±0.000 | 0.829±0.000 | 0.000 | 0.000 | 0.000 |
| FedProto | 0.907±0.000 | 0.829±0.000 | 0.000 | 0.000 | 0.000 |
| Scaffold | 0.907±0.000 | 0.829±0.000 | 0.000 | 0.000 | 0.000 |
| FairFed | 0.897±0.007 | 0.832±0.002 | 0.004 | 0.028 | 0.005 |
| PDFFed_DP | 0.894±0.002 | 0.828±0.006 | 0.008 | 0.027 | 0.008 |
| mFairFL | 0.890±0.023 | 0.830±0.001 | 0.004 | 0.038 | 0.009 |
| FedNova | 0.888±0.018 | 0.859±0.003 | 0.014 | 0.081 | 0.042 |
| FedRep | 0.887±0.025 | 0.825±0.012 | 0.009 | 0.039 | 0.018 |
| LoGoFair | 0.884±0.031 | 0.833±0.005 | 0.005 | 0.053 | 0.014 |
| **PDFFed** | **0.875±0.011** | 0.809±0.022 | 0.016 | 0.046 | 0.032 |
| FedFB | 0.873±0.012 | 0.839±0.006 | 0.017 | 0.089 | 0.036 |
| FederatedAverageWithPo | 0.873±0.043 | 0.829±0.003 | 0.010 | 0.073 | 0.018 |
| FedRenyi | 0.872±0.008 | 0.838±0.001 | 0.007 | 0.091 | 0.024 |
| FedProx | 0.827±0.014 | 0.835±0.003 | 0.017 | 0.181 | 0.079 |
| FedAvg | 0.823±0.009 | 0.848±0.003 | 0.016 | 0.199 | 0.055 |
| FedFair | 0.781±0.013 | 0.658±0.023 | 0.043 | 0.040 | 0.056 |

排除三个退化算法后，HM 前三为 **FairFed（0.897）> PDFFed_DP（0.894）> mFairFL（0.890）**，
PDFFed（0.875）排第 7；PDFFed 的 HM 与 FedFB / FederatedAverageWithPo / FedRenyi 处于同一档。

## 关键观察

1. **三个算法退化**：`FL_FairBatch`、`FedProto`、`Scaffold` 的 ACC 恰为 0.829，且 ACC_gap/DEO/SPD 全为 0。0.829 等于负类占比，说明模型退化为"全部预测为负类"的常数预测器；这三格不能作为算法有效性的证据。
2. **PDFFed 在该任务上不占优**：ACC 0.809 为未退化算法中最低，且低于多数类基线 0.829；三次 repeat 的 STD 达 0.022，是全部算法中最大。其 DEO 0.046 虽优于 FedAvg/FedProx/FedFB，但排序上落后于 FairFed（0.028）、PDFFed_DP（0.027）、mFairFL（0.038）、FedRep（0.039）、FedFair（0.040）。
3. **FedAvg / FedProx 的 DEO 最高**（0.199 / 0.181），是精度最高但组间最不公平的一类；FedNova 在保持 0.859 最高 ACC 的同时 DEO 仅 0.081，是精度-公平权衡最好的基线。
4. **FedFair 异常**：ACC 0.658±0.023 明显低于其余算法，需要单独核查其超参配置。

## 复现命令要点

```bash
python main_AUD_CLF.py \
  -dataset CREMA-D -split_strategy Speaker -task AUD_CLF -model_type AUDIOCNNRNN \
  -num_clients_K 20 -communication_round_I 200 -algorithm_epoch_T 2 \
  -batch_size 64 -test_batch_size 64 -test_eval_freq 50 \
  -optimize_method adam -learning_rate 0.001 \
  -exp_repeat_times 3 -parallel_repeats 1 -base_seed 42 -resume
```
