# Batch Size 消融实验数据（b64 vs b256）

本目录集中存放「真实批 64」与「真实批 256」的实验结果，用于分析 batch size 对
联邦学习公平性指标的影响，作为消融实验（ablation）的数据来源。

生成时间：2026-11-09
数据来源：`result_path/`（.30 + .31 两机 union 后的 2806 个结果 json）

## 一、口径定义（非常重要）

本仓库的 `batch_size` 语义在提交 `c9b96ada5` 前后不同，归档时统一换算成**有效批**：

| 代码版本 | 字段组合 | 真实前向批 | 累积步数 | 有效批 |
|---|---|---|---|---|
| 新（`experiment_config` 含 `mini_batch`） | `mini_batch=64` + `effective_batch_size=256` | 64 | 4 | **256** |
| 新 | `mini_batch=64` + `effective_batch_size=64` | 64 | 1 | **64** |
| 旧（只有 `batch_size`） | `batch_size=64` | 64 | 1 | **64** |
| 旧 | `batch_size=256` | 256 | 1 | **256** |

换算规则：有 `effective_batch_size` 时取它，否则取 `batch_size`。

## 二、目录结构

```
ablations/batch_size/
├── README.md              本文件
├── index_all_b64.csv      全部 292 个有效批=64 的结果清单
├── index_pairs.csv        61 个「同一格同时有 64 和 256」的配对对照表
├── b64/                   292 个 b64 结果 json（保持原 result_path 相对路径）
└── b256_matched/          61 个与 b64 一一配对的 b256 结果 json
```

`index_pairs.csv` 字段：`dataset, split, algorithm, K, acc_b64, acc_b256, delta_acc,
b64_path, b256_path, b64_schema_new, b256_schema_new`。

## 三、可比格与差异统计

全库有效批分布：

| 有效批 | 格数 |
|---|---|
| 256 | 2483 |
| 64 | 292 |
| 128 | 28 |
| 16 | 3 |

「同一 dataset/split/algorithm/K 同时存在 b64 与 b256」的可比格共 **61 个**：

| 数据集 | 可比格数 | 平均 \|ΔACC\| | 最大 \|ΔACC\| |
|---|---|---|---|
| LFWA+ | 44 | 1.49% | 7.90% |
| UTKFace | 13 | 4.82% | 7.89% |
| CelebA | 4 | 13.69% | 18.17% |

对照基准：两机跑**完全相同的配置**（PAD，`mini_batch=32 + eff=256`）时，
结果差异为 0.23% / 0.46%，可视为纯数值噪声上限（浮点累加顺序 + cuDNN 非确定性）。

**结论**：LFWA+ 的 1.49% 接近噪声量级，UTKFace 与 CelebA 的差异显著超出噪声，
说明 batch size 对后两者（尤其 CelebA 的 FedProto）影响真实存在。

## 四、使用注意

1. **b64 全量 292 格中，有相当一部分是音频数据集的原始设定**（`AudioMNIST` 64 格、
   `VCTK` 40 格、`CREMA-D` 48 格、`RAVDESS` 16 格等），它们的 64 不是为了消融而设，
   而是该数据集本身就用 64 跑。做 bs 消融时请只用 `index_pairs.csv` 里的 61 个格。
2. 61 个可比格里，b64 侧部分来自旧口径（只有 `batch_size`），b256 侧部分来自新口径
   （`mini_batch` + `effective_batch_size`）。`index_pairs.csv` 的
   `b64_schema_new` / `b256_schema_new` 列标明了每侧的代码版本，便于区分
   「批大小差异」与「是否梯度累积」两个变量。
3. **已知异常**：`CREMA-D` 有 4 格 ACC ≈ 0.171（6 类随机水平约 16.7%），为训练发散
   记录，不应作为有效数据使用。这批记录未纳入本目录的配对表。
