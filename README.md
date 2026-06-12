# Fairness Federated Learning Framework

[中文说明](#中文说明) | [English](#english)

---

<a id="english"></a>

## English

A general-purpose federated learning framework for **group fairness** research, supporting 20+ algorithms across tabular, image, and text classification tasks.

### Features

- **20+ FL algorithms**: Including fairness-aware methods (PDFFed, FairFed, FedFair, FedFB, FedMix, mFairFL, LoGoFair, etc.) and standard FL baselines (FedAvg, FedProx, SCAFFOLD, FedNova, etc.)
- **3 task types**: Tabular classification, Image classification, Text classification
- **10 datasets**: ADULT, COMPAS, DRUG, DUTCH, CelebA, UTKFace, FairFace, LFWA+, bios, moji
- **Flexible experiment configuration**: Support for different data partitioning strategies (Uniform, Dirichlet), client counts, and model architectures

### Project Structure

```
fairness_fl_code/
├── main.py                    # General entry point
├── main_Tabular_CLF.py        # Tabular classification
├── main_IMG_CLF.py            # Image classification
├── main_SENT_CLF.py           # Text classification
├── experiment.py              # Experiment class
├── run_experiments.py         # Batch experiment runner
├── setup_data.py              # Dataset download/pack tool
├── environment.yml            # Conda environment
│
├── algorithm/                 # 20+ federated learning algorithms
├── hypothesis/                # Model definitions (ANN, CNN, BERT, LR)
├── moudle/                    # Core modules (dataset, dataloader, config)
├── tool/                      # Utilities (logger, checkpoint, utils)
├── dataset/                   # Datasets
├── save_path/                 # Model checkpoints (gitignored)
└── log_path/                  # Experiment logs (gitignored)
```

### Environment Setup

**Option 1: Conda (Recommended)**

```bash
conda env create -f environment.yml
conda activate FL
```

**Option 2: pip**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers numpy pandas scikit-learn scipy tqdm Pillow mat73 matplotlib openpyxl pyarrow requests psutil
```

### Datasets

#### Bundled (included in repo)

| Dataset | Task | Sensitive Attribute |
|---------|------|---------------------|
| ADULT | Tabular_CLF | Gender, Race |
| COMPAS | Tabular_CLF | Race, Gender |
| DRUG | Tabular_CLF | Ethnicity, Gender |
| DUTCH | Tabular_CLF | Gender |
| bios | SENT_CLF | Gender |
| moji | SENT_CLF | Sentiment bias |
| CelebA (annotations) | IMG_CLF | Male attribute |

#### Downloadable (via `setup_data.py`)

| Dataset | Task | Size |
|---------|------|------|
| CelebA (images) | IMG_CLF | ~1.4 GB |
| UTKFace (images) | IMG_CLF | ~1.5 GB |
| FairFace (images) | IMG_CLF | ~500 MB |
| LFWA+ (images) | IMG_CLF | ~100 MB |

```bash
python setup_data.py --check    # Check status
python setup_data.py             # Download all missing
python setup_data.py --list      # List all datasets
```

### Quick Start

```bash
# Tabular
python main_Tabular_CLF.py

# Image (download images first)
python setup_data.py --datasets celeba
python main_IMG_CLF.py

# Text
python main_SENT_CLF.py
```

### Supported Algorithms

| Category | Algorithms |
|----------|-----------|
| Fair FL | PDFFed, FairFed, FedFair, FedFB, FedMix, NaiveMix, FL_FairBatch, mFairFL, Simple_mFairFL, FedRenyi, PraFFL, FedFACT, LoGoFair |
| Standard FL | FedAvg, FedProx, SCAFFOLD, FedNova, FedProto, FedRep |
| One-Shot FL | DOSFL, CoBoosting, OSFL |
| Baseline | SeparateTraining |

See [REFERENCES.md](REFERENCES.md) for paper links.

---

<a id="中文说明"></a>

## 中文说明

一个通用的**联邦学习群组公平性**研究框架，支持 20+ 算法，覆盖表格分类、图像分类、文本分类三类任务。

### 特性

- **20+ 联邦学习算法**：包括公平性方法（PDFFed, FairFed, FedFair, FedFB, FedMix, mFairFL, LoGoFair 等）和标准 FL 基线（FedAvg, FedProx, SCAFFOLD, FedNova 等）
- **3 种任务类型**：表格分类、图像分类、文本分类
- **10 个数据集**：ADULT, COMPAS, DRUG, DUTCH, CelebA, UTKFace, FairFace, LFWA+, bios, moji
- **灵活的实验配置**：支持不同数据划分策略（Uniform, Dirichlet）、客户端数量、模型架构

### 环境配置

**方式一：Conda（推荐）**

```bash
conda env create -f environment.yml
conda activate FL
```

**方式二：pip**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers numpy pandas scikit-learn scipy tqdm Pillow mat73 matplotlib openpyxl pyarrow requests psutil
```

### 数据集

#### 仓库自带（clone 即用）

| 数据集 | 任务 | 敏感属性 |
|--------|------|----------|
| ADULT | Tabular_CLF | 性别、种族 |
| COMPAS | Tabular_CLF | 种族、性别 |
| DRUG | Tabular_CLF | 民族、性别 |
| DUTCH | Tabular_CLF | 性别 |
| bios | SENT_CLF | 性别 |
| moji | SENT_CLF | 情感偏见 |
| CelebA（标注文件） | IMG_CLF | 男性属性 |

#### 需下载（通过 `setup_data.py`）

| 数据集 | 任务 | 大小 |
|--------|------|------|
| CelebA（图片） | IMG_CLF | ~1.4 GB |
| UTKFace（图片） | IMG_CLF | ~1.5 GB |
| FairFace（图片） | IMG_CLF | ~500 MB |
| LFWA+（图片） | IMG_CLF | ~100 MB |

```bash
python setup_data.py --check    # 检查状态
python setup_data.py             # 下载所有缺失数据集
python setup_data.py --list      # 列出所有数据集
```

### 快速开始

```bash
# 表格分类
python main_Tabular_CLF.py

# 图像分类（需先下载图片）
python setup_data.py --datasets celeba
python main_IMG_CLF.py

# 文本分类
python main_SENT_CLF.py
```

### 支持的算法

| 类别 | 算法 |
|------|------|
| 公平联邦学习 | PDFFed, FairFed, FedFair, FedFB, FedMix, NaiveMix, FL_FairBatch, mFairFL, Simple_mFairFL, FedRenyi, PraFFL, FedFACT, LoGoFair |
| 标准联邦学习 | FedAvg, FedProx, SCAFFOLD, FedNova, FedProto, FedRep |
| 单轮联邦学习 | DOSFL, CoBoosting, OSFL |
| 基线 | SeparateTraining |

论文链接见 [REFERENCES.md](REFERENCES.md)。

---

## Citation / 引用

```
@misc{fairness_fl_code,
  title={Fairness Federated Learning Framework},
  author={},
  year={2025},
  url={https://github.com/YOUR_GITHUB_USERNAME/fairness_fl_code}
}
```
