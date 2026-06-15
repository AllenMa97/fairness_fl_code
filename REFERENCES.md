# Algorithm & Dataset References / 算法与数据集参考文献

## Algorithms / 算法

### Federated Learning Baselines / 联邦学习基线

| Algorithm | Paper | Description |
|-----------|-------|-------------|
| **FedAvg** | McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS 2017. [PDF](https://arxiv.org/abs/1602.05629) | Standard federated averaging |
| **FedProx** | Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020. [PDF](https://arxiv.org/abs/1812.06127) | Proximal term to constrain local updates |
| **SCAFFOLD** | Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning", ICML 2020. [PDF](https://arxiv.org/abs/1910.06378) | Control variates for variance reduction |
| **FedNova** | Wang et al., "Federated Learning with Matched Averaging", ICLR 2020. [PDF](https://arxiv.org/abs/2007.07481) | Normalized averaging by local update steps |
| **FedProto** | Zhao et al., "Federated Learning with Non-IID Data via Local Global Prototypes", 2021. | Prototype-based federated aggregation |
| **FedRep** | Li et al., "Federated Learning with Representation Ensembling", 2021. | Representation alignment for FL |
| **SeparateTraining** | — | Non-federated independent training baseline |

### Fair Federated Learning / 公平联邦学习

| Algorithm | Paper | Description |
|-----------|-------|-------------|
| **PDFFed** | — | Prototype-Driven Fair Federated Learning. Global-local prototype alignment, classifier supervision, group representation contrast, selective upload |
| **FairFed** | Liang et al., "FairFed: A Fairness-Aware Federated Learning Framework", 2021. [PDF](https://arxiv.org/abs/2110.00857v3) | Fairness regularization in FL |
| **FedFair** | — | Fairness via sensitive attribute prediction difference. [PDF](https://arxiv.org/pdf/2109.05662) |
| **FedFB** | — | Weighted loss based fair FL. [PDF](https://arxiv.org/pdf/2110.15545) |
| **FedMix** | — | MixUp in FL feature space for fairness. [PDF](https://arxiv.org/pdf/2107.00233) |
| **NaiveMix** | — | Naive application of standard MixUp in FL. [PDF](https://arxiv.org/pdf/2107.00233) |
| **FL_FairBatch** | — | FairBatch resampling strategy in FL |
| **mFairFL** | — | Multi-fair FL with learnable hyperparameters. AAAI 2024. [PDF](https://arxiv.org/pdf/2312.05551) |
| **Simple_mFairFL** | — | Simplified baseline of mFairFL. [PDF](https://arxiv.org/pdf/2312.05551) |
| **FedRenyi** | Federated Rényi Fair Inference in Federated Heterogeneous System(https://proceedings.mlr.press/v286/ma25a.html) | Renyi entropy based fair federated inference |
| **PraFFL** | — | Preference/HyperNetwork based fair FL with personalized classifier heads |
| **FedFACT** | — | Cost-sensitive fair FL with dual variables |
| **LoGoFair** | — | Post-processing for local and global fairness in FL. [PDF](https://arxiv.org/pdf/2503.17231) |

### One-Shot Federated Learning / 单轮联邦学习

| Algorithm | Paper | Description |
|-----------|-------|-------------|
| **DOSFL** | — | Distilled One-Shot Federated Learning. [PDF](https://arxiv.org/abs/2009.07999v3) |
| **CoBoosting** | — | Data and ensemble co-boosting for one-shot FL. [PDF](https://openreview.net/pdf?id=tm8s3696Ox) |
| **OSFL** | Guha et al., "One-Shot Federated Learning: Theoretical Foundations and Algorithmic Connections", 2019. [PDF](https://arxiv.org/pdf/1902.11175v2) | Data-free knowledge distillation via ensemble of client models on synthetic noise inputs |

### Other / 其他

| Algorithm | Description |
|-----------|-------------|
| **ProxProbability** | Probability-based proximal term for FL |

---

## Datasets / 数据集

### Tabular Datasets / 表格数据集

| Dataset | Source | Task | Sensitive Attribute | Size |
|---------|--------|------|---------------------|------|
| **ADULT** | UCI Machine Learning Repository. [Link](https://archive.ics.uci.edu/ml/datasets/adult). Code adapted from [Renyi-Fair-Inference](https://github.com/optimization-for-data-driven-science/Renyi-Fair-Inference) | Income prediction (>50K) | Gender (Male/Female), Race (White/Non-white) | ~48 MB |
| **COMPAS** | ProPublica. [Analysis](https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb). [Data](https://github.com/propublica/compas-analysis) | Recidivism prediction | Race (African-American/Caucasian), Gender | ~5 MB |
| **DRUG** | UCI Machine Learning Repository. [Link](https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29) | Volatile substance abuse prediction | Ethnicity (White/Non-white), Gender | <1 MB |
| **DUTCH** | Dutch Census 2001. Referenced in [Fairness-aware Agnostic Federated Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6137304) | Income level (high/low) | Gender (Male/Female) | ~56 MB |

### Image Datasets / 图像数据集

| Dataset | Source | Task | Sensitive Attribute | Size |
|---------|--------|------|---------------------|------|
| **CelebA** | Liu et al., "Deep Learning Face Attributes in the Wild", ICCV 2015. [Website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | Attractiveness prediction (attr #3) | Male (attr #21) | ~1.4 GB (images) + ~50 MB (annotations) |
| **UTKFace** | [GitHub](https://susanqq.github.io/UTKFace/) | Gender prediction | Race (Black/Others) | ~1.5 GB |
| **FairFace** | Karkkainen et al., 2021. [Kaggle](https://www.kaggle.com/datasets/alexstvn/fairface) | Service test prediction | Gender (Male/Female) | ~500 MB |
| **LFWA+** | [Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pQ3NzdzRhVUhSams) | Attractiveness prediction (attr #2) | Gender (attr #20) | ~100 MB |

### Text Datasets / 文本数据集

| Dataset | Source | Task | Sensitive Attribute | Size |
|---------|--------|------|---------------------|------|
| **bios** | De-Arteaga et al., "Bias in Bios", FAT 2019. [HuggingFace](https://huggingface.co/datasets/LabHC/bias_in_bios) | Profession classification (binary) | Gender | ~86 MB |
| **moji** | — | Sentiment analysis | Sentiment bias | ~82 MB |
