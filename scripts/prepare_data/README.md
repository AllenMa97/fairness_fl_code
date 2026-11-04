# 数据准备脚本 / Data preparation

联邦公平性基准中两个「体积大、不入库」的数据集：VCTK（音频）与
PAD-UFES-20（医学图像）。原始数据不提交 git，按下面的脚本重新下载与解压。

| 数据集 | 任务 | 原始归档 | 解压后 | 脚本 |
|---|---|---|---|---|
| VCTK | `AUD_CLF` | `raw_data.zip` ~10.7 GB | `dataset/VCTK` ~10.2 GiB | `prepare_vctk.sh` |
| PAD-UFES-20 | `IMG_CLF` | `PAD-UFES-20.zip` ~3.6 GB | `dataset/PAD-UFES-20` ~6.7 GiB | `prepare_pad_ufes.sh` |

> 仓库 `.gitignore` 已忽略 `dataset/VCTK/` 与 `dataset/PAD-UFES-20/`，不要 `git add` 原始数据。

## VCTK

- 来源：22 kHz 版本 `https://huggingface.co/datasets/arampacha/vctk_22kHz`
  （脚本优先走 `hf-mirror.com` 镜像，失败自动切官方站）；
  说话人表 `speaker-info.txt` 来自 `CSTR-Edinburgh/vctk`（性别、口音）。
- 存放：`dataset/VCTK/`
- 目录结构：

```
dataset/VCTK/
├── speaker-info.txt          # 表头 ID AGE GENDER ACCENTS REGION COMMENT
├── p225/p225_001.wav ...
├── p226/...
└── s5/...                    # 官方补充说话人（部分镜像可能没有）
```

- 校验：`find dataset/VCTK -name '*.wav' | wc -l` 应为 **36940**；
  说话人目录 **94** 个；`speaker-info.txt` 111 行（含表头）；
  `du -sb dataset/VCTK` ≈ **10,928,439,432** 字节（不含归档）。
- 说明：代码按 `{speaker}_{utt}.wav` 解析，并用 `speaker-info.txt` 映射
  性别（标签）与英伦/海外口音（敏感属性）；`p280` 口音为 `Unknown`，加载时自动剔除。

```bash
bash scripts/prepare_data/prepare_vctk.sh
# 已有归档时可直接解压，脚本会自动复用 raw_data.zip
```

## PAD-UFES-20

- 来源：Mendeley Data，DOI `10.17632/zr7vgbcyr2.1`，
  官方 API `https://data.mendeley.com/public-api/zip/zr7vgbcyr2/download/1`。
- 存放：`dataset/PAD-UFES-20/`
- 目录结构：

```
dataset/PAD-UFES-20/
├── metadata.csv              # 每张图的诊断 / Fitzpatrick 肤色 / patient_id
└── images/
    ├── imgs_part_1/*.png     # 911 张
    ├── imgs_part_2/*.png     # 659 张
    └── imgs_part_3/*.png     # 728 张
```

- 校验：png 总数应为 **2298**；`metadata.csv` 应为 **316209** 字节；
  `du -sb dataset/PAD-UFES-20` ≈ **7,199,041,024** 字节（`images/`，不含归档）。
- 标签：`BCC/SCC/MEL` = 恶性，`ACK/NEV/SEK` = 良性；敏感属性为
  `fitspatrick` I–II vs III–VI，缺失肤色的样本加载时丢弃。

```bash
bash scripts/prepare_data/prepare_pad_ufes.sh
```

## 在两台训练机之间同步

数据只需在一台机器准备，另一台用内网 rsync 拉取（不要拉回本地 Windows）：

```bash
rsync -a --info=progress2 root@192.168.1.31:/root/fl/dataset/VCTK \
      root@192.168.1.31:/root/fl/dataset/PAD-UFES-20 /root/fl/dataset/
```
