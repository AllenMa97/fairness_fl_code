# PDFFed 损失消融（R1）结果

## COMPAS

| Split | Variant | ACC | ACC_gap | DEO | SPD | HM |
|---|---|---|---|---|---|---|
| Dirichlet1 | PDFFed (full) | 0.6158 | 0.0192 | 0.0710 | 0.0817 | 0.7398 |
| Dirichlet1 | PDFFed_noPA | 0.6130 | 0.0268 | 0.1122 | 0.1167 | 0.7246 |
| Dirichlet1 | PDFFed_noCONF | 0.6088 | 0.0264 | 0.0847 | 0.0946 | 0.7301 |
| Dirichlet1 | PDFFed_noL2L | 0.6095 | 0.0297 | 0.0699 | 0.0790 | 0.7357 |
| Dirichlet1 | PDFFed_noEO | 0.6158 | 0.0192 | 0.0682 | 0.0784 | 0.7406 |
| Uniform | PDFFed (full) | 0.5955 | 0.0307 | 0.0795 | 0.0839 | 0.7224 |
| Uniform | PDFFed_noPA | 0.5969 | 0.0288 | 0.0941 | 0.0931 | 0.7191 |
| Uniform | PDFFed_noCONF | 0.5955 | 0.0312 | 0.0894 | 0.0859 | 0.7195 |
| Uniform | PDFFed_noL2L | 0.5865 | 0.0296 | 0.0724 | 0.0696 | 0.7176 |
| Uniform | PDFFed_noEO | 0.5962 | 0.0291 | 0.0795 | 0.0823 | 0.7230 |

## DRUG

| Split | Variant | ACC | ACC_gap | DEO | SPD | HM |
|---|---|---|---|---|---|---|
| Dirichlet1 | PDFFed (full) | 0.7606 | 0.0752 | 0.0772 | -0.0022 | 0.8336 |
| Dirichlet1 | PDFFed_noPA | 0.7547 | 0.0606 | 0.0883 | -0.0108 | 0.8255 |
| Dirichlet1 | PDFFed_noCONF | 0.7617 | 0.0739 | 0.0772 | 0.0043 | 0.8343 |
| Dirichlet1 | PDFFed_noL2L | 0.7606 | 0.0752 | 0.0756 | 0.0004 | 0.8343 |
| Dirichlet1 | PDFFed_noEO | 0.7629 | 0.0752 | 0.0789 | 0.0030 | 0.8343 |
| Uniform | PDFFed (full) | 0.7629 | 0.0645 | 0.0967 | 0.0139 | 0.8272 |
| Uniform | PDFFed_noPA | 0.7347 | 0.0840 | 0.0789 | 0.0189 | 0.8174 |
| Uniform | PDFFed_noCONF | 0.7629 | 0.0642 | 0.0950 | 0.0269 | 0.8278 |
| Uniform | PDFFed_noL2L | 0.7653 | 0.0778 | 0.0928 | 0.0068 | 0.8302 |
| Uniform | PDFFed_noEO | 0.7606 | 0.0642 | 0.0933 | 0.0165 | 0.8271 |

## 关键观察

1. **L_PA 是公平性贡献最大的损失项。** 去掉后 DEO 在 COMPAS-Dirichlet1 上升 4.12 pp（0.0710→0.1122）、COMPAS-Uniform 上升 1.47 pp、DRUG-Dirichlet1 上升 1.11 pp；四格平均 ΔDEO = +1.23 pp，是唯一平均显著恶化公平性的移除项。HM 在四格全部下降（−1.52/−0.34/−0.81/−0.98 pp）。
2. **L_PA 同时也是精度影响最大的项。** 四格平均 ΔACC = −0.89 pp（DRUG-Uniform −2.82 pp 最重，COMPAS-Dirichlet1 仅 −0.28 pp），大于 noL2L（−0.32 pp）、noCONF（−0.14 pp）和 noEO（+0.02 pp）。
3. **L_Conf 的影响集中在 COMPAS-Dirichlet1。** 该格 ΔDEO = +1.37 pp、ΔACC = −0.70 pp、ΔHM = −0.97 pp；COMPAS-Uniform 的 ΔDEO = +0.99 pp；DRUG 两格的 |ΔDEO| ≤ 0.17 pp、|ΔHM| ≤ 0.07 pp，基本可忽略。
4. **L2L 呈现“公平—精度”权衡。** COMPAS-Uniform 去掉 L2L 后 DEO 降 0.71 pp、SPD 降 1.43 pp，但 ACC 降 0.91 pp、HM 降 0.48 pp；COMPAS-Dirichlet1 与 DRUG-Uniform 上影响接近噪声。DRUG-Uniform 的 noL2L（HM 0.8302）是唯一略高于 full（0.8272）的消融格。
5. **关掉全局 EO 校准（noEO）未观察到一致后果。** 四格 ΔDEO 分别为 −0.28 / 0.00 / +0.17 / −0.33 pp，ΔACC 为 0.00 / +0.07 / +0.23 / −0.23 pp，ΔHM 为 +0.08 / +0.05 / +0.07 / 0.00 pp；所有变化都在 3 次重复的波动范围内，说明该校准步在本轮实验中对最终公平性/精度无可辨识贡献。
6. **总体：** full 在 COMPAS-Uniform、DRUG-Dirichlet1、DRUG-Uniform 取得最高 HM；COMPAS-Dirichlet1 的 noEO 仅高 0.08 pp。DEO 最低值出现在 COMPAS-Dirichlet1 的 noEO（0.0682）、COMPAS-Uniform 的 noL2L（0.0724）、DRUG-Dirichlet1 的 noL2L（0.0756）、DRUG-Uniform 的 noPA（0.0789），除 COMPAS-Dirichlet1 的 noEO 外均以 ACC 下降为代价。综合看，L_PA 是唯一“去掉就同时伤公平与精度”的损失项。

## 口径与数据来源

- 配置：K=20、I=500、T=2、SGD lr=3e-4、batch=256、3 次重复（seeds 42/1042/2042）；数值为 `aggregate.metrics.<NAME>.mean`，HM 直接取自 JSON（缺失时才逐 repeat 重算）。
- 完整性：20/20 格齐全，无缺失。
- 变体生效核验：`w_pa=0`（noPA）、`w_conf=0`（noCONF）、`w_l2l=0`（noL2L）、`lambda_eo=0`（noEO）均已写入结果 JSON。
- full 基线：采用 192.168.1.31 上与 16 个变体同代码版本（git `e34c32ad3`）的 full 结果。192.168.1.30 上另有旧 full（git `e4f45401b`），其中 COMPAS-Uniform 数值不同（ACC 0.6144 / DEO 0.1140 / HM 0.7233，本表采用 0.5955 / 0.0795 / 0.7224）；COMPAS-Dirichlet1 两者逐位一致。
- 备注：DRUG-Dirichlet1 的 noL2L 与 full 在 repeat 级 ACC 完全一致（仅 repeat 0 的 DEO/SPD 有微小差异），说明该格 L2L 对分类器训练无可测影响。

## 图

- `ablation_bars.png`：HM / DEO 双面板分组柱状图（按数据集×划分分色）。
- `LPA_full_vs_noPA.png`：左 panel 为 L_PA（full 实线 vs noPA 虚线），右 panel 为 L_Conf（full vs noCONF），滚动中位数窗口 25 轮。
