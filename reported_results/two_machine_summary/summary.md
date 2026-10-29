# 两台机器（192.168.1.30 / 192.168.1.31）实验结果汇总

- **快照时间**：2026-10-22 09:33 (Asia/Shanghai)（状态探针）；结果扫描完成于 09:28
- **判据**：塌缩格 = `ACC <= floor + 0.01`，`floor` = 该格实际评测所用测试集的多数类占比，
  由每份结果 json 的 `aggregate.partition_fingerprints` 绑回
  `partition_cache/v2/<DS>/<fp>/metadata.json` 的逐客户端 `test_stats.label_counts` 求和得到
  （口径来自 `reported_results/collapse_audit/appendix_collapse_disclosure.md` §2）
- **合并规则**：cell key = `dataset/split/algo/hyp/K`；两台都有时取 `recorded_at` 最新的一份
  （审计 §9.7 的 newest 规则）
- **主口径**：同一 `(dataset, hyp, K)` 组内达到 `nsp_max` 个划分的算法；消融/超参扫描变体
  （`comb*`/`confX*`/`eoX*`/`pvc*`/`pve*`/`no*`/`lam*`/`eps*`）与他项目算法
  （PraFFL/FedFACT/DOSFL/OSFL/FedMix/NaiveMix/ProxProbability）一律排除
- **数据源**：`raw/cells_30.tsv`（933 行）、`raw/cells_31.tsv`（881 行）
- **重算**：`python analysis/build_two_machine_summary.py`

> 注意：此前用过的 `DEO=0 & SPD=0 & FR=1` 塌缩判据**不采用**。它在高 ACC 的健康格上产生假阳性
> （BANK 的 FedProx/PDFFed_DP/Scaffold、LFWA+ 的 FedRenyi、DRUG 的 LoGoFair），
> 又漏掉所有公平指标只是「很小」而非「恰好为 0」的真塌缩
> （BANK 的 FedFB 0.453、FairFace 与 UTKFace 全体）。


## T1 两台机器运行状态

> Probed live on both hosts at 2026-10-22 09:33 (Asia/Shanghai). Not derivable from the cell inventory.

| item | dot30 | dot31 |
|---|---|---|
| probe time | 2026-10-22 09:33:40 | 2026-10-22 09:33:57 |
| git HEAD | 6299c1e23 (Merge PR #67) | 6299c1e23 (Merge PR #67) |
| dirty files | 26 | 31 |
| running launchers | pareto30gpu (GPU0,2) 2d01h; k20fill30 (GPU1,3) 2d14h; redo/dutch30 (GPU0,1) 2d21h; redo/ab30 (GPU2,3,4) 2d21h | k20fill31 (GPU1,2) 2d15h; img31fill (GPU3) 1d00h; img31fill_b2 (GPU1) 20h; cpu31bank (CPU) 20h |
| our own jobs (VLLM_WORKER_RANK marker) | 0 | 0 |
| GPU util % | 0=93 1=99 2=99 3=99 4=0 5=19 | 0=99 1=100 2=100 3=3 4=99 5=99 |
| GPU mem MiB | 0=36297 1=37584 2=18494 3=23184 4=45745 5=14112 | 0=12716 1=29072 2=45374 3=23385 4=41748 5=41748 |
| idle-but-occupied GPU | GPU4: 45745 MiB @ 0% | GPU3: 23385 MiB @ 3% |
| result txt files | 943 | 1143 |
| txt written in last 3 h | 6 | 3 |
| disk | /dev/nvme0n1p2 1.4T/1.8T 85% | /dev/nvme0n1p2 1.3T/1.8T 74% |
| sent launcher (ours, terminated) | done.tsv: sent_batch 9 OK / 51 FAILED; sent_bios_b 17 OK / 19 FAILED | done.tsv empty |

## T2 地板、塌缩率与 PDFFed 系最好格（按塌缩率升序）

> floor = majority-class share of the test set the cells were actually evaluated on (bound per result json via aggregate.partition_fingerprints). A cell is COLLAPSE when ACC <= floor + 0.01. Sorted by collapse rate. ADULT/BANK/DRUG/COMPAS/DUTCH floors reproduce appendix_collapse_disclosure.md S3 exactly.

| dataset | floor | n_test | cells | cells_with_floor | collapsed | collapse_pct | no_floor | pdffed_best_acc | pdffed_best_excess_pp | pdffed_best_cell | pdffed_verdict | pdffed_dp_best_acc | pdffed_dp_best_excess_pp | pdffed_dp_best_cell | pdffed_dp_verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CelebA | 0.5042 | 19962 | 76 | 74 | 9 | 12.2 | 2 | - | - | - | no result | 0.691 | +18.7 | K30/Dirichlet1 | valid |
| COMPAS | 0.5397 | 478 | 324 | 291 | 44 | 15.1 | 33 | 0.616 | +7.6 | K20/Dirichlet1 | valid | 0.623 | +8.3 | K30/Dirichlet01 | valid |
| DUTCH | 0.5223 | 12084 | 184 | 160 | 28 | 17.5 | 24 | 0.794 | +27.2 | K20/Dirichlet1 | valid | 0.810 | +28.8 | K40/Dirichlet01 | valid |
| AudioMNIST | 0.5000 | 6000 | 64 | 64 | 19 | 29.7 | 0 | 0.928 | +42.8 | K20/Dirichlet1 | valid | 0.921 | +42.1 | K20/Dirichlet05 | valid |
| bios | MULTI | MULTI | 67 | 67 | 41 | 61.2 | 0 | 0.485 | -6.0 | K20/Dirichlet01 | COLLAPSE | 0.528 | -1.7 | K20/Dirichlet05 | COLLAPSE |
| LFWA+ | 0.6135 | 2629 | 82 | 80 | 66 | 82.5 | 2 | 0.606 | -0.8 | K30/Dirichlet05 | COLLAPSE | - | - | - | no result |
| BANK | 0.8776 | 13211 | 179 | 179 | 155 | 86.6 | 0 | 0.898 | +2.0 | K20/Dirichlet05 | valid | 0.897 | +1.9 | K40/Dirichlet01 | valid |
| CREMA-D_gender | 0.8293 | 1470 | 16 | 16 | 15 | 93.8 | 0 | 0.795 | -3.4 | K20/Speaker | COLLAPSE | 0.800 | -2.9 | K20/Speaker | COLLAPSE |
| RAVDESS | 0.8667 | 240 | 16 | 16 | 15 | 93.8 | 0 | 0.726 | -14.1 | K20/Speaker | COLLAPSE | 0.732 | -13.5 | K20/Speaker | COLLAPSE |
| CREMA-D | 0.8290 | 1468 | 46 | 46 | 44 | 95.7 | 0 | 0.809 | -2.0 | K20/Speaker | COLLAPSE | 0.829 | -0.0 | K20/Dirichlet1 | COLLAPSE |
| ADULT | 0.7638 | 16281 | 94 | 89 | 89 | 100.0 | 5 | 0.764 | +0.0 | K20/Dirichlet01 | COLLAPSE | 0.764 | +0.0 | K20/Dirichlet01 | COLLAPSE |
| DRUG | 0.7887 | 284 | 260 | 236 | 236 | 100.0 | 24 | 0.782 | -0.7 | K20/Dirichlet01 | COLLAPSE | 0.770 | -1.9 | K30/Uniform | COLLAPSE |
| FairFace | 0.5288 | 10954 | 34 | 34 | 34 | 100.0 | 0 | 0.476 | -5.3 | K20/Dirichlet01 | COLLAPSE | 0.475 | -5.4 | K20/Dirichlet01 | COLLAPSE |
| UTKFace | 0.5384 | 4822 | 85 | 85 | 85 | 100.0 | 0 | - | - | - | no result | - | - | - | no result |
| moji | MULTI | MULTI | 7 | 0 | 0 | NA | 7 | - | - | - | no result | - | - | - | no result |

## T3 PDFFed 系 vs 全场（主口径）

> Main calibre = algorithms reaching nsp_max partitions in that (dataset, hyp, K) group, with ablation/sweep variants and foreign algorithms excluded. Ranks are 1-based inside the group: HM and ACC descending, ACC_gap and DEO ascending (lower is better). Groups with nsp_max < 3 are dropped because ranks are not comparable.

| dataset | hyp | K | variant | n_algos_main | nsp_max | hm_rank | acc_rank | gap_rank | deo_rank | hm | acc | acc_gap | deo | valid_splits | exc_avg_pp | exc_max_pp | dhm_vs_fedavg | dacc_vs_fedavg | dhm_vs_best_baseline | best_baseline_by_hm | src |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AudioMNIST | AUDIOCNNRNN | 20 | PDFFed | 16 | 4 | 1 | 1 | 6 | 7 | 0.883 | 0.839 | 0.0123 | 0.0475 | 4/4 | +33.92 | +42.80 | +0.039 | +0.058 | +0.025 | FederatedAverageWithPo | .30@e4f45401b |
| AudioMNIST | AUDIOCNNRNN | 20 | PDFFed_DP | 16 | 4 | 5 | 6 | 9 | 6 | 0.844 | 0.774 | 0.0152 | 0.0355 | 4/4 | +27.38 | +42.10 | +0.000 | -0.007 | -0.014 | FederatedAverageWithPo | .30@e4f45401b |
| BANK | ANN | 20 | PDFFed | 14 | 4 | 1 | 1 | 2 | 6 | 0.936 | 0.889 | 0.0220 | 0.0132 | 2/4 | +1.19 | +2.04 | +0.005 | +0.006 | +0.002 | FedProx | .31@e34c32ad3 |
| BANK | ANN | 20 | PDFFed_DP | 14 | 4 | 3 | 5 | 3 | 4 | 0.934 | 0.885 | 0.0267 | 0.0105 | 2/4 | +0.69 | +1.54 | +0.003 | +0.001 | -0.000 | FedProx | .31@e34c32ad3 |
| BANK | ANN | 30 | PDFFed | 15 | 4 | 2 | 1 | 3 | 4 | 0.939 | 0.892 | 0.0227 | 0.0092 | 3/4 | +1.44 | +1.74 | +0.008 | +0.011 | +0.004 | FedProto | .30@e4f45401b |
| BANK | ANN | 30 | PDFFed_DP | 15 | 4 | 1 | 2 | 4 | 2 | 0.940 | 0.891 | 0.0243 | 0.0057 | 3/4 | +1.29 | +1.84 | +0.008 | +0.009 | +0.005 | FedProto | .30@e4f45401b |
| BANK | ANN | 40 | PDFFed | 15 | 4 | 1 | 1 | 4 | 6 | 0.939 | 0.892 | 0.0235 | 0.0095 | 4/4 | +1.46 | +1.64 | +0.006 | +0.012 | +0.005 | FedProto | .30@e4f45401b |
| BANK | ANN | 40 | PDFFed_DP | 15 | 4 | 2 | 2 | 3 | 2 | 0.938 | 0.889 | 0.0232 | 0.0065 | 3/4 | +1.11 | +1.94 | +0.006 | +0.009 | +0.004 | FedProto | .30@e4f45401b |
| COMPAS | ANN | 20 | PDFFed | 16 | 4 | 1 | 1 | 6 | 13 | 0.728 | 0.605 | 0.0245 | 0.0820 | 4/4 | +6.53 | +7.63 | +0.011 | +0.021 | +0.007 | FedRenyi | .31@e34c32ad3 |
| COMPAS | ANN | 20 | PDFFed_DP | 16 | 4 | 15 | 13 | 14 | 14 | 0.681 | 0.549 | 0.0418 | 0.0948 | 2/4 | +0.95 | +6.93 | -0.036 | -0.035 | -0.040 | FedRenyi | .31@e34c32ad3 |
| COMPAS | ANN | 30 | PDFFed | 14 | 4 | 1 | 1 | 3 | 13 | 0.719 | 0.598 | 0.0260 | 0.0958 | 4/4 | +5.80 | +6.53 | +0.008 | +0.021 | +0.008 | FedAvg | .30@4dc887b67 |
| COMPAS | ANN | 30 | PDFFed_DP | 14 | 4 | 2 | 3 | 2 | 11 | 0.715 | 0.586 | 0.0225 | 0.0832 | 3/4 | +4.66 | +8.33 | +0.003 | +0.009 | +0.003 | FedAvg | .30@4dc887b67 |
| COMPAS | ANN | 40 | PDFFed | 14 | 4 | 1 | 1 | 8 | 12 | 0.720 | 0.595 | 0.0300 | 0.0857 | 4/4 | +5.58 | +6.03 | +0.011 | +0.023 | +0.011 | FairFed | .30@4dc887b67 |
| COMPAS | ANN | 40 | PDFFed_DP | 14 | 4 | 10 | 3 | 13 | 14 | 0.701 | 0.579 | 0.0405 | 0.1098 | 3/4 | +3.91 | +6.03 | -0.008 | +0.006 | -0.008 | FairFed | .30@4dc887b67 |
| CelebA | BERTCLASSIFIER | 20 | PDFFed_DP | 3 | 4 | 1 | 3 | 3 | 1 | 0.750 | 0.609 | 0.1333 | 0.0183 | 4/4 | +10.46 | +17.28 | +0.030 | -0.104 | +0.030 | FedAvg | .31@4dc887b67 |
| CelebA | BERTCLASSIFIER | 30 | PDFFed_DP | 3 | 4 | 1 | 3 | 3 | 1 | 0.769 | 0.634 | 0.1042 | 0.0182 | 4/4 | +13.00 | +18.68 | +0.078 | -0.071 | +0.078 | FedAvg | .31@4dc887b67 |
| CelebA | BERTCLASSIFIER | 40 | PDFFed_DP | 3 | 4 | 1 | 3 | 3 | 1 | 0.765 | 0.627 | 0.1475 | 0.0125 | 4/4 | +12.28 | +16.78 | +0.070 | -0.037 | +0.070 | FedAvg | .31@4dc887b67 |
| DUTCH | ANN | 20 | PDFFed | 15 | 4 | 1 | 1 | 10 | 5 | 0.847 | 0.772 | 0.1108 | 0.0607 | 4/4 | +24.95 | +27.17 | +0.139 | +0.028 | +0.071 | mFairFL | .31@e34c32ad3 |
| DUTCH | ANN | 20 | PDFFed_DP | 15 | 4 | 2 | 10 | 11 | 4 | 0.807 | 0.710 | 0.1158 | 0.0485 | 4/4 | +18.77 | +27.57 | +0.099 | -0.034 | +0.031 | mFairFL | .31@e34c32ad3 |
| DUTCH | ANN | 30 | PDFFed | 13 | 4 | 1 | 2 | 8 | 3 | 0.856 | 0.778 | 0.1275 | 0.0483 | 4/4 | +25.59 | +25.97 | +0.112 | +0.076 | +0.071 | mFairFL | .30@e4f45401b |
| DUTCH | ANN | 30 | PDFFed_DP | 13 | 4 | 2 | 8 | 10 | 2 | 0.793 | 0.685 | 0.1665 | 0.0380 | 3/4 | +16.30 | +27.17 | +0.049 | -0.017 | +0.008 | mFairFL | .30@e4f45401b |
| DUTCH | ANN | 40 | PDFFed | 12 | 4 | 1 | 2 | 6 | 3 | 0.852 | 0.774 | 0.1200 | 0.0532 | 4/4 | +25.22 | +26.77 | +0.154 | +0.125 | +0.071 | mFairFL | .30@e4f45401b |
| DUTCH | ANN | 40 | PDFFed_DP | 12 | 4 | 3 | 6 | 9 | 2 | 0.773 | 0.663 | 0.1625 | 0.0405 | 3/4 | +14.12 | +28.77 | +0.075 | +0.014 | -0.007 | mFairFL | .30@e4f45401b |

## T4 主口径逐组明细

> One row per algorithm per group, ranked by HM. collapsed_splits counts how many of that algorithm's partitions are constant predictors. exc_avg_pp / exc_max_pp are ACC above the floor in percentage points.

| dataset | hyp | K | floor | nsp_max | rank | algo | hm | acc | acc_gap | deo | nsp | collapsed_splits | exc_avg_pp | exc_max_pp | src | is_pdffed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AudioMNIST | AUDIOCNNRNN | 20 | 0.5000 | 4 | 1 | PDFFed | 0.883 | 0.839 | 0.0123 | 0.0475 | 4 | 0 | +33.92 | +42.80 | .30@e4f45401b | yes |
| AudioMNIST | AUDIOCNNRNN | 20 | 0.5000 | 4 | 2 | FederatedAverageWithPo | 0.858 | 0.795 | 0.0318 | 0.0503 | 4 | 0 | +29.53 | +40.50 | .30@e4f45401b |  |
| AudioMNIST | AUDIOCNNRNN | 20 | 0.5000 | 4 | 3 | FedNova | 0.857 | 0.794 | 0.0220 | 0.0503 | 4 | 0 | +29.40 | +40.50 | .30@e4f45401b |  |
| AudioMNIST | AUDIOCNNRNN | 20 | 0.5000 | 4 | 4 | LoGoFair | 0.845 | 0.781 | 0.0290 | 0.0628 | 4 | 0 | +28.07 | +38.30 | .30@e4f45401b |  |
| AudioMNIST | AUDIOCNNRNN | 20 | 0.5000 | 4 | 5 | PDFFed_DP | 0.844 | 0.774 | 0.0152 | 0.0355 | 4 | 0 | +27.38 | +42.10 | .30@e4f45401b | yes |
| AudioMNIST | AUDIOCNNRNN | 20 | 0.5000 | 4 | 6 | FedAvg | 0.844 | 0.781 | 0.0340 | 0.0705 | 4 | 0 | +28.10 | +38.50 | .30@e4f45401b |  |
| AudioMNIST | AUDIOCNNRNN | 20 | 0.5000 | 4 | 7 | mFairFL | 0.826 | 0.750 | 0.0135 | 0.0517 | 4 | 0 | +25.05 | +37.70 | .30@e4f45401b |  |
| AudioMNIST | AUDIOCNNRNN | 20 | 0.5000 | 4 | 8 | FedRenyi | 0.754 | 0.723 | 0.0222 | 0.2090 | 4 | 0 | +22.27 | +27.10 | .30@e4f45401b |  |
| AudioMNIST | AUDIOCNNRNN | 20 | 0.5000 | 4 | 9 | FedProx | 0.718 | 0.588 | 0.0285 | 0.0638 | 4 | 1 | +8.82 | +14.40 | .30@e4f45401b |  |
| AudioMNIST | AUDIOCNNRNN | 20 | 0.5000 | 4 | 10 | FairFed | 0.706 | 0.588 | 0.0267 | 0.0857 | 4 | 0 | +8.75 | +18.90 | .30@e4f45401b |  |
| AudioMNIST | AUDIOCNNRNN | 20 | 0.5000 | 4 | 11 | FedRep | 0.685 | 0.547 | 0.0077 | 0.0513 | 4 | 3 | +4.72 | +19.00 | .30@e4f45401b |  |
| AudioMNIST | AUDIOCNNRNN | 20 | 0.5000 | 4 | 12 | FedProto | 0.684 | 0.530 | 0.0152 | 0.0235 | 4 | 2 | +3.00 | +6.60 | .30@e4f45401b |  |
| BANK | ANN | 20 | 0.8776 | 4 | 1 | PDFFed | 0.936 | 0.889 | 0.0220 | 0.0132 | 4 | 2 | +1.19 | +2.04 | .31@e34c32ad3 | yes |
| BANK | ANN | 20 | 0.8776 | 4 | 2 | FedProx | 0.934 | 0.885 | 0.0278 | 0.0110 | 4 | 3 | +0.77 | +1.84 | .31@6299c1e23 |  |
| BANK | ANN | 20 | 0.8776 | 4 | 3 | PDFFed_DP | 0.934 | 0.885 | 0.0267 | 0.0105 | 4 | 2 | +0.69 | +1.54 | .31@e34c32ad3 | yes |
| BANK | ANN | 20 | 0.8776 | 4 | 4 | FedProto | 0.933 | 0.879 | 0.0297 | 0.0063 | 4 | 4 | +0.17 | +0.54 | .31@6299c1e23 |  |
| BANK | ANN | 20 | 0.8776 | 4 | 5 | FedAvg | 0.931 | 0.884 | 0.0285 | 0.0165 | 4 | 4 | +0.61 | +0.94 | .31@6299c1e23 |  |
| BANK | ANN | 20 | 0.8776 | 4 | 6 | FairFed | 0.931 | 0.884 | 0.0285 | 0.0170 | 4 | 4 | +0.61 | +0.94 | .31@951792035 |  |
| BANK | ANN | 20 | 0.8776 | 4 | 7 | FedNova | 0.929 | 0.885 | 0.0280 | 0.0200 | 4 | 3 | +0.69 | +1.04 | .31@e34c32ad3 |  |
| BANK | ANN | 20 | 0.8776 | 4 | 8 | FedRep | 0.929 | 0.883 | 0.0288 | 0.0198 | 4 | 4 | +0.49 | +0.54 | .31@0ada0275b |  |
| BANK | ANN | 20 | 0.8776 | 4 | 9 | LoGoFair | 0.928 | 0.885 | 0.0278 | 0.0243 | 4 | 4 | +0.71 | +0.84 | .31@951792035 |  |
| BANK | ANN | 20 | 0.8776 | 4 | 10 | FL_FairBatch | 0.927 | 0.873 | 0.0292 | 0.0103 | 4 | 4 | -0.51 | -0.16 | .31@6da33067b |  |
| BANK | ANN | 20 | 0.8776 | 4 | 11 | mFairFL | 0.927 | 0.887 | 0.0272 | 0.0280 | 4 | 2 | +0.92 | +1.44 | .31@951792035 |  |
| BANK | ANN | 20 | 0.8776 | 4 | 12 | Scaffold | 0.922 | 0.867 | 0.0278 | 0.0165 | 4 | 4 | -1.01 | +0.84 | .31@e34c32ad3 |  |
| BANK | ANN | 30 | 0.8776 | 4 | 1 | PDFFed_DP | 0.940 | 0.891 | 0.0243 | 0.0057 | 4 | 1 | +1.29 | +1.84 | .30@e4f45401b | yes |
| BANK | ANN | 30 | 0.8776 | 4 | 2 | PDFFed | 0.939 | 0.892 | 0.0227 | 0.0092 | 4 | 1 | +1.44 | +1.74 | .30@e4f45401b | yes |
| BANK | ANN | 30 | 0.8776 | 4 | 3 | FedProto | 0.934 | 0.878 | 0.0310 | 0.0013 | 4 | 4 | +0.04 | +0.04 | .30@6299c1e23 |  |
| BANK | ANN | 30 | 0.8776 | 4 | 4 | FedProx | 0.932 | 0.879 | 0.0297 | 0.0090 | 4 | 4 | +0.17 | +0.34 | .30@6299c1e23 |  |
| BANK | ANN | 30 | 0.8776 | 4 | 5 | FairFed | 0.931 | 0.881 | 0.0297 | 0.0135 | 4 | 4 | +0.39 | +0.54 | .30@951792035 |  |
| BANK | ANN | 30 | 0.8776 | 4 | 6 | FedAvg | 0.931 | 0.881 | 0.0297 | 0.0135 | 4 | 4 | +0.39 | +0.54 | .30@e4f45401b |  |
| BANK | ANN | 30 | 0.8776 | 4 | 7 | FederatedAverageWithPo | 0.930 | 0.882 | 0.0290 | 0.0158 | 4 | 3 | +0.44 | +1.54 | .30@951792035 |  |
| BANK | ANN | 30 | 0.8776 | 4 | 8 | FedNova | 0.930 | 0.883 | 0.0292 | 0.0165 | 4 | 4 | +0.49 | +0.64 | .30@951792035 |  |
| BANK | ANN | 30 | 0.8776 | 4 | 9 | LoGoFair | 0.929 | 0.883 | 0.0295 | 0.0185 | 4 | 4 | +0.52 | +0.84 | .30@951792035 |  |
| BANK | ANN | 30 | 0.8776 | 4 | 10 | FedRep | 0.929 | 0.881 | 0.0290 | 0.0183 | 4 | 4 | +0.34 | +0.34 | .30@6299c1e23 |  |
| BANK | ANN | 30 | 0.8776 | 4 | 11 | mFairFL | 0.927 | 0.881 | 0.0292 | 0.0232 | 4 | 4 | +0.36 | +0.44 | .30@0ada0275b |  |
| BANK | ANN | 30 | 0.8776 | 4 | 12 | FL_FairBatch | 0.919 | 0.861 | 0.0283 | 0.0147 | 4 | 4 | -1.71 | -0.26 | .30@6da33067b |  |
| BANK | ANN | 40 | 0.8776 | 4 | 1 | PDFFed | 0.939 | 0.892 | 0.0235 | 0.0095 | 4 | 0 | +1.46 | +1.64 | .30@e4f45401b | yes |
| BANK | ANN | 40 | 0.8776 | 4 | 2 | PDFFed_DP | 0.938 | 0.889 | 0.0232 | 0.0065 | 4 | 1 | +1.11 | +1.94 | .30@e4f45401b | yes |
| BANK | ANN | 40 | 0.8776 | 4 | 3 | FedProto | 0.934 | 0.878 | 0.0310 | 0.0025 | 4 | 4 | +0.04 | +0.04 | .30@6299c1e23 |  |
| BANK | ANN | 40 | 0.8776 | 4 | 4 | FederatedAverageWithPo | 0.933 | 0.886 | 0.0275 | 0.0130 | 4 | 2 | +0.84 | +1.44 | .30@0ada0275b |  |
| BANK | ANN | 40 | 0.8776 | 4 | 5 | FairFed | 0.932 | 0.880 | 0.0297 | 0.0095 | 4 | 4 | +0.24 | +0.34 | .30@951792035 |  |
| BANK | ANN | 40 | 0.8776 | 4 | 6 | FedAvg | 0.932 | 0.880 | 0.0297 | 0.0095 | 4 | 4 | +0.24 | +0.34 | .30@e4f45401b |  |
| BANK | ANN | 40 | 0.8776 | 4 | 7 | FedProx | 0.931 | 0.879 | 0.0302 | 0.0095 | 4 | 4 | +0.12 | +0.14 | .30@6299c1e23 |  |
| BANK | ANN | 40 | 0.8776 | 4 | 8 | FedNova | 0.930 | 0.881 | 0.0295 | 0.0168 | 4 | 4 | +0.34 | +0.54 | .30@951792035 |  |
| BANK | ANN | 40 | 0.8776 | 4 | 9 | mFairFL | 0.929 | 0.880 | 0.0295 | 0.0145 | 4 | 4 | +0.24 | +0.34 | .30@0ada0275b |  |
| BANK | ANN | 40 | 0.8776 | 4 | 10 | LoGoFair | 0.929 | 0.880 | 0.0295 | 0.0165 | 4 | 4 | +0.27 | +0.44 | .30@0ada0275b |  |
| BANK | ANN | 40 | 0.8776 | 4 | 11 | FedRep | 0.926 | 0.878 | 0.0290 | 0.0203 | 4 | 4 | +0.04 | +0.14 | .30@6299c1e23 |  |
| BANK | ANN | 40 | 0.8776 | 4 | 12 | Scaffold | 0.910 | 0.859 | 0.0270 | 0.0312 | 4 | 4 | -1.91 | +0.34 | .30@0ada0275b |  |
| COMPAS | ANN | 20 | 0.5397 | 4 | 1 | PDFFed | 0.728 | 0.605 | 0.0245 | 0.0820 | 4 | 0 | +6.53 | +7.63 | .31@e34c32ad3 | yes |
| COMPAS | ANN | 20 | 0.5397 | 4 | 2 | FedRenyi | 0.721 | 0.582 | 0.0297 | 0.0500 | 4 | 0 | +4.23 | +5.33 | .31@6299c1e23 |  |
| COMPAS | ANN | 20 | 0.5397 | 4 | 3 | FairFed | 0.718 | 0.585 | 0.0235 | 0.0700 | 4 | 0 | +4.55 | +5.03 | .31@e34c32ad3 |  |
| COMPAS | ANN | 20 | 0.5397 | 4 | 4 | FedNova | 0.718 | 0.583 | 0.0232 | 0.0643 | 4 | 0 | +4.38 | +5.13 | .31@e34c32ad3 |  |
| COMPAS | ANN | 20 | 0.5397 | 4 | 5 | FedAvg | 0.717 | 0.584 | 0.0252 | 0.0688 | 4 | 0 | +4.46 | +5.13 | .31@e34c32ad3 |  |
| COMPAS | ANN | 20 | 0.5397 | 4 | 6 | FederatedAverageWithPo | 0.716 | 0.602 | 0.0303 | 0.1162 | 4 | 0 | +6.25 | +7.63 | .31@e34c32ad3 |  |
| COMPAS | ANN | 20 | 0.5397 | 4 | 7 | FedProto | 0.711 | 0.577 | 0.0210 | 0.0680 | 4 | 0 | +3.71 | +4.63 | .31@6da33067b |  |
| COMPAS | ANN | 20 | 0.5397 | 4 | 8 | FedProx | 0.711 | 0.567 | 0.0375 | 0.0450 | 4 | 0 | +2.75 | +4.93 | .31@e34c32ad3 |  |
| COMPAS | ANN | 20 | 0.5397 | 4 | 9 | FedRep | 0.704 | 0.572 | 0.0187 | 0.0810 | 4 | 0 | +3.25 | +4.43 | .31@e34c32ad3 |  |
| COMPAS | ANN | 20 | 0.5397 | 4 | 10 | FedFair | 0.702 | 0.554 | 0.0335 | 0.0343 | 4 | 1 | +1.38 | +1.73 | .31@e34c32ad3 |  |
| COMPAS | ANN | 20 | 0.5397 | 4 | 11 | FL_FairBatch | 0.698 | 0.546 | 0.0330 | 0.0278 | 4 | 3 | +0.60 | +1.03 | .31@e34c32ad3 |  |
| COMPAS | ANN | 20 | 0.5397 | 4 | 12 | FedFB | 0.694 | 0.537 | 0.0500 | 0.0130 | 4 | 4 | -0.27 | -0.27 | .31@e34c32ad3 |  |
| COMPAS | ANN | 30 | 0.5397 | 4 | 1 | FedRenyi | 0.723 | 0.581 | 0.0265 | 0.0425 | 2 | 0 | +4.18 | +4.93 | .30@e4f45401b |  |
| COMPAS | ANN | 30 | 0.5397 | 4 | 2 | PDFFed | 0.719 | 0.598 | 0.0260 | 0.0958 | 4 | 0 | +5.80 | +6.53 | .30@4dc887b67 | yes |
| COMPAS | ANN | 30 | 0.5397 | 4 | 3 | PDFFed_DP | 0.715 | 0.586 | 0.0225 | 0.0832 | 4 | 1 | +4.66 | +8.33 | .30@4dc887b67 | yes |
| COMPAS | ANN | 30 | 0.5397 | 4 | 4 | FedAvg | 0.711 | 0.577 | 0.0333 | 0.0675 | 4 | 0 | +3.71 | +4.33 | .30@4dc887b67 |  |
| COMPAS | ANN | 30 | 0.5397 | 4 | 5 | FairFed | 0.711 | 0.577 | 0.0335 | 0.0703 | 4 | 0 | +3.73 | +4.33 | .30@4dc887b67 |  |
| COMPAS | ANN | 30 | 0.5397 | 4 | 6 | FederatedAverageWithPo | 0.709 | 0.592 | 0.0377 | 0.1137 | 4 | 0 | +5.23 | +6.53 | .30@4dc887b67 |  |
| COMPAS | ANN | 30 | 0.5397 | 4 | 7 | FedNova | 0.708 | 0.572 | 0.0298 | 0.0655 | 4 | 0 | +3.21 | +4.13 | .30@4dc887b67 |  |
| COMPAS | ANN | 30 | 0.5397 | 4 | 8 | FedFair | 0.706 | 0.557 | 0.0285 | 0.0323 | 4 | 0 | +1.75 | +2.13 | .30@e4f45401b |  |
| COMPAS | ANN | 30 | 0.5397 | 4 | 9 | FedProto | 0.705 | 0.568 | 0.0225 | 0.0673 | 4 | 0 | +2.83 | +3.53 | .30@4dc887b67 |  |
| COMPAS | ANN | 30 | 0.5397 | 4 | 10 | FedRep | 0.703 | 0.568 | 0.0318 | 0.0717 | 4 | 0 | +2.78 | +3.93 | .30@4dc887b67 |  |
| COMPAS | ANN | 30 | 0.5397 | 4 | 11 | Scaffold | 0.692 | 0.560 | 0.0335 | 0.0920 | 4 | 1 | +2.08 | +3.63 | .30@4dc887b67 |  |
| COMPAS | ANN | 30 | 0.5397 | 4 | 12 | FedProx | 0.690 | 0.545 | 0.0295 | 0.0575 | 4 | 4 | +0.55 | +0.73 | .30@4dc887b67 |  |
| COMPAS | ANN | 40 | 0.5397 | 4 | 1 | PDFFed | 0.720 | 0.595 | 0.0300 | 0.0857 | 4 | 0 | +5.58 | +6.03 | .30@4dc887b67 | yes |
| COMPAS | ANN | 40 | 0.5397 | 4 | 2 | FedRenyi | 0.715 | 0.575 | 0.0190 | 0.0500 | 1 | 0 | +3.53 | +3.53 | .30@e4f45401b |  |
| COMPAS | ANN | 40 | 0.5397 | 4 | 3 | FairFed | 0.709 | 0.573 | 0.0290 | 0.0645 | 4 | 0 | +3.30 | +3.63 | .30@4dc887b67 |  |
| COMPAS | ANN | 40 | 0.5397 | 4 | 4 | FederatedAverageWithPo | 0.709 | 0.583 | 0.0340 | 0.0917 | 4 | 0 | +4.38 | +4.93 | .30@4dc887b67 |  |
| COMPAS | ANN | 40 | 0.5397 | 4 | 5 | FedAvg | 0.709 | 0.573 | 0.0285 | 0.0645 | 4 | 0 | +3.28 | +3.53 | .30@4dc887b67 |  |
| COMPAS | ANN | 40 | 0.5397 | 4 | 6 | FedNova | 0.709 | 0.570 | 0.0325 | 0.0605 | 4 | 0 | +3.06 | +3.33 | .30@4dc887b67 |  |
| COMPAS | ANN | 40 | 0.5397 | 4 | 7 | FedProto | 0.706 | 0.567 | 0.0217 | 0.0583 | 4 | 0 | +2.68 | +2.83 | .30@4dc887b67 |  |
| COMPAS | ANN | 40 | 0.5397 | 4 | 8 | FedFair | 0.705 | 0.556 | 0.0315 | 0.0315 | 4 | 0 | +1.60 | +2.03 | .30@e4f45401b |  |
| COMPAS | ANN | 40 | 0.5397 | 4 | 9 | Scaffold | 0.704 | 0.567 | 0.0307 | 0.0643 | 4 | 0 | +2.68 | +3.33 | .30@4dc887b67 |  |
| COMPAS | ANN | 40 | 0.5397 | 4 | 10 | FedRep | 0.704 | 0.565 | 0.0245 | 0.0607 | 4 | 0 | +2.50 | +2.73 | .30@4dc887b67 |  |
| COMPAS | ANN | 40 | 0.5397 | 4 | 11 | PDFFed_DP | 0.701 | 0.579 | 0.0405 | 0.1098 | 4 | 1 | +3.91 | +6.03 | .30@4dc887b67 | yes |
| COMPAS | ANN | 40 | 0.5397 | 4 | 12 | FedFB | 0.691 | 0.536 | 0.0473 | 0.0210 | 4 | 4 | -0.37 | -0.27 | .30@e4f45401b |  |
| CelebA | BERTCLASSIFIER | 2 | 0.5042 | 3 | 1 | FL_FairBatch | 0.669 | 0.534 | 0.1410 | 0.0887 | 3 | 1 | +2.98 | +10.48 | .31@4dc887b67 |  |
| CelebA | BERTCLASSIFIER | 20 | 0.5042 | 4 | 1 | FL_FairBatch | 0.776 | 0.679 | 0.0495 | 0.0915 | 2 | 0 | +17.43 | +17.58 | .31@e34c32ad3 |  |
| CelebA | BERTCLASSIFIER | 20 | 0.5042 | 4 | 2 | PDFFed_DP | 0.750 | 0.609 | 0.1333 | 0.0183 | 4 | 0 | +10.46 | +17.28 | .31@4dc887b67 | yes |
| CelebA | BERTCLASSIFIER | 20 | 0.5042 | 4 | 3 | FedRenyi | 0.742 | 0.691 | 0.0390 | 0.1950 | 1 | 0 | +18.68 | +18.68 | .31@e34c32ad3 |  |
| CelebA | BERTCLASSIFIER | 20 | 0.5042 | 4 | 4 | FederatedAverageWithPo | 0.742 | 0.678 | 0.1080 | 0.1550 | 2 | 0 | +17.43 | +20.78 | .31@4dc887b67 |  |
| CelebA | BERTCLASSIFIER | 20 | 0.5042 | 4 | 5 | FedProx | 0.730 | 0.700 | 0.0407 | 0.2330 | 3 | 0 | +19.61 | +21.28 | .31@4dc887b67 |  |
| CelebA | BERTCLASSIFIER | 20 | 0.5042 | 4 | 6 | FedAvg | 0.720 | 0.713 | 0.0813 | 0.2530 | 4 | 0 | +20.83 | +25.18 | .31@4dc887b67 |  |
| CelebA | BERTCLASSIFIER | 20 | 0.5042 | 4 | 7 | FairFed | 0.712 | 0.711 | 0.0880 | 0.2625 | 4 | 0 | +20.65 | +25.38 | .31@e34c32ad3 |  |
| CelebA | BERTCLASSIFIER | 20 | 0.5042 | 4 | 8 | FedNova | 0.702 | 0.684 | 0.1300 | 0.2445 | 2 | 0 | +17.98 | +25.18 | .31@4dc887b67 |  |
| CelebA | BERTCLASSIFIER | 20 | 0.5042 | 4 | 9 | mFairFL | 0.700 | 0.594 | 0.1700 | 0.1280 | 1 | 0 | +8.98 | +8.98 | .31@ |  |
| CelebA | BERTCLASSIFIER | 20 | 0.5042 | 4 | 10 | Scaffold | 0.670 | 0.504 | 0.4040 | 0.0000 | 2 | 2 | -0.02 | -0.02 | .31@4dc887b67 |  |
| CelebA | BERTCLASSIFIER | 20 | 0.5042 | 4 | 11 | FedProto | 0.669 | 0.506 | 0.3760 | 0.0130 | 1 | 1 | +0.18 | +0.18 | .31@e34c32ad3 |  |
| CelebA | BERTCLASSIFIER | 20 | 0.5042 | 4 | 12 | FedFair | 0.651 | 0.488 | 0.2000 | 0.0230 | 1 | 1 | -1.62 | -1.62 | .31@e34c32ad3 |  |
| CelebA | BERTCLASSIFIER | 30 | 0.5042 | 4 | 1 | PDFFed_DP | 0.769 | 0.634 | 0.1042 | 0.0182 | 4 | 0 | +13.00 | +18.68 | .31@4dc887b67 | yes |
| CelebA | BERTCLASSIFIER | 30 | 0.5042 | 4 | 2 | FL_FairBatch | 0.730 | 0.674 | 0.0580 | 0.2030 | 1 | 0 | +16.98 | +16.98 | .31@4dc887b67 |  |
| CelebA | BERTCLASSIFIER | 30 | 0.5042 | 4 | 3 | mFairFL | 0.708 | 0.584 | 0.2350 | 0.0760 | 1 | 0 | +7.98 | +7.98 | .31@ |  |
| CelebA | BERTCLASSIFIER | 30 | 0.5042 | 4 | 4 | FedProx | 0.706 | 0.709 | 0.0433 | 0.2930 | 3 | 0 | +20.48 | +22.18 | .31@4dc887b67 |  |
| CelebA | BERTCLASSIFIER | 30 | 0.5042 | 4 | 5 | FederatedAverageWithPo | 0.703 | 0.680 | 0.1505 | 0.2250 | 2 | 0 | +17.63 | +24.78 | .31@4dc887b67 |  |
| CelebA | BERTCLASSIFIER | 30 | 0.5042 | 4 | 6 | FedAvg | 0.691 | 0.705 | 0.0905 | 0.3063 | 4 | 0 | +20.11 | +23.78 | .31@4dc887b67 |  |
| CelebA | BERTCLASSIFIER | 30 | 0.5042 | 4 | 7 | FairFed | 0.689 | 0.704 | 0.0920 | 0.3080 | 4 | 0 | +19.95 | +23.58 | .31@e34c32ad3 |  |
| CelebA | BERTCLASSIFIER | 30 | 0.5042 | 4 | 8 | FedNova | 0.681 | 0.666 | 0.1485 | 0.2775 | 2 | 0 | +16.18 | +20.08 | .31@4dc887b67 |  |
| CelebA | BERTCLASSIFIER | 30 | 0.5042 | 4 | 9 | FedProto | 0.672 | 0.517 | 0.3660 | 0.0360 | 1 | 0 | +1.28 | +1.28 | .31@e34c32ad3 |  |
| CelebA | BERTCLASSIFIER | 30 | 0.5042 | 4 | 10 | Scaffold | 0.669 | 0.502 | 0.4040 | 0.0000 | 2 | 2 | -0.17 | -0.02 | .31@4dc887b67 |  |
| CelebA | BERTCLASSIFIER | 40 | 0.5042 | 4 | 1 | PDFFed_DP | 0.765 | 0.627 | 0.1475 | 0.0125 | 4 | 0 | +12.28 | +16.78 | .31@4dc887b67 | yes |
| CelebA | BERTCLASSIFIER | 40 | 0.5042 | 4 | 2 | FedProx | 0.714 | 0.647 | 0.1085 | 0.1835 | 2 | 0 | +14.28 | +19.08 | .31@4dc887b67 |  |
| CelebA | BERTCLASSIFIER | 40 | 0.5042 | 4 | 3 | FL_FairBatch | 0.701 | 0.640 | 0.0970 | 0.2180 | 1 | 0 | +13.58 | +13.58 | .31@4dc887b67 |  |
| CelebA | BERTCLASSIFIER | 40 | 0.5042 | 4 | 4 | mFairFL | 0.698 | 0.539 | 0.3350 | 0.0050 | 1 | 0 | +3.48 | +3.48 | .31@ |  |
| CelebA | BERTCLASSIFIER | 40 | 0.5042 | 4 | 5 | FedAvg | 0.695 | 0.664 | 0.1375 | 0.2410 | 4 | 0 | +15.98 | +22.78 | .31@4dc887b67 |  |
| CelebA | BERTCLASSIFIER | 40 | 0.5042 | 4 | 6 | FairFed | 0.693 | 0.662 | 0.1415 | 0.2433 | 4 | 0 | +15.83 | +22.78 | .31@e34c32ad3 |  |
| CelebA | BERTCLASSIFIER | 40 | 0.5042 | 4 | 7 | FedNova | 0.690 | 0.545 | 0.3170 | 0.0530 | 1 | 0 | +4.08 | +4.08 | .31@4dc887b67 |  |
| CelebA | BERTCLASSIFIER | 40 | 0.5042 | 4 | 8 | FederatedAverageWithPo | 0.665 | 0.600 | 0.2680 | 0.1520 | 2 | 1 | +9.58 | +19.68 | .31@4dc887b67 |  |
| CelebA | BERTCLASSIFIER | 40 | 0.5042 | 4 | 9 | Scaffold | 0.665 | 0.499 | 0.4040 | 0.0000 | 1 | 1 | -0.52 | -0.52 | .31@4dc887b67 |  |
| DUTCH | ANN | 20 | 0.5223 | 4 | 1 | PDFFed | 0.847 | 0.772 | 0.1108 | 0.0607 | 4 | 0 | +24.95 | +27.17 | .31@e34c32ad3 | yes |
| DUTCH | ANN | 20 | 0.5223 | 4 | 2 | PDFFed_DP | 0.807 | 0.710 | 0.1158 | 0.0485 | 4 | 0 | +18.77 | +27.57 | .31@e34c32ad3 | yes |
| DUTCH | ANN | 20 | 0.5223 | 4 | 3 | mFairFL | 0.775 | 0.747 | 0.1057 | 0.1515 | 4 | 0 | +22.42 | +26.27 | .31@0ada0275b |  |
| DUTCH | ANN | 20 | 0.5223 | 4 | 4 | FedProx | 0.773 | 0.728 | 0.0418 | 0.1623 | 4 | 0 | +20.52 | +22.87 | .31@6299c1e23 |  |
| DUTCH | ANN | 20 | 0.5223 | 4 | 5 | FedProto | 0.732 | 0.675 | 0.0948 | 0.1642 | 4 | 0 | +15.32 | +22.37 | .31@6299c1e23 |  |
| DUTCH | ANN | 20 | 0.5223 | 4 | 6 | FairFed | 0.708 | 0.744 | 0.0372 | 0.3100 | 4 | 0 | +22.17 | +23.57 | .31@0ada0275b |  |
| DUTCH | ANN | 20 | 0.5223 | 4 | 7 | FedAvg | 0.708 | 0.744 | 0.0370 | 0.3102 | 4 | 0 | +22.14 | +23.57 | .31@e34c32ad3 |  |
| DUTCH | ANN | 20 | 0.5223 | 4 | 8 | FedRep | 0.676 | 0.728 | 0.0675 | 0.3372 | 4 | 0 | +20.59 | +22.87 | .31@6299c1e23 |  |
| DUTCH | ANN | 20 | 0.5223 | 4 | 9 | FedNova | 0.667 | 0.753 | 0.0473 | 0.3875 | 4 | 0 | +23.09 | +25.27 | .31@0ada0275b |  |
| DUTCH | ANN | 20 | 0.5223 | 4 | 10 | LoGoFair | 0.666 | 0.766 | 0.0815 | 0.3493 | 4 | 0 | +24.32 | +26.57 | .31@0ada0275b |  |
| DUTCH | ANN | 20 | 0.5223 | 4 | 11 | FederatedAverageWithPo | 0.658 | 0.737 | 0.1047 | 0.3565 | 4 | 0 | +21.50 | +26.97 | .31@0ada0275b |  |
| DUTCH | ANN | 20 | 0.5223 | 4 | 12 | FL_FairBatch | 0.654 | 0.487 | 0.2900 | 0.0035 | 4 | 4 | -3.53 | -2.23 | .31@754c44326 |  |
| DUTCH | ANN | 30 | 0.5223 | 4 | 1 | PDFFed | 0.856 | 0.778 | 0.1275 | 0.0483 | 4 | 0 | +25.59 | +25.97 | .30@e4f45401b | yes |
| DUTCH | ANN | 30 | 0.5223 | 4 | 2 | PDFFed_DP | 0.793 | 0.685 | 0.1665 | 0.0380 | 4 | 1 | +16.30 | +27.17 | .30@e4f45401b | yes |
| DUTCH | ANN | 30 | 0.5223 | 4 | 3 | mFairFL | 0.785 | 0.759 | 0.0777 | 0.1573 | 4 | 0 | +23.64 | +24.57 | .30@4dc887b67 |  |
| DUTCH | ANN | 30 | 0.5223 | 4 | 4 | FedAvg | 0.744 | 0.702 | 0.0452 | 0.1852 | 4 | 0 | +18.00 | +21.37 | .30@4dc887b67 |  |
| DUTCH | ANN | 30 | 0.5223 | 4 | 5 | FairFed | 0.744 | 0.702 | 0.0452 | 0.1850 | 4 | 0 | +18.00 | +21.37 | .30@4dc887b67 |  |
| DUTCH | ANN | 30 | 0.5223 | 4 | 6 | FedRep | 0.716 | 0.711 | 0.0363 | 0.2707 | 4 | 0 | +18.87 | +20.77 | .30@e4f45401b |  |
| DUTCH | ANN | 30 | 0.5223 | 4 | 7 | FedProto | 0.701 | 0.666 | 0.1047 | 0.2023 | 4 | 0 | +14.34 | +25.87 | .30@e4f45401b |  |
| DUTCH | ANN | 30 | 0.5223 | 4 | 8 | FedNova | 0.700 | 0.737 | 0.0440 | 0.3065 | 4 | 0 | +21.47 | +25.37 | .30@4dc887b67 |  |
| DUTCH | ANN | 30 | 0.5223 | 4 | 9 | FedProx | 0.666 | 0.583 | 0.1700 | 0.1958 | 4 | 0 | +6.07 | +9.27 | .30@e4f45401b |  |
| DUTCH | ANN | 30 | 0.5223 | 4 | 10 | FL_FairBatch | 0.665 | 0.503 | 0.2730 | 0.0168 | 4 | 4 | -1.96 | -0.63 | .30@e4f45401b |  |
| DUTCH | ANN | 30 | 0.5223 | 4 | 11 | Scaffold | 0.651 | 0.642 | 0.1375 | 0.2645 | 4 | 1 | +11.95 | +23.67 | .30@e4f45401b |  |
| DUTCH | ANN | 30 | 0.5223 | 4 | 12 | FedFair | 0.644 | 0.492 | 0.2135 | 0.0615 | 4 | 4 | -2.98 | -2.93 | .30@e4f45401b |  |
| DUTCH | ANN | 40 | 0.5223 | 4 | 1 | PDFFed | 0.852 | 0.774 | 0.1200 | 0.0532 | 4 | 0 | +25.22 | +26.77 | .30@e4f45401b | yes |
| DUTCH | ANN | 40 | 0.5223 | 4 | 2 | mFairFL | 0.780 | 0.748 | 0.0467 | 0.1593 | 4 | 0 | +22.52 | +23.87 | .30@4dc887b67 |  |
| DUTCH | ANN | 40 | 0.5223 | 4 | 3 | PDFFed_DP | 0.773 | 0.663 | 0.1625 | 0.0405 | 4 | 1 | +14.12 | +28.77 | .30@e4f45401b | yes |
| DUTCH | ANN | 40 | 0.5223 | 4 | 4 | FedRep | 0.736 | 0.669 | 0.0597 | 0.1625 | 4 | 0 | +14.72 | +16.67 | .30@e4f45401b |  |
| DUTCH | ANN | 40 | 0.5223 | 4 | 5 | FedProto | 0.710 | 0.599 | 0.1707 | 0.0977 | 4 | 1 | +7.62 | +14.77 | .30@e4f45401b |  |
| DUTCH | ANN | 40 | 0.5223 | 4 | 6 | FedNova | 0.709 | 0.715 | 0.0488 | 0.2645 | 4 | 0 | +19.27 | +24.97 | .30@4dc887b67 |  |
| DUTCH | ANN | 40 | 0.5223 | 4 | 7 | FairFed | 0.698 | 0.650 | 0.1283 | 0.1787 | 4 | 0 | +12.77 | +24.97 | .30@4dc887b67 |  |
| DUTCH | ANN | 40 | 0.5223 | 4 | 8 | FedAvg | 0.698 | 0.650 | 0.1285 | 0.1785 | 4 | 0 | +12.74 | +24.97 | .30@4dc887b67 |  |
| DUTCH | ANN | 40 | 0.5223 | 4 | 9 | FL_FairBatch | 0.678 | 0.517 | 0.2205 | 0.0083 | 4 | 3 | -0.51 | +2.97 | .30@e4f45401b |  |
| DUTCH | ANN | 40 | 0.5223 | 4 | 10 | FedProx | 0.671 | 0.552 | 0.2027 | 0.1325 | 4 | 0 | +3.00 | +4.07 | .30@e4f45401b |  |
| DUTCH | ANN | 40 | 0.5223 | 4 | 11 | FederatedAverageWithPo | 0.670 | 0.784 | 0.0698 | 0.4005 | 4 | 0 | +26.20 | +27.27 | .30@4dc887b67 |  |
| DUTCH | ANN | 40 | 0.5223 | 4 | 12 | Scaffold | 0.668 | 0.643 | 0.0973 | 0.2585 | 4 | 1 | +12.12 | +18.87 | .30@e4f45401b |  |

_（每组仅列 HM 前 12，完整 179 行见 CSV）_

## T5 跨机/跨版本 ACC 不一致

> Same cell key present more than once with an ACC spread above 1e-5. Sorted by spread. copies_host_commit_acc lists every copy.

| dataset | split | algo | hyp | K | n_copies | hosts | d_acc | copies_host_commit_acc |
|---|---|---|---|---|---|---|---|---|
| BANK | Uniform | FedProx | ANN | 30 | 2 | 30+31 | 0.3990 | .30@951792035:0.8810 \| .31@e34c32ad3:0.4820 |
| DUTCH | Uniform | PDFFed_DP | ANN | 20 | 2 | 30+31 | 0.1210 | .30@e4f45401b:0.7740 \| .31@e34c32ad3:0.6530 |
| BANK | Dirichlet01 | FederatedAverageWithPo | ANN | 20 | 2 | 30+31 | 0.1080 | .30@e4f45401b:0.7300 \| .31@951792035:0.8380 |
| DUTCH | Dirichlet01 | FedProx | ANN | 20 | 2 | 30+31 | 0.1000 | .30@e4f45401b:0.6510 \| .31@6299c1e23:0.7510 |
| DUTCH | Dirichlet01 | FedProto | ANN | 20 | 2 | 30+31 | 0.0980 | .30@e4f45401b:0.6100 \| .31@6299c1e23:0.7080 |
| DUTCH | Dirichlet01 | FedRep | ANN | 20 | 2 | 30+31 | 0.0850 | .30@e4f45401b:0.6660 \| .31@6299c1e23:0.7510 |
| DUTCH | Uniform | Scaffold | ANN | 20 | 2 | 30+31 | 0.0740 | .30@e4f45401b:0.7970 \| .31@e34c32ad3:0.7230 |
| COMPAS | Dirichlet05 | FedAvg | ANN | 20 | 2 | 30+31 | 0.0740 | .30@e4f45401b:0.5910 \| .31@blank:0.5170 |
| DUTCH | Uniform | mFairFL | ANN | 20 | 2 | 30+31 | 0.0730 | .30@6299c1e23:0.7850 \| .31@e34c32ad3:0.7120 |
| COMPAS | Dirichlet1 | PDFFed_DP | ANN | 20 | 2 | 30+31 | 0.0710 | .30@e4f45401b:0.5860 \| .31@e34c32ad3:0.5150 |
| DUTCH | Uniform | FedFB | ANN | 20 | 2 | 30+31 | 0.0700 | .30@e4f45401b:0.4700 \| .31@6299c1e23:0.5400 |
| COMPAS | Dirichlet01 | FedAvg | ANN | 30 | 2 | 30+31 | 0.0680 | .30@4dc887b67:0.5830 \| .31@blank:0.5150 |
| COMPAS | Dirichlet1 | FedAvg | ANN | 20 | 2 | 30+31 | 0.0670 | .30@e4f45401b:0.5840 \| .31@blank:0.5170 |
| DUTCH | Uniform | FedRep | ANN | 20 | 2 | 30+31 | 0.0660 | .30@e4f45401b:0.7050 \| .31@e34c32ad3:0.6390 |
| DUTCH | Dirichlet1 | LoGoFair | ANN | 20 | 2 | 30+31 | 0.0650 | .30@e4f45401b:0.7140 \| .31@e34c32ad3:0.7790 |

_（仅列 TOP15，完整 94 行见 CSV）_

## T6 balance_sampling=1 的格

> The 2026-10-21 audit found 0 such cells out of 788; there are now 20, and all of them are still collapsed.

| dataset | algo | hyp | K | split | acc | floor | excess_pp | collapse | commit | host |
|---|---|---|---|---|---|---|---|---|---|---|
| CREMA-D_gender | FL_FairBatch | AUDIOCNNRNN | 20 | Speaker | 0.8290 | 0.829252 | -0.03 | COLLAPSE | e34c32ad3 | .31 |
| CREMA-D_gender | FedFB | AUDIOCNNRNN | 20 | Speaker | 0.4740 | 0.829252 | -35.53 | COLLAPSE | e34c32ad3 | .31 |
| CREMA-D_gender | FedProto | AUDIOCNNRNN | 20 | Speaker | 0.6310 | 0.829252 | -19.83 | COLLAPSE | e34c32ad3 | .31 |
| CREMA-D_gender | Scaffold | AUDIOCNNRNN | 20 | Speaker | 0.5590 | 0.829252 | -27.03 | COLLAPSE | e34c32ad3 | .31 |
| CREMA-D_gender | mFairFL | AUDIOCNNRNN | 20 | Speaker | 0.7190 | 0.829252 | -11.03 | COLLAPSE | e34c32ad3 | .31 |
| RAVDESS | FL_FairBatch | AUDIOCNNRNN | 20 | Speaker | 0.8670 | 0.866667 | +0.03 | COLLAPSE | e4f45401b | .30 |
| RAVDESS | FairFed | AUDIOCNNRNN | 20 | Speaker | 0.7290 | 0.866667 | -13.77 | COLLAPSE | e4f45401b | .30 |
| RAVDESS | FedAvg | AUDIOCNNRNN | 20 | Speaker | 0.7280 | 0.866667 | -13.87 | COLLAPSE | e4f45401b | .30 |
| RAVDESS | FedFB | AUDIOCNNRNN | 20 | Speaker | 0.5750 | 0.866667 | -29.17 | COLLAPSE | e4f45401b | .30 |
| RAVDESS | FedFair | AUDIOCNNRNN | 20 | Speaker | 0.6860 | 0.866667 | -18.07 | COLLAPSE | e4f45401b | .30 |
| RAVDESS | FedNova | AUDIOCNNRNN | 20 | Speaker | 0.7240 | 0.866667 | -14.27 | COLLAPSE | e4f45401b | .30 |
| RAVDESS | FedProto | AUDIOCNNRNN | 20 | Speaker | 0.6850 | 0.866667 | -18.17 | COLLAPSE | e4f45401b | .30 |
| RAVDESS | FedProx | AUDIOCNNRNN | 20 | Speaker | 0.8100 | 0.866667 | -5.67 | COLLAPSE | e4f45401b | .30 |
| RAVDESS | FedRep | AUDIOCNNRNN | 20 | Speaker | 0.8080 | 0.866667 | -5.87 | COLLAPSE | e4f45401b | .30 |
| RAVDESS | FederatedAverageWithPo | AUDIOCNNRNN | 20 | Speaker | 0.6500 | 0.866667 | -21.67 | COLLAPSE | e4f45401b | .30 |
| RAVDESS | LoGoFair | AUDIOCNNRNN | 20 | Speaker | 0.7920 | 0.866667 | -7.47 | COLLAPSE | e4f45401b | .30 |
| RAVDESS | PDFFed | AUDIOCNNRNN | 20 | Speaker | 0.7260 | 0.866667 | -14.07 | COLLAPSE | e4f45401b | .30 |
| RAVDESS | PDFFed_DP | AUDIOCNNRNN | 20 | Speaker | 0.7320 | 0.866667 | -13.47 | COLLAPSE | e4f45401b | .30 |
| RAVDESS | Scaffold | AUDIOCNNRNN | 20 | Speaker | 0.3070 | 0.866667 | -55.97 | COLLAPSE | e4f45401b | .30 |
| RAVDESS | mFairFL | AUDIOCNNRNN | 20 | Speaker | 0.7280 | 0.866667 | -13.87 | COLLAPSE | e4f45401b | .30 |

## T7 合并与覆盖统计

> Counting the two inventories in raw/ after the newest-recorded_at merge.

| metric | value | note |
|---|---|---|
| raw cells .30 | 933 | result_path scan |
| raw cells .31 | 881 | result_path scan |
| raw cells total | 1814 |  |
| unique cells after merge | 1534 | key = dataset/split/algo/hyp/K |
| cross-machine duplicate cells | 280 | same key present on both hosts |
| cross-copy ACC disagreements | 94 | spread > 1e-5 |
| max single-cell ACC spread | 0.3990 | BANK/Uniform/FedProx/ANN/30 |
| cells without floor | 97 | json missing or fingerprint not cached |
| cells judged collapsed | 880 | ACC <= floor + 1pp |
| cells judged valid | 557 |  |
| overall collapse rate | 61.2% | over cells with a floor |
| cells with balance_sampling=1 | 20 | 0 in the 2026-10-21 audit |
| datasets scanned | 15 |  |
| datasets usable for the main table | 5 | AudioMNIST, BANK, COMPAS, CelebA, DUTCH |
| main-calibre (dataset, hyp, K) groups | 14 | nsp_max >= 3, ablations excluded |
| distinct git commits in evidence | 10 | 0ada0275b, 4dc887b67, 6299c1e23, 6da33067b, 754c44326, 951792035, blank, c522dca6e, ce59163a3, e34c32ad3, e4f45401b |
| snapshot | 2026-10-22 09:33 (Asia/Shanghai) | status probe time; result scan finished 09:28 |

## T8 阻碍出表的结构性缺陷

> Each row is independently evidenced by the cell inventory.

| id | category | finding | quantified evidence | impact on reporting |
|---|---|---|---|---|
| D1 | dataset-level collapse | 9 of 15 datasets are wholly or nearly wholly collapsed; UTKFace and FairFace have EVERY algorithm below the floor, i.e. the failure is in the label/partition/evaluator, not in any algorithm. | ADULT 100%, DRUG 100%, FairFace 100%, UTKFace 100%, CREMA-D 96%, CREMA-D_gender 94%, RAVDESS 94%, BANK 87%, LFWA+ 82% | ADULT and DRUG must be removed from the main table (appendix S7). UTKFace / FairFace / LFWA+ / CREMA-D* / RAVDESS produce no usable number for any algorithm. |
| D2 | baseline is a no-op | FairFed reproduces FedAvg bit-for-bit, so the FairFed column carries no information about FairFed. | 33 (dataset, K, split) cells identical on ACC+ACC_gap+DEO: ADULT/K20/Dirichlet01, ADULT/K20/Dirichlet05, ADULT/K20/Dirichlet1, ADULT/K20/Uniform, BANK/K20/Dirichlet05, BANK/K20/Uniform, BANK/K30/Dirichlet01, BANK/K30/Dirichlet05 | FairFed cannot be used as a comparison point on those datasets until the implementation is fixed. |
| D3 | baseline is a no-op | mFairFL reproduces FedAvg bit-for-bit on some datasets. | 5 cells identical: ADULT/K20/Dirichlet01, ADULT/K20/Dirichlet05, ADULT/K4/Dirichlet05, CREMA-D/K20/Dirichlet1, RAVDESS/K20/Speaker | Same as D2 for mFairFL. |
| D4 | hyper-parameter not wired | FedRenyi lambda has no effect: lam0p25 / lam0p5 / lam1 / lam2 give identical numbers on DRUG. | 4 variant dirs, identical ACC/ACC_gap/DEO on DRUG | The Renyi fairness-strength ablation cannot be reported. |
| D5 | result-tree pollution | COMPAS and DRUG 20Clients directories mix PDFFed ablation variants and lambda sweeps with the main comparison, at nsp=1-2 only. | COMPAS K=20 holds 76 distinct algorithm directories | Any ranking that does not filter these is meaningless; they are excluded here by the ABLATION regex. |
| D6 | coverage hole | CelebA has no PDFFed (non-DP) and no LoGoFair result at all; UTKFace K=30/40 has no PDFFed-family result. | CelebA main algorithms present: FL_FairBatch, FairFed, FedAvg, FedFair, FedNova, FedProto, FedProx, FedRenyi, FederatedAverageWithPo, PDFFed_DP, Scaffold, mFairFL \| UTKFace algorithms present: FL_FairBatch, FairFed, FedAvg, FedFair, FedNova, FedProto, FedProx, FederatedAverageWithPo, PDFFed_DP_eps1, PDFFed_DP_eps100, PDFFed_DP_eps1000, PDFFed_DP_eps1e+06, PDFFed_DP_eps20, PDFFed_DP_eps50, PDFFed_DP_eps8, Scaffold | CelebA can only support a PDFFed_DP claim, and only 3 algorithms reach nsp=4 there, so the group is too thin to rank. |
| D7 | foreign algorithms | .31 result tree contains algorithms belonging to another project, some without ACC_gap/HM at all. | DOSFL, FedFACT, FedMix, NaiveMix, OSFL, PraFFL, ProxProbability | They show up as NaN rows and must be filtered before any ranking. |
| D8 | provenance | Cells whose json is missing or whose partition fingerprint is not in the local cache cannot be assigned a floor or a commit. | 97 cells: ADULT=5, COMPAS=33, CelebA=2, DRUG=24, DUTCH=24, LFWA+=2, moji=7 | They are excluded from every collapse statistic and cannot enter a single-commit table. |
| D9 | reproducibility | The same cell key run on two machines / two commits disagrees on ACC. | 94 cells disagree, max spread 0.3990 (BANK/Uniform/FedProx/ANN/30) | Fix per appendix S9.1: publish from ONE commit, do not re-run. Selection rule fixed a priori to newest recorded_at. |
| D10 | metric interpretation | PDFFed wins HM/ACC but ranks mid-to-low on ACC_gap and DEO in the same groups, so the HM win is driven by accuracy, not by narrowing unfairness. | see pdffed_head_to_head.csv columns gap_rank / deo_rank | Any claim of a fairness advantage must be made on CelebA PDFFed_DP only (DEO rank 1), and must not be inferred from HM. |
| D11 | mitigation did not help | balance_sampling=1 was turned on for 20 cells (0 in the 2026-10-21 audit) but every one of them is still collapsed. | 20 cells, all on RAVDESS (floor 0.8667) and CREMA-D_gender (floor 0.8293) | Label-balanced sampling does not rescue high-floor datasets; the collapse cause is elsewhere. |

## 底线结论

按地板判据（`ACC ≤ floor + 1pp`），当前 **1534 个唯一格**里只有 **5 个数据集**能进主结果表：
AudioMNIST、BANK、COMPAS、CelebA、DUTCH。

1. **PDFFed（非 DP）确实拿第一，但赢在 ACC 不在公平性。** 10 个主口径组里 PDFFed 有
   9 组 HM#1、8 组 ACC#1。优势最大的是 DUTCH（对 FedAvg ΔHM +0.11~+0.15，
   对最强基线 mFairFL +0.071），其次 AudioMNIST（对 Po +0.025）。但同组内 `gap_rank`
   与 `deo_rank` 系统性靠后（COMPAS DEO 13/16、DUTCH ACC_gap 10/15），**HM#1 是 ACC 抬起来的**。
2. **CelebA 的 PDFFed_DP 是唯一「DEO#1 + HM#1」的组合**（DEO 0.018/0.013 vs FedAvg 0.253/0.306），
   代价是 ACC 掉 3.7~10.4pp；且该组只有 3 个算法达到 nsp=4，样本量不足以支撑排名结论。
3. **BANK 的优势在噪声量级。** floor 0.8776，PDFFed 平均只越线 +1.19~+1.46pp
   （n_test=13211 → 1pp≈132 样本），且 4 个划分里仍有 1~2 个塌缩。按审计 §7 只能保留越线结果，
   并**同时披露**其余 13 个算法 ≤+1.02pp、10 个 0 越线，且给出 floor 作参照。
4. **五个模态零可用结果**：文本（bios）、人脸（UTKFace / FairFace / LFWA+）、
   语音情感（CREMA-D / CREMA-D_gender / RAVDESS）。bios floor = **0.5449**，
   PDFFed_DP 0.520 与 PDFFed 0.485 都在地板之下 —— 两者都是塌缩格，
   此前「PDFFed_DP 在文本上有真实优势」的说法**不成立**，那个「最低 ACC_gap」正是塌缩伪公平。
5. **ADULT、DRUG 按审计 §7 整体移出主结果表**（100% 塌缩，0 格越线）。
6. **出表方式**：不重跑，按单一 commit 出表（BANK/DRUG/COMPAS/DUTCH 用 `e4f45401b`，
   ADULT 用 `e34c32ad3`），同格取 `recorded_at` 最新。跨机分歧 94 处已列于 T5。

### 待办（阻塞项）

- CelebA 补 PDFFed（非 DP）与 LoGoFair，并把 FedAvg/FairFed 补齐到 nsp=4。
- 修 FairFed ≡ FedAvg、mFairFL ≡ FedAvg、FedRenyi λ 未生效三个基线缺陷（见 T8 D2/D3/D4）。
- UTKFace / FairFace 全算法低于地板，需先排查标签映射与评测器，属数据集级失效（T8 D1）。
- 97 格无版本指纹，无法归入任何 commit 口径（T8 D8）。
