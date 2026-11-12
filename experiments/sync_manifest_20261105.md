# 三端同步清单 · 2026-11-05

同步时间：`2026-11-05 09:55 ~ 10:35`（本机时钟）
范围：`2026-11-04 00:00` 之后写入的 `result_path/**` 与 `log_path/**`，外加 6 个 `parallel_logs/*/launch_record.txt` 索引文件。
汇聚点：本机 `d:\最新PDFFed\fairness_fl_code`；由 `.30`、`.31` 各自拉取，再由本机补齐到另一台。
**未搬**：`save_path` 断点（体积 65 GB，跨机无意义）、`parallel_logs` 历史大日志（仅搬 6 个本批 `launch_record.txt`）。

## 1. 结论

- 本轮新文件并集 **293 个**：可统一 **234 个**，已三端 sha256 全等；**同名不同源 59 个**按规则**按住未覆盖**（见 §3）。
- 排除按住项后，`.30` / `.31` / 本机三方在这 234 个文件上**文件数与内容完全一致**。
- 234 个可统一文件合计 **209,376,409 字节（约 199.7 MiB）**。

## 2. 逐批次校验（可统一部分，三端已一致）

| 批次 | 统一文件数 | 总字节 | 抽样 sha256（3 个） |
|---|---|---|---|
| ACSIncome | 133 | 185256566 | `log_path/ACSIncome/Dirichlet01/FL_FairBatch/.../1.txt` → `4691911f48bf0021`<br>`.../Dirichlet01/FedRep/.../1.txt` → `17b407013dc71a8b`<br>`.../Dirichlet01/FedProto/.../1.txt` → `12171f89d446286f` |
| AudioMNIST | 8 | 30981 | `result_path/AudioMNIST/Dirichlet01/FedFB/.../1.txt.json` → `cad34f942de0efe3`<br>`.../Dirichlet01/FedNova/.../1.txt.json` → `a622a523e69e1fdc`<br>`.../Dirichlet05/FedFB/.../1.txt.json` → `6c41febdcb448473` |
| CREMA-D | 6 | 23341 | `result_path/CREMA-D/Dirichlet1/FedFB/.../1.txt.json` → `af44a2d8026fa297`<br>`.../Dirichlet1/FedNova/.../1.txt.json` → `9c34f934be75bc60`<br>`.../Speaker/FedFB/.../1.txt.json` → `f12bd2a1f1a9ae05` |
| FairFace | 35 | 11214833 | `log_path/FairFace/Dirichlet01/FairFed/.../1.txt` → `3a2e1bb0507490c3`<br>`.../Dirichlet01/FedFB/.../1.txt` → `7063cc26ba584526`<br>`.../Dirichlet01/FedNova/.../1.txt` → `fd62abd4bb8b04fd` |
| LFWA+ | 12 | 3520276 | `log_path/LFWA+/Dirichlet01/FedNova/.../1.txt` → `d70874a4fa9590b1`<br>`.../Dirichlet01/FedNova/.../1.txt.conflict_20261104_2030` → `18569dc7d5fdacfb`<br>`.../Dirichlet05/FedNova/.../1.txt` → `8ff04ede1da79349` |
| PAD-UFES-20 | 31 | 8150682 | `log_path/PAD-UFES-20/Dirichlet01/FL_FairBatch/.../1.txt` → `4206dfa74e70adf1`（另 2 例见 `/tmp/manifest_*.tsv`） |
| RAVDESS | 2 | 7443 | `result_path/RAVDESS/Speaker/FedFB/.../1.txt.json` → `2f98625974613267`<br>`.../Speaker/FedNova/.../1.txt.json` → `ec9f2424af0f1d2d` |
| VCTK | 7 | 1172287 | `log_path/VCTK/Speaker/FedRenyi/.../1.txt` → `139c6caea66d8852`<br>`result_path/VCTK/Speaker/FedFB/.../1.txt.json` → `89543516837ee58a`<br>`.../FedNova/.../1.txt.json` → `eca185ac2aaf18c5` |
| **合计** | **234** | **209,376,409** | — |

> 抽样 sha256 为全 64 位摘要的前 16 位；全量 sha256 清单在两台服务器的 `/tmp/sha30b.tsv`、`/tmp/sha31b.tsv` 与本机 `C:\Users\GNT\AppData\Local\Temp\sha30b.tsv`、`sha31b.tsv`。

### 2.1 同步动作明细

| 动作 | 文件数 | 说明 |
|---|---|---|
| `.30` → 本机 → `.31` | 82 | `.30` 独有（或 `.30` 为同一运行的较新快照） |
| `.31` → 本机 → `.30` | 96 | `.31` 独有（或 `.31` 为较新快照） |
| 三端本来就同 sha | 56 | 无需搬运 |
| **合计** | **234** | — |

## 3. 按住未覆盖：同名不同源 59 个

判定：同名路径在两侧内容 sha256 不等。分三类：

| 类别 | 数量 | 说明 | 建议 |
|---|---|---|---|
| A. 同格不同运行（`log_path`） | 25 | 两侧首行时间戳不同，是同一格在两个批次/两台机上的两次独立运行 | 需用户决定保留哪一次；**未覆盖任何一侧** |
| B. 结果摘要 `1.txt` | 32 | 仅 `# code_version` 指纹行（fp/commit）不同，指标行完全一致；其中 3 例两侧大小差 1 字节，未逐字节核对 | 等价，可择一统一 |
| C. `bios` 结果 json | 2 | 3661 vs 4574 字节、3660 vs 4586 字节，**指标数值不同** | 真实冲突，需用户决定 |

### 3.1 明细

| # | 路径 | .30 大小 / 首行时间 | .31 大小 / 首行时间 | 本机大小 |
|---|---|---|---|---|
| 1 | `log_path/AudioMNIST/Dirichlet01/FedFB/AUDIOCNNRNN/20Clients/1.txt` | 378678 / 2026-11-04 19:37:06 | 406557 / 2026-10-17 11:15:21 | 406557 |
| 2 | `log_path/AudioMNIST/Dirichlet01/FedNova/AUDIOCNNRNN/20Clients/1.txt` | 387865 / 2026-11-04 19:39:05 | 453774 / 2026-10-17 11:15:21 | 453774 |
| 3 | `log_path/AudioMNIST/Dirichlet05/FedFB/AUDIOCNNRNN/20Clients/1.txt` | 381162 / 2026-11-04 19:40:22 | 374242 / 2026-10-17 12:47:44 | 374242 |
| 4 | `log_path/AudioMNIST/Dirichlet05/FedNova/AUDIOCNNRNN/20Clients/1.txt` | 387981 / 2026-11-04 19:41:35 | 381264 / 2026-10-17 11:49:33 | 381264 |
| 5 | `log_path/AudioMNIST/Dirichlet1/FedFB/AUDIOCNNRNN/20Clients/1.txt` | 381119 / 2026-11-04 19:42:48 | 374253 / 2026-10-17 14:23:51 | 374253 |
| 6 | `log_path/AudioMNIST/Dirichlet1/FedNova/AUDIOCNNRNN/20Clients/1.txt` | 387910 / 2026-11-04 19:44:00 | 381183 / 2026-10-17 14:13:25 | 381183 |
| 7 | `log_path/AudioMNIST/Uniform/FedFB/AUDIOCNNRNN/20Clients/1.txt` | 381071 / 2026-11-04 19:45:16 | 374246 / 2026-10-17 16:17:25 | 374246 |
| 8 | `log_path/AudioMNIST/Uniform/FedNova/AUDIOCNNRNN/20Clients/1.txt` | 387840 / 2026-11-04 19:46:27 | 381102 / 2026-10-17 16:11:02 | 381102 |
| 9 | `log_path/CREMA-D/Dirichlet1/FedFB/AUDIOCNNRNN/20Clients/1.txt` | 45332 / 2026-10-16 21:39:42 | 758231 / 2026-11-04 19:56:45 | 45332 |
| 10 | `log_path/CREMA-D/Dirichlet1/FedNova/AUDIOCNNRNN/20Clients/1.txt` | 46013 / 2026-10-16 21:26:39 | 773074 / 2026-11-04 19:58:00 | 46013 |
| 11 | `log_path/CREMA-D/Speaker/FedFB/AUDIOCNNRNN/20Clients/1.txt` | 744228 / 2026-10-18 07:18:52 | 757862 / 2026-11-04 19:54:11 | 744228 |
| 12 | `log_path/CREMA-D/Speaker/FedNova/AUDIOCNNRNN/20Clients/1.txt` | 758666 / 2026-10-18 00:17:29 | 772849 / 2026-11-04 19:55:25 | 758666 |
| 13 | `log_path/CREMA-D/Uniform/FedFB/AUDIOCNNRNN/20Clients/1.txt` | 45305 / 2026-10-16 21:40:41 | 757996 / 2026-11-04 19:59:14 | 45305 |
| 14 | `log_path/CREMA-D/Uniform/FedNova/AUDIOCNNRNN/20Clients/1.txt` | 45988 / 2026-10-16 21:26:47 | 773041 / 2026-11-04 20:00:46 | 45988 |
| 15 | `log_path/LFWA+/Dirichlet01/FedNova/BERTCLASSIFIER/40Clients/1.txt` | 310871 / 2026-11-04 20:32:10 | 368446 / 2026-10-24 16:06:49 | 368446 |
| 16 | `log_path/RAVDESS/Speaker/FedFB/AUDIOCNNRNN/20Clients/1.txt` | 783924 / 2026-10-20 08:45:06 | 758411 / 2026-11-04 19:51:44 | 783924 |
| 17 | `log_path/RAVDESS/Speaker/FedNova/AUDIOCNNRNN/20Clients/1.txt` | 797808 / 2026-10-20 09:13:08 | 772932 / 2026-11-04 19:52:57 | 797808 |
| 18 | `log_path/VCTK/Speaker/FedFB/AUDIOCNNRNN/20Clients/1.txt` | 1530034 / 2026-11-02 11:18:54 | 753704 / 2026-11-04 19:00:18 | 1530034 |
| 19 | `log_path/VCTK/Speaker/FedFB/AUDIOCNNRNN/30Clients/1.txt` | 954932 / 2026-11-02 18:44:23 | 954278 / 2026-11-04 19:00:18 | 954932 |
| 20 | `log_path/VCTK/Speaker/FedFB/AUDIOCNNRNN/40Clients/1.txt` | 1156176 / 2026-11-02 18:47:24 | 1156298 / 2026-11-04 19:00:18 | 1156176 |
| 21 | `log_path/VCTK/Speaker/FedNova/AUDIOCNNRNN/20Clients/1.txt` | 1751412 / 2026-11-02 10:27:28 | 775183 / 2026-11-04 19:00:18 | 1751412 |
| 22 | `log_path/VCTK/Speaker/FedNova/AUDIOCNNRNN/30Clients/1.txt` | 994290 / 2026-11-02 18:55:22 | 995861 / 2026-11-04 19:00:18 | 994290 |
| 23 | `log_path/VCTK/Speaker/FedNova/AUDIOCNNRNN/40Clients/1.txt` | 1212667 / 2026-11-02 19:02:50 | 1216178 / 2026-11-04 19:00:18 | 1212667 |
| 24 | `log_path/bios/Dirichlet01/FairFed/BERTCLASSIFIER/10Clients/1.txt` | 11201 / 2026-10-16 11:52:50 | 31030 / 2026-11-04 20:04:12 | 11201 |
| 25 | `log_path/bios/Dirichlet01/FedAvg/BERTCLASSIFIER/10Clients/1.txt` | 21748 / 2026-10-16 12:02:00 | 29364 / 2026-11-04 20:05:56 | 21748 |
| 26 | `result_path/AudioMNIST/Dirichlet01/FedFB/AUDIOCNNRNN/20Clients/1.txt` | 343 | 343 | 343 |
| 27 | `result_path/AudioMNIST/Dirichlet01/FedNova/AUDIOCNNRNN/20Clients/1.txt` | 362 | 362 | 362 |
| 28 | `result_path/AudioMNIST/Dirichlet05/FedFB/AUDIOCNNRNN/20Clients/1.txt` | 343 | 343 | 343 |
| 29 | `result_path/AudioMNIST/Dirichlet05/FedNova/AUDIOCNNRNN/20Clients/1.txt` | 362 | 362 | 362 |
| 30 | `result_path/AudioMNIST/Dirichlet1/FedFB/AUDIOCNNRNN/20Clients/1.txt` | 343 | 343 | 343 |
| 31 | `result_path/AudioMNIST/Dirichlet1/FedNova/AUDIOCNNRNN/20Clients/1.txt` | 362 | 362 | 362 |
| 32 | `result_path/AudioMNIST/Uniform/FedFB/AUDIOCNNRNN/20Clients/1.txt` | 343 | 343 | 343 |
| 33 | `result_path/AudioMNIST/Uniform/FedNova/AUDIOCNNRNN/20Clients/1.txt` | 362 | 362 | 362 |
| 34 | `result_path/CREMA-D/Dirichlet1/FedFB/AUDIOCNNRNN/20Clients/1.txt` | 343 | 343 | 343 |
| 35 | `result_path/CREMA-D/Dirichlet1/FedNova/AUDIOCNNRNN/20Clients/1.txt` | 362 | 362 | 362 |
| 36 | `result_path/CREMA-D/Speaker/FedFB/AUDIOCNNRNN/20Clients/1.txt` | 343 | 343 | 343 |
| 37 | `result_path/CREMA-D/Speaker/FedNova/AUDIOCNNRNN/20Clients/1.txt` | 361 | 361 | 361 |
| 38 | `result_path/CREMA-D/Uniform/FedFB/AUDIOCNNRNN/20Clients/1.txt` | 343 | 343 | 343 |
| 39 | `result_path/CREMA-D/Uniform/FedNova/AUDIOCNNRNN/20Clients/1.txt` | 362 | 361 | 362 |
| 40 | `result_path/FairFace/Dirichlet01/FedFB/BERTCLASSIFIER/30Clients/1.txt` | 343 | 343 | 343 |
| 41 | `result_path/FairFace/Dirichlet01/FedNova/BERTCLASSIFIER/30Clients/1.txt` | 361 | 361 | 361 |
| 42 | `result_path/LFWA+/Dirichlet01/FedNova/BERTCLASSIFIER/30Clients/1.txt` | 361 | 361 | 361 |
| 43 | `result_path/LFWA+/Dirichlet01/FedNova/BERTCLASSIFIER/40Clients/1.txt` | 361 | 361 | 361 |
| 44 | `result_path/LFWA+/Dirichlet05/FedNova/BERTCLASSIFIER/40Clients/1.txt` | 361 | 361 | 361 |
| 45 | `result_path/LFWA+/Dirichlet1/FedNova/BERTCLASSIFIER/30Clients/1.txt` | 361 | 361 | 361 |
| 46 | `result_path/LFWA+/Dirichlet1/FedNova/BERTCLASSIFIER/40Clients/1.txt` | 361 | 361 | 361 |
| 47 | `result_path/LFWA+/Uniform/FedNova/BERTCLASSIFIER/40Clients/1.txt` | 361 | 361 | 361 |
| 48 | `result_path/RAVDESS/Speaker/FedFB/AUDIOCNNRNN/20Clients/1.txt` | 344 | 343 | 344 |
| 49 | `result_path/RAVDESS/Speaker/FedNova/AUDIOCNNRNN/20Clients/1.txt` | 361 | 362 | 361 |
| 50 | `result_path/VCTK/Speaker/FedFB/AUDIOCNNRNN/20Clients/1.txt` | 343 | 343 | 343 |
| 51 | `result_path/VCTK/Speaker/FedFB/AUDIOCNNRNN/30Clients/1.txt` | 343 | 343 | 343 |
| 52 | `result_path/VCTK/Speaker/FedFB/AUDIOCNNRNN/40Clients/1.txt` | 343 | 343 | 343 |
| 53 | `result_path/VCTK/Speaker/FedNova/AUDIOCNNRNN/20Clients/1.txt` | 361 | 361 | 361 |
| 54 | `result_path/VCTK/Speaker/FedNova/AUDIOCNNRNN/30Clients/1.txt` | 361 | 361 | 361 |
| 55 | `result_path/VCTK/Speaker/FedNova/AUDIOCNNRNN/40Clients/1.txt` | 361 | 361 | 361 |
| 56 | `result_path/bios/Dirichlet01/FairFed/BERTCLASSIFIER/10Clients/1.txt` | 355 | 355 | 355 |
| 57 | `result_path/bios/Dirichlet01/FairFed/BERTCLASSIFIER/10Clients/1.txt.json` | 3661 | 4574 | 3661 |
| 58 | `result_path/bios/Dirichlet01/FedAvg/BERTCLASSIFIER/10Clients/1.txt` | 355 | 355 | 355 |
| 59 | `result_path/bios/Dirichlet01/FedAvg/BERTCLASSIFIER/10Clients/1.txt.json` | 3660 | 4586 | 3660 |

> 每格断点位置（`save_path/<ds>/<split>/<algo>/<model>/<K>Clients/experiment_state/<hash>/repeat_XX/checkpoint_latest.pt`）与逐格轮次见 `experiments/progress_snapshot_20261105.md` §1、§7。**本轮未搬 `save_path`**。

## 4. 附带同步的索引文件（6 个，三端一致）

`parallel_logs/ff30_ff_20261105_0758/launch_record.txt`、`parallel_logs/am30_20261104_1937/launch_record.txt`、`parallel_logs/vctk_rerun_20261104_1900/launch_record.txt`、`parallel_logs/bios31_20261104_2005/launch_record.txt`、`parallel_logs/ff31_ff_20261104_1915/launch_record.txt`、`parallel_logs/audio31_20261104_1948/launch_record.txt`。
