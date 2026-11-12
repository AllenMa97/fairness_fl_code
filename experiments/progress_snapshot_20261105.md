# 实验进度台账（停止前快照）· 2026-11-05

采集时间：`2026-11-05 09:20 ~ 09:24`（`.30` 09:22:08 / `.31` 09:24:30，各自以本机时钟为准）
采集方式：`nvidia-smi --query-compute-apps` 取 PID → `/proc/<pid>/{cwd,fd,environ}` 反推队列目录、`log_path` 格坐标；argv 已被 `apply_process_alias()` 改写为 `VLLM::Worker_TP*`，故不以 `pgrep -f` 为依据。
判定"实验类"的唯一口径：`/proc/<pid>/cwd == /root/fl` **且** 持有 `/root/fl/**` 文件句柄（vLLM/EngineCore 服务为 `cwd=/`、无 `/root/fl` 句柄）。
**§1–§4 为停止前的只读盘点结果（未改动、未删除任何文件，未发送任何信号）；§5–§7 为随后的优雅停止与停止后复核记录。**

---

## 1. 作业台账（一作业一行，共 30 个在跑作业）

进度列格式：`当前轮次/总轮次`；repeat 列格式：`k/3`（1-based）。
断点列：`experiment_state/<hash>` 下**当前 repeat 目录**的 `checkpoint_latest.pt`；`断点轮次`取自日志最后一条 `Checkpoint saved at round boundary N`。

### 1.1 `.30`（192.168.1.30）—— 23 个作业

| # | GPU | PID（训练主进程） | 数据集 | split | 算法 | K | 队列目录 | 启动时间 | 进度 | repeat | 断点 hash | 断点轮次 | ckpt mtime | ckpt 大小(B) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 798589 | ACSIncome | Dirichlet01 | FL_FairBatch | 20 | `acs_30_gpu3_r2_20261104_135009` | 2026-11-04 13:50:08 | 110/500 | 2/3 | `ec6a16157c2a00550119619389097db157740c4886bd7c43afcf09c7afb46728` | 101 | 2026-11-05 08:58:20 | 10,814,737 |
| 2 | 2 | 884727 | ACSIncome | Dirichlet01 | FL_FairBatch | 30 | `acs_30_g2_b3_20261104_150822` | 2026-11-04 15:08:29 | 88/500 | 2/3 | `d8f72f817dd37a77e6fc2711fcac992218e140d59f2c37fad2771170bbf1860f` | 81 | 2026-11-05 09:11:44 | 10,814,865 |
| 3 | 2 | 884468 | ACSIncome | Dirichlet01 | FL_FairBatch | 40 | `acs_30_g2_b3_20261104_150822` | 2026-11-04 15:08:21 | 84/500 | 2/3 | `7853fa20eb760e98571cabb14f3c72336f0def8d6477f8e504560c07d2646cbd` | 81 | 2026-11-05 09:18:45 | 10,814,865 |
| 4 | 3 | 635616 | ACSIncome | Dirichlet01 | FairFed | 40 | `acs_30_gpu3_b4_20261104_120227` | 2026-11-04 12:03:26 | 250/500 | 3/3 | `dcbe1fbe526b92d9db7831df409f4dcc7e11234f628b510a6a78dd64c6860fd1` | 241 | 2026-11-05 09:08:02 | 11,032,135 |
| 5 | 3 | 884496 | ACSIncome | Dirichlet01 | FedFB | 20 | `acs_30_g3_b3_20261104_150822` | 2026-11-04 15:08:21 | 189/500 | 1/3 | `9e4b43b9d72495504ad2a4754b18abbc0d4a3bea1a05b2ec5d159ad00e234f58` | 181 | 2026-11-05 08:50:48 | 10,923,563 |
| 6 | 1 | 846169 | ACSIncome | Dirichlet01 | FedFB | 30 | `acs_30_g1_fill2_20261104_143448` | 2026-11-04 14:35:19 | 454/500 | 1/3 | `9dcd183dc4cda3020cf99eaf30a2643ef5f1d127ed361cb9601b98412d23fe09` | 451 | 2026-11-05 09:14:42 | 10,981,209 |
| 7 | 0 | 884724 | ACSIncome | Dirichlet01 | FedProx | 40 | `acs_30_g0_b3_20261104_150822` | 2026-11-04 15:08:29 | 178/500 | 1/3 | `215791f43ba06655061bb49bb6aaf2ea4ce2b323f2157e6dcab65fec78b22890` | 171 | 2026-11-05 08:26:14 | 11,030,983 |
| 8 | 0 | 884523 | ACSIncome | Dirichlet01 | FedRep | 20 | `acs_30_g0_b3_20261104_150822` | 2026-11-04 15:08:21 | 316/500 | 1/3 | `5465fc5f440e5e667086f7e434e974f337313e3d38868883a3c119b0ab982fab` | 311 | 2026-11-05 08:53:34 | 10,925,035 |
| 9 | 1 | 884482 | ACSIncome | Dirichlet01 | FedRep | 30 | `acs_30_g1_b3_20261104_150822` | 2026-11-04 15:08:21 | 295/500 | 1/3 | `ecf16dc6a79cc661e0cd83d6ad4d2aa73096527b39ad7e55432f9e10c13719db` | 291 | 2026-11-05 08:59:27 | 10,978,777 |
| 10 | 1 | 884725 | ACSIncome | Dirichlet01 | FedRep | 40 | `acs_30_g1_b3_20261104_150822` | 2026-11-04 15:08:29 | 379/500 | 2/3 | `a1fed2f1ebe60e510c2dbd253d4e6ac1e9ba268a0f8131828d5f1619436ecf20` | 371 | 2026-11-05 08:53:23 | 11,034,055 |
| 11 | 3 | 566347 | ACSIncome | Dirichlet05 | FedProx | 30 | `acs_30_gpu3_gap4_20261104_111008` | 2026-11-04 11:10:08 | 352/500 | 2/3 | `4de6fc8830c6aa20e24783677c342cb6d8ead6bb6b20ac24a34cf62437ebd132` | 341 | 2026-11-05 09:20:42 | 10,979,737 |
| 12 | 0 | 846038 | ACSIncome | Dirichlet05 | FederatedAverageWithPo | 20 | `acs_30_g0_fill2_20261104_143448` | 2026-11-04 14:35:11 | 482/500 | 2/3 | `fdc7aec0766038a56bd4e0a3cef1fbe975e2ed59d7ccc555c5ad5d5dd19b8005` | 471 | 2026-11-05 09:20:18 | 10,942,059 |
| 13 | 2 | 846387 | ACSIncome | Dirichlet1 | FL_FairBatch | 30 | `acs_30_g2_fill2_20261104_143448` | 2026-11-04 14:35:27 | 89/500 | 2/3 | `ff790950370b5816672f1242da1699d1555cd601ce24029cf72ee615b2360b4e` | 81 | 2026-11-05 09:09:15 | 10,814,865 |
| 14 | 2 | 846168 | ACSIncome | Dirichlet1 | FairFed | 20 | `acs_30_g2_fill2_20261104_143448` | 2026-11-04 14:35:19 | 227/500 | 3/3 | `35f645b62234e1cc9b34f59ffa458c7e0b7bf67b5731cb688c2193f5d4b66661` | 221 | 2026-11-05 09:09:10 | 10,924,075 |
| 15 | 3 | 635140 | ACSIncome | Dirichlet1 | FairFed | 40 | `acs_30_gpu3_b4_20261104_120227` | 2026-11-04 12:03:06 | 241/500 | 3/3 | `561455835f9ecc83ca95ed7ae3b7cf7bd6367ea5419842462552c34085ff6c50` | 231 | 2026-11-05 09:02:31 | 11,031,943 |
| 16 | 2 | 845391 | ACSIncome | Dirichlet1 | FedRep | 20 | `acs_30_g2_fill2_20261104_143448` | 2026-11-04 14:34:47 | 476/500 | 2/3 | `447610005c33cc0e97d8516d4a6776246dd1bc051bc5c940e9da9972958067db` | 471 | 2026-11-05 09:02:48 | 10,926,955 |
| 17 | 3 | 633576 | ACSIncome | Uniform | FedProx | 30 | `acs_30_gpu3_b4_20261104_120227` | 2026-11-04 12:02:26 | 236/500 | 3/3 | `2d35da22081e16592f64c1321ad0cdeb4d22b824aeefeeafe97e7862cbad4db9` | 231 | 2026-11-05 09:12:56 | 10,978,073 |
| 18 | 0 | 845658 | ACSIncome | Uniform | FedRep | 30 | `acs_30_g0_fill2_20261104_143448` | 2026-11-04 14:34:55 | 256/500 | 2/3 | `4074665e2c6fdd13b30576904c16adb7c45b64cc736a69924dd98d6511ad267c` | 251 | 2026-11-05 09:07:26 | 10,978,201 |
| 19 | 4 | 1262958 | FairFace | Dirichlet05 | FedNova | 20 | `ff30_ff_20261105_0758` | 2026-11-05 07:58:30 | 45/50 | 1/3 | `59c17b6c811569aff57f98760ca077a5f771ee6c451b2b2b91fe06ab6d13335e` | 41 | 2026-11-05 09:16:36 | 20,871,505 |
| 20 | 5 | 1265842 | FairFace | Dirichlet05 | FedNova | 30 | `ff30_ff_20261105_0758` | 2026-11-05 08:01:29 | 38/50 | 1/3 | `74f477874df35ba9228c0e48e29f5df9a51aa23f8fc10d146f7d2f16a2d7c376` | 31 | 2026-11-05 09:10:31 | 20,924,863 |
| 21 | 4 | 1269290 | FairFace | Dirichlet1 | FedNova | 30 | `ff30_ff_20261105_0758` | 2026-11-05 08:04:39 | 41/50 | 1/3 | `ad462d0a4a427996ff47067b784cb4be86b2fc8e7c7b798fd5b038a45e7db5fa` | 31 | 2026-11-05 09:21:36 | 20,924,991 |
| 22 | 5 | 1272523 | FairFace | Uniform | FedFB | 30 | `ff30_ff_20261105_0758` | 2026-11-05 08:07:42 | 14/50 | 1/3 | `116b9208c292661cdc3dfb1b20fa55644eed92d9fb4b6e660eb9155aaf79e5d0` | 11 | 2026-11-05 09:09:58 | 20,924,671 |
| 23 | 5 | 1280904 | FairFace | Uniform | FedNova | 30 | `ff30_ff_20261105_0758` | 2026-11-05 08:13:03 | 32/50 | 1/3 | `efbbe078809e6fe441c061157ef376aeeb273d1f2232ac636d2d6d4a914d8287` | 21 | 2026-11-05 09:21:23 | 20,924,863 |

`.30` 上 18 个 ACSIncome 作业由 10 个 `bash scripts/launch_parallel.sh` 队列 launcher 派发（每个 launcher 顺序跑队列文件里的下一格），5 个 FairFace 作业是 `nohup` 直启、`ppid=1`。

### 1.2 `.31`（192.168.1.31）—— 7 个作业

| # | GPU | PID（训练主进程） | 数据集 | split | 算法 | K | 队列目录 | 启动时间 | 进度 | repeat | 断点 hash | 断点轮次 | ckpt mtime | ckpt 大小(B) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 24 | 0,1,2 | 1158592（GPU 子进程 1158858 / 1158861 / 1158862） | VCTK | Speaker | FedRenyi | 20 | `vctk_renyi_31_20261104_141141` | 2026-11-04 14:11:41 | 158/200 | 2/3 | `6a41c4ddcea87d150cf113853740e2b28c0dfc98ea49138a85a2c63e96eb05ea` | 151 | 2026-11-05 08:37:31 | 1,075,993 |
| 25 | 0 | 1453934 | ACSIncome | Dirichlet01 | FedRenyi | 20 | `acs_31_g0_b3_20261104_150834` | 2026-11-04 15:08:42 | 68/500 | 1/3 | `6b6c86fd329575742f8f2ac2b0c6b8c325827a900134d9266157ad16da537038` | 61 | 2026-11-05 08:00:30 | 10,924,395 |
| 26 | 0 | 1454697 | ACSIncome | Dirichlet01 | FedRenyi | 40 | `acs_31_g0_b3_20261104_150834` | 2026-11-04 15:08:50 | 69/500 | 1/3 | `cc35eba200d39bd94bb616cc964bbaae2479f8cb8370913ec3c023cd57498271` | 61 | 2026-11-05 07:52:33 | 11,033,671 |
| 27 | 1 | 1455307 | ACSIncome | Dirichlet01 | LoGoFair | 30 | `acs_31_g1_b3_20261104_150834` | 2026-11-04 15:08:58 | 223/500 | 2/3 | `cc3970167f59f1701b5530160042705d1401584ee2659b5c7fb695933a7441c9` | 221 | 2026-11-05 09:20:46 | 10,977,945 |
| 28 | 3 | 1453101 | ACSIncome | Dirichlet05 | FL_FairBatch | 20 | `acs_31_g3_b3_20261104_150834` | 2026-11-04 15:08:34 | 223/500 | 2/3 | `c593bcd0acc4b1d741670c409291c96335854ba69a1316b3f14df5b97527016c` | 221 | 2026-11-05 09:20:12 | 10,814,801 |
| 29 | 3 | 1453927 | ACSIncome | Dirichlet05 | FedRep | 30 | `acs_31_g3_b3_20261104_150834` | 2026-11-04 15:08:42 | 399/500 | 1/3 | `635ce3d883fa77e6bd6579d1f583841284afc6149bb84f3b5c7a827b4d290a63` | 391 | 2026-11-05 09:09:00 | 10,980,249 |
| 30 | 5 | 3089482 | FairFace | Dirichlet05 | FedFB | 30 | `ff31_ff_20261104_1915` | 2026-11-04 19:32:59 | 34/50 | 3/3 | `4cfc341d214e86777aedee9a0d421036979d4699b5c357d279d95bf9d3a71b27` | 31 | 2026-11-05 09:15:53 | 20,924,927 |

**采集期间自然结束（不计入 30 个在跑作业）**：`.31` GPU5 `VCTK/Speaker/FedFB/AUDIOCNNRNN/40Clients`（PID 2809866）于 `09:04:48` 完成最后一轮（200/200、repeat 3/3），已写出 `metrics.json` 与 `result_path/.../1.txt.json`，进程已退出。

---

## 2. 汇总

| 批次归属 | 在跑格数 | 分布 | 最低进度格 |
|---|---|---|---|
| ACSIncome | **23** | `.30` 18 + `.31` 5；split: D01 11 / D05 4 / D1 5 / Uniform 3 | **68/500（repeat 1/3）**：`.31` D01/FedRenyi/K20 |
| FairFace | **6** | `.30` 5 + `.31` 1 | **14/50（repeat 1/3）**：`.30` Uniform/FedFB/K30 |
| VCTK | **1** | `.31` GPU0–2 多卡 | **158/200（repeat 2/3）**：Speaker/FedRenyi/K20 |
| 其他 | 0 | — | — |
| **合计** | **30** | `.30` 23 + `.31` 7 | 全局最低：**FairFace 14/50** |

按算法：FL_FairBatch 5、FedRep 7、FairFed 5、FedProx 4、FedFB 3、FedRenyi 3、FedNova 4、LoGoFair 1、FederatedAverageWithPo 1、VCTK-FedRenyi 1（其中 FedFB/FedNova 直通类共 7 格，其余为累积类）。

---

## 3. 断点可用性（`-resume` 判定）

判定口径：`checkpoint_latest.pt` 存在 **且** 位于当前 repeat 目录（`experiment_state/<hash>/repeat_XX/`），**且** 日志最后一条 `Checkpoint saved at round boundary N` 的 hash/repdir 与该文件一致。本项目的 checkpoint 写入频率为**每 10 轮一次**（`checkpoint_save_freq=10`），因此 ckpt mtime 落后日志 1~10 轮属正常，不作为"断点失效"依据。

| 结果 | 格数 | 说明 |
|---|---|---|
| **断点可用（30/30）** | 30 | 全部 30 个作业的 active repeat 目录下 `checkpoint_latest.pt` 均存在，hash 与日志末条 boundary 一致，size 与模型规模吻合（ANN ≈ 10.8–11.0 MB；BERTCLASSIFIER ≈ 20.9 MB；VCTK AUDIOCNNRNN ≈ 1.08 MB） |
| **断点不可用，需重跑** | **0** | 无 |

逐格断点轮次已列于第 1 节（`断点轮次`），续跑将从该 boundary 轮次恢复，而不是从台账里的"当前轮次"。最保守的是 `.31` D01/FedRenyi/20 与 /40（boundary 61，当前 68/69），以及 `.30` FedProx/40（boundary 171，当前 178）。

其他核对项：

- `save_path` 下**没有**任何 `.tmp` / `.part` / 0 字节文件；`result_path` 下同样为 0（采集时全量扫描 30 格）。
- `result_path/<...>/1.txt.json` 对 30 个在跑格**均不存在**（终态结果仅在整格 3 个 repeat 全部结束后写出），故不能据此判断断点好坏；已完成的格子（如 VCTK FedFB/K40）json 可正常 `json.load`。
- 训练 repeat 目录下的 `metrics.json` 只在**该 repeat 结束时**写出；在跑格的当前 repeat 尚无 `metrics.json`，属正常现象，不作为异常。
- 多 hash 目录：同格 `experiment_state` 下存在历史 hash（10 月旧跑），当前活跃 hash 已在上表标注；旧 hash 的 ckpt 不作为本作业断点。

---

## 4. 停止对象预登记（仅登记，尚未发送信号）

### 4.1 保护清单（绝对不动）

| 主机 | PID | GPU | 显存 | 判据 |
|---|---|---|---|---|
| `.31` | 1065441 | 4 | 44,888 MiB | `cwd=/`，`cmd=VLLM::EngineCore`，Hunyuan/vLLM 服务 |
| `.31` | 1065581 / 1065582 / 1065583 / 1065584 | 0 / 1 / 2 / 3 | 各 24,118 MiB | 同上，vLLM Worker |
| `.30` | 175441 / 175442 / 175443 / 175444 | 0 / 1 / 2 / 3 | 各 39,890 MiB | 同上 |
| `.30` | 18622 / 20673 | 4 / 5 | 5,272 / 4,942 MiB | `VLLM::EngineCore`，`cwd=/` |

### 4.2 计划发送 `SIGTERM` 的对象

`.30`：

- 5 个 FairFace 训练主进程：1262958、1265842、1269290、1272523、1280904（`ppid=1`，直启）
- 10 个队列 launcher 根：566281、633490、798549、845123、845125、845128、884325、884331、884338、884346（均为 `ppid=1` 的 `bash scripts/launch_parallel.sh`；先停根，避免子进程退出后 launcher 继续起队列里的下一格）
- 18 个 ACSIncome 训练主进程（第 1 节 PID 列）

`.31`：

- 5 个 ACS launcher 根：1452876（g0，含 2 格）、1452881（g1）、1452899（g3，含 2 格）
- 5 个 ACS 训练主进程：1453934、1454697、1455307、1453101、1453927
- FairFace 训练主进程 3089482（`ppid=1`）
- VCTK FedRenyi 家族：1158592（主进程，`ppid=1`）、1158858、1158861、1158862（GPU 子进程）、1158857（resource_tracker）

### 4.3 归属不确定、暂不处理

- `.31` PID **1158859 / 1158860**（`name=VLLM::Worker_TP0`、`ppid=1`，持有 `log_path/VCTK/Speaker/FedRenyi/AUDIOCNNRNN/20Clients/1.txt` 句柄、无 GPU 上下文、日志自 2026-11-04 14:11:45 起未更新）。高度疑似 VCTK-FedRenyi 首次启动遗留的孤儿进程，但不在 GPU 上、不影响实验吞吐，**停止阶段先不动**，待主进程退出后观察是否自行消失；若仍驻留，再单独报告由用户决定。
- 各训练进程下 `mem=0`、短命的 `VLLM::Worker_TP*` 子进程（DataLoader worker，09:21–09:24 持续 fork/回收）不单独发送信号，随父进程退出而回收。

---

## 5. 停止执行记录（SIGTERM，未使用 SIGKILL）

| 主机 | 发 TERM 时刻 | 对象 | PID 数 | 结果 |
|---|---|---|---|---|
| `.30` | 2026-11-05 09:29:44 | 10 个 launcher 根 → 18 个 worker shell → 23 个训练主进程 | 51 | 5 秒内全部退出，无残留 |
| `.31` | 2026-11-05 09:30:15 ~ 09:30:21 | 3 个 launcher 根 → 5 个 worker shell → 6 个训练主进程 + VCTK 家族 5 个 | 19 | 5 秒内全部退出，无残留 |

- 每个 PID 在发信号前均做了归属校验（`cwd=/root/fl` 或 `cmd` 含 `launch_parallel.sh`；`cwd=/` 的一律拒绝），校验结果全部为 `OK`，无拒绝、无跳过。
- 先停 launcher 根、再停 worker shell、最后停训练主进程，确保队列 launcher 不会在子进程退出后继续领取下一格。
- **未发送任何 `SIGKILL` / `kill -9`；无超时未退出对象，无需用户裁决。**
- 保护对象（vLLM / EngineCore / Hunyuan / `.31` GPU4 服务）全部存活，PID 与显存不变。

---

## 6. 停止后复核

### 6.1 残留实验进程

| 主机 | `cwd=/root/fl` 或 `cmd` 含 `launch_parallel.sh` 的进程 | 结论 |
|---|---|---|
| `.30` | **NONE** | 无残留 |
| `.31` | **NONE** | 无残留 |

`.31` 采集阶段标记"归属不确定"的 1158859 / 1158860 在停止前已自行消失，无需处理。

### 6.2 停止后资源水位（实测）

| 主机 | 时刻 | GPU0 | GPU1 | GPU2 | GPU3 | GPU4 | GPU5 | RAM used/total | available | load1 |
|---|---|---|---|---|---|---|---|---|---|---|
| `.30` | 09:31:19 | 39,902 | 39,902 | 39,902 | 39,902 | 5,284 | 4,952 | 41 / 503 GB | 455 GB | 6.48 |
| `.31` | 09:32:03 | 24,130 | 24,130 | 24,128 | 24,128 | 44,898 | 1 | 41 / 503 GB | 458 GB | 3.80 |

- 两机 GPU 利用率均为 **0%**。
- 两机剩余显存占用全部来自服务：`.30` GPU0–3 各 39,890 MiB（vLLM）+ GPU4 5,272 / GPU5 4,942 MiB（EngineCore）；`.31` GPU0–3 各 24,118 MiB（vLLM）+ GPU4 44,888 MiB（Hunyuan）。
- RAM 从停止前的 `.30` 246 GB、`.31` 98 GB 回落到各 **41 GB**，远低于 430 GB 红线。

### 6.3 flush 与可疑产物

| 检查项 | 结果 | 说明 |
|---|---|---|
| TERM 后 `checkpoint_latest.pt` 是否有更新 | **0/30 更新** | 全部 30 个 ckpt 的 mtime 均早于 TERM 时刻。训练主进程未注册 `SIGTERM` 处理器（全仓仅 `tool/worker.py`、`_warm_cache.py` 注册），SIGTERM 等价立即终止，不会触发额外的 checkpoint 落盘。最后断点仍为最近一次 10 轮边界，文件完整可读。 |
| TERM 后 `metrics.json` 是否有更新 | **0/30 更新** | 在跑格当前 repeat 尚未结束；已存在的 `metrics.json` 均为历史 repeat 的产物（mtime 2026-11-03 ~ 11-05 早些时候）。 |
| TERM 后 `1.txt` 是否有更新 | 29/30 的 mtime 早于 TERM；`.31` VCTK-FedRenyi 为 09:30:23（比 TERM 晚约 2 秒） | 该格属进程退出时的最后一次缓冲刷写，非新增训练轮次。 |
| `result_path` / `save_path` 下 `.tmp` / `.part` / 0 字节 json | **NONE**（30 格全扫） | 无半写文件 |
| `checkpoint_latest.pt` 是否小于 1 KB | **0 个** | 最小 1,075,993 B（VCTK），最大 20,924,991 B（FairFace） |
| 结果 json 可 `json.load` | `.30` 近 24h 18/18 通过；`.31` 20/20 通过 | 无损坏 json |
| 在跑格的 `result_path/1.txt.json` | 30/30 不存在 | 属正常：终态 json 仅在整格 3 个 repeat 全部结束后写出 |

未发现任何可疑产物，未删除任何文件。

---

## 7. 停止位置与续跑清单

"停时轮次"为日志最后一条 `Communication Round`；"断点轮次"为最后一次 `Checkpoint saved at round boundary`，**`-resume` 将从断点轮次恢复**，因此每格会重跑 `停时轮次 − 断点轮次` 轮（1~10 轮，由 `checkpoint_save_freq=10` 决定）。

| # | 主机 | 格坐标 | 停时轮次 | repeat | 断点轮次 | 重跑轮数 | 断点 |
|---|---|---|---|---|---|---|---|
| 1 | `.30` | ACSIncome/Dirichlet01/FL_FairBatch/K20 | 112/500 | 2/3 | 111 | 1 | 可用 |
| 2 | `.30` | ACSIncome/Dirichlet01/FL_FairBatch/K30 | 89/500 | 2/3 | 81 | 8 | 可用 |
| 3 | `.30` | ACSIncome/Dirichlet01/FL_FairBatch/K40 | 89/500 | 2/3 | 81 | 8 | 可用 |
| 4 | `.30` | ACSIncome/Dirichlet01/FairFed/K40 | 254/500 | 3/3 | 251 | 3 | 可用 |
| 5 | `.30` | ACSIncome/Dirichlet01/FedFB/K20 | 190/500 | 1/3 | 181 | 9 | 可用 |
| 6 | `.30` | ACSIncome/Dirichlet01/FedFB/K30 | 457/500 | 1/3 | 451 | 6 | 可用 |
| 7 | `.30` | ACSIncome/Dirichlet01/FedProx/K40 | 179/500 | 1/3 | 171 | 8 | 可用 |
| 8 | `.30` | ACSIncome/Dirichlet01/FedRep/K20 | 318/500 | 1/3 | 311 | 7 | 可用 |
| 9 | `.30` | ACSIncome/Dirichlet01/FedRep/K30 | 296/500 | 1/3 | 291 | 5 | 可用 |
| 10 | `.30` | ACSIncome/Dirichlet01/FedRep/K40 | 381/500 | 2/3 | 371 | 10 | 可用 |
| 11 | `.30` | ACSIncome/Dirichlet05/FedProx/K30 | 355/500 | 2/3 | 351 | 4 | 可用 |
| 12 | `.30` | ACSIncome/Dirichlet05/FederatedAverageWithPo/K20 | 484/500 | 2/3 | 481 | 3 | 可用 |
| 13 | `.30` | ACSIncome/Dirichlet1/FL_FairBatch/K30 | 94/500 | 2/3 | 91 | 3 | 可用 |
| 14 | `.30` | ACSIncome/Dirichlet1/FairFed/K20 | 231/500 | 3/3 | 221 | 10 | 可用 |
| 15 | `.30` | ACSIncome/Dirichlet1/FairFed/K40 | 245/500 | 3/3 | 241 | 4 | 可用 |
| 16 | `.30` | ACSIncome/Dirichlet1/FedRep/K20 | 478/500 | 2/3 | 471 | 7 | 可用 |
| 17 | `.30` | ACSIncome/Uniform/FedProx/K30 | 240/500 | 3/3 | 231 | 9 | 可用 |
| 18 | `.30` | ACSIncome/Uniform/FedRep/K30 | 258/500 | 2/3 | 251 | 7 | 可用 |
| 19 | `.30` | FairFace/Dirichlet05/FedNova/K20 | 50/50 | 1/3 | 41 | 9 | 可用 |
| 20 | `.30` | FairFace/Dirichlet05/FedNova/K30 | 43/50 | 1/3 | 41 | 2 | 可用 |
| 21 | `.30` | FairFace/Dirichlet1/FedNova/K30 | 45/50 | 1/3 | 41 | 4 | 可用 |
| 22 | `.30` | FairFace/Uniform/FedFB/K30 | 15/50 | 1/3 | 11 | 4 | 可用 |
| 23 | `.30` | FairFace/Uniform/FedNova/K30 | 35/50 | 1/3 | 31 | 4 | 可用 |
| 24 | `.31` | VCTK/Speaker/FedRenyi/K20 | 159/200 | 2/3 | 151 | 8 | 可用 |
| 25 | `.31` | ACSIncome/Dirichlet01/FedRenyi/K20 | 68/500 | 1/3 | 61 | 7 | 可用 |
| 26 | `.31` | ACSIncome/Dirichlet01/FedRenyi/K40 | 70/500 | 1/3 | 61 | 9 | 可用 |
| 27 | `.31` | ACSIncome/Dirichlet01/LoGoFair/K30 | 227/500 | 2/3 | 221 | 6 | 可用 |
| 28 | `.31` | ACSIncome/Dirichlet05/FL_FairBatch/K20 | 230/500 | 2/3 | 221 | 9 | 可用 |
| 29 | `.31` | ACSIncome/Dirichlet05/FedRep/K30 | 402/500 | 1/3 | 401 | 1 | 可用 |
| 30 | `.31` | FairFace/Dirichlet05/FedFB/K30 | 36/50 | 3/3 | 31 | 5 | 可用 |

**断点不可用、需重跑：0 格。** 30 格均可从表中"断点轮次"用 `-resume` 续跑。

需人工留意的两处（不影响续跑）：

1. `.31` VCTK-FedRenyi/K20 的 `log_path/.../1.txt` 自 2026-11-04 14:11:45 起未再更新，当前进度只能从其 queue 的 `parallel_logs/vctk_renyi_31_20261104_141141/run.log` 读取（159/200、repeat 2/3、断点 151）。续跑时日志会重新写回 `1.txt`，属历史日志分叉，不是丢数据。
2. `.30` FairFace/Dirichlet05/FedNova/K20 停在 50/50（repeat 1/3 末轮），本次被 TERM 打断在 repeat 收尾阶段，未写出该 repeat 的 `metrics.json`；续跑会从断点 41 重跑最后 9 轮后重新收尾，不影响最终 3-seed 结果的完整性。
