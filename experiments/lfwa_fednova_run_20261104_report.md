# LFWA+ / FedNova 批量重跑报告（2026-11-04，`.30` GPU4/GPU5）

## 一、结论

| 项目 | 结果 |
|---|---|
| 拉起作业数 | **6 个**（GPU4 三格、GPU5 三格，串行排队逐个启动） |
| 待跑坐标 | 取自 [pending_runs.md](file:///d:/最新PDFFed/fairness_fl_code/experiments/pending_runs.md) 第 34/35/38/41/42/44 行，6/6 全部覆盖，无缺格 |
| 批量参数 | 全部 `-mini_batch 128`，省略 `-batch_size`；配置打印均为 `batch_size=256 / accumulation_steps=2 / effective_batch_size=256` |
| 累积验证 | **已用运行时计数证实生效**：Dirichlet01/K40 实测 micro-batch 494 次、`optimizer.step()` 340 次，与按各 client 数据量解析出的理论值（494 / 340）**逐位吻合**；无累积时 step 数应等于 494，实测少 154 次 |
| 08:55 完成情况 | **6 格 × 3 repeats 全部自然跑完并在 22:33 前落盘**，08:55 时无可停作业，**未发送任何 SIGTERM**（也就没有部分完成的 repeat） |
| 结果文件 | 6 个 `result_path/.../1.txt.json` 全部生成（20:57 / 21:12 / 21:44 / 21:46 / 22:23 / 22:33） |
| 资源 | 每作业约 3.7–3.8 GB 显存，单卡峰值占用 11.4 GB / 49.1 GB（GPU4）、8.7 GB（GPU5），安全垫远超 2 GB；主机 RAM used 332–350 GB / 503 GB（≤ 430 ✓） |
| 既有作业 | **未 TERM 任何既有作业**，GPU4/5 上原有 AudioMNIST 与 VLLM EngineCore 全程存活，作业退出后占用回落到基线 |

---

## 二、启动前的占用核查与争用情况

20:04–20:16 首次核查（`nvidia-smi` + `/proc/*/cmdline` + `/proc/*/fd`）：

| 卡 | 既有占用 | 本批可用余量 |
|---|---|---|
| GPU4 | 4 × AudioMNIST(FedFB，各约 600 MiB) + 1 × VLLM EngineCore 约 5.3 GB ≈ **7.7 GB** | 约 41 GB |
| GPU5 | 4 × AudioMNIST(FedNova，各约 600 MiB) + 1 × VLLM EngineCore 约 4.9 GB ≈ **4.95 GB** | 约 44 GB |
| GPU0–3 | ACSIncome 系列，占用 46.0–46.9 GB / 49.1 GB | 未触碰 |

- 争用结论：GPU4/5 均未被占满，LFWA+ 单作业 3.7–3.8 GB 可直接放入，**未发生挤占**。
- 本批作业全部退出后复测：GPU4 **7691 MiB**、GPU5 **4952 MiB**，与开跑前基线一致，证明既有作业未受影响（GPU5 既有占用在 20:36–22:23 之间保持约 4.95 GB 不变）。

> 说明：训练进程 argv 在远端被改写为 `VLLM::Worker_TP*`/`VLLM::EngineCore`，`cmdline` 无法用于识别；因此作业识别改用 `/proc/<pid>/fd` 中打开 `log_path/LFWA+...` 的描述符，并用 `Tgid == pid` 去掉线程，未按名字误杀任何进程。

---

## 三、参数与累积验证

### 3.1 逐格坐标（原表 vs 本次）

| # | 数据集 | split | 算法 | 模型 | K | 原表名义 bs | 本次 effective | `mini_batch` | `accumulation_steps` |
|---|---|---|---|---|---|---|---|---|---|
| 34 | LFWA+ | Dirichlet01 | FedNova | BERTCLASSIFIER | 30 | 32（B 类无累积） | **256** | 128 | 2 |
| 35 | LFWA+ | Dirichlet01 | FedNova | BERTCLASSIFIER | 40 | 64（B 类无累积） | **256** | 128 | 2 |
| 38 | LFWA+ | Dirichlet05 | FedNova | BERTCLASSIFIER | 40 | 64 | **256** | 128 | 2 |
| 41 | LFWA+ | Dirichlet1 | FedNova | BERTCLASSIFIER | 30 | 32 | **256** | 128 | 2 |
| 42 | LFWA+ | Dirichlet1 | FedNova | BERTCLASSIFIER | 40 | 64 | **256** | 128 | 2 |
| 44 | LFWA+ | Uniform | FedNova | BERTCLASSIFIER | 40 | 64 | **256** | 128 | 2 |

其余超参全部沿用 LFWA+ 既有设定（未做任何改动）：`-communication_round_I 50`、`-algorithm_epoch_T 2`、`-exp_repeat_times 3`、`learning_rate 0.0003`、`optimize_method sgd`、`test_batch_size 256`、`dataloader_num_workers None`、`test_eval_freq 5`、`base_seed 42`、`checkpoint_save_freq 10`、`use_amp auto`、`-final_artifact_policy metrics_only`。`-batch_size` 未传（避免废止别名 WARNING）。

### 3.2 配置打印（6 格一致）

```
****** mini_batch : 128 ******
****** batch_size : 256 ******
****** accumulation_steps : 2 ******
****** effective_batch_size : 256 ******
```
（`128 × 2 = 256` 满足要求。）

### 3.3 运行时计数证据（非仅看配置）

探针方式：通过 `PYTHONPATH` 注入 `/root/fl_ops/probe/sitecustomize.py`，monkeypatch `torch.optim` 各具体优化器子类的 `step` 与 `DataLoader.__iter__`，每 5 秒原子写快照；**未修改仓库训练代码**（仓库 `git` 版本 `6299c1e`，dirty 570，与本地工作区一致）。

抽取 **GPU5 Dirichlet01/K40**（pid 1956450）做逐项核对：

| 口径 | micro-batch 数 | optimizer.step() 数 | step : micro-batch |
|---|---|---|---|
| 理论值（按各 client 数据量解析，`floor(size/128)`，每 2 个 micro-batch 记 1 step） | 494 | 340 | 1 : 1.452 |
| 探针实测 | **494** | **340** | **1 : 1.452** |
| 若无累积（对照） | 494 | 494 | 1 : 1 |

- 实测与理论**完全一致**；相比无累积少 154 次 step，可判定累积真实生效。
- 其他格的整段运行快照（全部来自新探针）：

| 格 | micro-batch | optimizer.step() | 比值 |
|---|---|---|---|
| GPU5 D1/K40（3 repeats 全程） | 3078 | 1880 | 0.611 |
| GPU4 Uniform/K40（3 repeats 全程） | 3600 | 2400 | 0.667 |
| GPU4 D05/K40（中途快照） | 2112 | 1348 | 0.638 |
| GPU5 D1/K30（中途快照） | 1480 | 824 | 0.557 |

**为什么比值不是严格的 1:2**：LFWA+ 的 Dirichlet 划分下多数 client 只有 ≤128 个样本（仅 1 个 micro-batch），而训练循环在每个 client 的**最后一个 micro-batch 处强制 step**（`(batch_id+1)%accumulation_steps==0 or 最后一个 batch`），因此小 client 天然拉高 step 占比；大 client（如 3815 样本 → 30 micro-batch → 15 step）恰为 1:2。Discichlet01/K40 一格的逐 client 理论推导与实测逐位吻合，已足以排除"累积未生效"。

> 既有代码语义提醒（未改动）：梯度是**累加**而非按 `accumulation_steps` 平均（`FedNova.py` 中 loss 未除以累积步数），与项目既有约定一致；本次只按要求把 effective batch 提到 256。

### 3.4 首轮 loss（各格 setup 后首批 client）

| 格 | client | Epoch 1 | Epoch 2 |
|---|---|---|---|
| D01/K30 | 27 | 1.6423 | 1.4164 |
| D01/K40 | 19 | 0.4236 | 0.3214 |
| D05/K40 | 19 | 0.00679 | 0.00676 |
| D1/K30 | 27 | 0.00703 | 0.00686 |
| D1/K40 | 19 | 0.00541 | 0.00535 |
| Uniform/K40 | 19 | 0.00796 | 0.00864 |

（D01/K30 首个 client 的 1.64 是该划分下该 client 的初始 loss 实测值，未做任何修饰。）

---

## 四、逐次启动时间线

策略：每卡由 `scripts/launch_parallel.sh` 串行排队（`PER_GPU_LIMIT=1`），起 1 个等 60–90 秒再起下一个；`-cuda 4/5` 由 launcher 注入，`OMP_NUM_THREADS=28`。

| 时间 | 卡 | 格 | 事件 |
|---|---|---|---|
| 20:04–20:12 | 4/5 | — | 占用核查、查重（同名目录/同 seed 双进程）、探针自测 |
| 20:12:41 | 4 | D01/K30 | 启动（GPU4 首个；该格使用初版探针，step 计数不适用，故累积证据改用 D01/K40） |
| 20:16–20:30 | 5 | D01/K40 | 发现同格双进程（本批旧 launcher 退出后其孤儿 worker 继续起作业）：两个 PID 同时写同一 `1.txt`；仅 TERM **本批自身**的重复进程与孤儿 worker（未触碰任何他人作业），污染产物改名归档为 `1.txt.conflict_20261104_2030` / `experiment_state.conflict_20261104_2030`，**不计入结果** |
| 20:32:10 | 5 | D01/K40 | 用含新探针的 launcher 重新启动 |
| 20:57:32 | 4 | D01/K30 | 完成：3 repeats + `30Clients/1.txt.json` |
| 20:57:38 | 4 | D05/K40 | 启动（等待 6 s，接续上一格退出） |
| 21:12:24 | 5 | D01/K40 | 完成：3 repeats + `40Clients/1.txt.json` |
| 21:12:30 | 5 | D1/K30 | 启动 |
| 21:44:25 | 4 | D05/K40 | 完成：3 repeats + `40Clients/1.txt.json` |
| 21:44:31 | 4 | Uniform/K40 | 启动 |
| 21:46:07 | 5 | D1/K30 | 完成：3 repeats + `30Clients/1.txt.json` |
| 21:46:14 | 5 | D1/K40 | 启动 |
| 22:23:23 | 5 | D1/K40 | 完成：3 repeats + `40Clients/1.txt.json` |
| 22:33:24 | 4 | Uniform/K40 | 完成：3 repeats + `40Clients/1.txt.json`；launcher 输出 `all done; worker_failures=0` |

单格耗时约 33–49 分钟（`I=50`、`T=2`、3 repeats）；同卡两格之间的启动间隔受上一格实际结束时间支配（6 秒级），满足"逐个启动、起后观察再起下一个"的要求。

每格启动后 60–90 秒的观察记录（汇总口径，非逐分钟采样）：进程存活；该卡显存增加约 3.7–3.8 GB；主机 `free -g` used 342–350 GB（available 144–152 GB）；首轮 loss 见 3.4 节。

---

## 五、完成核对明细

### 5.1 逐 repeat `metrics.json`（`save_path/.../experiment_state/<hash>/repeat_XX/`）

| 格 | repeat_00 | repeat_01 | repeat_02 | 结果 json |
|---|---|---|---|---|
| D01/K30 | 20:26:49 ACC 0.5907 DEO 0.1424 | 20:41:33 ACC 0.5261 DEO 0.0447 | 20:57:31 ACC 0.6135 DEO 0.0000 | 20:57:32 |
| D01/K40 | 20:45:02 ACC 0.6135 DEO 0.0034 | 20:58:48 ACC 0.6135 DEO 0.0000 | 21:12:23 ACC 0.4230 DEO 0.0142 | 21:12:24 |
| D05/K40 | 21:13:10 ACC 0.6272 DEO 0.1537 | 21:28:44 ACC 0.6238 DEO 0.1213 | 21:44:25 ACC 0.6238 DEO 0.3170 | 21:44:25 |
| D1/K30 | 21:23:51 ACC 0.6333 DEO 0.2193 | 21:34:51 ACC 0.6352 DEO 0.2808 | 21:46:07 ACC 0.6459 DEO 0.3050 | 21:46:07 |
| D1/K40 | 21:58:37 ACC 0.6398 DEO 0.2358 | 22:11:00 ACC 0.6337 DEO 0.1939 | 22:23:22 ACC 0.6402 DEO 0.2696 | 22:23:23 |
| Uniform/K40 | 22:00:50 ACC 0.6348 DEO 0.3406 | 22:17:21 ACC 0.6428 DEO 0.2802 | 22:33:23 ACC 0.6364 DEO 0.3612 | 22:33:24 |

18/18 个 `metrics.json` 全部存在且可解析，`repeat_idx` 0/1/2 齐全。

### 5.2 `result_path/.../1.txt.json`（聚合 3 repeats）

| 格 | 记录时间 | reps | ACC | DEO | SPD | HM | FR |
|---|---|---|---|---|---|---|---|
| D01/K30 | 2026-11-04 20:57:32 | 3 | 0.5768 | 0.0624 | 0.0625 | 0.7128 | 0.9376 |
| D01/K40 | 2026-11-04 21:12:24 | 3 | 0.5500 | 0.0059 | 0.0094 | 0.7040 | 0.9941 |
| D05/K40 | 2026-11-04 21:44:25 | 3 | 0.6250 | 0.1973 | 0.1707 | 0.7007 | 0.8027 |
| D1/K30 | 2026-11-04 21:46:07 | 3 | 0.6381 | 0.2683 | 0.2388 | 0.6812 | 0.7317 |
| D1/K40 | 2026-11-04 22:23:23 | 3 | 0.6379 | 0.2331 | 0.2047 | 0.6961 | 0.7669 |
| Uniform/K40 | 2026-11-04 22:33:24 | 3 | 0.6380 | 0.3273 | 0.3151 | 0.6545 | 0.6727 |

### 5.3 `checkpoint_latest.pt` 说明（重要，避免误判为"未落盘"）

- 运行期间确实每 10 轮写一次 checkpoint，且每 repeat 收尾必然写"round boundary 50"：6 格日志均可见 `Checkpoint saved at round boundary 1/11/21/31/41/50: .../repeat_XX/checkpoint_latest.pt`。
- 但本次沿用既有 `-final_artifact_policy metrics_only`，`repeat` 完成时由 `finalize_repeat_artifacts`（[checkpoint.py](file:///d:/最新PDFFed/fairness_fl_code/tool/checkpoint.py#L816-L826)）**主动删除** `checkpoint_latest.pt`。因此 6 格完成后目录中按设计不再保留 checkpoint；唯一残留的一个是 20:30 归档的冲突目录内文件。
- 结论：`metrics.json` 与 `1.txt.json` 均已落盘，checkpoint 属策略性删除，非写盘失败。

---

## 六、08:55 优雅停止核对

- 全部 6 格在 **22:33:24** 自然结束，**远早于 08:55**；按任务约定"若作业在 08:55 前已自然跑完，正常退出即可，不要额外 kill"，**未发送任何 SIGTERM/kill**。
- 22:33:55 的自动完成核对（`/root/fl_ops/lfwa_done_report.txt`）：`remaining_lfwa_pids=0 remaining_launchers=0`，两个 launcher 均输出 `all done; worker_failures=0`。
- 22:37 复核：无本批训练进程、无本批 launcher 残留；GPU4/GPU5 显存回落至既有基线。
- 预置的 08:50/08:55 watchdog 已在本批自然完成后**主动解除**（pid 2001214 于 22:37:53 收到 TERM 退出）。原因：watchdog 的进程识别会匹配所有打开 `LFWA+` 日志的进程，若保留到明早，可能误伤其他会话新起的 LFWA 作业，违背"不得 TERM 他人在跑作业"。因此本批不需要、也不再保留该终止器。

完成状态（如实）：

| 格 | 状态 |
|---|---|
| D01/K30、D01/K40、D05/K40、D1/K30、D1/K40、Uniform/K40 | **全部完成**（3/3 repeats + 聚合 json） |
| 部分完成 | **无** |
| 未完成 | **无** |
| 不计入结果 | D01/K40 的 20:16–20:30 双进程污染产物（已改名 `.conflict_20261104_2030` 归档） |

---

## 七、其他如实说明

1. **GPU4 第一格（D01/K30）** 使用的是初版探针（只 patch 了 `torch.optim.Optimizer` 基类），`step` 计数恒为 0，故该格的累积运行时证据不可用；改用 D01/K40 的逐位吻合证据代表本批。配置打印四字段在 6 格日志中均正确。
2. **批量语义**：`mini_batch=128 × accumulation_steps=2 = 256` 已满足目标 effective batch；仓库的 loss 未做 1/2 平均（累加语义），本次未修改代码，仅记录该事实。
3. **D01/K40 repeat_02** 的 ACC 0.4230 明显低于同格其它 repeat（0.6135/0.6135），为真实产出、未作处理；如需剔除或复跑请另行决策。
4. 本次只新增/使用了运维脚本（远端 `/root/fl_ops/`、本地 `%TEMP%\fl_ops\`），**仓库代码、`dataloader_num_workers`、超参均未改动**；`pending_runs.md` 未修改。
