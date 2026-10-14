# ε 对照实验结果归档 / Epsilon-Control Ablation Result Archive

> 归档时间 / Archived: 2026-10-14 ~ 2026-10-15
> 实验机 / Machine: 192.168.1.31（纯 CPU 运行 / CPU-only）
> 代码版本 / Code revision: `main` 含 PR #30/#31/#32/#33/#36/#37（DP 组合折算求根、B 标定、图像 B=24、GPU 轮转超配修复）

本目录是 **PDFFed_DP 隐私预算 ε 对照实验**的完整留档，用于论文中「隐私预算 vs. 效用/公平性」一节的报告。
This directory is the complete archive of the **PDFFed_DP privacy-budget ε control experiment**, kept for the
"privacy budget vs. utility/fairness" section of the paper.

---

## 1. 目录结构 / Directory Layout

```
epsilon_control_2026-10/
├── README.md                      # 本文件 / this file
├── summary.md                     # 指标汇总表（自动生成）/ aggregated metric tables (auto-generated)
├── diagnostics/
│   └── dp_noise_diagnostics.csv   # 逐配置的 DP 加噪诊断 / per-config DP noise diagnostics
├── result_path/                   # 原始指标输出 / raw metric outputs
│   ├── COMPAS/Dirichlet05/PDFFed_DP_eps{1,8,20,50,100,1e+06}/ANN/20Clients/{1,2,3}.txt
│   ├── DRUG/Dirichlet05/PDFFed_DP_eps{1,8,20,50,100,1e+06}/ANN/20Clients/{1,2,3}.txt
│   ├── COMPAS/Dirichlet05/PDFFed_{fix,k25}/ANN/20Clients/{1,2,3}.txt   # 非 DP 参照 / non-DP references
│   ├── DRUG/Dirichlet05/PDFFed_{fix,k25}/ANN/20Clients/{1,2,3}.txt
│   └── UTKFace/Dirichlet05/PDFFed_DP_eps{1,8,20,50,100,1000,1e+06}/BERTCLASSIFIER/20Clients/1.txt
│   └── UTKFace/Dirichlet05/{FedAvg,FedProx,FedNova,FederatedAverageWithPo}/BERTCLASSIFIER/{4,20,30,40}Clients/*.txt
│       # 含 K 扫描参照（4/20/30/40 客户端）；K=20 行才与 ε 对照同口径
│       # includes extra K-sweep baselines (K = 4/20/30/40); only the K=20 row matches the ε-control setting
└── parallel_logs/eps_ctrl/        # 运行日志 / run logs
    ├── launcher.out / extra_launcher.out
    ├── tab1.log … tab11.log        # 表格臂 / tabular arm
    ├── extra1.log extra2.log extra3.log   # ε=1e6 与图像 ε=1000 补充臂 / supplementary arms
    └── img1.log                    # 图像臂 / image arm
```

> 说明 / Note：图像实验使用的模型是**图像分类网络**；目录名 `BERTCLASSIFIER` 是历史命名遗留，与 BERT 无关。
> The image experiments use an **image classification network**; the directory name `BERTCLASSIFIER` is a
> legacy naming artifact and is unrelated to BERT.

---

## 2. 实验口径 / Experimental Protocol

| 项 / Item | 表格 / Tabular | 图像 / Image |
|---|---|---|
| 数据集 / Dataset | COMPAS, DRUG | UTKFace |
| 任务 / Task | Tabular_CLF | IMG_CLF |
| 算法 / Algorithm | PDFFed_DP | PDFFed_DP |
| 模型 / Model | ANN | CNN + 分类头 (dir: `BERTCLASSIFIER`) |
| 划分 / Split | Dirichlet05 | Dirichlet05 |
| 客户端数 / K | 20 | 20 |
| 通信轮 / I | 100 | 50 |
| 本地轮 / T | 2 | 2 |
| 重复 / repeats | 3 | 3 |
| 学习率 / lr | 3e-4 (SGD) | 3e-4 (SGD) |
| batch size | 256 | 256 |
| 设备 / Device | CPU (`cuda=""`) | CPU (`cuda=""`) |
| 扫描量 / Sweep | ε ∈ {1, 8, 20, 50, 100, 1e6} | ε ∈ {1, 8, 20, 50, 100, 1000, 1e6} |
| 先验界 / B_prior | 1.0 | 24.0 |
| 噪声口径 / σ | `σ = Δ₂·√(2·ln(2/δ))/ε_round`，`Δ₂ = B·√d / m` | 同左 |
| 隐私参数 / DP params | δ = 1e-5 | δ = 1e-5 |

ε = 1e6 一行是**数值上的非隐私端点（近似无噪声）**，用于观测 DP 噪声项被压到可忽略时的行为；
ε = 1000（仅图像）同理，作为图像臂的额外端点。
The ε = 1e6 row is a **numerically non-private endpoint** (noise ≈ 0), used to observe behaviour when the DP
noise term becomes negligible; ε = 1000 (image only) serves the same role as an extra endpoint.

**改动范围 / Scope of change**：本对照实验**只改 ε 这一个隐私预算旋钮**，噪声注入机制、裁剪上界算法、
原型聚合流程均未改动。`dp_epsilon` 在 `epsilon_control_2026-10` 中作为**总隐私预算**，单轮预算
`ε_round` 由 DRV 强组合等式的唯一正根二分反解得到（PR #30），非旧实现的「只解第一项」近似。
**Only the ε knob was varied**; the noise-injection mechanism, clipping bound logic and prototype
aggregation were left untouched. `dp_epsilon` here denotes the **total budget**, and the per-round budget
`ε_round` is obtained by bisection on the full DRV composition equation (PR #30), not by the old
"first-term-only" approximation.

---

## 3. 主要结论 / Key Findings

完整数值见 [summary.md](summary.md)。Below is the condensed reading of `summary.md`.

### 3.1 ADULT 上的「常数预测器塌缩」不是 DP 机制的通病 / The ADULT collapse does not generalize

在 ADULT 上曾观测到 ACC 精确等于多数类占比、DEO/SPD 精确为 0、FR 精确为 1.0 的**常数预测器塌缩**。
本次在 COMPAS / DRUG / UTKFace 三套数据上**均未复现**：DEO/SPD 普遍非零、FR < 1、且指标随 ε 变化。
⇒ 该塌缩是 **ADULT 数据集特异**现象，不能据此断言「DP 噪声破坏 PDFFed 的原型机制」。
The constant-predictor collapse observed on ADULT (ACC exactly equal to the majority-class ratio, DEO/SPD
exactly 0, FR exactly 1.0) **did not reproduce** on COMPAS / DRUG / UTKFace. It is therefore an
**ADULT-specific** phenomenon, not a generic failure of the DP mechanism.

### 3.2 不存在单调的隐私–效用曲线，且方向一致地「反直觉」/ No monotone privacy–utility curve

三套数据上「ε 越大效用越高」的直觉**都不成立**，最优 ε 落在中间或最小端：

| 数据集 / Dataset | HM 最优 ε / best ε by HM | 趋势 / trend |
|---|---|---|
| COMPAS | ε = 1（HM 0.722） | ε 增大 HM 单调下滑至 0.651（1e6） |
| DRUG | ε = 20（HM 0.828） | ε 增大后有回落，1e6 降至 0.713 |
| UTKFace | ε = 1（HM 0.665） | ε 增大 HM 单调下滑至 0.631（1e6） |

即「加噪更少」并没有换来更好的效用，甚至在两个数据集上单调变差。这说明在本文口径下，
**DP 噪声不是效用损失的主导来源**（见 3.4 的机制解释与 3.3 的幅度对比）。
"Less noise" did **not** buy better utility — on two of three datasets the trend is monotonically worse.
This indicates that, under the present protocol, **DP noise is not the dominant source of utility loss**
(see 3.4 for the mechanism and 3.3 for the magnitude comparison).

### 3.3 ε ≥ 1 时 DP 相对非 DP 的效用损失很小 / Small DP penalty for ε ≥ 1

* UTKFace：DP 最好 HM = 0.665（ε=1）vs. FedProx 0.683 → 差约 2.6%；vs. FedAvg 0.675 → 差约 1.5%。
* COMPAS：DP 最好 HM = 0.722 vs. 非 DP `PDFFed_fix` 0.716 → **持平甚至略优**。
* DRUG：DP 最好 HM = 0.828（ε=20）vs. 非 DP `PDFFed_fix` 0.827 → 持平。

⇒ 只要 ε ≥ 1，DP 版本与非 DP 版本在本文口径下基本可打平，隐私保护**没有付出显著的效用代价**。
For ε ≥ 1 the DP variant is essentially on par with its non-DP counterpart; privacy protection costs
little utility under this protocol.

### 3.4 机制落点：先验界 B 与实测幅值 B_obs 的失配度 / Mechanism: the B vs. B_obs mismatch

关键量是**先验硬上界 B** 与**实测特征逐维最大幅值 B_obs**：

| 数据集 / Dataset | B (先验) | B_obs (实测) | B_obs/B | 结果 / outcome |
|---|---|---|---|---|
| ADULT | 1.0 | ≈ 0.17 ~ 0.22 | ≈ 0.2 | 信噪比 < 1 → **塌缩** |
| COMPAS | 1.0 | ≈ 0.41 ~ 0.55 | ≈ 0.5 | 存活 |
| DRUG | 1.0 | ≈ 0.52 ~ 0.78 | ≈ 0.7 | 存活 |
| UTKFace | 24.0 | ≈ 5.6 ~ 13.4 | ≈ 0.3 ~ 0.6 | 存活 |

`Δ₂ = B·√d / m` 中 B 被高估的倍数会**整体抬升噪声**（σ ∝ B）。ADULT 的 B 相对 B_obs 放大约 5 倍，
使 σ 远超原型本身的量级，原型被噪声淹没 → 塌缩；其余数据集 B/B_obs 更接近，σ 与原型同量级，机制可用。
A loose prior `B` inflates σ proportionally (`σ ∝ B`). On ADULT, B overestimates B_obs by ≈ 5×, pushing σ
far above the prototype magnitude, so the prototypes are drowned in noise → collapse. On the other
datasets B/B_obs is tighter (≈ 0.3–0.7), σ is comparable to the prototype scale, and the mechanism works.

### 3.5 稀疏槽位的噪声放大 / Noise amplification at sparse slots

`m` 为 (group × label) 槽位的样本数，`Δ₂ ∝ 1/m`。Dirichlet05 下若干槽位 `m = 0`（空张量，已加保护）
或 `m` 极小（如 1），这些槽位的 σ 可达封顶值 1976.55（见 `diagnostics/dp_noise_diagnostics.csv` 的 `sigma_max`）。
极端稀疏槽位的原型在噪声下不可用，是后续可改进的方向（**不属于本次改动范围**）。
`m` is the per-(group × label) slot count and `Δ₂ ∝ 1/m`. Under Dirichlet05 some slots have `m = 0` (empty
tensors, guarded) or `m = 1`, where σ reaches the 1976.55 cap. Prototypes in extremely sparse slots are
unusable under noise — a future improvement direction, **outside the scope of this change**.

---

## 4. DP 加噪诊断字段说明 / Field Reference for `diagnostics/dp_noise_diagnostics.csv`

由各作业日志中的 `DP noise injected` 行解析得到（取**第 1 轮前 20 个客户端**，即最坏口径）。
Parsed from the `DP noise injected` lines (first round, first 20 clients — the worst-case snapshot).

| 字段 / Field | 含义 / Meaning |
|---|---|
| `dataset` | 数据集名 / dataset name |
| `epsilon_total` | 总隐私预算 ε / total privacy budget |
| `epsilon_round` | 单轮预算 `ε_round`，由 DRV 强组合等式二分反解 / per-round budget from the DRV composition equation |
| `T` | 参与组合的通信轮数（= I）/ composition rounds (= I) |
| `B_prior` | 算法使用的先验硬上界 B / prior hard bound used by the algorithm |
| `B_obs_min` / `B_obs_max` | 特征逐维实测最大幅值的跨客户端最小/最大值 / min & max measured per-dim feature amplitude |
| `d` | 原型维度 / prototype dimension（表格 64，图像 512） |
| `m_min` / `m_max` | (group × label) 槽位样本数的最小/最大值 / min & max slot sample counts |
| `sigma_min` / `sigma_median` / `sigma_max` | 各槽位噪声标准差的最小/中位/最大值 / min, median, max per-slot noise σ |

---

## 5. 复现方式 / Reproduction

```bash
# 表格臂 / tabular arm（示例 ε=20，COMPAS）
python main_Tabular_CLF.py -algorithm PDFFed_DP -dataset COMPAS -model_type ANN \
  -split_strategy Dirichlet05 -num_clients_K 20 -communication_round_I 100 \
  -algorithm_epoch_T 2 -cuda "" -dp_epsilon 20 -dp_delta 1e-5 -dp_feature_bound 1.0 -dp_rounds 100

# 图像臂 / image arm（示例 ε=20，UTKFace）
python main_IMG_CLF.py -algorithm PDFFed_DP -dataset UTKFace \
  -split_strategy Dirichlet05 -num_clients_K 20 -communication_round_I 50 \
  -algorithm_epoch_T 2 -cuda "" -dp_epsilon 20 -dp_delta 1e-5 -dp_feature_bound 24 -dp_rounds 50
```

各 ε 的作业串行编排由 `scripts/launch_parallel.sh <gpu_ids_csv> <cmds_file>` 承担（PR #37 起为
`flock` 原子计数 + 单卡串行，避免同卡多作业叠加导致 OOM）。

---

## 6. 代码版本与数据可用性 / Code Provenance & Data Validity

* **代码指纹 / Code fingerprint**：所有结果文件首行带 `# code_version: fp=d1b1aaef9bcd commit=4dc887b67 dirty=True`
  （`dirty=True` 表示运行时工作区含未提交改动，即本次 DP 修复分支）。
* 本目录同时被 tarball 一并取回了少量**非 ε 对照**的历史记录，报告时必须甄别：
  * `result_path/{COMPAS,DRUG}/Dirichlet05/FedAvg/ANN/20Clients/4.txt` 与
    `.../FedProx/ANN/20Clients/4.txt` 是**桩数据**（两者 ACC/DEO/SPD/FR/HM 完全相同、STD 全为 0.0，
    仅 Gpu Seconds / Communication Cost 不同），**不可作为基线报告**；`summary.md` 已将其排除。
  * `result_path/UTKFace/Dirichlet05/{FedAvg,FedProx,FedNova,FederatedAverageWithPo}/BERTCLASSIFIER/{4,30,40}Clients/`
    是**独立的 K 扫描**记录，与 ε 对照（K=20）口径不同，仅可作参考，不可与 ε 行同表混排。
  * 只有 `.../BERTCLASSIFIER/20Clients/` 下的四个基线（FedAvg / FedProx / FedNova / FederatedAverageWithPo，
    文件名 `4.txt`，其中 `4` 是该基线的实验编号而非 repeat 序号）与 ε 对照口径一致，已用于 `summary.md`。

---

## 7. 论文报告注意事项 / Notes for Paper Reporting

1. **画像口径要写全**：纯 CPU、3 repeats、Dirichlet05/K=20、T=2、I 分别为 100（表格）/50（图像）。
   CPU 与 GPU 的公平性指标差异通常小于 run-to-run 方差，但设备口径仍需在正文说明。
2. **不要报告「隐私–效用单调权衡」**：本文口径下不存在该单调性（3.2）。
3. **ADULT 塌缩需单独说明**：它是 B 与 B_obs 失配导致的 σ 量级问题，属数据集/特征尺度特异，
   应作为「先验界标定敏感性」讨论，而非 DP 机制的普适性结论（3.1、3.4）。
4. **B 的标定必须用全量口径实测**：图像任务上小样本冒烟会严重低估 B_obs（冒烟 ≈6.4~7.7，
   全量 ≈13.0~13.6），据此定的 B 会在全量下余量不足。
5. `summary.md` 与 `diagnostics/*.csv` 均由原始 `result_path/` 与日志自动生成，可随实验补充重跑。
