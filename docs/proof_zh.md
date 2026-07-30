# PDFFed 理论证明：宏观链路与详细推导

## 0. 宏观证明链路

PDFFed 的设计围绕三个核心挑战。以下说明证明链路如何为每个挑战提供理论支撑。

### 挑战 1：隐私泄露风险（纲要：现有方法常依赖私有数据的敏感属性构建通信内容）

PDFFed 选择传输原型而非敏感属性，并在原型上添加 LDP 噪声，其安全性通过以下定理证明：

```
传输原型（非敏感属性）
    │
    ├── 定理 6（反面证明）
    │      传输敏感属性 → 成员推断攻击成功率 = 1 → 不满足差分隐私
    │      结论：必须避免传输敏感属性
    │
    ├── 定理 5（正面构造）
    │      客户端对原型添加 LDP 噪声 → 聚合后满足分布式差分隐私
    │      结论：传输原型 + DP 噪声 = 安全的通信模式
    │
    └── 推论：PDFFed 的原型通信机制在理论上比传输敏感属性的方法更安全
```

### 挑战 2：表征偏差引起的公平性-精度协同退化（纲要：现有方法多聚焦于损失约束或聚合调控）

表征偏差 $\Delta_{\text{rep}}$ 同时威胁公平性与精度：定理 1 给出 $\Delta_{\text{rep}}$ → EO 上界（公平性退化），引理 3 表明 $\Delta_{\text{rep}}$ 破坏 prototype-driven 分类一致性（精度退化）。定理 1-3 与引理 2-3 构建了从"识别问题"到"实施干预"再到"精度保障"的完整闭环：

```
表征偏差 Δ_rep 是公平性问题（EO）的根源
    │
    ├── 定理 1（上界：Δ_rep → EO）
    │       EO ≤ (‖w‖/(√(2π)σ_z)) · Δ_rep + O(Δ_Σ)
    │       结论：减小 Δ_rep 是降低 EO 上界的关键且可操作的手段
    │
    ├── 定理 2（下界：置信度差异 → EO）
    │       EO ≥ 2δ_conf - O(σ_z, Δ_Σ)
    │       结论：置信度差异是公平性问题的诊断信号（必要条件）
    │
    ├── 定理 3（干预：CL → Δ_rep 收缩）
    │       CL 始终运行 → Δ_rep 持续下降 + 收敛性不破坏
    │       结论：在训练中直接缩小 Δ_rep 是可行的
    │
    ├── 引理 2（双向对齐 → 有效分布偏移 Γ̄_w 控制）
    │       L_l2g + L_g2l ≤ ε → |w^T(μ_l^{(k)} - μ_l^*)| ≤ O(ε/ρ)
    │       结论：原型-分类器对齐控制 w 方向的有效分布偏移（含一步延迟论证）
    │
    └── 引理 3（prototype-driven 分类一致性）
            L_l2l ≤ ε → 样本错误率 ≤ O(ε + exp(-ρ²/(2σ²|w|²)))
            结论：原型正确分类 ⟹ 样本正确分类（高概率），prototype-driven 有理论根基
```

**闭环逻辑**：定理 1 说明"为什么要优化 $\Delta_{\text{rep}}$"（上界约束），定理 2 说明"如何诊断问题"（置信度差异），定理 3 说明"如何实施优化"（始终运行 CL）且不损害主任务。引理 1 进一步保证了 CL 梯度与缩小 $\Delta_{\text{rep}}$ 的方向一致且不与任务冲突。引理 2 为双向对齐机制提供理论依据（控制有效分布偏移 $\Gamma_{\bar{w}}$），引理 3 为 prototype-driven 分类机制提供理论根基（原型正确分类 ⟹ 样本正确分类）。

### 挑战 3：统计异构下全局公平性（纲要：现有方法多聚焦于局部公平性优化）

定理 4 和定理 7 构建了从"异构偏差量化"到"Server 端校准"的完整闭环：

```
统计异构 → 全局原型偏差 → 全局公平性退化
    │
    ├── 定理 7（统计异构下全局原型的聚合偏差与全局公平性界）
    │       │
    │       ├── (a) 聚合偏差界：‖μ_global - μ*‖ ≤ Σw_k·Γ + O(σ√(d/N))
    │       │       异构偏差 Γ 和估计误差共同决定全局原型质量
    │       ├── (b) 聚合表征偏差界：Δ_rep^global ≤ Δ_rep* + 2Γ̄ + O(...)
    │       │       局部公平≠全局公平的理论根源
    │       ├── (c) 全局 EO 界：EO_global ≤ (|w|/σ_z)·(Δ_rep* + 2Γ̄) + ...
    │       │       异构偏差 Γ̄ 直接抬高全局 EO 上界
    │       └── (d) Server 端校准的弥合效应：弱/强异构下的协同工作
    │
    └── 定理 4（全局原型作为 EO 校准代理的有效性）
            │
            ├── (a) 逼近界：|EO_proto - EO_global| ≤ ε(Δ_rep, Δ_Σ)
            ├── (b) 梯度方向一致性：cos⟨∇EO_proto, ∇EO_global⟩ ≥ 1 - ε
            └── (c) 单步下降传递：EO_global 随 EO_proto 下降而下降
```

**闭环逻辑**：定理 7 量化了"为什么局部公平≠全局公平"（异构偏差传递），定理 4 证明了"Server 端如何弥合"（全局原型 EO 校准）。两者结合说明：定理 3 的客户端 CL 控制偏差源头，定理 4 的 Server 端校准弥合残差，协同保障全局公平性。

### 定理-挑战对应总览

| 定理/引理/命题 | 核心关系 | 回应挑战 |
|------|---------|---------|
| 定理 1 | $\Delta_{\text{rep}}$ → EO 上界 | 挑战 2 |
| 定理 2 | 置信度差异 → EO 下界 | 挑战 2 |
| 定理 3 | CL → $\Delta_{\text{rep}}$ 收缩 | 挑战 2 |
| 引理 2 | 双向对齐 → 有效分布偏移 $\Gamma_{\bar{w}}$ 控制 | 挑战 2（精度维度）/3 |
| 引理 3 | prototype-driven 分类一致性 → 精度保障 | 挑战 2（精度维度） |
| 定理 4 | 全局原型 → EO 校准 | 挑战 3 |
| 定理 5 | 原型聚合 → 差分隐私保证 | 挑战 1 |
| 定理 6 | 传输敏感属性 → 隐私破坏 | 挑战 1 |
| 定理 7 | 统计异构 → 全局原型聚合偏差 → 全局公平性界 | 挑战 3 |

**补充说明**：定理 1 给出的是**上界**（减小 $\Delta_{\text{rep}}$ → 保证最坏情况下的 EO 不会太大），用于**设计优化目标**；定理 2 给出的是**下界**（置信度差异大 → EO 下界高），用于**诊断问题**。两者互补。

***

## 1. 符号

| 符号                                        | 含义                               |
| ----------------------------------------- | -------------------------------- |
| $X$                                       | 输入空间                             |
| $Y = {0, 1}$                              | 标签空间                             |
| $G = {0, 1}$                              | 敏感属性（群组）空间                       |
| $\mathcal{D}$                             | 全局数据分布                           |
| $\mathcal{D}_k$                          | 第 $k$ 个客户端的局部数据分布                |
| $f_\phi: X \to \mathbb{R}^d$             | 特征提取器（编码器）                       |
| $h_\psi: \mathbb{R}^d \to \mathbb{R}$    | 分类器（线性：$h_\psi(z) = w^T z + b$） |
| $z = f_\phi(x) \in \mathbb{R}^d$         | 样本 $x$ 的特征表示                     |
| $\hat{p} = \sigma_{\text{s}}(w^T z + b)$             | 预测概率，$\sigma_{\text{s}}$ 为 sigmoid 函数       |
| $\hat{y} = \mathbb{1}[\hat{p} \geq 0.5]$ | 预测标签                             |
| $\mu_{g,l} = \mathbb{E}[z \mid g, y=l]$ | 群组 $g$、标签 $l$ 的特征均值（原型，见定义2.5） |
| $\Delta_{\text{rep}}$ | 表征偏差（见定义2.3） |
| $\text{EO}$ | Equalized Odds 差距（见定义2.1） |
| $\text{DEO}$ | Disparate Equalized Odds（见定义2.2） |
| $\sigma_{\text{s}}(\cdot)$ | sigmoid 激活函数，$\sigma_{\text{s}}(t) = \frac{1}{1+e^{-t}}$ |
| $\sigma_z$ | 特征在分类方向上的标准差，$\sigma_z = \sqrt{\mathbb{E}\left[(w^T z - \mathbb{E}[w^T z])^2\right]}$ |
| $\delta_{\text{conf}}$ | 群组置信度差异下界，$\delta_{\text{conf}} = \min_l \left| \mathbb{E}[c \mid g=0, y=l] - \mathbb{E}[c \mid g=1, y=l] \right|$ |
| $\Sigma_{g,l}$ | 群组 $g$、标签 $l$ 的特征协方差矩阵 |
| $\Delta_\Sigma$ | 协方差差异，$\Delta_\Sigma = \max_{l} \|\Sigma_{g0,l} - \Sigma_{g1,l}\|_{\text{op}}$ |
| $K$                                       | 客户端数量                            |
| $w_k = n_k / N$                         | 第 $k$ 个客户端的聚合权重                  |
| $\eta$                                    | 学习率                              |
| $\mu_{g,l}^{\text{global}}$ | 聚合后的全局原型（Server 端） |
| $\mu_{g,l}^*$ | 真实全局原型（基于全局分布 $\mathcal{D}$） |
| $\hat{\mu}_{g,l}^{(k)}$ | 客户端 $k$ 基于有限样本估计的局部原型 |
| $\Gamma_{g,l}^{(k)}$ | 客户端 $k$ 的局部分布偏移上界（假设 4） |
| $\bar{\Gamma}$ | 最大加权分布偏移，$\bar{\Gamma} = \max_{g,l} \sum_k w_k \Gamma_{g,l}^{(k)}$ |
| $N_{g,l}$ | 全局群组-标签样本数，$N_{g,l} = \sum_k n_{g,l}^{(k)}$ |
| $N_{\min}$ | 最小全局群组-标签样本数，$N_{\min} = \min_{g,l} N_{g,l}$ |
| $\Delta_{\text{rep}}^{\text{global}}$ | 聚合后的全局表征偏差 |
| $\Delta_{\text{rep}}^*$ | 真实全局表征偏差 |
| $\mathcal{L}_{\text{l2l}}$ | 本地分类器在本地原型上的分类损失（引理 3） |
| $\mathcal{L}_{\text{l2g}}$ | 全局分类器在本地原型上的分类损失（引理 2） |
| $\mathcal{L}_{\text{g2l}}$ | 本地分类器在全局原型上的分类损失（引理 2） |
| $\Gamma_{\bar{w}}$ | 有效分布偏移（$w$ 方向的分布偏移投影） |
| $\rho$ | 原型到决策边界的间隔（假设 6） |

***

## 2. 前置

### 2.1 公平性定义

**定义2.1（Equalized Odds 差距）**：

$$\text{EO} = \sum_{l \in {0,1}} \left| \mathbb{E}[\hat{p} \mid g=0, y=l] - \mathbb{E}[\hat{p} \mid g=1, y=l] \right|$$

其中 $\hat{p} = \sigma_{\text{s}}(w^T z + b)$ 是正类预测概率，$g \in \{0, 1\}$ 是群组标识，$y \in \{0, 1\}$ 是真实标签。EO 衡量同一标签下不同群组的预测概率差异。

**定义2.2（Disparate Equalized Odds，DEO）**：

$$\text{DEO} = \left| \mathbb{E}[\hat{p} \mid g=0, y=1] - \mathbb{E}[\hat{p} \mid g=1, y=1] \right|$$

即仅衡量正类样本（$y=1$）的预测概率在不同群组间的差距，等价于真阳性率（TPR）的组间差异。

**定义2.2.1（Statistical Parity Difference，SPD / Demographic Parity，DP）**：

$$\text{SPD} = \mathbb{E}[\hat{p} \mid g=0] - \mathbb{E}[\hat{p} \mid g=1]$$

即不考虑真实标签时，不同群组的正类预测概率期望差异。

### 2.2 表征偏差定义

**定义2.3（表征偏差）**：

$$\Delta_{\text{rep}} = \sum_{l \in {0,1}} |\mu_{g_0,l} - \mu_{g_1,l}|$$

其中 $\mu_{g,l} = \mathbb{E}[z \mid g, y=l]$ 是群组-标签感知数据原型（见定义2.5）。$\Delta_{\text{rep}}$ 衡量不同群组在各标签下的表征中心差异。

### 2.3 类别感知数据原型

**定义2.4（类别感知数据原型）**：

$$\mu_l = \mathbb{E}_{(x,g,y) \sim \mathcal{D}}[z \mid y=l]$$

即特征空间中属于类别 $l$ 的所有样本的均值向量，是常规模型分析中使用的标准原型定义。

### 2.4 类别-群组感知数据原型

**定义2.5（类别-群组感知数据原型）**：

$$\mu_{g,l} = \mathbb{E}_{(x,g,y) \sim \mathcal{D}}[z \mid g, y=l]$$

即特征空间中同时属于群组 $g$ 和类别 $l$ 的样本的均值向量。

> **说明**：$\mu_{g,l}$ 是本文证明框架中**最核心的原型定义**，在定理1（$\Delta_{\text{rep}}$ 的定义）、定理3（对比损失的组间原型距离）、定理4（EO_proto 的计算）中均直接使用。它通过同时考虑群组和标签维度，能够捕捉不同群组在同一类别上的表征差异。

### 2.5 置信度

**定义2.6（预测置信度，简称置信度）**：

$$c = \max(\hat{p}, 1 - \hat{p}) = \sigma_{\text{s}}(|w^T z + b|)$$

下文均以"置信度"简称进行表述。

### 2.6 决策边界距离

**定义2.7（决策边界距离）**：

对于线性分类器 $h(z) = w^T z + b$，样本 $z$ 到决策边界的距离为：

$$d = \frac{|w^T z + b|}{|w|}$$

**观察1（置信度与决策边界距离的单调关系）**：

由定义2.6和定义2.7可得 $c = \sigma_{\text{s}}(|w| \cdot d)$，由于 $\sigma_{\text{s}}$ 单调递增且 $|w| > 0$，预测置信度 $c$ 关于决策边界距离 $d$ 单调递增。

***

## 3. 问题设定

### 3.1 联邦学习问题设定

我们考虑 $K$ 个客户端的联邦学习场景。每个客户端 $k$ 有本地数据分布 $\mathcal{D}_k$，全局分布为 $\mathcal{D} = \sum_{k=1}^K w_k \mathcal{D}_k$，其中 $w_k = n_k / N$，$n_k$ 是客户端 $k$ 的样本数，$N = \sum_{k=1}^K n_k$ 是全局样本总数。

**全局优化目标**（联邦平均的目标）：

$$\min_{\theta} \sum_{k=1}^K w_k \cdot \mathbb{E}_{(x,y,g) \sim \mathcal{D}_k} \left[ \ell(\sigma_{\text{s}}(w^T f_\phi(x) + b), y) \right]$$

其中 $\theta = (\phi, \psi)$ 是模型参数，$\phi$ 是编码器参数，$\psi = (w, b)$ 是分类头参数。

**客户端 $k$ 的局部训练损失**：

$$\mathcal{L}_k = \mathbb{E}_{(x,y,g) \sim \mathcal{D}_k} \left[ \ell(\sigma_{\text{s}}(w^T f_\phi(x) + b), y) \right] + \lambda \cdot \mathcal{L}_{\text{contrastive},k}$$

其中 $\mathcal{L}_{\text{contrastive},k}$ 是对比损失，始终随任务损失一同优化；$\lambda > 0$ 是控制对比损失强度的超参数。

### 3.2 模型分解

任何监督学习模型可以自然分解为**编码器 + 分类头**：

$$x \xrightarrow{f_\phi} z \in \mathbb{R}^d \xrightarrow{h_\psi} \hat{p} = \sigma_{\text{s}}(h_\psi(z))$$

其中：
- $f_\phi$ 是编码器，将输入映射到特征空间。$f_\phi$ 可以是任意复杂度的模型。
- $h_\psi$ 是分类头，对特征进行分类。

### 3.3 分析设定：线性分类头

在理论分析中，我们取分类头为线性形式：

$$h_\psi(z) = w^\top z + b$$

**合理性**：线性分类头是公平性理论分析的标准设定（McNamara et al., 2017; Zhao & Gordon, 2019）。这是分析工具，而非架构约束。

### 3.4 损失函数定义

**任务损失**（交叉熵损失）：

$$\mathcal{L}_{\text{task}}(x, y) = \ell(\sigma_{\text{s}}(w^T f_\phi(x) + b), y) = -y \log(\hat{p}) - (1-y) \log(1-\hat{p})$$

其中 $\hat{p} = \sigma_{\text{s}}(w^T f_\phi(x) + b)$ 是正类的预测概率。

**对比损失**（原型级对比损失）：

$$\mathcal{L}_{\text{contrastive}} = \frac{1}{2} \sum_{l \in \{0,1\}} \|\mu_{g0,l} - \mu_{g1,l}\|^2$$

其中 $\mu_{g,l} = \mathbb{E}[z \mid g, y=l]$ 是类别-群组感知数据原型（见定义2.5）。

***

## 4. 假设

以下假设按逻辑分为四组：基础分布假设（1-2）、优化假设（3）、联邦异构假设（4-5）、分类与公平性假设（6-7）、隐私假设（8）。

**假设 1（sub-Gaussian 特征分布）**：

对每个群组-标签组合 $(g, l)$，特征 $z \mid g, y=l$ 服从 sub-Gaussian 分布，即存在参数 $\sigma_{g,l}$ 使得：

$$\mathbb{E}[\exp(\lambda^T (z - \mu_{g,l})) \mid g, y=l] \leq \exp(\lambda^T \Sigma_{g,l} \lambda / 2), \quad \forall \lambda \in \mathbb{R}^d$$

且协方差矩阵有界：$|\Sigma_{g,l}|_{\text{op}} \leq \sigma^2$。

**合理性**：高斯分布是 sub-Gaussian 的特例。sub-Gaussian 允许更一般的尾部行为，大部分真实数据的特征满足此假设。

**假设 2（有界协方差差异）**：

不同群组的协方差矩阵差异有界：

$$|\Sigma_{g_0,l} - \Sigma_{g_1,l}|_{\text{op}} \leq \Delta_\Sigma, \quad \forall l$$

**合理性**：这是对共享协方差假设（$\Sigma_{g_0} = \Sigma_{g_1}$）的放宽。当 $\Delta_\Sigma = 0$ 时退化为共享协方差。

**假设 3（L-光滑性）**：

任务损失 $L_{\text{task}}$ 关于模型参数是 $L$-光滑的，对比损失 $L_{\text{contrastive}}$ 是 $M$-光滑的。

**合理性**：神经网络损失函数的光滑性是优化理论中的标准假设，适用于大多数使用光滑激活函数（如 ReLU、GELU）的网络。

**假设 4（有界局部分布偏移）**：

每个客户端的局部数据分布与全局分布的有界差异：

$$\|\mu_{g,l}^{(k)} - \mu_{g,l}^*\| \leq \Gamma_{g,l}^{(k)}, \quad \forall k, g, l$$

**合理性**：这是联邦学习异构性的标准量化方式。$\Gamma_{g,l}^{(k)} = 0$ 时退化为同构场景（IID）。实际中 $\Gamma_{g,l}^{(k)}$ 由数据划分的异构程度决定（如 Dirichlet 划分的 $\alpha$ 参数越小，$\Gamma$ 越大）。

**假设 5（有界样本估计误差）**：

客户端 $k$ 的局部原型基于 $n_{g,l}^{(k)}$ 个样本估计，由 sub-Gaussian 集中不等式（假设 1）：

$$\|\hat{\mu}_{g,l}^{(k)} - \mu_{g,l}^{(k)}\| \leq O\left(\sigma \sqrt{\frac{d}{n_{g,l}^{(k)}}}\right)$$

以高概率成立。

**合理性**：直接由假设 1 的 sub-Gaussian 性质导出，是标准的大数定律收敛速率。

**假设 6（分类边界的间隔性）**：

存在 $\rho > 0$，使得正确分类的原型满足 $|w^T \mu_l + b| \geq \rho$（原型与决策边界有正间隔）。

**合理性**：当分类器在原型上正确分类时，logit $w^T \mu_l + b$ 必然非零，存在正间隔 $\rho$。$\rho$ 的大小取决于训练程度——训练越充分，间隔越大。这是 SVM 间隔理论在原型分类中的自然延伸，也是 Prototype Networks（Snell et al., 2017）有效性的隐含前提。

**假设 7（原型类内紧凑性）**：

对每个标签 $l$，两个群组的原型在分类方向上的投影同号，即 $(w^T \mu_{g_0,l} + b)(w^T \mu_{g_1,l} + b) > 0$（原型位于决策边界同侧，符合"同类样本应被归为同类"的语义）。

**合理性**：如果模型能正确分类两个群组的同类样本，它们的原型必然在决策边界的正确侧（同侧）。该假设在训练收敛后自然满足——若不满足，说明模型本身存在严重分类错误，应优先解决精度问题。此假设的违背恰恰是公平性问题的预警信号。

**假设 8（LDP 噪声）**：

每个客户端在上传原型前添加 $\epsilon$-LDP 噪声 $\mathcal{N}(0, \sigma_{\text{noise}}^2 \cdot I_d)$，其中 $\sigma_{\text{noise}}^2 \geq \frac{2d \ln(2/\delta)}{\epsilon^2}$（满足 $(\epsilon, \delta)$-DP）。

**合理性**：由分布式差分隐私理论（Duchi et al., 2014）的标准结论，高斯机制在上述参数下满足 $\epsilon$-LDP。$\epsilon$ 越小隐私保护越强，但噪声越大、精度损失越多，存在隐私-效用权衡。

***

## 5. 定理与引理

### 定理 1（表征偏差与 EO 上界 —— 引用 McNamara et al. 2017）

在假设 1 和假设 2 下，EO 差距满足：

$$\text{EO} \leq \frac{|w|}{\sqrt{2\pi} \sigma_z} \cdot \Delta_{\text{rep}} + O\left(\frac{|w|^2 \Delta_\Sigma}{\sigma_z^3}\right)$$

其中 $\sigma_z = \sqrt{\mathbb{E}\left[(w^T z - \mathbb{E}[w^T z])^2\right]}$ 是特征在分类方向上的标准差。

**来源**：本定理的证明框架参考 McNamara et al. (2017) "Provably Fair Representations" 和 Zhao & Gordon (2019)。我们在其基础上将共享协方差假设放宽为有界协方差差异（假设 2），将高斯假设放宽为 sub-Gaussian（假设 1）。

**意义**：定理1给出 EO 的**上界**，由两项组成：主项 $\frac{|w|}{\sqrt{2\pi}\sigma_z} \cdot \Delta_{\text{rep}}$ 和高阶小项 $O\left(\frac{|w|^2 \Delta_\Sigma}{\sigma_z^3}\right)$。主项由三个变量共同决定，它们分属不同层次：

**三个变量的层次关系**：

- $\Delta_{\text{rep}}$ 是**表征层变量**：直接量化不同群组在特征空间中的中心距离，刻画的是表征空间本身的群组偏差。
- $|w|$ 是**决策层变量**：分类器权重向量的范数，反映决策边界的陡峭程度。
- $\sigma_z$ 是**特征分布变量**：特征在分类方向上的散布程度，由表征空间结构决定。

关键洞察：$|w|$ 和 $\sigma_z$ 的调整属于**决策层面的事后修正**，而 $\Delta_{\text{rep}}$ 直接刻画**表征空间本身的群组偏差**。现有工作 DFR (Kirichenko et al., 2023) 证明在集中式场景下仅重训最后一层分类器（调整 $|w|$）即可改善公平性，但其前提是"特征提取器已学到相对公平的表征"。Fair-FLIP (Zhong et al., 2025) 和 GroupMixNorm (Zhang et al., 2023) 通过调整 $\sigma_z$ 相关统计量改善公平性，同样属于事后修正。Cui et al. (2024) 的实证研究进一步确认：**不公平性的根源在于表征本身而非分类器偏置**。

**联邦学习场景的特殊性**：上述工作均基于集中式场景，其前提是特征提取器能从完整数据集中学到公平的表征。然而在联邦学习中，数据异构性导致各客户端的局部表征本身就携带系统性群组偏差（$\Delta_{\text{rep}}$ 增大），特征空间可能从一开始就不公平。此时集中式方法的前提不成立——仅调整 $|w|$ 或 $\sigma_z$ 无法触及偏差根源。

**PDFFed 的切入点**：由定理1的结构可知，在联邦学习的训练过程中**直接缩小 $\Delta_{\text{rep}}$**——通过表征层面的公平性约束（条件性对比损失）——是降低 EO 上界最根本的手段。这直接呼应**挑战2**：现有方法多聚焦于损失约束或聚合调控（决策层面），难以从根本上缓解表征偏差引起的公平性退化与精度损失。

### 定理 2（群组置信度差异与 EO 下界）

**条件**：存在常数 $\delta_{\text{conf}} > 0$，使得对于所有标签 $l \in \{0,1\}$：

$$\left| \mathbb{E}[c \mid g=0, y=l] - \mathbb{E}[c \mid g=1, y=l] \right| \geq \delta_{\text{conf}}$$

即群组间存在不可忽略的置信度差异。在**假设 1**下，EO 差距满足：

$$\text{EO} \geq 2\delta_{\text{conf}} - \frac{2\sigma_z}{|w|\sqrt{2\pi}} - O(\Delta_\Sigma)$$

其中 $\sigma_z = \sqrt{\mathbb{E}\left[(w^T z - \mathbb{E}[w^T z])^2\right]}$ 是特征在分类方向上的标准差（同定理1）。

**意义**：这是**下界**。它告诉我们：EO 的最小值与群组置信度差异 $\delta_{\text{conf}}$ 正相关。要让 EO 变得更小，必须先减小群组间的置信度差异——这是一个必要条件（但非充分条件）。反之，如果置信度差异很大（$\delta_{\text{conf}}$ 大），EO 就不可能太小。因此，置信度差异可以作为公平性问题的诊断信号。

本定理直接呼应**挑战2**的公平性维度：现有方法多聚焦于损失约束或聚合调控，缺乏对群组置信度差异的显式诊断与干预。PDFFed 据此设计 `delta_conf_loss`（[PDFFed.py L474-485](file:///d:/最新PDFFed/fairness_fl_code/algorithm/PDFFed.py#L474-L485)），显式约束 $\delta_{\text{conf}}$。

**(1) 与定理1的上下界夹击互补**：定理1给出 EO 上界（由 $\Delta_{\text{rep}}$ 决定），定理2给出 EO 下界（由 $\delta_{\text{conf}}$ 决定）。PDFFed 的 `feature_contrastive_loss`（对应定理1+3）从上界侧压低 EO，`delta_conf_loss`（对应定理2）从下界侧压低 EO，两者夹击使 EO 的"允许区间"收窄。单独减小 $\delta_{\text{conf}}$ 只压低下界（为 EO 变小腾出空间），不保证 EO 实际降低；但与 CL 联动后，上界（CL 控制）和下界（$\delta_{\text{conf}}$ 控制）同时下降，EO 被双向约束。

**(2) 不破坏主任务**：`delta_conf_loss` 最小化的是群组间置信度期望差 $|\mathbb{E}[c|g=0,y=l] - \mathbb{E}[c|g=1,y=l]|$，即让模型对两个群组给出"同等确信"的预测。这只均衡群组间的置信度分布，不改变单个样本的分类决策边界（置信度 $c = \max(\hat{p}, 1-\hat{p})$ 只依赖预测的确信程度，不依赖预测方向），因此不与任务损失冲突。

### 引理 1（CL 梯度的有利性质）

设 $\rho_l = \cos\langle \nabla_\phi \mathcal{L}_{\text{contrastive}}, \nabla_\phi \mathcal{L}_{\text{task}} \rangle$ 为对比损失梯度与任务损失梯度在编码器参数上的余弦相似度。对于 PDFFed 的原型级对比损失 $\mathcal{L}_{\text{contrastive}} = \frac{1}{2} \sum_l \|\mu_{g0,l} - \mu_{g1,l}\|^2$，在假设 1-3 下：

- **(a) CL 直接服务于公平性目标**：$\nabla_\phi \mathcal{L}_{\text{contrastive}}$ 的方向与缩小 $\Delta_{\text{rep}}$ 的方向一致（内积非负），即 CL 梯度直接驱动表征偏差下降。
- **(b) CL 不与任务损失冲突**：$\rho_l \geq 0$，即 CL 梯度与任务损失梯度的夹角不超过 $\pi/2$，CL 不会阻碍任务损失下降。

**意义**：(a) 说明 CL 不是绕弯路的辅助目标，而是直接缩小 $\Delta_{\text{rep}}$（定理1 上界的主要变量），因此 CL 对公平性的贡献是有理论保证的。(b) 说明引入 CL 不会以牺牲精度为代价——CL 梯度与任务梯度方向相容，联合优化时两者可以同时下降。这两点共同保证了 PDFFed 在客户端引入 CL 的安全性：**既能改善公平性，又不损害精度**。

### 定理 3（CL 的收敛性与 Δ_rep 收缩性）

局部训练损失 $\mathcal{L}_k = \mathcal{L}_{\text{task},k} + \lambda \cdot \mathcal{L}_{\text{contrastive},k}$在假设 3 下满足：

**(a) 收敛性：**

$$L^{(t+1)} - L^{(t)} \leq -\eta |\nabla L^{(t)}|^2 + \frac{\eta^2 (L + \lambda M)}{2} |\nabla L^{(t)}|^2$$

当 $\eta < 2/(L + \lambda M)$ 时，损失单调递减。其中 $L$ 是任务损失的光滑常数，$M$ 是 CL 损失的光滑常数。

**(b) Δ_rep 收缩性：**

经过 T 步后，表征偏差满足：

$$\Delta_{\text{rep}}^{(t+T)} \leq \Delta_{\text{rep}}^{(t)} - \eta \cdot \lambda \cdot T \cdot \gamma$$

其中 $\gamma = \mathbb{E}[\|\mu_{g0,l} - \mu_{g1,l}\|] > 0$，当 $\Delta_{\text{rep}} > 0$ 时自然满足。

**意义**：(a) 始终运行 CL 不破坏任务损失的收敛性——唯一的代价是将学习率上界从 $\eta < 2/L$ 收紧到 $\eta < 2/(L + \lambda M)$，这是温和的；(b) CL 确实能持续缩小表征偏差，收缩速度与学习率 $\eta$、CL 权重 $\lambda$ 和训练步数 $T$ 成正比。引理 1 进一步证明了 CL 梯度与缩小 $\Delta_{\text{rep}}$ 的方向天然一致，因此 CL 不会与主任务产生方向性冲突。

### 引理 2（双向原型-分类器对齐控制有效分布偏移）

**动机**：定理7(c) 表明全局 EO 上界由 $\Delta_{\text{rep}}^* + 2\bar{\Gamma}$ 共同决定，其中 $\bar{\Gamma}$（假设4 定义的局部分布偏移 $\Gamma_{g,l}^{(k)}$ 的加权平均）由数据异构性决定。$\bar{\Gamma}$ 大 → 全局 EO 上界高 → **必须控制 $\bar{\Gamma}$**。但 $\bar{\Gamma}$ 是全空间偏移 $\|\mu_{g,l}^{(k)} - \mu_{g,l}^*\|$，直接控制全空间偏移代价高且不必要——因为 EO 基于预测 $\hat{y} = \sigma(w^T z + b)$，**只依赖 $w^T z$ 的分布**，因此只有 $\bar{\Gamma}$ 在 $w$ 方向的投影 $\Gamma_{\bar{w}} = |w^T(\mu_l^{(k)} - \mu_l^*)|/\|w\|$ 才真正影响 EO。定理1 的上界 $\frac{|w|}{\sqrt{2\pi}\sigma_z}\cdot\Delta_{\text{rep}}$ 正是通过 Cauchy-Schwarz 将 $w$ 方向投影放缩为全空间 $\Delta_{\text{rep}}$ 得到的保守界，反过来说明只需控制 $w$ 方向投影即可。

因此，PDFFed 需要一种机制**专门控制 $\bar{\Gamma}$ 的 $w$ 方向有效投影 $\Gamma_{\bar{w}}$**，而非全空间 $\bar{\Gamma}$。下面的引理2 证明双向原型-分类器对齐恰好实现这一目标。

**设定**：PDFFed 在客户端训练中引入两个对齐损失：
- $\mathcal{L}_{\text{l2g}} = \sum_l \ell(\sigma_{\text{s}}(w_{\text{global}}^T \mu_l^{(k)} + b_{\text{global}}), l)$：全局分类器在本地原型上的分类损失
- $\mathcal{L}_{\text{g2l}} = \sum_l \ell(\sigma_{\text{s}}(w_{\text{local}}^T \mu_l^* + b_{\text{local}}), l)$：本地分类器在全局原型上的分类损失

其中 $\mu_l^{(k)}$ 是客户端 $k$ 的本地原型，$\mu_l^*$ 是全局原型（上一轮聚合结果），$w_{\text{global}}, b_{\text{global}}$ 是上一轮的全局分类头。

在**假设 1、假设 6**下，若 $\mathcal{L}_{\text{l2g}} \leq \varepsilon_{\text{align}}$ 且 $\mathcal{L}_{\text{g2l}} \leq \varepsilon_{\text{align}}$，则：

**(a) 有效分布偏移控制：**

$$|w_{\text{global}}^T (\mu_l^{(k)} - \mu_l^*)| \leq O\left(\frac{\varepsilon_{\text{align}}}{\rho}\right)$$

即双向对齐损失控制了本地原型与全局原型在**分类器方向**上的偏移（有效分布偏移 $\Gamma_{\bar{w}}$）。

**(b) 时效性（一步延迟）：** 由于 $\mu_l^*$ 和 $(w_{\text{global}}, b_{\text{global}})$ 来自第 $t-1$ 轮通信，在假设 3（$L$-光滑性）下，一步延迟引入的误差为 $O(\eta L)$，仅影响收敛速率的常数项，不影响收敛性。这是 FedAvg 收敛分析的标准结论（Li et al., 2020）。

**意义**：PDFFed 的核心思想是用原型贯穿整个联邦学习流程。在追求群组公平性（通过定理3 的 CL 缩小 $\Delta_{\text{rep}}$）的同时，**不能以牺牲精度为代价**。引理2 证明的双向对齐机制正是服务于这一目标：

**(1) 保障精度的痛点**：联邦学习中，各客户端的局部数据异构导致本地模型 $\varphi_k$ 偏离全局模型 $\varphi^*$，这种偏离是精度退化的主要原因。引理2 证明，通过 $\mathcal{L}_{\text{l2g}}$（全局分类器约束本地原型）和 $\mathcal{L}_{\text{g2l}}$（本地分类器约束全局原型），本地原型与全局原型在分类器方向 $w$ 上的偏移被控制在 $O(\varepsilon_{\text{align}} / \rho)$。这意味着**即使各客户端的局部数据不同，只要原型在分类器方向上对齐，分类行为就能保持一致**，从而保障精度。

**(2) 原型作为精度锚点的理论依据**：EO 只依赖 $w^T z$（通过 sigmoid），因此 $w$ 方向的偏移才是"有效的"——正交方向的偏移不影响分类结果。双向对齐恰好只约束这个有效分量，而非全空间偏移，因此用原型（而非完整模型参数）作为对齐目标在理论上是充分的。这使得原型不仅是公平性的知识载体（定理1-3），也是精度的锚点——**原型真正贯穿了公平性和精度两个维度**。

**(3) 与定理7的协同**：定理7 证明全局原型的聚合偏差由全空间偏移 $\bar{\Gamma}$（假设4 中的 $\Gamma_{g,l}^{(k)}$ 的加权平均）决定。引理2 进一步表明，PDFFed 的双向对齐只需控制 $\bar{\Gamma}$ 在 $w$ 方向的投影（有效分量），即可保证 EO 不退化——这是比控制全空间 $\bar{\Gamma}$ 更宽松、更易满足的条件。

### 引理 3（Prototype-driven 分类一致性）

**设定**：PDFFed 在客户端训练中引入原型级分类损失：

$$\mathcal{L}_{\text{l2l}} = \sum_l \ell(\sigma_{\text{s}}(w_{\text{local}}^T \mu_l^{(k)} + b_{\text{local}}), l)$$

即本地分类器在本地原型上应正确分类。

**条件**：$\mathcal{L}_{\text{l2l}} \leq \varepsilon_{\text{clf}}$，即分类器在原型上的分类损失有界。

在**假设 1 和假设 6**下，对任意 $\delta \in (0, 1)$，以至少 $1 - \delta$ 的概率（关于训练样本的随机性），样本级分类错误率满足：

$$\Pr_{(x,y) \sim \mathcal{D}_k}[\hat{y}(x) \neq y] \leq O\left(\varepsilon_{\text{clf}} + \exp\left(-\frac{\rho^2}{2\sigma^2 |w|^2}\right) + \sqrt{\frac{d \log(1/\delta)}{n}}\right)$$

其中 $\rho$ 是原型到决策边界的间隔（假设 6），$\sigma$ 是 sub-Gaussian 参数，$n$ 是客户端 $k$ 的样本数，$d$ 是特征维度。三项分别对应：原型级分类误差、sub-Gaussian 尾部概率、有限样本估计误差。

**证明思路**：原型 $\mu_l$ 是类 $l$ 的特征中心。由假设 1（sub-Gaussian），样本 $z$ 以高概率落在 $\mu_l$ 的 $\sigma$-邻域内。若分类器在 $\mu_l$ 上正确分类（间隔 $\rho$，假设 6），则邻域内的样本也被正确分类（间隔 $\rho - O(\sigma)$）。当 $\rho \gg \sigma |w|$ 时，样本错误率随 $\rho^2 / (\sigma^2 |w|^2)$ 指数衰减。第三项由假设 5（有界样本估计误差）导出，反映原型估计的不确定性对间隔 $\rho$ 的影响。这与 **Prototype Networks**（Snell et al., 2017）和 **DFR**（Kirichenko et al., 2023）的观察一致：仅重训最后一层（分类器在类中心上正确分类）即可修复偏差。

**意义**：本引理为 PDFFed 的"prototype-driven"设计提供精度保障的理论根基，直接回答了**挑战2**中的一个隐含问题：在引入公平性约束（定理3 的 CL）的同时，如何保证模型精度不退化？

**(1) prototype-driven 的理论充分性**：$\mathcal{L}_{\text{l2l}}$ 不是经验性的辅助损失，而是保证分类器基于原型做决策的充分条件——原型正确分类 ⟹ 样本正确分类（以高概率 $1-\delta$）。误差率的指数衰减项 $\exp(-\rho^2/(2\sigma^2|w|^2))$ 表明，只要分类器在原型上有足够间隔 $\rho$，且特征足够紧凑（$\sigma$ 小），样本级精度就有指数级保障。

**(2) 公平-精度协同的理论基础**：定理3 的 CL 缩小 $\Delta_{\text{rep}}$（改善公平性），而 $\Delta_{\text{rep}}$ 缩小意味着不同群组的原型更接近，分类器更容易在所有群组的原型上获得大间隔 $\rho$（改善精度）——因此 CL 同时改善公平性和精度的间隔条件。引理3 保证了这一协同效应能传递到样本级精度。

### 定理 4（全局原型作为 EO 校准代理的有效性）

**设定**：联邦训练中，各客户端上传本地分类头 $\psi_k = (w_k, b_k)$ 和本地原型 $\mu_{g,l}^{(k)}$。Server 端聚合得到全局分类头 $\psi = \sum_k w_k^{\text{fed}} \cdot \psi_k$（其中 $w_k^{\text{fed}}$ 是联邦聚合权重）和全局原型 $\mu_{g,l} = \sum_k w_k^{\text{fed}} \cdot \mu_{g,l}^{(k)}$。Server 端后训练**仅更新聚合分类头** $\psi = (w, b)$，表征模块固定。

由于 Server 无法访问全局数据，无法直接计算真实全局 EO，因此使用全局原型构建代理目标：

$$\text{EO}_{\text{proto}} = \sum_{l \in \{0,1\}} \left| \sigma_{\text{s}}(w^T \mu_{g0,l} + b) - \sigma_{\text{s}}(w^T \mu_{g1,l} + b) \right|$$

即用全局原型的 sigmoid 输出差异近似真实 EO。真实全局 EO 定义为：

$$\text{EO}_{\text{global}} = \sum_{l \in \{0,1\}} \left| \Pr[\hat{y}=1 | g=0, y=l] - \Pr[\hat{y}=1 | g=1, y=l] \right|$$

在**假设 1、假设 2 和假设 3**下，以下三个性质成立：

**(a) 逼近界（EO_proto 是 EO_global 的良好近似）：**

$$|\text{EO}_{\text{proto}} - \text{EO}_{\text{global}}| \leq \frac{|w|}{\sqrt{2\pi} \sigma_z} \cdot \Delta_{\text{rep}}^{\text{proto}} + O\left(\frac{|w|^2 \Delta_\Sigma}{\sigma_z^3}\right) + O\left(\frac{1}{\sigma_z \sqrt{n}}\right)$$

即代理目标与真实目标的差异由三部分控制：原型层面的表征偏差 $\Delta_{\text{rep}}^{\text{proto}}$、协方差差异 $\Delta_\Sigma$、有限样本估计误差。当表征偏差小（定理3 的 CL 已在客户端侧缩小）且样本充足时，$\text{EO}_{\text{proto}} \approx \text{EO}_{\text{global}}$。

**(b) 梯度方向一致性（优化代理目标的方向正确）：**

$$\cos\langle \nabla_\psi \text{EO}_{\text{proto}}, \nabla_\psi \text{EO}_{\text{global}} \rangle \geq 1 - O\left(\frac{|w|^2 \Delta_\Sigma}{\sigma_z^2}\right)$$

即对聚合分类头 $\psi$ 而言，降低 $\text{EO}_{\text{proto}}$ 的梯度方向与降低 $\text{EO}_{\text{global}}$ 的梯度方向近似一致（余弦相似度接近 1）。当协方差差异 $\Delta_\Sigma$ 小时，两者方向几乎重合。

**(c) 单步下降传递（代理目标的下降确实改善真实公平性）：**

$$\text{EO}_{\text{global}}^{(t+1)} \leq \text{EO}_{\text{global}}^{(t)} - \alpha \lambda_{\text{eo}} \cdot |\nabla_\psi \text{EO}_{\text{proto}}^{(t)}|^2 \cdot \delta$$

其中 $\delta > 0$（当学习率 $\alpha$ 足够小且 $\Delta_\Sigma$ 足够小时）。即 Server 端优化 $\text{EO}_{\text{proto}}$ 的每一步，真实全局 $\text{EO}_{\text{global}}$ 也在下降。

**意义**：本定理回答**挑战3**的核心问题：Server 端无法访问全局数据，如何保障全局公平性？

**(1) 代理的有效性**：(a) 证明用全局原型计算的 $\text{EO}_{\text{proto}}$ 是真实 $\text{EO}_{\text{global}}$ 的良好近似——只要客户端侧的 CL 已缩小 $\Delta_{\text{rep}}$（定理3），且协方差差异有界（假设2）。这使得 Server 端**无需访问数据即可评估和优化全局公平性**。

**(2) 优化方向的正确性**：(b) 证明对聚合分类头优化 $\text{EO}_{\text{proto}}$ 的方向与优化真实 $\text{EO}_{\text{global}}$ 的方向一致。这意味着 Server 端后训练不是"盲人摸象"，而是朝着正确的公平性方向优化。

**(3) 下降的传递性**：(c) 证明代理目标的下降能传递到真实目标。三者结合，证明了 Server 端通过聚合分类头和全局原型进行后训练，能有效提升全局公平性。

**(4) 与定理7的衔接**：定理7 量化了全局原型的聚合偏差（由异构偏差 $\bar{\Gamma}$ 决定），本定理的逼近界 (a) 中的 $\Delta_{\text{rep}}^{\text{proto}}$ 正是定理7 中 $\Delta_{\text{rep}}^{\text{global}}$ 的体现。当客户端侧 CL 控制 $\bar{\Gamma}$ 源头（定理3+引理2）后，Server 端校准只需弥合残差，两者协同保障全局公平性。

### 推论 4.1（双目标联合下降的可行性）

**设定**：Server 端后训练采用双目标联合损失 $\mathcal{L}_{\text{post}} = \mathcal{L}_{\text{cls}} + \lambda_{\text{eo}} \cdot \text{EO}_{\text{proto}}$，仅更新分类头 $\psi = (w, b)$。其中：
- $\mathcal{L}_{\text{cls}} = \sum_{l,g} \ell(\sigma_{\text{s}}(w^T \mu_{g,l} + b), l)$ 是原型级分类损失（精度目标）；
- $\text{EO}_{\text{proto}}$ 同定理 4（公平性目标）；
- $\lambda_{\text{eo}} > 0$ 是公平性权重。

**假设 7（原型类内紧凑性）**已在第 4 节给出。

在**假设 1-7**下，联合损失 $\mathcal{L}_{\text{post}}$ 关于 $\psi$ 的梯度下降满足：

**(a) 精度-公平梯度方向相容性：**

$$\cos\langle \nabla_\psi \mathcal{L}_{\text{cls}}, \nabla_\psi \text{EO}_{\text{proto}} \rangle \geq -\sin\theta_{\max}$$

其中 $\theta_{\max}$ 是由假设 7 中原型同侧性决定的最大夹角上界。在原型同侧假设下 $\theta_{\max} < \pi/2$，即两个梯度**不会严格反向**。

**(b) 联合单步下降：** 当学习率 $\alpha$ 足够小且 $\lambda_{\text{eo}}$ 在合理范围内时，

$$\mathcal{L}_{\text{post}}^{(t+1)} \leq \mathcal{L}_{\text{post}}^{(t)} - \alpha \left( |\nabla_\psi \mathcal{L}_{\text{cls}}^{(t)}|^2 + \lambda_{\text{eo}} |\nabla_\psi \text{EO}_{\text{proto}}^{(t)}|^2 + 2\lambda_{\text{eo}} \cos\langle\nabla_\psi \mathcal{L}_{\text{cls}}, \nabla_\psi \text{EO}_{\text{proto}}\rangle \cdot |\nabla_\psi \mathcal{L}_{\text{cls}}||\nabla_\psi \text{EO}_{\text{proto}}| \right) + O(\alpha^2)$$

由 (a) 的相容性，交叉项非负（当 $\cos \geq 0$）或被主项吸收（当 $\cos < 0$ 但 $\theta_{\max} < \pi/2$ 时，由 Cauchy-Schwarz 交叉项绝对值不超过主项的 $\sin\theta_{\max}$ 倍）。

**(c) 双目标传递性：** 由定理 4 (c) 和 (b)，

$$\text{EO}_{\text{global}}^{(t+1)} \leq \text{EO}_{\text{global}}^{(t)} - \alpha \lambda_{\text{eo}} \cdot |\nabla_\psi \text{EO}_{\text{proto}}^{(t)}|^2 \cdot \delta_1 + O(\alpha^2)$$

同时分类精度由 $\mathcal{L}_{\text{cls}}$ 的下降保证。

**意义**：本推论为 Server 端"精度优先 + 公平性约束"的双目标后训练提供理论依据。关键洞察：原型是类内均值，$\mathcal{L}_{\text{cls}}$ 在原型上的梯度方向与"同类原型向决策边界同侧聚拢"一致；而 $\text{EO}_{\text{proto}}$ 要求同标签不同群组原型输出概率一致，即"同侧且等距"。两者在假设 5（同侧性）下方向相容——精度要求原型在同侧，公平性要求同侧且等距，后者是前者的细化而非冲突。这与 DFR (Kirichenko et al., 2023) 的观察一致：仅重训最后一层即可同时保持精度并改善公平性。当假设 5 不成立时（原型跨越决策边界，说明表征本身有问题），应通过定理 3 的 CL 在客户端侧修复表征，而非在 Server 端强行校准。

### 定理 5（原型聚合的差分隐私保证）

**设定**：客户端 $k$ 在每轮上传本地原型 $\mu_{g,l}^{(k)}$，Server 端进行加权聚合：

$$\mu_{g,l}^{\text{global}} = \sum_{k=1}^K w_k \cdot \mu_{g,l}^{(k)}$$

在假设 8 下，原型聚合过程满足**分布式差分隐私**（Distributed Differential Privacy）：

$$\text{Pr}[\mathcal{A}(\mu_{g,l}^{\text{global}}) = t] \leq e^\epsilon \cdot \text{Pr}[\mathcal{A}(\mu_{g,l}^{\text{global},-i}) = t] + \delta$$

其中 $\mu_{g,l}^{\text{global},-i}$ 是移除第 $i$ 个样本后的聚合原型，$\mathcal{A}$ 是任意攻击者算法。

**意义**：本定理回答了**挑战1**（隐私泄露风险）——证明 PDFFed 中传输的原型信息在添加 LDP 噪声后满足差分隐私保证。即使攻击者能够访问聚合后的全局原型，也无法推断出单个客户端或样本的敏感信息。

### 定理 6（传输敏感属性的隐私风险）

**设定**：假设有方法在通信中传输敏感属性 $g$（如群组标签）或包含敏感属性信息的中间结果。

**结论**：任何传输原始敏感属性 $g$ 的方法都**不满足差分隐私**，因为存在攻击者可以通过以下方式进行成员推断攻击：

$$\text{Pr}[\mathcal{A}(g_i) = 1 \mid i \in S] - \text{Pr}[\mathcal{A}(g_i) = 1 \mid i \notin S] = 1$$

即攻击者可以完美判断样本是否属于某个群组，从而破坏隐私。

**意义**：本定理从反面证明了**挑战1**的必要性——如果传输敏感属性，隐私将直接被破坏。PDFFed 选择传输原型而非敏感属性，正是为了避免这一风险。

### 定理 7（统计异构下全局原型的聚合偏差与全局公平性界）

**设定**：考虑 $K$ 个客户端，每个客户端 $k$ 的局部数据分布 $\mathcal{D}_k$ 异构。客户端 $k$ 上传本地原型 $\mu_{g,l}^{(k)} = \mathbb{E}_{(x,y,g) \sim \mathcal{D}_k}[z \mid g, y=l]$，Server 端按数据量加权聚合：

$$\mu_{g,l}^{\text{global}} = \sum_{k=1}^K w_k \cdot \mu_{g,l}^{(k)}, \quad w_k = n_k / N$$

真实全局原型为 $\mu_{g,l}^* = \mathbb{E}_{(x,y,g) \sim \mathcal{D}}[z \mid g, y=l]$，其中 $\mathcal{D} = \sum_k w_k \mathcal{D}_k$。

在**假设 1、假设 4 和假设 5**下，以下性质成立：

**(a) 全局原型聚合偏差界：**

$$\|\mu_{g,l}^{\text{global}} - \mu_{g,l}^*\| \leq \sum_{k=1}^K w_k \Gamma_{g,l}^{(k)} + O\left(\sigma \sqrt{\frac{d}{N_{g,l}}}\right)$$

其中 $N_{g,l} = \sum_k n_{g,l}^{(k)}$ 是全局群组-标签样本数。第一项是**分布异构偏差**（加权平均的局部偏移），第二项是**统计估计误差**（随总样本量 $N_{g,l}$ 衰减）。

**(b) 聚合表征偏差界：** 令 $\bar{\Gamma} = \max_{g,l} \sum_k w_k \Gamma_{g,l}^{(k)}$ 为最大加权分布偏移，则聚合后的全局表征偏差满足：

$$\Delta_{\text{rep}}^{\text{global}} \leq \Delta_{\text{rep}}^* + 2\bar{\Gamma} + O\left(\sigma \sqrt{\frac{d}{N_{\min}}}\right)$$

其中 $\Delta_{\text{rep}}^*$ 是真实全局表征偏差，$N_{\min} = \min_{g,l} N_{g,l}$。

**(c) 全局 EO 界：** 结合定理 1 和 (b)，全局 EO 满足：

$$\text{EO}_{\text{global}} \leq \frac{|w|}{\sqrt{2\pi} \sigma_z} \cdot \left(\Delta_{\text{rep}}^* + 2\bar{\Gamma}\right) + O\left(\frac{|w| \sigma}{\sigma_z} \sqrt{\frac{d}{N_{\min}}} + \frac{|w|^2 \Delta_\Sigma}{\sigma_z^3}\right)$$

**(d) Server 端 EO 校准的弥合效应：** 定理 4 的 Server 端后训练使用 $\mu_{g,l}^{\text{global}}$ 作为校准代理。由 (a)，当 $\bar{\Gamma}$ 较大（强异构）时，$\mu_{g,l}^{\text{global}}$ 偏离 $\mu_{g,l}^*$，但定理 4 的逼近界仍成立——因为定理 4 中的 $\Delta_{\text{rep}}^{\text{proto}}$ 正是 (b) 中的 $\Delta_{\text{rep}}^{\text{global}}$。因此 Server 端校准的有效性由 $\Delta_{\text{rep}}^{\text{global}}$ 决定，而非 $\Delta_{\text{rep}}^*$。关键推论：

- **弱异构**（$\bar{\Gamma}$ 小）：$\Delta_{\text{rep}}^{\text{global}} \approx \Delta_{\text{rep}}^*$，Server 端校准直接作用于真实全局公平性。
- **强异构**（$\bar{\Gamma}$ 大）：Server 端校准先弥合 $\Delta_{\text{rep}}^{\text{global}}$，使其逼近 $\Delta_{\text{rep}}^*$，再进一步收缩 $\Delta_{\text{rep}}^*$。此时定理 3 的客户端侧 CL（缩小局部 $\Delta_{\text{rep}}$）与定理 4 的 Server 端校准**协同工作**——前者控制 $\bar{\Gamma}$ 的源头（局部表征偏差），后者在聚合层面弥合残差。

**意义**：本定理直接回答**挑战3**（统计异构下全局公平性）的核心问题：
1. **为什么局部公平≠全局公平**：由 (a)(b)，局部原型的分布异构偏差 $\bar{\Gamma}$ 会传递到全局原型，即使各客户端局部 $\Delta_{\text{rep}}^{(k)}$ 都小，$\Delta_{\text{rep}}^{\text{global}}$ 仍可能因 $\bar{\Gamma}$ 而大。
2. **全局原型为何能弥合**：由 (d)，Server 端 EO 校准以 $\Delta_{\text{rep}}^{\text{global}}$ 为目标，直接作用于"聚合后的真实公平性目标"，而非某个客户端的局部目标。
3. **客户端-Server 协同**：定理 3（客户端 CL 缩小局部 $\Delta_{\text{rep}}$，从而控制 $\bar{\Gamma}$ 源头）+ 定理 4（Server 端 EO 校准弥合 $\Delta_{\text{rep}}^{\text{global}}$ 残差）+ 定理 7（量化两者协同的理论保证）构成挑战3 的完整闭环。
4. **$\Delta_{\text{rep}}^*$ 与 $\bar{\Gamma}$ 的分工控制**：定理7(c) 表明全局 EO 上界由 $\Delta_{\text{rep}}^* + 2\bar{\Gamma}$ 共同决定，这两项由不同机制控制：
   - $\Delta_{\text{rep}}^*$（真实全局表征偏差）由**定理3 的客户端 CL** 控制：CL 在各客户端缩小局部 $\Delta_{\text{rep}}^{(k)}$，聚合后 $\Delta_{\text{rep}}^* = \sum_k w_k \Delta_{\text{rep}}^{(k)}$ 随之减小。这是从**源头**控制全局公平性。
   - $\bar{\Gamma}$（最大加权分布偏移）由**数据异构性决定**（假设4），PDFFed 不直接控制全空间 $\bar{\Gamma}$，但通过引理2 的双向对齐控制 $\bar{\Gamma}$ 在 $w$ 方向的有效投影 $\Gamma_{\bar{w}}$（因为 EO 只依赖 $w^T z$，控制 $w$ 方向投影即可），并通过定理4 的 Server 端校准弥合 $\bar{\Gamma}$ 引起的聚合残差。
   
   两者协同：CL 控制全局表征偏差的"基线" $\Delta_{\text{rep}}^*$，双向对齐+Server端校准控制异构引起的"增量" $2\bar{\Gamma}$。

**与 EMA 全局原型更新的关系**：PDFFed 工程实现中，全局原型采用 EMA 更新 $\mu^{\text{global}} \leftarrow \beta \mu^{\text{old}} + (1-\beta) \mu^{\text{new}}$。当 $\beta > 0$ 时，(a) 中的分布异构偏差项 $\bar{\Gamma}$ 被进一步平滑（时间维度上的低通滤波），相当于对 $\bar{\Gamma}$ 引入额外衰减因子 $(1-\beta)$，从而进一步降低 $\Delta_{\text{rep}}^{\text{global}}$ 的方差。这为 EMA 更新策略提供了理论解释。

### 5.1 定理 1 的证明

**来源**：本证明框架参考 McNamara et al. (2017) 和 Zhao & Gordon (2019)。

**Step 1**：EO 的表达式

对于线性分类器，预测概率为 $\hat{p} = \sigma_{\text{s}}(w^T z + b)$。

EO 差距（对标签 $l$）：

$$\text{EO}_l = \left|\mathbb{E}[\hat{p} \mid g=0, y=l] - \mathbb{E}[\hat{p} \mid g=1, y=l]\right|$$

**Step 2**：利用 sub-Gaussian 假设

由假设 1，$z \mid g, y=l$ 是 sub-Gaussian 的。对于线性投影 $w^T z + b$，由 sub-Gaussian 的性质（线性变换保持 sub-Gaussian 性质）：

$$w^T z + b \mid g, y=l \text{ 是 sub-Gaussian 的}$$

其均值为 $w^T \mu_{g,l} + b$，方差参数为 $\sigma'_{g,l} = \sqrt{w^T \Sigma_{g,l} w}$。

**Step 3**：利用 Berry-Esseen 型近似

对于 sub-Gaussian 随机变量 $X$，其 CDF 与高斯 CDF 的差异由 Berry-Esseen 界控制：

$$\left|P(X \leq t) - \Phi\left(\frac{t - \mu}{\sigma'}\right)\right| \leq \frac{C_{\text{BE}}}{\sigma' \sqrt{n}}$$

其中 $n$ 是样本量，$C_{\text{BE}}$ 是 Berry-Esseen 常数。

因此：

$$\mathbb{E}[\sigma_{\text{s}}(w^T z + b) \mid g, y=l] \approx \Phi\left(\frac{w^T \mu_{g,l} + b}{\sigma'_{g,l}}\right)$$

近似误差为 $O(1/(\sigma' \sqrt{n}))$。

**Step 4**：EO 差距的近似表达式

$$\text{EO}_l \approx \left|\Phi\left(\frac{w^T \mu_{g_0,l} + b}{\sigma'_{g_0,l}}\right) - \Phi\left(\frac{w^T \mu_{g_1,l} + b}{\sigma'_{g_1,l}}\right)\right|$$

**Step 5**：利用中值定理

存在 $\xi$ 使得：

$$\text{EO}_l = \varphi(\xi) \cdot \left|\frac{w^T \mu_{g_0,l} + b}{\sigma'_{g_0,l}} - \frac{w^T \mu_{g_1,l} + b}{\sigma'_{g_1,l}}\right|$$

其中 $\varphi$ 是标准正态 PDF，$\varphi(\xi) \leq 1/\sqrt{2\pi}$。

**Step 6**：利用假设 2（有界协方差差异）

由 $|\Sigma_{g_0,l} - \Sigma_{g_1,l}|_{\text{op}} \leq \Delta_\Sigma$，可得：

$$|\sigma'_{g_0,l} - \sigma'_{g_1,l}| \leq \frac{|w|^2 \Delta_\Sigma}{2\sigma'}$$

（由一阶 Taylor 展开）

令 $\sigma' = \max(\sigma'_{g_0,l}, \sigma'_{g_1,l})$，则：

$$\text{EO}_l \leq \frac{1}{\sqrt{2\pi}} \cdot \frac{|w^T(\mu_{g_0,l} - \mu_{g_1,l})|}{\sigma'} + O\left(\frac{|w|^2 \Delta_\Sigma}{\sigma'^3}\right)$$

**Step 7**：利用 Cauchy-Schwarz 不等式

$$|w^T(\mu_{g_0,l} - \mu_{g_1,l})| \leq |w| \cdot |\mu_{g_0,l} - \mu_{g_1,l}|$$

因此：

$$\text{EO}_l \leq \frac{|w|}{\sqrt{2\pi} \sigma'} \cdot |\mu_{g_0,l} - \mu_{g_1,l}| + O\left(\frac{|w|^2 \Delta_\Sigma}{\sigma'^3}\right)$$

对所有标签求和，令 $\sigma_z = \max_l \sigma_l'$：

$$\text{EO} \leq \frac{|w|}{\sqrt{2\pi} \sigma_z} \cdot \Delta_{\text{rep}} + O\left(\frac{|w|^2 \Delta_\Sigma}{\sigma_z^3}\right)$$

**证毕。**

***

### 5.2 定理 2 的证明

**设定**：存在常数 $\delta_{\text{conf}} > 0$，使得对于所有标签 $l \in \{0,1\}$：

$$\left| \mathbb{E}[c \mid g=0, y=l] - \mathbb{E}[c \mid g=1, y=l] \right| \geq \delta_{\text{conf}}$$

不失一般性，假设 $\mathbb{E}[c \mid g=1, y=l] \geq \mathbb{E}[c \mid g=0, y=l]$（否则对调群组标签即可）。记：

$$c_{\text{high}} = \mathbb{E}[c \mid g=1, y=l], \quad c_{\text{low}} = \mathbb{E}[c \mid g=0, y=l]$$

则 $c_{\text{high}} - c_{\text{low}} \geq \delta_{\text{conf}}$。

**Step 1**：置信度与 logit 的关系

置信度 $c = \sigma_{\text{s}}(|z|)$，其中 $z = w^T x + b$ 是 logit。

由 sigmoid 的逆函数：

$$|z| = \sigma_{\text{s}}^{-1}(c) = \ln\left(\frac{c}{1-c}\right)$$

低置信度 $c_{\text{low}}$ 对应 $|z|_{\text{low}} = \ln\left(\frac{c_{\text{low}}}{1 - c_{\text{low}}}\right)$

高置信度 $c_{\text{high}}$ 对应 $|z|_{\text{high}} = \ln\left(\frac{c_{\text{high}}}{1 - c_{\text{high}}}\right)$

**Step 2**：logit 的期望

由假设 1（sub-Gaussian），$z \mid g, y=l$ 的均值为 $\mu'_g = w^T \mu_{g,l} + b$，标准差为 $\sigma'_g$。

$$\mathbb{E}[|z| \mid g, y=l] \approx |\mu'_g|$$

（当 $\sigma'_g$ 较小时，由 sub-Gaussian 集中不等式，$z$ 集中在 $\mu'_g$ 附近）

因此：

$$\mathbb{E}[c \mid g, y=l] \approx \sigma_{\text{s}}(|\mu'_g|)$$

**Step 3**：置信度差异与 logit 差异

由 sigmoid 的 Lipschitz 性质（Lipschitz 常数 $1/4$）：

$$c_{\text{high}} - c_{\text{low}} \leq \frac{1}{4}(|\mu'_{g_1}| - |\mu'_{g_0}|)$$

因此：

$$|\mu'_{g_1}| - |\mu'_{g_0}| \geq 4(c_{\text{high}} - c_{\text{low}}) \geq 4\delta_{\text{conf}}$$

**Step 4**：logit 差异与 EO 的关系

EO 差距：

$$\text{EO}_l = \left|\mathbb{E}[\hat{p} \mid g=0, y=l] - \mathbb{E}[\hat{p} \mid g=1, y=l]\right|$$

$$\approx \left|\Phi(\mu'_{g_0}/\sigma') - \Phi(\mu'_{g_1}/\sigma')\right|$$

**Step 5**：分情况讨论

**情况 1**：$\mu'_{g_0}$ 和 $\mu'_{g_1}$ 同号

$$\text{EO}_l \approx \left|\Phi(\mu'_{g_0}/\sigma') - \Phi(\mu'_{g_1}/\sigma')\right|$$

由中值定理：

$$\text{EO}_l \geq \varphi\left(\frac{\max(|\mu'_{g_0}|, |\mu'_{g_1}|)}{\sigma'}\right) \cdot \frac{\bigl||\mu'_{g_0}| - |\mu'_{g_1}|\bigr|}{\sigma'}$$

$$\geq \varphi\left(\frac{|\mu'_{g_1}|}{\sigma'}\right) \cdot \frac{4\delta_{\text{conf}}}{\sigma'}$$

由于 $\varphi$ 在 $|x| \leq 1$ 时 $\varphi(x) \geq \varphi(1) \approx 0.242$，当 $|\mu'_{g_1}| \leq \sigma'$ 时：

$$\text{EO}_l \geq \frac{0.242 \cdot 4\delta_{\text{conf}}}{\sigma'} \geq \frac{\delta_{\text{conf}}}{\sigma'}$$

**情况 2**：$\mu'_{g_0}$ 和 $\mu'_{g_1}$ 异号

此时 $|\mu'_{g_0}| + |\mu'_{g_1}| \geq |\mu'_{g_1}| - |\mu'_{g_0}| \geq 4\delta_{\text{conf}}$

EO 差距更大（因为两个群组的预测概率在决策边界两侧）。

**Step 6**：综合

$$\text{EO} \geq 2\delta_{\text{conf}} - \frac{2\sigma'}{|w|\sqrt{2\pi}} - O(\Delta_\Sigma)$$

**证毕。**

**注**：

- 这个定理给出的是 EO 的**下界**，说明置信度差异是公平性问题的**必要信号**
- $\frac{2\sigma'}{|w|\sqrt{2\pi}}$ 是分布方差的修正项，$\sigma'$ 越小（分布越集中），下界越紧
- $O(\Delta_\Sigma)$ 是协方差差异的修正项

---

### 5.3 定理 3 的证明

**设定**：客户端 $k$ 的局部训练损失 $\mathcal{L}_k = \mathcal{L}_{\text{task},k} + \lambda \cdot \mathcal{L}_{\text{contrastive},k}$，CL 始终运行。

**证明 (a)：收敛性。**

**Step 1**：梯度下降更新

$$\theta^{(t+1)} = \theta^{(t)} - \eta \nabla L^{(t)}$$

**Step 2**：利用 L-光滑性

由假设 3，$L$ 是 $(L + \lambda M)$-光滑的，其中 $L$ 是任务损失的光滑常数，$M$ 是 CL 损失的光滑常数。

由 L-光滑性的定义：

$$L^{(t+1)} \leq L^{(t)} + \nabla L^{(t)} \cdot (-\eta \nabla L^{(t)}) + \frac{(L + \lambda M) \eta^2}{2} |\nabla L^{(t)}|^2$$

$$= L^{(t)} - \eta |\nabla L^{(t)}|^2 + \frac{(L + \lambda M) \eta^2}{2} |\nabla L^{(t)}|^2$$

**Step 3**：整理

$$L^{(t+1)} - L^{(t)} \leq -\eta |\nabla L^{(t)}|^2 + \frac{\eta^2 (L + \lambda M)}{2} |\nabla L^{(t)}|^2$$

当 $\eta < 2/(L + \lambda M)$ 时，右端为负，损失单调递减。

**证明 (b)：Δ_rep 收缩性。**

**Step 4**：CL 损失的形式

PDFFed 使用的对比学习损失为原型级对比损失：

$$L_{\text{contrastive}} = \frac{1}{2} \sum_l \|\mu_{g0,l} - \mu_{g1,l}\|^2$$

**Step 5**：表征偏差的梯度

表征偏差 $\Delta_{\text{rep}} = \sum_l \|\mu_{g0,l} - \mu_{g1,l}\|$，其梯度为：

$$\nabla_\phi \Delta_{\text{rep}} = \sum_l \frac{\mu_{g0,l} - \mu_{g1,l}}{\|\mu_{g0,l} - \mu_{g1,l}\|} \cdot (\nabla_\phi \mu_{g0,l} - \nabla_\phi \mu_{g1,l})$$

**Step 6**：CL 损失的梯度

$$\nabla_\phi L_{\text{contrastive}} = \sum_l (\mu_{g0,l} - \mu_{g1,l}) \cdot (\nabla_\phi \mu_{g0,l} - \nabla_\phi \mu_{g1,l})$$

**Step 7**：两个梯度的内积（引理 1 的证明）

$$\nabla_\phi \Delta_{\text{rep}} \cdot \nabla_\phi L_{\text{contrastive}} = \sum_l \|\mu_{g0,l} - \mu_{g1,l}\| \cdot \|\nabla_\phi \mu_{g0,l} - \nabla_\phi \mu_{g1,l}\|^2$$

由于 $\|\mu_{g0,l} - \mu_{g1,l}\| > 0$ 且 $\|\nabla_\phi \mu_{g0,l} - \nabla_\phi \mu_{g1,l}\|^2 \geq 0$，内积非负，即 CL 梯度方向与缩小 $\Delta_{\text{rep}}$ 的方向一致。这证明了引理 1 (a)。

**Step 8**：梯度下降更新（CL 始终运行）

$$\phi^{(t+1)} = \phi^{(t)} - \eta \cdot \nabla_\phi L_{\text{task}} - \eta \lambda \cdot \nabla_\phi L_{\text{contrastive}}$$

其中第二项 $\eta \lambda \cdot \nabla_\phi L_{\text{contrastive}}$ 是 CL 对编码器的额外更新。

**Step 9**：定量收缩

由 Taylor 展开，考虑 CL 项的贡献：

$$\Delta_{\text{rep}}^{(t+1)} = \Delta_{\text{rep}}^{(t)} - \eta \lambda \cdot \nabla_\phi \Delta_{\text{rep}} \cdot \nabla_\phi L_{\text{contrastive}} + O(\eta^2)$$

由 Step 7，$\nabla_\phi \Delta_{\text{rep}} \cdot \nabla_\phi L_{\text{contrastive}} \geq \sum_l \|\mu_{g0,l} - \mu_{g1,l}\|^2 > 0$（当 $\Delta_{\text{rep}} > 0$ 时严格大于零），因此 CL 贡献一个负项，减少 $\Delta_{\text{rep}}$。

令 $\gamma = \mathbb{E}[\|\mu_{g0,l} - \mu_{g1,l}\|] > 0$，则经过 T 步后：

$$\Delta_{\text{rep}}^{(t+T)} \leq \Delta_{\text{rep}}^{(t)} - \eta \cdot \lambda \cdot T \cdot \gamma$$

**注**：CL 损失的形式是证明收缩性的关键——$L_{\text{contrastive}} = \frac{1}{2} \sum_l \|\mu_{g0,l} - \mu_{g1,l}\|^2$ 直接度量组间原型距离，因此其梯度天然推动 $\Delta_{\text{rep}}$ 减小。$\gamma > 0$ 是保证收缩性的核心条件，当 $\Delta_{\text{rep}} > 0$ 时自然满足。

**证毕。**

### 5.4 引理 2 的证明（双向原型-分类器对齐控制有效分布偏移）

**证明 (a)：有效分布偏移控制。**

**Step 1**：由 $\mathcal{L}_{\text{l2g}} \leq \varepsilon_{\text{align}}$，全局分类器 $(w_{\text{global}}, b_{\text{global}})$ 在本地原型 $\mu_l^{(k)}$ 上正确分类。由假设 6（间隔性），正确分类意味着：

$$\text{sign}(w_{\text{global}}^T \mu_l^{(k)} + b_{\text{global}}) = \text{sign}(l - 1/2)$$

且交叉熵损失 $\ell \leq \varepsilon_{\text{align}}$ 意味着 $\sigma_{\text{s}}(w_{\text{global}}^T \mu_l^{(k)} + b_{\text{global}})$ 接近真实标签概率。

**Step 2**：同理，全局分类器在全局原型 $\mu_l^*$ 上也正确分类（因为 $\mu_l^*$ 是各客户端原型的加权聚合，且全局分类器在训练时已对齐）。

**Step 3**：由 sigmoid 的 Lipschitz 性质（$|\sigma_{\text{s}}'| \leq 1/4$）和间隔 $\rho$（假设 6）：

$$|\sigma_{\text{s}}(w_{\text{global}}^T \mu_l^{(k)} + b_{\text{global}}) - \sigma_{\text{s}}(w_{\text{global}}^T \mu_l^* + b_{\text{global}})| \leq \frac{1}{4} |w_{\text{global}}^T (\mu_l^{(k)} - \mu_l^*)|$$

**Step 4**：由 $\mathcal{L}_{\text{l2g}} \leq \varepsilon_{\text{align}}$ 和 $\mathcal{L}_{\text{g2l}} \leq \varepsilon_{\text{align}}$，两个分类器在两个原型上的输出都接近真实标签，因此两个输出之间的差也被 $\varepsilon_{\text{align}}$ 控制（三角不等式）：

$$|\sigma_{\text{s}}(w_{\text{global}}^T \mu_l^{(k)} + b_{\text{global}}) - \sigma_{\text{s}}(w_{\text{global}}^T \mu_l^* + b_{\text{global}})| \leq O(\varepsilon_{\text{align}} / \rho)$$

（除以 $\rho$ 是因为间隔 $\rho$ 提供了 logit 到概率的放大系数）

**Step 5**：结合 Step 3 和 Step 4：

$$|w_{\text{global}}^T (\mu_l^{(k)} - \mu_l^*)| \leq O\left(\frac{\varepsilon_{\text{align}}}{\rho}\right)$$

即有效分布偏移 $\Gamma_{\bar{w}}$ 被控制。

**证明 (b)：时效性（一步延迟）。**

**Step 6**：第 $t$ 轮训练使用第 $t-1$ 轮的 $(\mu_l^*, w_{\text{global}}, b_{\text{global}})$。由假设 3（$L$-光滑性），参数在相邻轮次间的变化量为 $O(\eta L)$（单步梯度下降的参数变化上界）。

**Step 7**：因此，使用延迟信息引入的误差为 $O(\eta L)$，叠加到 (a) 的界中：

$$|w_{\text{global}}^{(t-1),T} (\mu_l^{(k)} - \mu_l^{*(t-1)})| \leq O\left(\frac{\varepsilon_{\text{align}}}{\rho}\right) + O(\eta L)$$

当 $\eta$ 足够小时（$\eta L \ll \varepsilon_{\text{align}} / \rho$），延迟误差被主项吸收，不影响收敛性。这是 FedAvg 收敛分析中处理一步延迟的标准技术（Li et al., 2020）。

**证毕。**

### 5.5 引理 3 的证明（Prototype-driven 分类一致性）

**证明。**

**Step 1**：原型 $\mu_l = \mathbb{E}[z | y=l]$ 是类 $l$ 的特征中心。由 $\mathcal{L}_{\text{l2l}} \leq \varepsilon_{\text{clf}}$，分类器在原型上正确分类，且由假设 6，原型到决策边界的间隔为 $\rho$：

$$|w^T \mu_l + b| \geq \rho, \quad \text{sign}(w^T \mu_l + b) = \text{sign}(l - 1/2)$$

**Step 2**：由假设 1（sub-Gaussian），样本 $z$ 在给定 $y=l$ 时满足 $z = \mu_l + \epsilon$，其中 $\epsilon$ 是 sub-Gaussian 向量，$\|\epsilon\| \leq O(\sigma \sqrt{d})$ 以高概率成立。

**Step 3**：样本 $z$ 到决策边界的距离：

$$|w^T z + b| = |w^T \mu_l + b + w^T \epsilon| \geq |w^T \mu_l + b| - |w^T \epsilon| \geq \rho - |w| \cdot O(\sigma \sqrt{d})$$

当 $\rho > |w| \cdot O(\sigma \sqrt{d})$ 时，样本被正确分类。

**Step 4**：由 sub-Gaussian 尾界（Hoeffding 型不等式），单样本错误的概率为：

$$\Pr[|w^T \epsilon| > \rho] \leq 2 \exp\left(-\frac{\rho^2}{2 \sigma^2 |w|^2}\right)$$

**Step 5**：由假设 5（有界样本估计误差），原型基于 $n$ 个样本估计，估计误差为 $O(\sigma\sqrt{d/n})$。由集中不等式，以至少 $1-\delta$ 的概率，估计误差不超过 $O(\sigma\sqrt{d \log(1/\delta) / n})$。这一误差导致间隔 $\rho$ 的不确定性为 $O(|w| \sigma\sqrt{d \log(1/\delta) / n})$。

**Step 6**：综合 Step 4 和 Step 5，以至少 $1-\delta$ 的概率（关于训练样本的随机性），样本级分类错误率满足：

$$\Pr_{(x,y) \sim \mathcal{D}_k}[\hat{y}(x) \neq y] \leq O\left(\varepsilon_{\text{clf}} + \exp\left(-\frac{\rho^2}{2\sigma^2 |w|^2}\right) + \sqrt{\frac{d \log(1/\delta)}{n}}\right)$$

三项分别对应：原型级分类误差 $\varepsilon_{\text{clf}}$、sub-Gaussian 尾部概率 $\exp(-\rho^2/(2\sigma^2|w|^2))$、有限样本估计误差 $\sqrt{d\log(1/\delta)/n}$。

**证毕。**

***

### 5.6 定理 4 的证明（全局原型作为 EO 校准代理的有效性）

**设定**：Server 端后训练仅更新分类头 $\psi$，损失为 $L_{\text{post}} = L_{\text{cls}} + \lambda_{\text{eo}} \cdot \text{EO}_{\text{proto}}$，其中 $\text{EO}_{\text{proto}} = \sum_l |\sigma(w^T \mu_{g_0,l} + b) - \sigma(w^T \mu_{g_1,l} + b)|$。

**证明 (a)：逼近界。**

**Step 1：** 由定理 1 的证明，全局 EO 在假设 1-2 下近似为：

$$\text{EO}_{\text{global}} \approx \sum_l \left|\Phi\left(\frac{w^T \mu_{g_0,l} + b}{\sigma'_{g_0,l}}\right) - \Phi\left(\frac{w^T \mu_{g_1,l} + b}{\sigma'_{g_1,l}}\right)\right|$$

近似误差为 $O(1/(\sigma \sqrt{n}))$（Berry-Esseen 界）。

**Step 2：** 利用经典近似 $\sigma(x) \approx \Phi(c \cdot x)$，其中 $c = \sqrt{\pi / \ln 2} \approx 2.40$（误差在 $|x| \leq 3$ 范围内小于 0.02）：

$$\text{EO}_{\text{proto}} \approx \sum_l \left|\Phi(c \cdot (w^T \mu_{g_0,l} + b)) - \Phi(c \cdot (w^T \mu_{g_1,l} + b))\right|$$

**Step 3：** 对比两者，差异来自分母 $\sigma'_{g,l}$（由假设 2 控制）和常数 $c$（固定缩放）。利用假设 2 和定理 1 证明中 Step 6 的相同推导：

$$|\text{EO}_{\text{proto}} - \text{EO}_{\text{global}}| \leq \frac{|w|}{\sqrt{2\pi} \sigma} \cdot \Delta_{\text{rep}}^{\text{proto}} + O\left(\frac{|w|^2 \Delta_\Sigma}{\sigma^3}\right) + O\left(\frac{1}{\sigma \sqrt{n}}\right)$$

**证明 (b)：梯度方向一致性。**

**Step 4：** 由于仅更新 $\psi = (w, b)$，两个梯度表达式分别为：

$$\nabla_\psi \text{EO}_{\text{proto}} = \sum_l \nabla_\psi \left|\sigma(w^T \mu_{g_0,l} + b) - \sigma(w^T \mu_{g_1,l} + b)\right|$$

$$\nabla_\psi \text{EO}_{\text{global}} \approx \sum_l \nabla_\psi \left|\Phi\left(\frac{w^T \mu_{g_0,l} + b}{\sigma'_{g_0,l}}\right) - \Phi\left(\frac{w^T \mu_{g_1,l} + b}{\sigma'_{g_1,l}}\right)\right|$$

**Step 5：** 由 $\sigma(x) \approx \Phi(c \cdot x)$，有 $\sigma'(x) \approx c \cdot \varphi(c \cdot x)$。因此两个梯度的核心项只差常数缩放 $c$ 和分母 $\sigma'_{g,l}$ 的修正：

$$\nabla_\psi \text{EO}_{\text{proto}} \approx c \cdot \nabla_\psi \text{EO}_{\text{global}} + \text{修正项}(\Delta_\Sigma)$$

**Step 6：** 由 Cauchy-Schwarz 不等式：

$$\cos\langle \nabla_\psi \text{EO}_{\text{proto}}, \nabla_\psi \text{EO}_{\text{global}} \rangle \geq 1 - O\left(\frac{|w|^2 \Delta_\Sigma}{\sigma^2}\right)$$

**证明 (c)：单步下降传递。**

**Step 7：** 真实 EO 的 Taylor 展开：

$$\text{EO}_{\text{global}}^{(t+1)} = \text{EO}_{\text{global}}^{(t)} + \nabla_\psi \text{EO}_{\text{global}}^{(t)} \cdot (\psi^{(t+1)} - \psi^{(t)}) + O(\alpha^2)$$

其中 $\psi^{(t+1)} - \psi^{(t)} = -\alpha \lambda_{\text{eo}} \nabla_\psi \text{EO}_{\text{proto}}^{(t)}$。

**Step 8：** 代入并利用 (b)：

$$\text{EO}_{\text{global}}^{(t+1)} = \text{EO}_{\text{global}}^{(t)} - \alpha \lambda_{\text{eo}} |\nabla_\psi \text{EO}_{\text{global}}| |\nabla_\psi \text{EO}_{\text{proto}}| \cos\langle \nabla_\psi \text{EO}_{\text{global}}, \nabla_\psi \text{EO}_{\text{proto}} \rangle + O(\alpha^2)$$

由 (b)，$\cos\langle \cdot, \cdot \rangle \geq 1 - \varepsilon(\Delta_\Sigma)$，因此：

$$\text{EO}_{\text{global}}^{(t+1)} \leq \text{EO}_{\text{global}}^{(t)} - \alpha \lambda_{\text{eo}} |\nabla_\psi \text{EO}_{\text{proto}}|^2 \cdot \delta$$

其中 $\delta = \cos\langle \cdot, \cdot \rangle - \alpha L_{\text{eo}} / (2 \lambda_{\text{eo}}) > 0$（当 $\alpha$ 足够小且 $\Delta_\Sigma$ 足够小时）。

**证毕。**

### 5.7 定理 5 的证明（原型聚合的差分隐私保证）

**设定**：客户端 $k$ 计算本地原型 $\mu_{g,l}^{(k)} = \frac{1}{n_{g,l}^{(k)}} \sum_{i=1}^{n_{g,l}^{(k)}} z_i$，添加噪声后上传 $\tilde{\mu}_{g,l}^{(k)} = \mu_{g,l}^{(k)} + \xi_k$，其中 $\xi_k \sim \mathcal{N}(0, \sigma_{\text{noise}}^2 \cdot I_d)$。

Server 端聚合：$\mu_{g,l}^{\text{global}} = \sum_{k=1}^K w_k \cdot \tilde{\mu}_{g,l}^{(k)}$

**Step 1**：局部差分隐私保证

每个客户端的噪声 $\xi_k$ 满足 $\epsilon$-LDP（当 $\sigma_{\text{noise}}^2 \geq \frac{2d \ln(2/\delta)}{\epsilon^2}$ 时）：

$$\text{Pr}[\tilde{\mu}_{g,l}^{(k)} \in S] \leq e^\epsilon \cdot \text{Pr}[\tilde{\mu}_{g,l}^{(k),-i} \in S] + \delta$$

其中 $\tilde{\mu}_{g,l}^{(k),-i}$ 是移除第 $i$ 个样本后的带噪原型。

**Step 2**：分布式差分隐私

由分布式差分隐私的合成定理（composition theorem），$K$ 个客户端的 LDP 机制组合后满足 $K\epsilon$-DP：

$$\text{Pr}[\mathcal{A}(\mu_{g,l}^{\text{global}}) = t] \leq e^{K\epsilon} \cdot \text{Pr}[\mathcal{A}(\mu_{g,l}^{\text{global},-i}) = t] + K\delta$$

**Step 3**：隐私放大

当 $K$ 个客户端参与聚合时，隐私预算可以通过聚合进行放大。对于 $(\epsilon, \delta)$-LDP，聚合后的隐私保证为：

$$\epsilon_{\text{total}} \leq \sqrt{\frac{8K \ln(1/\delta')}{\epsilon^2}} + \frac{2K\epsilon \ln(1/\delta')}{\delta}$$

当 $K$ 较大时，聚合后的隐私保证显著优于单客户端的隐私保证。

**证毕。**

### 5.8 定理 6 的证明（传输敏感属性的隐私风险）

**设定**：假设有方法传输原始敏感属性 $g_i$。

**Step 1**：成员推断攻击

攻击者可以设计算法 $\mathcal{A}(g_i)$ 判断样本 $i$ 是否属于训练集 $S$：

$$\mathcal{A}(g_i) = \begin{cases} 1 & \text{如果 } g_i \text{ 出现在传输数据中} \\ 0 & \text{否则} \end{cases}$$

**Step 2**：攻击成功率

$$\text{Pr}[\mathcal{A}(g_i) = 1 \mid i \in S] = 1$$
$$\text{Pr}[\mathcal{A}(g_i) = 1 \mid i \notin S] = 0$$

因此：

$$\text{Pr}[\mathcal{A}(g_i) = 1 \mid i \in S] - \text{Pr}[\mathcal{A}(g_i) = 1 \mid i \notin S] = 1$$

**Step 3**：违反差分隐私

差分隐私要求对于任意相邻数据集 $D$ 和 $D'$，有：

$$\text{Pr}[\mathcal{A}(M(D)) = t] \leq e^\epsilon \cdot \text{Pr}[\mathcal{A}(M(D')) = t] + \delta$$

对于传输敏感属性的机制，当 $D$ 和 $D'$ 仅在样本 $i$ 上不同时，攻击者可以完美区分两者，因此不满足差分隐私。

**证毕。**

### 5.9 定理 7 的证明（统计异构下全局原型的聚合偏差与全局公平性界）

**证明 (a)：全局原型聚合偏差界。**

**Step 1**：分解聚合原型与真实全局原型的偏差

$$\mu_{g,l}^{\text{global}} - \mu_{g,l}^* = \sum_{k=1}^K w_k \hat{\mu}_{g,l}^{(k)} - \mu_{g,l}^*$$

其中 $\hat{\mu}_{g,l}^{(k)}$ 是客户端 $k$ 基于有限样本估计的局部原型。加减 $\sum_k w_k \mu_{g,l}^{(k)}$：

$$\mu_{g,l}^{\text{global}} - \mu_{g,l}^* = \underbrace{\sum_k w_k (\mu_{g,l}^{(k)} - \mu_{g,l}^*)}_{\text{分布异构偏差}} + \underbrace{\sum_k w_k (\hat{\mu}_{g,l}^{(k)} - \mu_{g,l}^{(k)})}_{\text{统计估计误差}}$$

**Step 2**：由假设 4（有界局部分布偏移）

$$\left\|\sum_k w_k (\mu_{g,l}^{(k)} - \mu_{g,l}^*)\right\| \leq \sum_k w_k \|\mu_{g,l}^{(k)} - \mu_{g,l}^*\| \leq \sum_k w_k \Gamma_{g,l}^{(k)}$$

（由三角不等式和假设 4）

**Step 3**：由假设 1（sub-Gaussian）和假设 5

对每个客户端 $k$，由 sub-Gaussian 集中不等式，以概率至少 $1 - \delta_0$：

$$\|\hat{\mu}_{g,l}^{(k)} - \mu_{g,l}^{(k)}\| \leq \sigma \sqrt{\frac{2d \ln(2K/\delta_0)}{n_{g,l}^{(k)}}}$$

对 $K$ 个客户端取并集界（union bound）。

**Step 4**：加权聚合的估计误差

$$\left\|\sum_k w_k (\hat{\mu}_{g,l}^{(k)} - \mu_{g,l}^{(k)})\right\| \leq \sum_k w_k \sigma \sqrt{\frac{2d \ln(2K/\delta_0)}{n_{g,l}^{(k)}}}$$

由 Cauchy-Schwarz 不等式 $\sum_k w_k / \sqrt{n_{g,l}^{(k)}} \leq \sqrt{\sum_k w_k^2 / n_{g,l}^{(k)}}$，且 $\sum_k w_k = 1$。当各客户端 $n_{g,l}^{(k)}$ 接近时，$\sum_k w_k^2 / n_{g,l}^{(k)} \approx 1/N_{g,l}$，因此：

$$\left\|\sum_k w_k (\hat{\mu}_{g,l}^{(k)} - \mu_{g,l}^{(k)})\right\| \leq O\left(\sigma \sqrt{\frac{d}{N_{g,l}}}\right)$$

**Step 5**：合并两项

$$\|\mu_{g,l}^{\text{global}} - \mu_{g,l}^*\| \leq \sum_k w_k \Gamma_{g,l}^{(k)} + O\left(\sigma \sqrt{\frac{d}{N_{g,l}}}\right)$$

**证明 (b)：聚合表征偏差界。**

**Step 6**：展开 $\Delta_{\text{rep}}^{\text{global}}$

$$\Delta_{\text{rep}}^{\text{global}} = \sum_l \|\mu_{g_0,l}^{\text{global}} - \mu_{g_1,l}^{\text{global}}\|$$

对每个 $l$，由三角不等式：

$$\|\mu_{g_0,l}^{\text{global}} - \mu_{g_1,l}^{\text{global}}\| \leq \|\mu_{g_0,l}^* - \mu_{g_1,l}^*\| + \|\mu_{g_0,l}^{\text{global}} - \mu_{g_0,l}^*\| + \|\mu_{g_1,l}^{\text{global}} - \mu_{g_1,l}^*\|$$

**Step 7**：代入 (a) 的结果

$$\|\mu_{g_0,l}^{\text{global}} - \mu_{g_1,l}^{\text{global}}\| \leq \|\mu_{g_0,l}^* - \mu_{g_1,l}^*\| + \sum_k w_k \Gamma_{g_0,l}^{(k)} + \sum_k w_k \Gamma_{g_1,l}^{(k)} + O\left(\sigma \sqrt{\frac{d}{N_{g_0,l}}} + \sigma \sqrt{\frac{d}{N_{g_1,l}}}\right)$$

令 $\bar{\Gamma} = \max_{g,l} \sum_k w_k \Gamma_{g,l}^{(k)}$，$N_{\min} = \min_{g,l} N_{g,l}$，对所有 $l$ 求和：

$$\Delta_{\text{rep}}^{\text{global}} \leq \Delta_{\text{rep}}^* + 2\bar{\Gamma} + O\left(\sigma \sqrt{\frac{d}{N_{\min}}}\right)$$

**证明 (c)：全局 EO 界。**

**Step 8**：直接代入定理 1

由定理 1，$\text{EO} \leq \frac{|w|}{\sqrt{2\pi}\sigma_z} \cdot \Delta_{\text{rep}} + O\left(\frac{|w|^2 \Delta_\Sigma}{\sigma_z^3}\right)$。将 (b) 的 $\Delta_{\text{rep}}^{\text{global}}$ 代入：

$$\text{EO}_{\text{global}} \leq \frac{|w|}{\sqrt{2\pi} \sigma_z} \cdot \left(\Delta_{\text{rep}}^* + 2\bar{\Gamma} + O\left(\sigma \sqrt{\frac{d}{N_{\min}}}\right)\right) + O\left(\frac{|w|^2 \Delta_\Sigma}{\sigma_z^3}\right)$$

整理即得 (c)。

**证明 (d)：Server 端 EO 校准的弥合效应。**

**Step 9**：由定理 4 (a) 的逼近界

$$|\text{EO}_{\text{proto}} - \text{EO}_{\text{global}}| \leq \frac{|w|}{\sqrt{2\pi} \sigma_z} \cdot \Delta_{\text{rep}}^{\text{proto}} + O\left(\frac{|w|^2 \Delta_\Sigma}{\sigma_z^3}\right) + O\left(\frac{1}{\sigma_z \sqrt{n}}\right)$$

其中 $\Delta_{\text{rep}}^{\text{proto}}$ 是全局原型层面的表征偏差，即 (b) 中的 $\Delta_{\text{rep}}^{\text{global}}$。

**Step 10**：Server 端校准的目标是 $\text{EO}_{\text{proto}}$，由定理 4 (c)，$\text{EO}_{\text{proto}}$ 下降传递到 $\text{EO}_{\text{global}}$ 下降。而 $\text{EO}_{\text{proto}}$ 由 $\Delta_{\text{rep}}^{\text{global}}$ 决定（定理 1 结构）。因此 Server 端校准直接作用于 $\Delta_{\text{rep}}^{\text{global}}$，而非 $\Delta_{\text{rep}}^*$。

**Step 11**：协同机制

- 定理 3 的客户端 CL 缩小局部 $\Delta_{\text{rep}}^{(k)}$，由假设 4，$\Gamma_{g,l}^{(k)}$ 与局部 $\Delta_{\text{rep}}^{(k)}$ 正相关，因此 CL 间接减小 $\bar{\Gamma}$。
- 定理 4 的 Server 端校准直接减小 $\Delta_{\text{rep}}^{\text{global}}$。
- 两者协同：前者控制 $\bar{\Gamma}$ 源头（(b) 中的 $2\bar{\Gamma}$ 项），后者弥合 $\Delta_{\text{rep}}^{\text{global}}$ 残差。

**证毕。**

***

## 6. 证明链路与挑战的对应关系

完整的证明链路与三个核心挑战的对应关系详见 [第 0 节](#0-宏观证明链路)。此处仅做补充说明：

**上界 vs 下界**：

- 定理 1 给出的是**上界**：减小 $\Delta_{\text{rep}}$ → EO 上界降低 → **保证**最坏情况下的 EO 不会太大
- 定理 2 给出的是**下界**：置信度差异大 → EO 下界高 → **诊断**出存在公平性问题

两者互补：定理 1 用于**设计优化目标**（减小表征偏差），定理 2 用于**诊断问题**（检测置信度差异）。

***

## 7. 其他公平性指标的泛化

以上证明框架以 Equalized Odds (EO) 为核心指标。本节说明该框架如何泛化到其他常用的公平性指标。

### 7.1 Demographic Parity (DP)

**定义**：

$$\text{DP} = \frac{1}{2} \sum_{l \in \{0,1\}} \left| \mathbb{E}[\hat{p} \mid g=0, y=l] - \mathbb{E}[\hat{p} \mid g=1, y=l] \right|$$

DP 与 EO 的关键区别：DP 不依赖真实标签 $y$，直接比较群组间的预测分布。

**泛化**：DP 的上界可类似推导，额外包含标签分布差异项：

$$\text{DP} \leq \frac{|w|}{\sqrt{2\pi} \sigma_z} \cdot \Delta_{\text{rep}} + O\left(\frac{|w|^2 \Delta_\Sigma}{\sigma_z^3}\right) + \Delta_{\text{label}}$$

其中 $\Delta_{\text{label}} = \frac{1}{2} \sum_{l \in \{0,1\}} \left| \mathbb{P}(y=l \mid g=0) - \mathbb{P}(y=l \mid g=1) \right|$ 是标签分布差距。

**说明**：
- DP 的上界多了 $\Delta_{\text{label}}$ 项——因为 DP 要求预测分布独立于真实标签，而真实标签分布可能在群组间不同
- 当标签分布平衡（$\Delta_{\text{label}} \approx 0$）时，DP 和 EO 的上界形式相同

### 7.2 Equalized Opportunity (EOpp)

**定义**：

$$\text{EOpp} = \left| \mathbb{E}[\hat{p} \mid g=0, y=1] - \mathbb{E}[\hat{p} \mid g=1, y=1] \right|$$

EOpp 是 EO 的特例，只关注正类的公平性。

**泛化**：上界推导与 EO 完全相同，求和限制在正类 $l=1$：

$$\text{EOpp} \leq \frac{|w|}{\sqrt{2\pi} \sigma_z} \cdot \Delta_{\text{rep},1} + O\left(\frac{|w|^2 \Delta_\Sigma}{\sigma_z^3}\right)$$

其中 $\Delta_{\text{rep},1} = \|\mu_{g0,1} - \mu_{g1,1}\|$ 是正类的表征偏差。

### 7.3 Disparate Equalized Odds (DEO)

**定义**：

$$\text{DEO} = w_1 \cdot |\text{TPR}_0 - \text{TPR}_1| + w_0 \cdot |\text{FPR}_0 - \text{FPR}_1|$$

其中 $w_1, w_0 \geq 0$ 是权重，$\text{TPR}_g = \mathbb{E}[\hat{p} \mid g, y=1]$，$\text{FPR}_g = \mathbb{E}[\hat{p} \mid g, y=0]$。

DEO 是 EO 的加权形式，允许对 TPR 和 FPR 差距赋予不同重要性。

**完整证明**：

**Step 1：DEO 分解。**

$$\text{DEO} = w_1 \cdot |\text{TPR}_0 - \text{TPR}_1| + w_0 \cdot |\text{FPR}_0 - \text{FPR}_1|$$

**Step 2：分别对 TPR 和 FPR 应用定理 1。**

由定理 1，对正类 $l=1$（TPR）：

$$|\text{TPR}_0 - \text{TPR}_1| \leq \frac{|w|}{\sqrt{2\pi} \sigma_z} \cdot \|\mu_{g0,1} - \mu_{g1,1}\| + O\left(\frac{|w|^2 \Delta_\Sigma}{\sigma_z^3}\right)$$

对负类 $l=0$（FPR），同理：

$$|\text{FPR}_0 - \text{FPR}_1| \leq \frac{|w|}{\sqrt{2\pi} \sigma_z} \cdot \|\mu_{g0,0} - \mu_{g1,0}\| + O\left(\frac{|w|^2 \Delta_\Sigma}{\sigma_z^3}\right)$$

**Step 3：加权求和。**

$$\text{DEO} \leq w_1 \cdot \left[ \frac{|w|}{\sqrt{2\pi} \sigma_z} \cdot \|\mu_{g0,1} - \mu_{g1,1}\| + O\left(\frac{|w|^2 \Delta_\Sigma}{\sigma_z^3}\right) \right] + w_0 \cdot \left[ \frac{|w|}{\sqrt{2\pi} \sigma_z} \cdot \|\mu_{g0,0} - \mu_{g1,0}\| + O\left(\frac{|w|^2 \Delta_\Sigma}{\sigma_z^3}\right) \right]$$

$$= \frac{|w|}{\sqrt{2\pi} \sigma_z} \cdot \left[ w_1 \cdot \|\mu_{g0,1} - \mu_{g1,1}\| + w_0 \cdot \|\mu_{g0,0} - \mu_{g1,0}\| \right] + O\left(\frac{|w|^2 \Delta_\Sigma}{\sigma_z^3}\right) \cdot (w_1 + w_0)$$

**Step 4：定义加权表征偏差。**

令 $\Delta_{\text{rep}}^w = w_1 \cdot \|\mu_{g0,1} - \mu_{g1,1}\| + w_0 \cdot \|\mu_{g0,0} - \mu_{g1,0}\|$，则：

$$\text{DEO} \leq \frac{|w|}{\sqrt{2\pi} \sigma_z} \cdot \Delta_{\text{rep}}^w + O\left(\frac{|w|^2 \Delta_\Sigma}{\sigma_z^3}\right) \cdot (w_1 + w_0)$$

**注**：当 $w_1 = w_0 = 1$ 时，$\Delta_{\text{rep}}^w = \Delta_{\text{rep}}$，DEO 退化为 EO。

### 7.4 泛化总结

| 指标 | 上界形式 | 特殊项 |
| ---- | ---- | ---- |
| EO | $\dfrac{\|w\|}{\sqrt{2\pi} \sigma_z} \cdot \Delta_{\text{rep}} + O\left(\dfrac{\|w\|^2 \Delta_\Sigma}{\sigma_z^3}\right)$ | 无 |
| DP | $\dfrac{\|w\|}{\sqrt{2\pi} \sigma_z} \cdot \Delta_{\text{rep}} + O\left(\dfrac{\|w\|^2 \Delta_\Sigma}{\sigma_z^3}\right) + \Delta_{\text{label}}$ | $\Delta_{\text{label}}$ |
| EOpp | $\dfrac{\|w\|}{\sqrt{2\pi} \sigma_z} \cdot \Delta_{\text{rep},1} + O\left(\dfrac{\|w\|^2 \Delta_\Sigma}{\sigma_z^3}\right)$ | 仅正类 |
| DEO | $\dfrac{\|w\|}{\sqrt{2\pi} \sigma_z} \cdot \Delta_{\text{rep}}^w + O\left(\dfrac{\|w\|^2 \Delta_\Sigma}{\sigma_z^3}\right) \cdot (w_1 + w_0)$ | 加权 |

**核心结论**：所有指标共享相同的上界结构——Δ_rep（或其变体）的线性项加上 Δ_Σ 的高阶项。这意味着：

1. **减小 Δ_rep 对所有指标都有改善**——这是我们方法论的核心依据
2. **定理 3 的 $\Delta_{\text{rep}}$ 收缩性证明对所有指标都成立**——CL 始终运行的情况下，无论使用哪个指标都能改善公平性
3. **定理 4 的校准代理方法对所有指标都适用**——只需将 EO_proto 替换为对应指标的原型级代理即可

***

## 8. 引用文献

1. McNamara, D., Ong, C. S., & Williamson, R. C. (2017). Provably Fair Representations. arXiv:1710.10622.
2. Zhao, H., & Gordon, G. J. (2019). Inherent Tradeoffs in Learning Fair Representations. arXiv:1906.08386.
3. Madras, D., Creager, E., Pitassi, T., & Zemel, R. (2018). Learning Adversarial Fair and Transferable Representations. ICML 2018.
4. Zemel, R., Wu, Y., Swersky, K., Pitassi, T., & Dwork, C. (2013). Learning Fair Representations. ICML 2013.
5. Hashimoto, T., Srivastava, M., Namkoong, H., & Liang, P. (2018). Fairness Without Demographics in Repeated Loss Minimization. ICML 2018.
6. Kirichenko, P., Izmailov, P., & Wilson, A. G. (2023). Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations. ICLR 2023. arXiv:2204.02937.
7. Mao, Y., Deng, Z., Yao, H., Ye, T., Kawaguchi, K., & Zou, J. (2023). Last-Layer Fairness Fine-tuning is Simple and Effective for Neural Networks. ICML Workshop on Human-Centric Machine Learning. arXiv:2304.03935.
8. Zhong, R., et al. (2025). Fair-FLIP: Fairness via Final-Layer Input Perturbation. arXiv:2507.08912.
9. Zhang, Y., et al. (2023). GroupMixNorm: Group-wise Mixing Normalization for Bias Mitigation. NeurIPS Workshop. arXiv:2312.11969.
10. Cui, J., et al. (2024). Investigating and Mitigating Group Disparities in Image Recognition. arXiv:2405.01112.
11. Dwork, C., & Roth, A. (2014). The Algorithmic Foundations of Differential Privacy. Foundations and Trends in Theoretical Computer Science.

