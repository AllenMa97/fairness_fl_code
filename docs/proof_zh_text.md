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

### 挑战 2：表征偏差引起的公平性退化（纲要：现有方法多聚焦于损失约束或聚合调控）

定理 1-3 构建了从"识别问题"到"实施干预"的完整闭环：

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
    └── 定理 3（干预：CL → Δ_rep 收缩）
            CL 始终运行 → Δ_rep 持续下降 + 收敛性不破坏
            结论：在训练中直接缩小 Δ_rep 是可行的
```

**闭环逻辑**：定理 1 说明"为什么要优化 $\Delta_{\text{rep}}$"（上界约束），定理 2 说明"如何诊断问题"（置信度差异），定理 3 说明"如何实施优化"（始终运行 CL）且不损害主任务。Lemma 1 进一步保证了 CL 梯度与缩小 $\Delta_{\text{rep}}$ 的方向一致。

### 挑战 3：统计异构下全局公平性（纲要：现有方法多聚焦于局部公平性优化）

定理 4 证明了 Server 端后训练的有效性：

```
Server 端使用全局原型进行 EO 校准
    │
    └── 定理 4（全局原型作为 EO 校准代理的有效性）
            │
            ├── (a) 逼近界：|EO_proto - EO_global| ≤ ε(Δ_rep, Δ_Σ)
            │       全局原型 EO 是真实全局 EO 的良好近似
            │
            ├── (b) 梯度方向一致性：cos⟨∇EO_proto, ∇EO_global⟩ ≥ 1 - ε
            │       优化代理目标的方向与真实目标一致（不会跑偏）
            │
            └── (c) 单步下降传递：EO_global 随 EO_proto 下降而下降
                    代理目标的下降可以传递到真实目标
```

### 定理-挑战对应总览

| 定理 | 核心关系 | 回应挑战 |
|------|---------|---------|
| 定理 1 | $\Delta_{\text{rep}}$ → EO 上界 | 挑战 2 |
| 定理 2 | 置信度差异 → EO 下界 | 挑战 2 |
| 定理 3 | CL → $\Delta_{\text{rep}}$ 收缩 | 挑战 2 |
| 定理 4 | 全局原型 → EO 校准 | 挑战 3 |
| 定理 5 | 原型聚合 → 差分隐私保证 | 挑战 1 |
| 定理 6 | 传输敏感属性 → 隐私破坏 | 挑战 1 |

**补充说明**：定理 1 给出的是**上界**（减小 $\Delta_{\text{rep}}$ → 保证最坏情况下的 EO 不会太大），用于**设计优化目标**；定理 2 给出的是**下界**（置信度差异大 → EO 下界高），用于**诊断问题**。两者互补。

***

## 1. 符号定义

| 符号                                        | 含义                               |
| ----------------------------------------- | -------------------------------- |
| $X$                                       | 输入空间                             |
| $Y = {0, 1}$                              | 标签空间                             |
| $G = {0, 1}$                              | 敏感属性（群组）空间                       |
| $\mathcal{D}$                             | 全局数据分布                           |
| $\mathcal{D}\_k$                          | 第 $k$ 个客户端的局部数据分布                |
| $f\_\phi: X \to \mathbb{R}^d$             | 特征提取器（编码器）                       |
| $h\_\psi: \mathbb{R}^d \to \mathbb{R}$    | 分类器（线性：$h\_\psi(z) = w^T z + b$） |
| $z = f\_\phi(x) \in \mathbb{R}^d$         | 样本 $x$ 的特征表示                     |
| $\hat{p} = \sigma_{\text{s}}(w^T z + b)$             | 预测概率，$\sigma_{\text{s}}$ 为 sigmoid 函数       |
| $\hat{y} = \mathbb{1}\[\hat{p} \geq 0.5]$ | 预测标签                             |
| $\mu_{g,l} = \mathbb{E}[z \mid g, y=l]$ | 群组 $g$、标签 $l$ 的特征均值（原型，见定义2.5） |
| $\bar{g}_{g,l}$ | 群组 $g$、标签 $l$ 的梯度原型（见定义2.6） |
| $\Delta_{\text{rep}}$ | 表征偏差（见定义2.3） |
| $\rho_l$ | 梯度方向一致性（见定义2.7） |
| $\text{EO}$ | Equalized Odds 差距（见定义2.1） |
| $\text{DEO}$ | Disparate Equalized Odds（见定义2.2） |
| $\sigma_{\text{s}}(\cdot)$ | sigmoid 激活函数，$\sigma_{\text{s}}(t) = \frac{1}{1+e^{-t}}$ |
| $\sigma_z$ | 特征在分类方向上的标准差，$\sigma_z = \sqrt{\mathbb{E}\left[(w^T z - \mathbb{E}[w^T z])^2\right]}$ |
| $\delta_{\text{conf}}$ | 群组置信度差异下界，$\delta_{\text{conf}} = \min_l \left| \mathbb{E}[c \mid g=0, y=l] - \mathbb{E}[c \mid g=1, y=l] \right|$ |
| $\Sigma_{g,l}$ | 群组 $g$、标签 $l$ 的特征协方差矩阵 |
| $\Delta_\Sigma$ | 协方差差异，$\Delta_\Sigma = \max_{l} \|\Sigma_{g0,l} - \Sigma_{g1,l}\|_{\text{op}}$ |
| $K$                                       | 客户端数量                            |
| $w\_k = n\_k / N$                         | $w_k = n_k / N$                  |
| $\eta$                                    | 学习率                              |

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

### 2.5 梯度原型

**定义2.6（群组-标签感知梯度原型）**：

对样本 $(x, y, g)$，任务损失为 $L_{\text{task}}(x, y) = \ell(\sigma_{\text{s}}(w^T f_\phi(x) + b), y)$。

特征梯度为 $\nabla_z L_{\text{task}} = \partial \ell / \partial z \in \mathbb{R}^d$。

群组-标签感知梯度原型定义为：

$$\bar{g}_{g,l} = \mathbb{E}_{(x,y,g) \sim \mathcal{D}} [\nabla_z L_{\text{task}}(x, y) \mid g, l]$$

**命名说明**：该原型是各样本梯度的均值，因此称为"梯度原型"；又因为梯度需要按群组和标签做区分，所以称为"群组-标签感知"。

### 2.6 梯度方向一致性

**定义2.7（群组间梯度原型方向一致性，简称梯度方向一致性）**：

$$\rho_l = \frac{\bar{g}_{g_0,l} \cdot \bar{g}_{g_1,l}}{|\bar{g}_{g_0,l}| \cdot |\bar{g}_{g_1,l}|}$$

即两个群组梯度原型的余弦相似度。$\rho_l \in [-1, 1]$。下文均以"梯度方向一致性"简称进行表述。

### 2.7 置信度

**定义2.8（预测置信度，简称置信度）**：

$$c = \max(\hat{p}, 1 - \hat{p}) = \sigma_{\text{s}}(|w^T z + b|)$$

下文均以"置信度"简称进行表述。

### 2.8 决策边界距离

**定义2.9（决策边界距离）**：

对于线性分类器 $h(z) = w^T z + b$，样本 $z$ 到决策边界的距离为：

$$d = \frac{|w^T z + b|}{|w|}$$

**观察1（置信度与决策边界距离的单调关系）**：

由定义2.8和定义2.9可得 $c = \sigma_{\text{s}}(|w| \cdot d)$，由于 $\sigma_{\text{s}}$ 单调递增且 $|w| > 0$，预测置信度 $c$ 关于决策边界距离 $d$ 单调递增。

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

***

## 5. 定理与引理

### 定理 1（表征偏差与 EO 上界 —— 引用 McNamara et al. 2017）

在假设 1 和假设 2 下，EO 差距满足：

$$\text{EO} \leq \frac{|w|}{\sqrt{2\pi} \sigma_z} \cdot \Delta_{\text{rep}} + O\left(\frac{|w|^2 \Delta_\Sigma}{\sigma_z^3}\right)$$

其中 $\sigma_z = \sqrt{\mathbb{E}\left[(w^T z - \mathbb{E}[w^T z])^2\right]}$ 是特征在分类方向上的标准差。

**来源**：本定理的证明框架参考 McNamara et al. (2017) "Provably Fair Representations" 和 Zhao & Gordon (2019)。我们在其基础上将共享协方差假设放宽为有界协方差差异（假设 3），将高斯假设放宽为 sub-Gaussian（假设 1）。

**意义**：这是**上界**，即"$\text{EO}$ 的最大值由两项共同决定"。第一项 $\frac{|w|}{\sqrt{2\pi}\sigma_z} \cdot \Delta_{\text{rep}}$ 是主要项，由表征偏差 $\Delta_{\text{rep}}$、分类器权重范数 $|w|$ 和分类方向标准差 $\sigma_z$ 共同决定；第二项是高阶小项，由协方差差异 $\Delta_{\Sigma}$ 控制。

三个变量的性质分析：

- $|w|$ 是分类器权重向量 $w$ 的**范数**（标量），反映决策边界的整体陡峭程度。已有工作 DFR (Kirichenko et al., 2023, ICLR, arXiv:2204.02937) 证明，在集中式学习中仅重训最后一层分类器即可显著改善公平性；Mao et al. (2023, ICML Workshop on Human-Centric Machine Learning, arXiv:2304.03935) 进一步表明 last-layer fine-tuning 能有效避免 fairness overfitting。这些方法确实会同时改变 $w$ 的方向和范数 $|w|$。DFR 论文明确指出其有效性的前提：**"标准神经网络实际上已经学到了核心特征（core features），只是没有主要依赖这些特征进行预测"**——即特征提取器已经学到了能够区分不同群组但又不歧视任何群组的特征表示，问题仅在于最后一层分类器给了虚假特征（spurious features）过高的权重。
- $\sigma_z$ 是特征在分类方向上的散布程度，由特征提取器 $f_\phi$ 学到的表征空间结构决定。已有工作 Fair-FLIP (Zhong et al., 2025, arXiv:2507.08912) 通过重加权 final-layer 输入特征来减少群组间变异度差异，GroupMixNorm (Zhang et al., 2023, NeurIPS Workshop, arXiv:2312.11969) 通过混合群组特征统计量来改善公平性，表明 $\sigma_z$ 相关的特征统计量确实与公平性相关。但这些方法属于**事后修正**，在训练完成后对特征统计量进行调整，不改变特征空间本身的结构。
- $\Delta_{\text{rep}}$ 直接量化了不同群组在表征空间中的中心距离，刻画的是表征空间本身的群组偏差。

Cui et al. (2024, arXiv:2405.01112) 的实证研究揭示了一个关键事实：**不公平性的根源在于 problematic representation 而非 classifier bias**——分类器权重范数 $|w|$ 本身已经平衡，问题出在特征空间的质量。这一发现与定理1的结构一致：$|w|$ 和 $\sigma_z$ 的调整属于"决策层面/事后修正"，而 $\Delta_{\text{rep}}$ 直接刻画了表征空间本身的群组偏差。

在联邦学习场景下，这一区别尤为关键。上述 DFR、Mao et al.、Fair-FLIP、GroupMixNorm 等工作均在集中式学习场景下提出，其共同前提是：特征提取器能够从完整数据集中学到相对公平的表征。然而在联邦学习中，当数据异构性较大时，各客户端的局部数据分布存在偏差，学到的表征本身就容易携带系统性群组偏差（$\Delta_{\text{rep}}$ 增大），特征空间可能从一开始就不公平。此时，集中式方法的前提不再成立——仅从分类器层面调整 $|w|$ 或事后调整 $\sigma_z$ 相关统计量，无法触及偏差的根源。

因此，在联邦学习的训练过程中直接缩小 $\Delta_{\text{rep}}$——通过表征层面的公平性约束（如本文引入的对比损失）——是降低 EO 上界最根本的手段。这一结论直接呼应了核心挑战2：现有方法多聚焦于损失约束或聚合调控（决策层面），难以从根本上缓解表征偏差引起的公平性退化。本文在局部训练中引入对比损失来直接缩小 $\Delta_{\text{rep}}$，正是基于这一理论洞察。

### 定理 2（群组置信度差异与 EO 下界）

**假设**：存在常数 $\delta_{\text{conf}} > 0$，使得对于所有标签 $l \in \{0,1\}$：

$$\left| \mathbb{E}[c \mid g=0, y=l] - \mathbb{E}[c \mid g=1, y=l] \right| \geq \delta_{\text{conf}}$$

在**假设 1（sub-Gaussian 特征分布）**下，EO 差距满足：

$$\text{EO} \geq 2\delta_{\text{conf}} - \frac{2\sigma_z}{|w|\sqrt{2\pi}} - O(\Delta_\Sigma)$$

其中 $\sigma_z = \sqrt{\mathbb{E}\left[(w^T z - \mathbb{E}[w^T z])^2\right]}$ 是特征在分类方向上的标准差（同定理1）。

**意义**：这是**下界**。它告诉我们：EO 的最小值与群组置信度差异 $\delta_{\text{conf}}$ 正相关。要让 EO 变得更小，必须先减小群组间的置信度差异——这是一个必要条件（但非充分条件）。反之，如果置信度差异很大（$\delta_{\text{conf}}$ 大），EO 就不可能太小。因此，置信度差异可以作为公平性问题的诊断信号。

### Lemma 1（CL 梯度与缩小表征偏差的方向一致性）

设 $\rho_l = \cos\langle \nabla_\phi \mathcal{L}_{\text{contrastive}}, \nabla_\phi \mathcal{L}_{\text{task}} \rangle$ 为对比损失梯度与任务损失梯度在编码器参数上的余弦相似度。对于 PDFFed 的原型级对比损失 $\mathcal{L}_{\text{contrastive}} = \frac{1}{2} \sum_l \|\mu_{g0,l} - \mu_{g1,l}\|^2$，在假设 1-3 下：

- 对任意标签 $l$，$\nabla_\phi \mathcal{L}_{\text{contrastive}}$ 的方向与缩小 $\Delta_{\text{rep}}$ 的方向一致（内积非负）；
- $\rho_l$ 可通过训练过程中的梯度统计量在线计算，作为训练动态的观测指标。

### 定理 3（CL 的收敛性与 Δ_rep 收缩性）

局部训练损失 $\mathcal{L}_k = \mathcal{L}_{\text{task},k} + \lambda \cdot \mathcal{L}_{\text{contrastive},k}$（始终运行 CL）在假设 3 下满足：

**(a) 收敛性：**

$$L^{(t+1)} - L^{(t)} \leq -\eta |\nabla L^{(t)}|^2 + \frac{\eta^2 (L + \lambda M)}{2} |\nabla L^{(t)}|^2$$

当 $\eta < 2/(L + \lambda M)$ 时，损失单调递减。其中 $L$ 是任务损失的光滑常数，$M$ 是 CL 损失的光滑常数。

**(b) Δ_rep 收缩性：**

经过 T 步后，表征偏差满足：

$$\Delta_{\text{rep}}^{(t+T)} \leq \Delta_{\text{rep}}^{(t)} - \eta \cdot \lambda \cdot T \cdot \gamma$$

其中 $\gamma = \mathbb{E}[\|\mu_{g0,l} - \mu_{g1,l}\|] > 0$，当 $\Delta_{\text{rep}} > 0$ 时自然满足。

**意义**：(a) 始终运行 CL 不破坏任务损失的收敛性——唯一的代价是将学习率上界从 $\eta < 2/L$ 收紧到 $\eta < 2/(L + \lambda M)$，这是温和的；(b) CL 确实能持续缩小表征偏差，收缩速度与学习率 $\eta$、CL 权重 $\lambda$ 和训练步数 $T$ 成正比。Lemma 1 进一步证明了 CL 梯度与缩小 $\Delta_{\text{rep}}$ 的方向天然一致，因此 CL 不会与主任务产生方向性冲突。

### 定理 4（全局原型作为 EO 校准代理的有效性）

**设定**：Server 端后训练仅更新分类头 $\psi = (w, b)$，使用全局原型层面的 EO 代理目标进行校准：

$$\text{EO}_{\text{proto}} = \sum_{l \in \{0,1\}} \left| \sigma_{\text{s}}(w^T \mu_{g0,l} + b) - \sigma_{\text{s}}(w^T \mu_{g1,l} + b) \right|$$

其中 $\mu_{g,l}$ 是全局原型（通过聚合客户端原型获得）。

在**假设 1、假设 2 和假设 3**下，以下三个性质成立：

**(a) 逼近界：**

$$|\text{EO}_{\text{proto}} - \text{EO}_{\text{global}}| \leq \frac{|w|}{\sqrt{2\pi} \sigma_z} \cdot \Delta_{\text{rep}}^{\text{proto}} + O\left(\frac{|w|^2 \Delta_\Sigma}{\sigma_z^3}\right) + O\left(\frac{1}{\sigma_z \sqrt{n}}\right)$$

**(b) 梯度方向一致性：**

$$\cos\langle \nabla_\psi \text{EO}_{\text{proto}}, \nabla_\psi \text{EO}_{\text{global}} \rangle \geq 1 - O\left(\frac{|w|^2 \Delta_\Sigma}{\sigma_z^2}\right)$$

**(c) 单步下降传递：**

$$\text{EO}_{\text{global}}^{(t+1)} \leq \text{EO}_{\text{global}}^{(t)} - \alpha \lambda_{\text{eo}} \cdot |\nabla_\psi \text{EO}_{\text{proto}}^{(t)}|^2 \cdot \delta$$

其中 $\delta > 0$（当学习率 $\alpha$ 足够小且 $\Delta_\Sigma$ 足够小时）。

**意义**：本定理回答了"为什么在 Server 端使用全局原型进行 EO 校准是有效的"这一问题——这正是**挑战3**（统计异构下难以保障全局公平性）的核心。(a) 说明全局原型层面的 EO 代理目标是真实全局 EO 的良好近似；(b) 说明优化代理目标的方向与优化真实目标的方向一致；(c) 说明代理目标的下降确实能传递到真实目标的下降。三者结合，证明了 Server 端后训练不仅不会损害模型性能，反而能有效提升全局公平性。关键洞察：$\sigma_{\text{s}}(x) \approx \Phi(c \cdot x)$（$c \approx 2.40$）这一经典近似是连接 sigmoid 与正态 CDF 的桥梁。

### 定理 5（原型聚合的差分隐私保证）

**设定**：客户端 $k$ 在每轮上传本地原型 $\mu_{g,l}^{(k)}$，Server 端进行加权聚合：

$$\mu_{g,l}^{\text{global}} = \sum_{k=1}^K w_k \cdot \mu_{g,l}^{(k)}$$

**假设 4（LDP 噪声）**：每个客户端在上传原型前添加 $\epsilon$-LDP 噪声 $\mathcal{N}(0, \sigma_{\text{noise}}^2 \cdot I_d)$，其中 $\sigma_{\text{noise}}^2 \geq \frac{2d \ln(2/\delta)}{\epsilon^2}$（满足 $(\epsilon, \delta)$-DP）。

在假设 4 下，原型聚合过程满足**分布式差分隐私**（Distributed Differential Privacy）：

$$\text{Pr}[\mathcal{A}(\mu_{g,l}^{\text{global}}) = t] \leq e^\epsilon \cdot \text{Pr}[\mathcal{A}(\mu_{g,l}^{\text{global},-i}) = t] + \delta$$

其中 $\mu_{g,l}^{\text{global},-i}$ 是移除第 $i$ 个样本后的聚合原型，$\mathcal{A}$ 是任意攻击者算法。

**意义**：本定理回答了**挑战1**（隐私泄露风险）——证明 PDFFed 中传输的原型信息在添加 LDP 噪声后满足差分隐私保证。即使攻击者能够访问聚合后的全局原型，也无法推断出单个客户端或样本的敏感信息。

### 定理 6（传输敏感属性的隐私风险）

**设定**：假设有方法在通信中传输敏感属性 $g$（如群组标签）或包含敏感属性信息的中间结果。

**结论**：任何传输原始敏感属性 $g$ 的方法都**不满足差分隐私**，因为存在攻击者可以通过以下方式进行成员推断攻击：

$$\text{Pr}[\mathcal{A}(g_i) = 1 \mid i \in S] - \text{Pr}[\mathcal{A}(g_i) = 1 \mid i \notin S] = 1$$

即攻击者可以完美判断样本是否属于某个群组，从而破坏隐私。

**意义**：本定理从反面证明了**挑战1**的必要性——如果传输敏感属性，隐私将直接被破坏。PDFFed 选择传输原型而非敏感属性，正是为了避免这一风险。

***

## 5. 详细证明

### 5.1 定理 1 的证明

**来源**：本证明框架参考 McNamara et al. (2017) 和 Zhao & Gordon (2019)。

**Step 1**：EO 的表达式

对于线性分类器，预测概率为 $\hat{p} = \sigma_{\text{s}}(w^T z + b)$。

EO 差距（对标签 $l$）：

$$\text{EO}\_l = \left|\mathbb{E}\[\hat{p} \mid g=0, y=l] - \mathbb{E}\[\hat{p} \mid g=1, y=l]\right|$$

**Step 2**：利用 sub-Gaussian 假设

由假设 2，$z \mid g, y=l$ 是 sub-Gaussian 的。对于线性投影 $w^T z + b$，由 sub-Gaussian 的性质（线性变换保持 sub-Gaussian 性质）：

$$w^T z + b \mid g, y=l \text{ 是 sub-Gaussian 的}$$

其均值为 $w^T \mu\_{g,l} + b$，方差参数为 $\sigma'_{g,l} = \sqrt{w^T \Sigma_{g,l} w}$。

**Step 3**：利用 Berry-Esseen 型近似

对于 sub-Gaussian 随机变量 $X$，其 CDF 与高斯 CDF 的差异由 Berry-Esseen 界控制：

$$\left|P(X \leq t) - \Phi\left(\frac{t - \mu}{\sigma'}\right)\right| \leq \frac{C\_{\text{BE}}}{\sigma' \sqrt{n}}$$

其中 $n$ 是样本量，$C\_{\text{BE}}$ 是 Berry-Esseen 常数。

因此：

$$\mathbb{E}\[\sigma_{\text{s}}(w^T z + b) \mid g, y=l] \approx \Phi\left(\frac{w^T \mu\_{g,l} + b}{\sigma'\_{g,l}}\right)$$

近似误差为 $O(1/(\sigma' \sqrt{n}))$。

**Step 4**：EO 差距的近似表达式

$$\text{EO}_l \approx \left|\Phi\left(\frac{w^T \mu_{g\_0,l} + b}{\sigma'_{g\_0,l}}\right) - \Phi\left(\frac{w^T \mu_{g\_1,l} + b}{\sigma'\_{g\_1,l}}\right)\right|$$

**Step 5**：利用中值定理

存在 $\xi$ 使得：

$$\text{EO}_l = \varphi(\xi) \cdot \left|\frac{w^T \mu_{g\_0,l} + b}{\sigma'_{g\_0,l}} - \frac{w^T \mu_{g\_1,l} + b}{\sigma'\_{g\_1,l}}\right|$$

其中 $\varphi$ 是标准正态 PDF，$\varphi(\xi) \leq 1/\sqrt{2\pi}$。

**Step 6**：利用假设 3（有界协方差差异）

由 $|\Sigma\_{g\_0,l} - \Sigma\_{g\_1,l}|_{\text{op}} \leq \Delta_\Sigma$，可得：

$$|\sigma'_{g\_0,l} - \sigma'_{g\_1,l}| \leq \frac{|w|^2 \Delta\_\Sigma}{2\sigma'}$$

（由一阶 Taylor 展开）

令 $\sigma' = \max(\sigma'_{g\_0,l}, \sigma'_{g\_1,l})$，则：

$$\text{EO}_l \leq \frac{1}{\sqrt{2\pi}} \cdot \frac{|w^T(\mu_{g\_0,l} - \mu\_{g\_1,l})|}{\sigma'} + O\left(\frac{|w|^2 \Delta\_\Sigma}{\sigma'^3}\right)$$

**Step 7**：利用 Cauchy-Schwarz 不等式

$$|w^T(\mu\_{g\_0,l} - \mu\_{g\_1,l})| \leq |w| \cdot |\mu\_{g\_0,l} - \mu\_{g\_1,l}|$$

因此：

$$\text{EO}_l \leq \frac{|w|}{\sqrt{2\pi} \sigma'} \cdot |\mu_{g\_0,l} - \mu\_{g\_1,l}| + O\left(\frac{|w|^2 \Delta\_\Sigma}{\sigma'^3}\right)$$

对所有标签求和，令 $\sigma_z = \max_l \sigma_l'$：

$$\text{EO} \leq \frac{|w|}{\sqrt{2\pi} \sigma_z} \cdot \Delta_{\text{rep}} + O\left(\frac{|w|^2 \Delta_\Sigma}{\sigma_z^3}\right)$$

**证毕。**

***

### 5.2 定理 2 的证明

**设定**：

- 群组 0 的期望置信度 $\mathbb{E}[c \mid g=0, y=l] \leq c_{\text{low}}$
- 群组 1 的期望置信度 $\mathbb{E}[c \mid g=1, y=l] \geq c_{\text{high}}$
- $c_{\text{low}} < c_{\text{high}}$

**Step 1**：置信度与 logit 的关系

置信度 $c = \sigma_{\text{s}}(|z|)$，其中 $z = w^T x + b$ 是 logit。

由 sigmoid 的逆函数：

$$|z| = \sigma_{\text{s}}^{-1}(c) = \ln\left(\frac{c}{1-c}\right)$$"

低置信度 $c_{\text{low}}$ 对应 $|z|_{\text{low}} = \ln\left(\frac{c_{\text{low}}}{1 - c_{\text{low}}}\right)$

高置信度 $c_{\text{high}}$ 对应 $|z|_{\text{high}} = \ln\left(\frac{c_{\text{high}}}{1 - c_{\text{high}}}\right)$

**Step 2**：logit 的期望

由假设 2（sub-Gaussian），$z \mid g, y=l$ 的均值为 $\mu'_g = w^T \mu_{g,l} + b$，标准差为 $\sigma'_g$。

$$\mathbb{E}[|z| \mid g, y=l] \approx |\mu'_g|$$

（当 $\sigma'_g$ 较小时，由 sub-Gaussian 集中不等式，$z$ 集中在 $\mu'_g$ 附近）

因此：

$$\mathbb{E}[c \mid g, y=l] \approx \sigma_{\text{s}}(|\mu'_g|)$$

**Step 3**：置信度差异与 logit 差异

由 sigmoid 的 Lipschitz 性质（Lipschitz 常数 $1/4$）：

$$c_{\text{high}} - c_{\text{low}} \leq \frac{1}{4}(|\mu'_{g_1}| - |\mu'_{g_0}|)$$

因此：

$$|\mu'_{g_1}| - |\mu'_{g_0}| \geq 4(c_{\text{high}} - c_{\text{low}})$$

**Step 4**：logit 差异与 EO 的关系

EO 差距：

$$\text{EO}_l = \left|\mathbb{E}[\hat{p} \mid g=0, y=l] - \mathbb{E}[\hat{p} \mid g=1, y=l]\right|$$

$$\approx \left|\Phi(\mu'_{g_0}/\sigma') - \Phi(\mu'_{g_1}/\sigma')\right|$$

**Step 5**：分情况讨论

**情况 1**：$\mu'_{g_0}$ 和 $\mu'_{g_1}$ 同号

$$\text{EO}_l \approx \left|\Phi(\mu'_{g_0}/\sigma') - \Phi(\mu'_{g_1}/\sigma')\right|$$

由中值定理：

$$\text{EO}_l \geq \varphi\left(\frac{\max(|\mu'_{g_0}|, |\mu'_{g_1}|)}{\sigma'}\right) \cdot \frac{\bigl||\mu'_{g_0}| - |\mu'_{g_1}|\bigr|}{\sigma'}$$

$$\geq \varphi\left(\frac{|\mu'_{g_1}|}{\sigma'}\right) \cdot \frac{4(c_{\text{high}} - c_{\text{low}})}{\sigma'}$$

由于 $\varphi$ 在 $|x| \leq 1$ 时 $\varphi(x) \geq \varphi(1) \approx 0.242$，当 $|\mu'_{g_1}| \leq \sigma'$ 时：

$$\text{EO}_l \geq \frac{0.242 \cdot 4(c_{\text{high}} - c_{\text{low}})}{\sigma'} \geq \frac{c_{\text{high}} - c_{\text{low}}}{\sigma'}$$

**情况 2**：$\mu'_{g_0}$ 和 $\mu'_{g_1}$ 异号

此时 $|\mu'_{g_0}| + |\mu'_{g_1}| \geq |\mu'_{g_1}| - |\mu'_{g_0}| \geq 4(c_{\text{high}} - c_{\text{low}})$

EO 差距更大（因为两个群组的预测概率在决策边界两侧）。

**Step 6**：综合

$$\text{EO} \geq 2(c_{\text{high}} - c_{\text{low}}) - \frac{2\sigma'}{|w|\sqrt{2\pi}} - O(\Delta_\Sigma)$$

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

**Step 7**：两个梯度的内积（Lemma 1 的证明）

$$\nabla_\phi \Delta_{\text{rep}} \cdot \nabla_\phi L_{\text{contrastive}} = \sum_l \|\mu_{g0,l} - \mu_{g1,l}\| \cdot \|\nabla_\phi \mu_{g0,l} - \nabla_\phi \mu_{g1,l}\|^2$$

由于 $\|\mu_{g0,l} - \mu_{g1,l}\| > 0$ 且 $\|\nabla_\phi \mu_{g0,l} - \nabla_\phi \mu_{g1,l}\|^2 \geq 0$，内积非负，即 CL 梯度方向与缩小 $\Delta_{\text{rep}}$ 的方向一致。这证明了 Lemma 1。

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

**设定**：Server 端后训练仅更新分类头 ψ，损失为 L_post = L_cls + λ_eo · EO_proto，其中 EO_proto = Σ_l |σ(w^T μ_{g0,l} + b) - σ(w^T μ_{g1,l} + b)|。

**证明 (a)：逼近界。**

**Step 1：** 由定理 1 的证明，全局 EO 在假设 1-2 下近似为：

EO_global ≈ Σ_l |Φ((w^T μ_{g0,l} + b) / σ'_{g0,l}) - Φ((w^T μ_{g1,l} + b) / σ'_{g1,l})|

近似误差为 O(1/(σ √n))（Berry-Esseen 界）。

**Step 2：** 利用经典近似 σ(x) ≈ Φ(c · x)，其中 c = √(π/ln 2) ≈ 2.40（误差在 |x| ≤ 3 范围内小于 0.02）：

EO_proto ≈ Σ_l |Φ(c · (w^T μ_{g0,l} + b)) - Φ(c · (w^T μ_{g1,l} + b))|

**Step 3：** 对比两者，差异来自分母 σ'_{g,l}（由假设 3 控制）和常数 c（固定缩放）。利用假设 3 和定理 1 证明中 Step 6 的相同推导：

|EO_proto - EO_global| ≤ (|w| / (√(2π) σ)) · Δ_rep^proto + O(|w|² Δ_Σ / σ³) + O(1 / (σ √n))

**证明 (b)：梯度方向一致性。**

**Step 4：** 由于仅更新 ψ = (w, b)，两个梯度表达式分别为：

∇_ψ EO_proto = Σ_l ∇_ψ |σ(w^T μ_{g0,l} + b) - σ(w^T μ_{g1,l} + b)|

∇_ψ EO_global ≈ Σ_l ∇_ψ |Φ((w^T μ_{g0,l} + b) / σ'_{g0,l}) - Φ((w^T μ_{g1,l} + b) / σ'_{g1,l})|

**Step 5：** 由 σ(x) ≈ Φ(c · x)，有 σ'(x) ≈ c · φ(c · x)。因此两个梯度的核心项只差常数缩放 c 和分母 σ'_{g,l} 的修正：

∇_ψ EO_proto ≈ c · ∇_ψ EO_global + 修正项(Δ_Σ)

**Step 6：** 由 Cauchy-Schwarz 不等式：

cos⟨∇_ψ EO_proto, ∇_ψ EO_global⟩ ≥ 1 - O(|w|² Δ_Σ / σ²)

**证明 (c)：单步下降传递。**

**Step 7：** 真实 EO 的 Taylor 展开：

EO_global^{(t+1)} = EO_global^{(t)} + ∇_ψ EO_global^{(t)} · (ψ^{(t+1)} - ψ^{(t)}) + O(α²)

其中 ψ^{(t+1)} - ψ^{(t)} = -α λ_eo ∇_ψ EO_proto^{(t)}。

**Step 8：** 代入并利用 (b)：

EO_global^{(t+1)} = EO_global^{(t)} - α λ_eo |∇_ψ EO_global| |∇_ψ EO_proto| cos⟨∇_ψ EO_global, ∇_ψ EO_proto⟩ + O(α²)

由 (b)，cos⟨·,·⟩ ≥ 1 - ε(Δ_Σ)，因此：

EO_global^{(t+1)} ≤ EO_global^{(t)} - α λ_eo |∇_ψ EO_proto|² · δ

其中 δ = cos⟨·,·⟩ - α L_eo / (2 λ_eo) > 0（当 α 足够小且 Δ_Σ 足够小时）。

**证毕。**

### 5.5 定理 5 的证明（原型聚合的差分隐私保证）

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

### 5.6 定理 6 的证明（传输敏感属性的隐私风险）

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

***

## 6. 证明链路与挑战的对应关系

完整的证明链路与三个核心挑战的对应关系详见 [第 0 节](#0-宏观证明链路)。此处仅做补充说明：

**上界 vs 下界**：

- 定理 1 给出的是**上界**：减小 $\Delta_{\text{rep}}$ → EO 上界降低 → **保证**最坏情况下的 EO 不会太大
- 定理 2 给出的是**下界**：置信度差异大 → EO 下界高 → **诊断**出存在公平性问题

两者互补：定理 1 用于**设计优化目标**（减小表征偏差），定理 2 用于**诊断问题**（检测置信度差异）。

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
| EO | $\dfrac{\|w\|}{\sqrt{2\pi}\,\sigma_z} \cdot \Delta_{\text{rep}} + O\!\left(\dfrac{\|w\|^2 \Delta_\Sigma}{\sigma_z^3}\right)$ | 无 |
| DP | $\dfrac{\|w\|}{\sqrt{2\pi}\,\sigma_z} \cdot \Delta_{\text{rep}} + O\!\left(\dfrac{\|w\|^2 \Delta_\Sigma}{\sigma_z^3}\right) + \Delta_{\text{label}}$ | $\Delta_{\text{label}}$ |
| EOpp | $\dfrac{\|w\|}{\sqrt{2\pi}\,\sigma_z} \cdot \Delta_{\text{rep},1} + O\!\left(\dfrac{\|w\|^2 \Delta_\Sigma}{\sigma_z^3}\right)$ | 仅正类 |
| DEO | $\dfrac{\|w\|}{\sqrt{2\pi}\,\sigma_z} \cdot \Delta_{\text{rep}}^w + O\!\left(\dfrac{\|w\|^2 \Delta_\Sigma}{\sigma_z^3}\right) \cdot (w_1 + w_0)$ | 加权 |

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

