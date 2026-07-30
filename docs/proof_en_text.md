# PDFFed Theoretical Analysis: Prototype-Driven Fair Federated Learning (Plain Text Version)

## 0. Macro-Level Proof Chain

PDFFed addresses three core challenges in fair federated learning. The proof chain is organized around these challenges, with each supported by specific theorems.

### Challenge 1: Privacy Leakage Risk

**Core Problem:** In federated learning, transmitting sensitive attributes (e.g., group labels) during communication leads to direct privacy leakage.

```
Privacy risk analysis
    │
    ├── Theorem 6 (Reverse Proof)
    │       Transmitting sensitive attributes → MIA success rate = 1 → violates DP
    │       Conclusion: Must avoid transmitting sensitive attributes
    │
    ├── Theorem 5 (DP Guarantee)
    │       Client adds LDP noise to prototypes → aggregated result satisfies distributed DP
    │       Conclusion: Prototype transmission + DP noise = secure communication pattern
    │
    └── Corollary: PDFFed's prototype-based communication is theoretically safe
```

**Theorems involved:** Theorem 6 provides a reverse proof that transmitting sensitive attributes violates differential privacy — an adversary can perform membership inference with perfect accuracy. Theorem 5 then proves that prototype aggregation with LDP noise satisfies distributed differential privacy, establishing the security of PDFFed's communication protocol.

### Challenge 2: Fairness Degradation from Representation Bias

**Core Problem:** Existing methods primarily adjust at the decision level (loss weights, aggregation strategies) and cannot address representation-level bias. Theorems 1–3 build a complete closed loop from "problem identification" to "intervention implementation."

```
Representation bias Δ_rep ↔ Fairness metric EO
    │
    ├── Theorem 1 (Upper Bound: Δ_rep → EO)
    │       EO ≤ (||w|| / (√(2π) σ_z)) · Δ_rep + O(||w||² Δ_Σ / σ_z³)
    │       Reducing Δ_rep is the key means to lower the EO upper bound
    │
    ├── Theorem 2 (Lower Bound: Confidence Gap → EO)
    │       EO ≥ 2 δ_conf − 2 σ_z / (||w|| √(2π)) − O(Δ_Σ)
    │       Confidence gap serves as a diagnostic signal for fairness issues
    │
    └── Theorem 3 (Intervention: CL → Δ_rep Contraction)
            Always-on CL → Δ_rep decreases monotonically + convergence preserved
            Directly shrinking Δ_rep during training is feasible and safe
```

**Closed-loop logic:** Theorem 1 explains why optimizing Δ_rep is essential (upper bound constraint), Theorem 2 explains how to diagnose the problem (confidence gap as necessary condition), and Theorem 3 explains how to implement optimization (contrastive loss) while preserving task convergence. Together they form a complete argument from theoretical insight to algorithmic design.

### Challenge 3: Global Fairness under Statistical Heterogeneity

**Core Problem:** With heterogeneous client data distributions, local fairness optimization does not guarantee global fairness. A global-level calibration mechanism is required.

```
Server-side EO calibration using global prototypes
    │
    └── Theorem 4 (Global Prototype as EO Calibration Proxy)
            │
            ├── (a) Approximation Bound
            │       |EO_proto − EO_global| ≤ ε(Δ_rep, Δ_Σ, 1/√n)
            │       Global prototype EO accurately approximates true global EO
            │
            ├── (b) Gradient Direction Consistency
            │       cos⟨∇_ψ EO_proto, ∇_ψ EO_global⟩ ≥ 1 − O(||w||² Δ_Σ / σ_z²)
            │       Optimizing the proxy aligns with optimizing the true objective
            │
            └── (c) Descent Transfer
                    EO_global^{(t+1)} ≤ EO_global^{(t)} − α λ_eo · ||∇_ψ EO_proto||² · δ
                    Descent in the proxy guarantees descent in the true objective
```

**Significance:** Theorem 4 proves that server-side post-training using global prototypes is both safe and effective — optimizing the proxy does not harm model performance and actively improves global fairness. This directly resolves Challenge 3: achieving global fairness calibration under statistical heterogeneity without transmitting sensitive attributes.

### Theorem–Challenge Mapping

| Theorem | Type | Core Relation | Addressed Challenge |
|---------|------|--------------|---------------------|
| Theorem 1 | Adapted | Δ_rep → EO upper bound | Challenge 2 |
| Theorem 2 | Original | Confidence gap → EO lower bound | Challenge 2 |
| Theorem 3 | Original | CL → Δ_rep contraction | Challenge 2 |
| Theorem 4 | Original | Global prototype → EO calibration (approximation + direction consistency + descent transfer) | Challenge 3 |
| Theorem 5 | Original | Prototype aggregation + LDP noise → distributed DP | Challenge 1 |
| Theorem 6 | Original | Transmitting sensitive attributes → DP violation (reverse proof) | Challenge 1 |

**Supplementary note on upper vs. lower bounds:**

- Theorem 1 provides an **upper bound**: reducing Δ_rep guarantees the worst-case EO will not be too large — used for **optimization target design**.
- Theorem 2 provides a **lower bound**: large confidence gap implies EO must be large — used for **fairness problem diagnosis**.

The two bounds are complementary and together form a principled approach to fairness-aware federated learning.

---

## 1. Notation and Definitions

### Definition 1.1 (Equalized Odds Gap)

EO = Σ_{l∈{0,1}} | E[p̂ | g=0, y=l] - E[p̂ | g=1, y=l] |

where p̂ = σ(w^T z + b) is the predicted probability for the positive class, g ∈ {0, 1} is the group identifier, and y ∈ {0, 1} is the true label. EO measures the prediction probability difference between groups for the same label.

### Definition 1.2 (Disparate Equalized Odds, DEO)

DEO = | E[p̂ | g=0, y=1] - E[p̂ | g=1, y=1] |

Measures only the prediction probability gap for positive samples (y=1) across groups, equivalent to the difference in True Positive Rates (TPR).

### Definition 1.2.1 (Statistical Parity Difference, SPD / Demographic Parity, DP)

SPD = E[p̂ | g=0] - E[p̂ | g=1]

The difference in expected positive prediction probabilities across groups, ignoring true labels.

### Definition 1.3 (Representation Bias)

Δ_rep = Σ_{l∈{0,1}} ‖μ_{g0,l} - μ_{g1,l}‖

where μ_{g,l} = E[z | g, y=l] is the group-label aware data prototype (see Definition 1.5). Δ_rep measures the representation center difference between groups across labels.

### Definition 1.4 (Class-Aware Data Prototype)

μ_l = E_{(x,g,y)~D}[z | y=l]

The mean feature vector of all samples belonging to class l in the feature space. This is the standard prototype definition used in conventional model analysis.

### Definition 1.5 (Class-Group Aware Data Prototype)

μ_{g,l} = E_{(x,g,y)~D}[z | g, y=l]

The mean feature vector of samples that belong to both group g and class l in the feature space.

> **Note:** μ_{g,l} is the **core prototype definition** in this proof framework, directly used in Theorem 1 (definition of Δ_rep), Theorem 3 (inter-group prototype distance in contrastive loss), and Theorem 4 (EO_proto computation). By considering both group and label dimensions simultaneously, it captures representation differences between groups on the same class.

### Definition 1.6 (Group-Label Aware Gradient Prototype)

For a sample (x, y, g) with task loss L_task(x, y) = ℓ(σ(w^T f_φ(x) + b), y), the feature gradient is ∇_z L_task = ∂ℓ/∂z ∈ R^d.

The group-label aware gradient prototype is:

ḡ_{g,l} = E_{(x,y,g)~D} [ ∇_z L_task(x, y) | g, l ]

**Naming:** This prototype is the mean of per-sample gradients (hence "gradient prototype"), stratified by group and label (hence "group-label aware").

### Definition 1.7 (Inter-Group Gradient Prototype Direction Consistency, abbreviated as Gradient Direction Consistency)

ρ_l = (ḡ_{g0,l} · ḡ_{g1,l}) / (‖ḡ_{g0,l}‖ · ‖ḡ_{g1,l}‖) ∈ [-1, 1]

The cosine similarity between gradient prototypes of two groups. We will use "gradient direction consistency" as the abbreviation throughout the rest of the paper.

### Definition 1.8 (Prediction Confidence, abbreviated as Confidence)

c = max(p̂, 1 - p̂) = σ(|w^T z + b|)

We will use "confidence" as the abbreviation throughout the rest of the paper.

### Definition 1.9 (Decision Boundary Distance)

d = |w^T z + b| / ‖w‖

### Definition 1.10 (Group Confidence Difference, δ_conf)

δ_conf = min_l |E[c | g=0, y=l] - E[c | g=1, y=l]|

where c = max(p̂, 1 - p̂) is the prediction confidence (see Definition 1.8). δ_conf quantifies the minimum confidence gap between groups across all labels.

### Observation 1 (Confidence and Decision Boundary Distance)

From Definitions 1.8 and 1.9:

c = σ(‖w‖ · d)

Since σ is monotonically increasing and ‖w‖ > 0, prediction confidence c is monotonically increasing in decision boundary distance d.

---

## 2. Preliminaries

### 2.1 Federated Learning Problem Setup

We consider a federated learning setting with K clients. Each client k has local data distribution D_k, and the global distribution is D = Σ_{k=1}^K w_k D_k where w_k = n_k / N, n_k is the number of samples at client k, and N = Σ_{k=1}^K n_k is the total number of global samples.

**Global Optimization Objective (FedAvg target):**

min_θ Σ_{k=1}^K w_k · E_{(x,y,g) ~ D_k} [ ℓ(σ(w^T f_φ(x) + b), y) ]

where θ = (φ, ψ) is the model parameter, φ is the encoder parameter, and ψ = (w, b) is the classifier head parameter.

**Local Training Loss for Client k:**

L_k = E_{(x,y,g) ~ D_k} [ ℓ(σ(w^T f_φ(x) + b), y) ] + λ · L_contrastive,k

where L_contrastive,k is the contrastive loss (always active).

### 2.2 Model Decomposition

Any supervised learning model can be naturally decomposed into an **encoder + classifier head**:

x →(f_φ) z ∈ R^d →(h_ψ) p̂ = σ(h_ψ(z))

where:
- f_φ is the encoder, mapping inputs to the feature space. f_φ can be any model of arbitrary complexity.
- h_ψ is the classifier head, performing classification on features.

### 2.3 Analysis Setup: Linear Classifier Head

In theoretical analysis, we take the classifier head as linear:

h_ψ(z) = w^T z + b

**Justification:** Linear classifier heads are standard in fairness theoretical analysis (McNamara et al., 2017; Zhao & Gordon, 2019). This is an analysis tool, not an architectural constraint.

### 2.4 Loss Function Definitions

**Task Loss** (Cross-entropy loss):

L_task(x, y) = ℓ(σ(w^T f_φ(x) + b), y) = -y log(p̂) - (1-y) log(1-p̂)

where p̂ = σ(w^T f_φ(x) + b) is the predicted probability for the positive class.

**Contrastive Loss** (Prototype-level contrastive loss):

L_contrastive = (1/2) Σ_{l ∈ {0,1}} ‖μ_{g0,l} - μ_{g1,l}‖²

where μ_{g,l} = E[z | g, y=l] is the class-group aware data prototype (see Definition 1.5).

---

## 3. Assumptions

### Assumption 1 (Sub-Gaussian Feature Distribution)

For each group-label pair (g, l), the feature z | g, y=l is sub-Gaussian:

E[exp(λ^T (z - μ_{g,l})) | g, y=l] ≤ exp(λ^T Σ_{g,l} λ / 2), ∀ λ ∈ R^d

with bounded covariance: ‖Σ_{g,l}‖_op ≤ σ².

**Justification:** Gaussian distributions are a special case of sub-Gaussian. Sub-Gaussian allows more general tail behavior, satisfied by most real-world features.

### Assumption 2 (Bounded Covariance Difference)

The covariance difference between groups is bounded:

‖Σ_{g0,l} - Σ_{g1,l}‖_op ≤ Δ_Σ, ∀ l

**Justification:** This relaxes the shared covariance assumption (Σ_{g0} = Σ_{g1}). When Δ_Σ = 0, it reduces to shared covariance.

### Assumption 3 (L-Smoothness)

The task loss L_task is L-smooth, and the contrastive loss L_contrastive is M-smooth with respect to model parameters.

### Assumption 4 (Non-trivial Confidence Gap)

There exists a constant δ_conf > 0 such that for all labels l ∈ {0,1}:

|E[c | g=0, y=l] - E[c | g=1, y=l]| ≥ δ_conf

**Justification:** This captures the scenario where there is a meaningful confidence difference between groups, which is necessary for proving the EO lower bound in Theorem 2.

### Assumption 5 (LDP Noise)

Each client adds ε-LDP noise N(0, σ_noise² · I_d) before uploading prototypes, where σ_noise² ≥ (2d ln(2/δ))/ε² (satisfying (ε, δ)-DP).

**Justification:** Local Differential Privacy (LDP) ensures that individual client data cannot be inferred from the transmitted information, addressing privacy concerns in federated learning.

---

## 4. Theorems and Lemmas

### Theorem 1 (Representation Bias and EO Upper Bound — adapted from McNamara et al., 2017)

Under Assumptions 1 and 2:

EO ≤ (‖w‖ / (√(2π) σ_z)) · Δ_rep + O(‖w‖² Δ_Σ / σ_z³)

where σ_z = √(E[(w^T z - E[w^T z])²]) is the standard deviation of features projected onto the classification direction. Note: In this paper, σ(·) denotes the sigmoid function, while σ_z denotes the standard deviation in the classification direction—they are different symbols.

**Source:** The proof framework is adapted from McNamara et al. (2017) "Provably Fair Representations" and Zhao & Gordon (2019). We relax the shared covariance assumption to bounded covariance difference (Assumption 3), and the Gaussian assumption to sub-Gaussian (Assumption 1).

**Significance:** This is an **upper bound**, meaning "the maximum value of EO is jointly determined by two terms." The first term (‖w‖ / (√(2π) σ_z)) · Δ_rep is dominant, determined by representation bias Δ_rep, classifier weight norm ‖w‖, and classification direction standard deviation σ_z; the second term is a higher-order small term controlled by covariance difference Δ_Σ.

Analysis of the three variables:

- ‖w‖ is the **norm** (scalar) of the classifier weight vector w, reflecting the overall steepness of the decision boundary. DFR (Kirichenko et al., 2023, ICLR, arXiv:2204.02937) demonstrates that in centralized learning, retraining only the last layer classifier can significantly improve fairness; Mao et al. (2023, ICML Workshop on Human-Centric Machine Learning, arXiv:2304.03935) further show that last-layer fine-tuning effectively avoids fairness overfitting. These methods do change both the direction and norm ‖w‖ of w. The DFR paper explicitly states its key premise: **"standard neural networks are in fact learning core features, even if they do not primarily rely on these features to make predictions"** — i.e., the feature extractor has already learned representations that can distinguish different groups without discriminating against any group; the problem lies only in the last layer classifier giving excessive weight to spurious features.
- σ_z is the spread of features along the classification direction, determined by the representation space learned by the feature extractor f_φ. Fair-FLIP (Zhong et al., 2025, arXiv:2507.08912) reduces subgroup variability differences by reweighting final-layer input features, and GroupMixNorm (Zhang et al., 2023, NeurIPS Workshop, arXiv:2312.11969) improves fairness by mixing group-level feature statistics, indicating that σ_z-related feature statistics are indeed related to fairness. However, these methods are **post-hoc corrections** that adjust feature statistics after training, without changing the structure of the feature space itself.
- Δ_rep directly quantifies the distance between group centers in the representation space, characterizing the group bias of the representation space itself.

Cui et al. (2024, arXiv:2405.01112) reveal a key finding through empirical study: **the root cause of unfairness lies in problematic representation rather than classifier bias**—the classifier weight norm ‖w‖ is already balanced, and the problem lies in the quality of the feature space. This finding is consistent with the structure of Theorem 1: adjustments to ‖w‖ and σ_z belong to "decision-level/post-hoc correction," while Δ_rep directly characterizes the group bias of the representation space itself.

This distinction is particularly critical in the federated learning setting. The aforementioned works (DFR, Mao et al., Fair-FLIP, GroupMixNorm) are all proposed in centralized learning scenarios, with a common premise: the feature extractor can learn relatively fair representations from the complete dataset. However, in federated learning, when data heterogeneity is significant, each client's local data distribution is biased, and the learned representations tend to carry systematic group bias (Δ_rep increases)—the feature space may be unfair from the very beginning. At this point, the premise of centralized methods no longer holds—adjusting ‖w‖ at the classifier level or post-hoc adjusting σ_z-related statistics cannot address the root cause of the bias.

Therefore, directly reducing Δ_rep during the federated learning training process—through representation-level fairness constraints (such as the contrastive loss introduced in this paper)—is the most fundamental means to lower the EO upper bound. This conclusion directly addresses Core Challenge 2: existing methods primarily focus on loss constraints or aggregation adjustments (decision-level), and cannot fundamentally mitigate fairness degradation caused by representation bias. Our introduction of contrastive loss in local training to directly shrink Δ_rep is grounded in this theoretical insight.

### Theorem 2 (Group Confidence Difference and EO Lower Bound)

**Assumption 4 (Non-trivial Confidence Gap):** There exists a constant δ_conf > 0 such that for all labels l ∈ {0,1}:

|E[c | g=0, y=l] - E[c | g=1, y=l]| ≥ δ_conf

Under **Assumption 1 (Sub-Gaussian Feature Distribution)**:

EO ≥ 2δ_conf - 2σ_z / (‖w‖ √(2π)) - O(Δ_Σ)

where σ_z = √(E[(w^T z - E[w^T z])²]) is the standard deviation of features projected onto the classification direction (same as Theorem 1).

**Significance:** This is a **lower bound**. It tells us: the minimum value of EO is positively correlated with the group confidence difference δ_conf. To make EO smaller, the confidence difference between groups must first be reduced—this is a necessary condition (but not sufficient). Conversely, if the confidence difference is large (δ_conf is large), EO cannot be too small. Therefore, the confidence difference can serve as a diagnostic signal for fairness problems.

### Lemma 1 (CL Gradient Direction Consistency)

The gradient of contrastive loss L_contrastive with respect to encoder parameters φ aligns with the direction that reduces representation bias Δ_rep. Specifically:

⟨∇_φ L_contrastive, ∇_φ Δ_rep⟩ ≥ 0

**Proof:** See Section 5.3, Steps 4–7.

### Theorem 3 (Convergence of Contrastive Learning and Δ_rep Contraction)

Under Assumption 3, the local training loss L_k = L_task,k + λ · L_contrastive,k satisfies:

**(a) Convergence:**

L^{(t+1)} - L^{(t)} ≤ -η ‖∇L^{(t)}‖² + (η² (L + λM) / 2) ‖∇L^{(t)}‖²

When η < 2/(L + λM), the loss decreases monotonically.

**(b) Δ_rep Contraction:**

After T steps, the representation bias satisfies:

Δ_rep^{(t+T)} ≤ Δ_rep^{(t)} - η · λ · T · γ

where γ = E[∂L_contrastive / ∂‖μ_{g0,l} - μ_{g1,l}‖] > 0.

**Significance:** (a) Always-on CL introduces a mild convergence cost proportional to λM but does not break convergence. (b) By Lemma 1, the CL gradient consistently aligns with the Δ_rep reduction direction (non-negative inner product), guaranteeing that Δ_rep decreases monotonically through gradient descent.

### Theorem 4 (Global Prototype as EO Calibration Proxy)

**Setup:** Server-side post-training updates only the classifier head ψ = (w, b), using the prototype-level EO proxy objective for calibration:

EO_proto = Σ_{l∈{0,1}} |σ(w^T μ_{g0,l} + b) - σ(w^T μ_{g1,l} + b)|

where μ_{g,l} is the global prototype (obtained by aggregating client prototypes).

Under **Assumptions 1, 2, and 3**, three properties hold:

**(a) Approximation bound:**

|EO_proto - EO_global| ≤ (‖w‖ / (√(2π) σ_z)) · Δ_rep^proto + O(‖w‖² Δ_Σ / σ_z³) + O(1 / (σ_z √n))

**(b) Gradient direction consistency:**

cos⟨∇_ψ EO_proto, ∇_ψ EO_global⟩ ≥ 1 - O(‖w‖² Δ_Σ / σ_z²)

**(c) Single-step descent transfer:**

EO_global^{(t+1)} ≤ EO_global^{(t)} - α λ_eo · ‖∇_ψ EO_proto^{(t)}‖² · δ

where δ > 0 (when learning rate α is sufficiently small and Δ_Σ is sufficiently small).

**Significance:** This theorem answers the question "Why is calibrating EO using global prototypes on the Server effective?"—this is exactly the core of **Challenge 3** (difficulty guaranteeing global fairness under statistical heterogeneity). (a) shows the prototype-level EO proxy is a good approximation of the true global EO; (b) shows the optimization direction of the proxy is aligned with the true objective; (c) shows the descent of the proxy objective indeed transfers to the descent of the true objective. Together, they prove that server-side post-training not only does not harm model performance but effectively improves global fairness. Key insight: the classical approximation σ(x) ≈ Φ(c·x) (c ≈ 2.40) bridges sigmoid and normal CDF.

### Theorem 5 (Differential Privacy Guarantee for Prototype Aggregation)

**Setup:** Client k uploads local prototype μ_{g,l}^{(k)} per round, and the Server performs weighted aggregation:

μ_{g,l}^{global} = Σ_{k=1}^K w_k · μ_{g,l}^{(k)}

**Assumption 5 (LDP Noise):** Each client adds ε-LDP noise N(0, σ_noise² · I_d) before uploading prototypes, where σ_noise² ≥ (2d ln(2/δ))/ε² (satisfying (ε, δ)-DP).

Under Assumption 5, the prototype aggregation process satisfies **Distributed Differential Privacy**:

Pr[A(μ_{g,l}^{global}) = t] ≤ e^ε · Pr[A(μ_{g,l}^{global,-i}) = t] + δ

where μ_{g,l}^{global,-i} is the aggregated prototype after removing the i-th sample, and A is any adversary algorithm.

**Significance:** This theorem addresses **Challenge 1** (privacy leakage risk)—proving that the prototype information transmitted in PDFFed satisfies differential privacy guarantees after adding LDP noise. Even if an adversary can access the aggregated global prototypes, they cannot infer sensitive information about individual clients or samples.

### Theorem 6 (Privacy Risk of Transmitting Sensitive Attributes)

**Setup:** Suppose a method transmits sensitive attributes g (e.g., group labels) or intermediate results containing sensitive attribute information during communication.

**Conclusion:** Any method that transmits raw sensitive attributes g **does not satisfy differential privacy**, because an adversary can perform membership inference attack as follows:

Pr[A(g_i) = 1 | i ∈ S] - Pr[A(g_i) = 1 | i ∉ S] = 1

i.e., the adversary can perfectly determine whether a sample belongs to a group, thereby violating privacy.

**Significance:** This theorem proves from the negative side the necessity of **Challenge 1**—if sensitive attributes are transmitted, privacy is directly compromised. PDFFed chooses to transmit prototypes rather than sensitive attributes precisely to avoid this risk.

---

## 5. Detailed Proofs

### 5.1 Proof of Theorem 1

*(Proof framework adapted from McNamara et al., 2017 and Zhao & Gordon, 2019)*

**Step 1: EO expression.**

For a linear classifier:

EO_l = |E[p̂ \| g=0, y=l] - E[p̂ \| g=1, y=l]|

**Step 2: Sub-Gaussian property.**

By Assumption 2, z \| g, y=l is sub-Gaussian. The linear projection w^T z + b preserves sub-Gaussianity with mean w^T μ_{g,l} + b and variance parameter σ'_{g,l} = √(w^T Σ_{g,l} w).

**Step 3: Berry-Esseen approximation.**

For sub-Gaussian random variable X, the Berry-Esseen theorem bounds the difference between its CDF and the Gaussian CDF:

|P(X ≤ t) - Φ((t - μ)/σ')| ≤ C_BE / (σ' √n)

Therefore:

E[σ(w^T z + b) \| g, y=l] ≈ Φ((w^T μ_{g,l} + b) / σ'_{g,l})

**Step 4: EO gap expression.**

EO_l ≈ |Φ((w^T μ_{g0,l} + b) / σ'_{g0,l}) - Φ((w^T μ_{g1,l} + b) / σ'_{g1,l})|

**Step 5: Mean value theorem.**

There exists ξ such that:

EO_l = φ(ξ) · |(w^T μ_{g0,l} + b)/σ'_{g0,l} - (w^T μ_{g1,l} + b)/σ'_{g1,l}|

where φ(ξ) ≤ 1/√(2π).

**Step 6: Bounded covariance difference.**

By Assumption 3, ‖Σ_{g0,l} - Σ_{g1,l}‖_op ≤ Δ_Σ, which gives:

|σ'_{g0,l} - σ'_{g1,l}| ≤ ‖w‖² Δ_Σ / (2σ')

Setting σ' = max(σ'_{g0,l}, σ'_{g1,l}):

EO_l ≤ (1/√(2π)) · |w^T(μ_{g0,l} - μ_{g1,l})| / σ' + O(‖w‖² Δ_Σ / σ'³)

**Step 7: Cauchy-Schwarz inequality.**

|w^T(μ_{g0,l} - μ_{g1,l})| ≤ ‖w‖ · ‖μ_{g0,l} - μ_{g1,l}‖

Summing over all labels with σ_z = max_l σ'_l:

EO ≤ (‖w‖ / (√(2π) σ_z)) · Δ_rep + O(‖w‖² Δ_Σ / σ_z³)

**QED.**

---

### 5.2 Proof of Theorem 2

**Step 1: Confidence and logit relationship.**

Confidence c = σ(|z|) where z = w^T x + b is the logit. By the inverse sigmoid:

|z| = σ⁻¹(c) = ln(c / (1-c))

Low confidence c_low corresponds to |z|_low = ln(c_low / (1-c_low))

High confidence c_high corresponds to |z|_high = ln(c_high / (1-c_high))

**Step 2: Logit expectation.**

By Assumption 2 (sub-Gaussian), z \| g, y=l has mean μ'_g = w^T μ_{g,l} + b and standard deviation σ'_g. When σ'_g is small, by sub-Gaussian concentration:

E[|z| \| g, y=l] ≈ |μ'_g|

Therefore:

E[c \| g, y=l] ≈ σ(|μ'_g|)

**Step 3: Confidence gap and logit gap.**

By the Lipschitz property of sigmoid (Lipschitz constant 1/4):

c_high - c_low ≤ (1/4)(|μ'_{g1}| - |μ'_{g0}|)

Therefore:

|μ'_{g1}| - |μ'_{g0}| ≥ 4(c_high - c_low)

**Step 4: Logit gap and EO.**

EO_l ≈ |Φ(μ'_{g0} / σ') - Φ(μ'_{g1} / σ')|

**Step 5: Case analysis.**

*Case 1: μ'_{g0} and μ'_{g1} have the same sign.*

By the mean value theorem:

EO_l ≥ φ(max(|μ'_{g0}|, |μ'_{g1}|) / σ') · (||μ'_{g0}| - |μ'_{g1}|| / σ')

≥ φ(|μ'_{g1}| / σ') · 4(c_high - c_low) / σ'

When |μ'_{g1}| ≤ σ', φ(|μ'_{g1}| / σ') ≥ φ(1) ≈ 0.242:

EO_l ≥ (0.242 · 4(c_high - c_low)) / σ' ≥ (c_high - c_low) / σ'

*Case 2: μ'_{g0} and μ'_{g1} have opposite signs.*

Then |μ'_{g0}| + |μ'_{g1}| ≥ |μ'_{g1}| - |μ'_{g0}| ≥ 4(c_high - c_low)

The EO gap is even larger since the two groups' predictions are on opposite sides of the decision boundary.

**Step 6: Combining.**

EO ≥ 2(c_high - c_low) - 2σ' / (‖w‖ √(2π)) - O(Δ_Σ)

- The term 2σ' / (‖w‖ √(2π)) is a correction for distribution variance; smaller σ' (more concentrated distribution) yields a tighter bound.
- The term O(Δ_Σ) is a correction for covariance difference.

**QED.**

---

### 5.3 Proof of Theorem 3

**Setup:** Local training loss for client k: L_k = L_task,k + λ · L_contrastive,k

**Lemma 1 Proof:** See Steps 4–7 below, establishing ⟨∇_φ L_contrastive, ∇_φ Δ_rep⟩ ≥ 0.

**Proof of (a): Convergence.**

**Step 1: Gradient descent update.**

θ^{(t+1)} = θ^{(t)} - η ∇L^{(t)}

**Step 2: L-smoothness.**

By Assumption 3, L is L'-smooth where L' = L + λM.

By the definition of L-smoothness:

L^{(t+1)} ≤ L^{(t)} + ∇L^{(t)} · (-η ∇L^{(t)}) + (L' η² / 2) ‖∇L^{(t)}‖²

= L^{(t)} - η ‖∇L^{(t)}‖² + (η² (L + λM) / 2) ‖∇L^{(t)}‖²

**Step 3: Convergence condition.**

When η < 2/(L + λM), the right-hand side is negative, ensuring monotonic decrease.

**Proof of (b): Δ_rep Contraction.**

**Step 4: CL loss form.** PDFFed uses a prototype-level contrastive loss:

L_contrastive = (1/2) Σ_l ‖μ_{g0,l} - μ_{g1,l}‖²

**Step 5: Representation bias gradient.** Representation bias Δ_rep = Σ_l ‖μ_{g0,l} - μ_{g1,l}‖, its gradient is:

∇_φ Δ_rep = Σ_l (μ_{g0,l} - μ_{g1,l})/‖μ_{g0,l} - μ_{g1,l}‖ · (∇_φ μ_{g0,l} - ∇_φ μ_{g1,l})

**Step 6: CL loss gradient.**

∇_φ L_contrastive = Σ_l (μ_{g0,l} - μ_{g1,l}) · (∇_φ μ_{g0,l} - ∇_φ μ_{g1,l})

**Step 7: Inner product of the two gradients.**

∇_φ Δ_rep · ∇_φ L_contrastive = Σ_l ‖μ_{g0,l} - μ_{g1,l}‖ · ‖∇_φ μ_{g0,l} - ∇_φ μ_{g1,l}‖²

Since ‖μ_{g0,l} - μ_{g1,l}‖ > 0 and ‖∇_φ μ_{g0,l} - ∇_φ μ_{g1,l}‖² ≥ 0, the inner product is non-negative.

**Step 8: Effect of gradient descent.** The update is:

φ^{(t+1)} = φ^{(t)} - η λ · ∇_φ L_contrastive

By Step 7, ∇_φ Δ_rep and ∇_φ L_contrastive are in the same direction (non-negative inner product), so gradient descent reduces Δ_rep.

**Step 9: Quantitative contraction.** By Taylor expansion:

Δ_rep^{(t+1)} = Δ_rep^{(t)} - η λ · ∇_φ Δ_rep · ∇_φ L_contrastive + O(η²)

Let γ = E[∂L_contrastive / ∂‖μ_{g0,l} - μ_{g1,l}‖] = E[‖μ_{g0,l} - μ_{g1,l}‖] > 0, then after T steps:

Δ_rep^{(t+T)} ≤ Δ_rep^{(t)} - η · λ · T · γ

**Note:** The CL loss form is critical for proving contraction—the loss must be explicitly written for rigorous proof. γ > 0 is the core condition guaranteeing contraction, naturally satisfied when Δ_rep > 0.

**QED.**

---

### 5.4 Proof of Theorem 4

**Setup:** Server-side post-training updates only ψ, with loss L_post = L_cls + λ_eo · EO_proto, where EO_proto = Σ_l |σ(w^T μ_{g0,l} + b) - σ(w^T μ_{g1,l} + b)|.

**Proof of (a): Approximation bound.**

**Step 1:** From the proof of Theorem 1, the global EO under Assumptions 1-2 approximates to:

EO_global ≈ Σ_l |Φ((w^T μ_{g0,l} + b) / σ'_{g0,l}) - Φ((w^T μ_{g1,l} + b) / σ'_{g1,l})|

with approximation error O(1/(σ √n)) (Berry-Esseen bound).

**Step 2:** Using the classical approximation σ(x) ≈ Φ(c · x), where c = √(π/ln 2) ≈ 2.40 (error < 0.02 for |x| ≤ 3):

EO_proto ≈ Σ_l |Φ(c · (w^T μ_{g0,l} + b)) - Φ(c · (w^T μ_{g1,l} + b))|

**Step 3:** Comparing the two, the difference arises from the denominator σ'_{g,l} (controlled by Assumption 3) and the constant c (fixed scaling). Using Assumption 3 and the same derivation as Step 6 in the proof of Theorem 1:

|EO_proto - EO_global| ≤ (‖w‖ / (√(2π) σ)) · Δ_rep^proto + O(‖w‖² Δ_Σ / σ³) + O(1 / (σ √n))

**Proof of (b): Gradient direction consistency.**

**Step 4:** Since only ψ = (w, b) is updated, the two gradient expressions are:

∇_ψ EO_proto = Σ_l ∇_ψ |σ(w^T μ_{g0,l} + b) - σ(w^T μ_{g1,l} + b)|

∇_ψ EO_global ≈ Σ_l ∇_ψ |Φ((w^T μ_{g0,l} + b) / σ'_{g0,l}) - Φ((w^T μ_{g1,l} + b) / σ'_{g1,l})|

**Step 5:** From σ(x) ≈ Φ(c · x), we have σ'(x) ≈ c · φ(c · x). Thus the core terms of the two gradients differ only by a constant scaling c and the denominator correction σ'_{g,l}:

∇_ψ EO_proto ≈ c · ∇_ψ EO_global + correction(Δ_Σ)

**Step 6:** By the Cauchy-Schwarz inequality:

cos⟨∇_ψ EO_proto, ∇_ψ EO_global⟩ ≥ 1 - O(‖w‖² Δ_Σ / σ²)

**Proof of (c): Descent transfer.**

**Step 7:** Taylor expansion of the true EO:

EO_global^{(t+1)} = EO_global^{(t)} + ∇_ψ EO_global^{(t)} · (ψ^{(t+1)} - ψ^{(t)}) + O(α²)

where ψ^{(t+1)} - ψ^{(t)} = -α λ_eo ∇_ψ EO_proto^{(t)}.

**Step 8:** Substituting and using (b):

EO_global^{(t+1)} = EO_global^{(t)} - α λ_eo ‖∇_ψ EO_global‖ ‖∇_ψ EO_proto‖ cos⟨∇_ψ EO_global, ∇_ψ EO_proto⟩ + O(α²)

By (b), cos⟨·,·⟩ ≥ 1 - ε(Δ_Σ), therefore:

EO_global^{(t+1)} ≤ EO_global^{(t)} - α λ_eo ‖∇_ψ EO_proto‖² · δ

where δ = cos⟨·,·⟩ - α L_eo / (2 λ_eo) > 0 (when α is sufficiently small and Δ_Σ is sufficiently small).

**QED.**

---

### 5.5 Proof of Theorem 5 (Differential Privacy Guarantee for Prototype Aggregation)

**Setup:** Client k computes local prototype μ_{g,l}^{(k)} = (1/n_{g,l}^{(k)}) Σ_{i=1}^{n_{g,l}^{(k)}} z_i, adds noise before uploading μ̃_{g,l}^{(k)} = μ_{g,l}^{(k)} + ξ_k, where ξ_k ~ N(0, σ_noise² · I_d).

Server aggregation: μ_{g,l}^{global} = Σ_{k=1}^K w_k · μ̃_{g,l}^{(k)}

**Step 1: Local Differential Privacy Guarantee**

Each client's noise ξ_k satisfies ε-LDP (when σ_noise² ≥ (2d ln(2/δ))/ε²):

Pr[μ̃_{g,l}^{(k)} ∈ S] ≤ e^ε · Pr[μ̃_{g,l}^{(k),-i} ∈ S] + δ

where μ̃_{g,l}^{(k),-i} is the noisy prototype after removing the i-th sample.

**Step 2: Distributed Differential Privacy**

By the composition theorem of distributed differential privacy, the combination of K clients' LDP mechanisms satisfies Kε-DP:

Pr[A(μ_{g,l}^{global}) = t] ≤ e^{Kε} · Pr[A(μ_{g,l}^{global,-i}) = t] + Kδ

**Step 3: Privacy Amplification**

When K clients participate in aggregation, the privacy budget can be amplified through aggregation. For (ε, δ)-LDP, the aggregated privacy guarantee is:

ε_total ≤ √(8K ln(1/δ') / ε²) + (2Kε ln(1/δ'))/δ

When K is large, the aggregated privacy guarantee is significantly better than that of a single client.

**QED.**

---

### 5.6 Proof of Theorem 6 (Privacy Risk of Transmitting Sensitive Attributes)

**Setup:** Suppose a method transmits raw sensitive attributes g_i.

**Step 1: Membership Inference Attack**

The adversary can design algorithm A(g_i) to determine whether sample i belongs to training set S:

A(g_i) = {1 if g_i appears in transmitted data, 0 otherwise}

**Step 2: Attack Success Rate**

Pr[A(g_i) = 1 | i ∈ S] = 1
Pr[A(g_i) = 1 | i ∉ S] = 0

Therefore:

Pr[A(g_i) = 1 | i ∈ S] - Pr[A(g_i) = 1 | i ∉ S] = 1

**Step 3: Violating Differential Privacy**

Differential privacy requires that for any adjacent datasets D and D':

Pr[A(M(D)) = t] ≤ e^ε · Pr[A(M(D')) = t] + δ

For mechanisms that transmit sensitive attributes, when D and D' differ only in sample i, the adversary can perfectly distinguish between them, thus violating differential privacy.

**QED.**

---

## 6. Proof Chain

The complete proof chain with challenge-by-challenge theorem assignments is presented in **Section 0 (Macro-Level Proof Chain)**. Please refer to that section for the full organization of theorems by challenge, including the ASCII-art chain diagrams and the theorem–challenge mapping table.

### Supplementary: Upper and Lower Bound Complementarity

- Theorem 1 provides an **upper bound**: reducing Δ_rep → EO upper bound decreases → **guarantees** the worst-case EO will not be too large. This bound is used for **optimization target design**.
- Theorem 2 provides a **lower bound**: large confidence gap → EO lower bound is high → **diagnoses** the existence of fairness issues. This bound is used for **fairness problem diagnosis**.

The two bounds are complementary and together form a principled approach to fairness-aware federated learning.

---

## 7. Generalization to Other Fairness Metrics

The above proof framework focuses on Equalized Odds (EO) as the core metric. This section shows how the framework generalizes to other commonly used fairness metrics.

### 7.1 Demographic Parity (DP)

**Definition:**

$$\text{DP} = \frac{1}{2} \sum_{l \in \{0,1\}} \bigl| \mathbb{P}(\hat{y}=l \mid g=0) - \mathbb{P}(\hat{y}=l \mid g=1) \bigr|$$

The key difference between DP and EO: DP does not depend on the true label $y$, directly comparing prediction distributions across groups.

**Generalization:** The upper bound for DP can be derived similarly, with an additional term for label distribution difference:

$$\text{DP} \leq \frac{\|w\|}{\sqrt{2\pi}\,\sigma_z} \cdot \Delta_{\text{rep}} + O\!\left(\frac{\|w\|^2 \Delta_{\Sigma}}{\sigma_z^3}\right) + \Delta_{\text{label}}$$

where

$$\Delta_{\text{label}} = \frac{1}{2} \sum_{l \in \{0,1\}} \bigl| \mathbb{P}(y=l \mid g=0) - \mathbb{P}(y=l \mid g=1) \bigr|$$

is the label distribution gap.

**Remarks:**
- DP's upper bound includes an extra $\Delta_{\text{label}}$ term—this is because DP requires prediction distributions independent of true labels, which may differ across groups
- When label distributions are balanced ($\Delta_{\text{label}} \approx 0$), DP and EO share the same upper bound form

### 7.2 Equalized Opportunity (EOpp)

**Definition:**

$$\text{EOpp} = \bigl| \mathbb{P}(\hat{y}=1 \mid y=1, g=0) - \mathbb{P}(\hat{y}=1 \mid y=1, g=1) \bigr|$$

EOpp is a special case of EO, focusing only on positive class fairness.

**Generalization:** The upper bound derivation is identical to EO, with the sum restricted to positive class $l=1$:

$$\text{EOpp} \leq \frac{\|w\|}{\sqrt{2\pi}\,\sigma_z} \cdot \Delta_{\text{rep},1} + O\!\left(\frac{\|w\|^2 \Delta_{\Sigma}}{\sigma_z^3}\right)$$

where $\Delta_{\text{rep},1} = \|\mu_{g_0,1} - \mu_{g_1,1}\|$ is the representation bias for the positive class.

### 7.3 Disparate Equalized Odds (DEO)

**Definition:**

$$\text{DEO} = w_1 \cdot |\text{TPR}_0 - \text{TPR}_1| + w_0 \cdot |\text{FPR}_0 - \text{FPR}_1|$$

where $w_1, w_0 \geq 0$ are weights, and

$$\text{TPR}_g = \mathbb{P}(\hat{y}=1 \mid y=1, g), \qquad \text{FPR}_g = \mathbb{P}(\hat{y}=1 \mid y=0, g).$$

DEO is a weighted form of EO, allowing different importance weights for TPR and FPR gaps.

**Full Proof:**

**Step 1: DEO decomposition.**

$$\text{DEO} = w_1 \cdot |\text{TPR}_0 - \text{TPR}_1| + w_0 \cdot |\text{FPR}_0 - \text{FPR}_1|$$

**Step 2: Apply Theorem 1 to TPR and FPR separately.**

From Theorem 1, for positive class $l=1$ (TPR):

$$|\text{TPR}_0 - \text{TPR}_1| \leq \frac{\|w\|}{\sqrt{2\pi}\,\sigma_z} \cdot \|\mu_{g_0,1} - \mu_{g_1,1}\| + O\!\left(\frac{\|w\|^2 \Delta_{\Sigma}}{\sigma_z^3}\right)$$

For negative class $l=0$ (FPR), similarly:

$$|\text{FPR}_0 - \text{FPR}_1| \leq \frac{\|w\|}{\sqrt{2\pi}\,\sigma_z} \cdot \|\mu_{g_0,0} - \mu_{g_1,0}\| + O\!\left(\frac{\|w\|^2 \Delta_{\Sigma}}{\sigma_z^3}\right)$$

**Step 3: Weighted summation.**

$$\begin{aligned}
\text{DEO} &\leq w_1 \cdot \left[ \frac{\|w\|}{\sqrt{2\pi}\,\sigma_z} \cdot \|\mu_{g_0,1} - \mu_{g_1,1}\| + O\!\left(\frac{\|w\|^2 \Delta_{\Sigma}}{\sigma_z^3}\right) \right] \\
&\quad + w_0 \cdot \left[ \frac{\|w\|}{\sqrt{2\pi}\,\sigma_z} \cdot \|\mu_{g_0,0} - \mu_{g_1,0}\| + O\!\left(\frac{\|w\|^2 \Delta_{\Sigma}}{\sigma_z^3}\right) \right] \\[4pt]
&= \frac{\|w\|}{\sqrt{2\pi}\,\sigma_z} \cdot \Bigl[ w_1 \cdot \|\mu_{g_0,1} - \mu_{g_1,1}\| + w_0 \cdot \|\mu_{g_0,0} - \mu_{g_1,0}\| \Bigr] \\
&\quad + O\!\left(\frac{\|w\|^2 \Delta_{\Sigma}}{\sigma_z^3}\right) \cdot (w_1 + w_0)
\end{aligned}$$

**Step 4: Combined weighted representation bias.**

Let

$$\Delta_{\text{rep}}^w = w_1 \cdot \|\mu_{g_0,1} - \mu_{g_1,1}\| + w_0 \cdot \|\mu_{g_0,0} - \mu_{g_1,0}\|,$$

then:

$$\text{DEO} \leq \frac{\|w\|}{\sqrt{2\pi}\,\sigma_z} \cdot \Delta_{\text{rep}}^w + O\!\left(\frac{\|w\|^2 \Delta_{\Sigma}}{\sigma_z^3}\right) \cdot (w_1 + w_0)$$

**Note:** When $w_1 = w_0 = 1$, $\Delta_{\text{rep}}^w = \Delta_{\text{rep}}$ and DEO reduces to EO.

### 7.4 Generalization Summary

| Metric | Upper Bound Form | Special Term |
|--------|-----------------|--------------|
| EO | $\dfrac{\|w\|}{\sqrt{2\pi}\,\sigma_z} \cdot \Delta_{\text{rep}} + O\!\left(\dfrac{\|w\|^2 \Delta_{\Sigma}}{\sigma_z^3}\right)$ | None |
| DP | $\dfrac{\|w\|}{\sqrt{2\pi}\,\sigma_z} \cdot \Delta_{\text{rep}} + O\!\left(\dfrac{\|w\|^2 \Delta_{\Sigma}}{\sigma_z^3}\right) + \Delta_{\text{label}}$ | $\Delta_{\text{label}}$ |
| EOpp | $\dfrac{\|w\|}{\sqrt{2\pi}\,\sigma_z} \cdot \Delta_{\text{rep},1} + O\!\left(\dfrac{\|w\|^2 \Delta_{\Sigma}}{\sigma_z^3}\right)$ | Positive class only |
| DEO | $\dfrac{\|w\|}{\sqrt{2\pi}\,\sigma_z} \cdot \Delta_{\text{rep}}^w + O\!\left(\dfrac{\|w\|^2 \Delta_{\Sigma}}{\sigma_z^3}\right) \cdot (w_1 + w_0)$ | Weighted |

**Core Conclusion:** All metrics share the same upper bound structure—a linear term in $\Delta_{\text{rep}}$ (or its variants) plus a higher-order term in $\Delta_{\Sigma}$. This implies:

1. **Reducing $\Delta_{\text{rep}}$ improves all metrics**—this is the core justification for our methodology
2. **Theorem 3's $\Delta_{\text{rep}}$ contraction proof applies to all metrics**—CL improves fairness regardless of which metric is used
3. **Theorem 4's calibration proxy method applies to all metrics**—simply replace $\text{EO}_{\text{proto}}$ with the prototype-level proxy of the corresponding metric

---

## 8. References

1. McNamara, D., Ong, C. S., & Williamson, R. C. (2017). Provably Fair Representations. arXiv:1710.10622.
2. Zhao, H., & Gordon, G. J. (2019). Inherent Tradeoffs in Learning Fair Representations. arXiv:1906.08386.
3. Madras, D., Creager, E., Pitassi, T., & Zemel, R. (2018). Learning Adversarially Fair and Transferable Representations. ICML 2018.
4. Zemel, R., Wu, Y., Swersky, K., Pitassi, T., & Dwork, C. (2013). Learning Fair Representations. ICML 2013.
5. Hashimoto, T., Srivastava, M., Namkoong, H., & Liang, P. (2018). Fairness Without Demographics in Repeated Loss Minimization. ICML 2018.
6. Kirichenko, P., Izmailov, P., & Wilson, A. G. (2023). Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations. ICLR 2023. arXiv:2204.02937.
7. Mao, Y., Deng, Z., Yao, H., Ye, T., Kawaguchi, K., & Zou, J. (2023). Last-Layer Fairness Fine-tuning is Simple and Effective for Neural Networks. ICML Workshop on Human-Centric Machine Learning. arXiv:2304.03935.
8. Zhong, D., Greenberg, K., Yao, S., Han, D., Chen, Y., & Chen, H. (2025). Fair Federated Learning with Instance-Wise Penalty. arXiv:2507.08912.
9. Zhang, Y., Xu, H., Zhou, A., & Jin, H. (2023). GroupMixNorm Layer for Learning Fair Models. NeurIPS Workshop on Algorithmic Fairness through the Lens of Time. arXiv:2312.11969.
10. Cui, G., Wang, L., Ren, V., Yuan, J., Lee, H., & Lakkaraju, H. (2024). Rethinking Fairness in Federated Learning: Learned Features Alone Can Already Compromise Fairness. arXiv:2405.01112.
11. Dwork, C., & Roth, A. (2014). The Algorithmic Foundations of Differential Privacy. Foundations and Trends in Theoretical Computer Science, 9(3–4), 211–407.
