import re

with open(r'd:\最新PDFFed\fairness_fl_code\docs\proof_zh.md', 'r', encoding='utf-8') as f:
    c = f.read()

print(f"Original length: {len(c)}")

# ============================================================
# PART 1: STRUCTURAL CHANGES
# ============================================================

# 1a. Replace global optimization objective + local training loss block
# Old block starts at "**全局优化目标**" and ends after the local training loss explanation
old_global_block = """**全局优化目标**（联邦平均的目标）：

$$\\min_{\\theta} \\sum_{k=1}^K w_k \\cdot \\mathbb{E}_{(x,y,g) \\sim \\mathcal{D}_k} \\left[ \\ell(\\varsigma(w^T f_\\phi(x) + b), y) \\right]$$

其中 $\\theta = (\\phi, \\psi)$ 是模型参数，$\\phi$ 是编码器参数，$\\psi = (w, b)$ 是分类头参数。

**客户端 $k$ 的局部训练损失**：

$$\\mathcal{L}_k = \\mathbb{E}_{(x,y,g) \\sim \\mathcal{D}_k} \\left[ \\ell(\\varsigma(w^T f_\\phi(x) + b), y) \\right] + \\lambda \\cdot \\mathbb{I}(\\rho_l < \\tau) \\cdot \\mathcal{L}_{\\text{contrastive},k}$$

其中 $\\mathcal{L}_{\\text{contrastive},k}$ 是条件性对比损失（当 $\\rho_l < \\tau$ 时激活），$\\mathbb{I}(\\cdot)$ 是指示函数。"""

new_global_block = """**全局优化目标**（联邦学习的目标）：

$$\\min_{\\theta} \\mathcal{F}(\\theta) = \\sum_{k=1}^K \\frac{n_k}{N} \\cdot f_k(\\theta), \\quad f_k(\\theta) = \\mathbb{E}_{(x,y,g) \\sim \\mathcal{D}_k} \\left[ \\ell(f_\\theta(x), y) \\right]$$

其中 $\\theta = (\\phi, \\psi)$ 是模型参数，$f_\\theta(x) = \\varsigma(w^T f_\\phi(x) + b)$ 是模型对输入 $x$ 的正类预测概率，$n_k = |\\mathcal{D}_k|$ 是客户端 $k$ 的样本数，$N = \\sum_{k=1}^K n_k$ 是全局样本总数。

**定义2.10 任务损失**（交叉熵损失）：

$$\\mathcal{L}_{\\text{task}}(x, y) = \\ell(\\varsigma(w^T f_\\phi(x) + b), y) = -y \\log(\\hat{p}) - (1-y) \\log(1-\\hat{p})$$

其中 $\\hat{p} = \\varsigma(w^T f_\\phi(x) + b)$ 是正类预测概率。

**定义2.11 对比损失**（原型级对比损失）：

$$\\mathcal{L}_{\\text{contrastive}} = \\frac{1}{2} \\sum_{l \\in \\{0,1\\}} \\|\\mu_{g0,l} - \\mu_{g1,l}\\|^2$$

其中 $\\mu_{g,l} = \\mathbb{E}[z \\mid g, y=l]$ 是群组-标签感知数据原型（见定义2.5）。该损失通过直接惩罚同一类别下不同群组的原型距离，减小表征偏差 $\\Delta_{\\text{rep}}$。

**客户端 $k$ 的局部训练损失**（在任务损失基础上增加公平性正则项）：

$$\\mathcal{L}_k = f_k(\\theta) + \\lambda \\cdot \\mathbb{I}(\\rho_l < \\tau) \\cdot \\mathcal{L}_{\\text{contrastive},k} = \\mathbb{E}_{(x,y,g) \\sim \\mathcal{D}_k} \\left[ \\mathcal{L}_{\\text{task}}(x, y) \\right] + \\lambda \\cdot \\mathbb{I}(\\rho_l < \\tau) \\cdot \\mathcal{L}_{\\text{contrastive},k}$$

其中 $\\mathcal{L}_{\\text{contrastive},k}$ 是条件性对比损失（当 $\\rho_l < \\tau$ 时激活），$\\mathbb{I}(\\cdot)$ 是指示函数。"""

assert old_global_block in c, "Old global block not found!"
c = c.replace(old_global_block, new_global_block)
print("Part 1a: Global+local block replaced.")

# 1b. Remove old task loss and contrastive loss definitions (now moved above)
# They appear between "**定义2.9（决策边界距离）**" and "**观察1**"
old_def_block = """

**定义2.10 任务损失**（交叉熵损失）：

$$\\mathcal{L}_{\\text{task}}(x, y) = \\ell(\\varsigma(w^T f_\\phi(x) + b), y) = -y \\log(\\hat{p}) - (1-y) \\log(1-\\hat{p})$$

其中 $\\hat{p} = \\varsigma(w^T f_\\phi(x) + b)$ 是正类预测概率。

**定义2.11 对比损失**（原型级对比损失）：

$$\\mathcal{L}_{\\text{contrastive}} = \\frac{1}{2} \\sum_{l \\in \\{0,1\\}} \\|\\mu_{g0,l} - \\mu_{g1,l}\\|^2$$

其中 $\\mu_{g,l} = \\mathbb{E}[z \\mid g, y=l]$ 是群组-标签感知数据原型（见定义2.5）。该损失通过直接惩罚同一类别下不同群组的原型距离，减小表征偏差 $\\Delta_{\\text{rep}}$。

"""

assert old_def_block in c, "Old def 2.10/2.11 block not found!"
c = c.replace(old_def_block, "\n\n")
print("Part 1b: Old def 2.10/2.11 removed.")


# ============================================================
# PART 2: SYNTAX FIXES (all formula corruptions from regex mishaps)
# ============================================================

# Pattern: garbage "\frac{\|w\|^2 \Delta_\Sigma}{\sigma_z^3}..." inserted into formulas
# These are due to a misapplied regex that partially matched sigma/Sigma terms

fixes = [
    # --- Fix 1: Corrupted \rho_l (gradient direction consistency) ---
    (
        r'\frac{\|w\|^2 \Delta_\Sigma}{\sigma_z^3}bar{g}_{g0,l} \cdot \bar{g}_{g1,l}}{\|\bar{g}_{g0,l}\| \cdot \|\bar{g}_{g1,l}\|}',
        r'\frac{\bar{g}_{g0,l} \cdot \bar{g}_{g1,l}}{\|\bar{g}_{g0,l}\| \cdot \|\bar{g}_{g1,l}\|}'
    ),
    # --- Fix 2: Corrupted sub-Gaussian assumption ---
    (
        r"e^{\frac{\|w\|^2 \Delta_\Sigma}{\sigma_z^3}lambda^T \sigma_{g,l} \lambda}{2}}",
        r"e^{\frac{\lambda^T \Sigma_{g,l} \lambda}{2}}"
    ),
    # --- Fix 3: Theorem 1 formula ---
    (
        r"O\left(\frac{\|w\|^2 \Delta_\Sigma}{\sigma_z^3}|w\|^2 \Delta_\Sigma|w\|^2 \Delta_{\varsigma}}{\sigma_z^3}\right)",
        r"O\left(\frac{\|w\|^2 \Delta_{\varsigma}}{\sigma_z^3}\right)"
    ),
    # --- Fix 4: Theorem 3(a) convergence ---
    (
        r"\frac{\|w\|^2 \Delta_\Sigma}{\sigma_z^3}eta^2 (L + \lambda M)}{2}",
        r"\frac{\eta^2 (L + \lambda M)}{2}"
    ),
    # --- Fix 5: Missing \frac before \partial in Theorem 3(b) ---
    (
        r"\partial \mathcal{L}_{\text{contrastive}}}{\partial \|\mu_{g0,l} - \mu_{g1,l}\|}",
        r"\frac{\partial \mathcal{L}_{\text{contrastive}}}{\partial \|\mu_{g0,l} - \mu_{g1,l}\|}"
    ),
    # --- Fix 6: Theorem 4(a) - same O() corruption ---
    # Already covered by Fix 3 (same pattern)
    # --- Fix 7: Theorem 4(b) - O() with sigma_z^2 ---
    (
        r"O\left(\frac{\|w\|^2 \Delta_\Sigma}{\sigma_z^3}|w\|^2 \Delta_\Sigma|w\|^2 \Delta_{\varsigma}}{\sigma_z^2}\right)",
        r"O\left(\frac{\|w\|^2 \Delta_{\varsigma}}{\sigma_z^2}\right)"
    ),
    # --- Fix 8: Theorem 4(c) delta ---
    (
        r"\frac{\|w\|^2 \Delta_\Sigma}{\sigma_z^3}alpha L_{\text{eo}}}{2\lambda_{\text{eo}}}",
        r"\frac{\alpha L_{\text{eo}}}{2\lambda_{\text{eo}}}"
    ),
    # --- Fix 9: Proof 5.1 Step 6 ---
    (
        r"\frac{\|w\|^2 \Delta_\Sigma}{\sigma_z^3}|w\|^2 \Delta_{\varsigma}}{2\sigma_z}",
        r"\frac{\|w\|^2 \Delta_{\varsigma}}{2\sigma_z}"
    ),
    # --- Fix 10: Proof 5.2 Step 4 --- multiple \Phi corruptions ---
    (
        r"\frac{\|w\|^2 \Delta_\Sigma}{\sigma_z^3}mu'_{g0}}{\sigma_z}",
        r"\frac{\mu'_{g0}}{\sigma_z}"
    ),
    (
        r"\frac{\|w\|^2 \Delta_\Sigma}{\sigma_z^3}mu'_{g1}}{\sigma_z}",
        r"\frac{\mu'_{g1}}{\sigma_z}"
    ),
    # --- Fix 11: Proof 5.2 Step 5 phi corruption ---
    (
        r"\frac{\|w\|^2 \Delta_\Sigma}{\sigma_z^3}max(|\mu'_{g0}|, |\mu'_{g1}|)}{\sigma_z}",
        r"\frac{\max(|\mu'_{g0}|, |\mu'_{g1}|)}{\sigma_z}"
    ),
    # --- Fix 12: Proof 5.3 Step 5 gradient of Delta_rep ---
    (
        r"\frac{\|w\|^2 \Delta_\Sigma}{\sigma_z^3}mu_{g0,l} - \mu_{g1,l}}{\|\mu_{g0,l} - \mu_{g1,l}\|}",
        r"\frac{\mu_{g0,l} - \mu_{g1,l}}{\|\mu_{g0,l} - \mu_{g1,l}\|}"
    ),
    # --- Fix 13: Proof 5.4 Step 10 gradient direction consistency ---
    (
        r"O\left(\frac{\|w\|^2 \Delta_\Sigma}{\sigma_z^3}|w\|^2 \Delta_\Sigma|w\|^2 \Delta_\Sigma}{\sigma^2}\right)",
        r"O\left(\frac{\|w\|^2 \Delta_\Sigma}{\sigma^2}\right)"
    ),
    # --- Fix 14: Missing \frac before \eta^2 in Proof 5.3 (line 498) ---
    # Already covered by Fix 4 (same pattern)
]

for i, (old, new) in enumerate(fixes):
    count = c.count(old)
    if count > 0:
        c = c.replace(old, new)
        print(f"  Fix {i+1}: replaced {count} occurrence(s)")
    else:
        print(f"  Fix {i+1}: NOT FOUND (may already be fixed)")

# --- Simple residual corruption check (no regex to avoid escape issues) ---
print("\n--- Residual corruption check ---")
corrupt_snippets = [
    r'\|w\|^2 \Delta_\Sigma}{\sigma_z^3}bar{',
    r'\|w\|^2 \Delta_\Sigma}{\sigma_z^3}eta^',
    r'\|w\|^2 \Delta_\Sigma}{\sigma_z^3}mu\'',
    r'\|w\|^2 \Delta_\Sigma}{\sigma_z^3}max',
    r'\|w\|^2 \Delta_\Sigma}{\sigma_z^3}alpha',
    r'\|w\|^2 \Delta_\Sigma}{\sigma_z^3}lambda',
    r'\|w\|^2 \Delta_\Sigma}{\sigma_z^3}|w\|^2',
    r'\partial \mathcal{L}_{\text{contrastive}}}{\partial',
]
found_any = False
for snippet in corrupt_snippets:
    count = c.count(snippet)
    if count > 0:
        found_any = True
        for i, line in enumerate(c.split('\n')):
            if snippet in line:
                print(f"  Line {i+1}: {line.strip()[:150]}")
if not found_any:
    print("  None found - all clean!")

with open(r'd:\最新PDFFed\fairness_fl_code\docs\proof_zh.md', 'w', encoding='utf-8') as f:
    f.write(c)

print(f"\nFinal length: {len(c)}")
print("Done!")
