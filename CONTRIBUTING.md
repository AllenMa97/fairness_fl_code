# 贡献指南 / Contributing Guide

感谢你对 `fairness_fl_code`（公平联邦学习研究框架）感兴趣。本仓库是学术研究项目，所有代码合入前都会经过维护者审阅——维护者有权接受、要求修改或关闭任何 Pull Request。

Thank you for your interest in `fairness_fl_code` (a fairness-aware federated learning research framework). This is an academic research project: every change is reviewed by the maintainer, who may accept, request changes, or close any Pull Request.

> 中文优先 / Chinese First：本文件同时提供中文与英文说明，中文部分在前。
> This file is bilingual; the Chinese section comes first.

---

## 目录 / Table of Contents

- [报告问题 / Reporting Issues](#报告问题--reporting-issues)
- [贡献代码（Fork + Pull Request） / Submitting Code](#贡献代码fork--pull-request--submitting-code)
- [开发约定 / Development Conventions](#开发约定--development-conventions)
- [Pull Request 检查清单 / PR Checklist](#pull-request-检查清单--pr-checklist)
- [遇到困难？联系我们 / Still Stuck? Contact Us](#遇到困难联系我们--still-stuck-contact-us)

---

## 报告问题 / Reporting Issues

在 [Issues](https://github.com/AllenMa97/fairness_fl_code/issues) 页面新建 issue：

- 标题简明描述问题（可用中英文）。
- Bug 报告请附上：报错信息、复现命令或脚本、相关算法与数据集、运行环境（Python / PyTorch 版本）。
- 功能建议请说明使用场景与预期行为。

Create an issue on the [Issues](https://github.com/AllenMa97/fairness_fl_code/issues) page:

- Use a concise, descriptive title (Chinese or English).
- For bugs, include: the error message, the reproduction command/script, the algorithm and dataset involved, and your environment (Python / PyTorch versions).
- For feature requests, describe the use case and expected behavior.

---

## 贡献代码（Fork + Pull Request） / Submitting Code

**重要**：直接 `git push` 到本仓库需要协作者权限，普通贡献者会得到 `Permission denied`。请务必使用 Fork + Pull Request 流程（对公开仓库默认开放，无需任何额外授权）。

**Important**: pushing directly to this repository requires collaborator permission; regular contributors will get `Permission denied`. Always use the Fork + Pull Request workflow (open to everyone by default on a public repository — no extra permission needed).

以下命令以 PowerShell / bash 均可执行（按你的系统环境二选一即可）：

The commands below work in both PowerShell and bash (choose whatever matches your environment):

```bash
# 1. 在 GitHub 网页上把仓库 Fork 到自己账号下
#    Click "Fork" on the GitHub page of this repository

# 2. 克隆自己的 fork，并关联上游仓库
git clone https://github.com/<your-username>/fairness_fl_code.git
cd fairness_fl_code
git remote add upstream https://github.com/AllenMa97/fairness_fl_code.git

# 3. 建一个功能分支（不要直接改 main）
git checkout -b feat/your-feature

# 4. 提交并推送到你自己的 fork
git add .
git commit -m "feat(algorithms): Describe what you changed"
git push origin feat/your-feature

# 5. 在 GitHub 上从该分支发起 Pull Request（目标仓库选 AllenMa97/fairness_fl_code 的 main）
#    Open a Pull Request from that branch to AllenMa97/fairness_fl_code:main

# 6. 后续同步上游更新（可选 / optional）
git fetch upstream
git rebase upstream/main
```

维护者会审阅你的 PR；如果暂时不合并，不代表你的贡献不被认可，可能是进度或规划上的安排，请耐心沟通。The maintainer will review your PR. If it is not merged right away, it does not mean your contribution is unwelcome — it may simply be a matter of roadmap or timing; feel free to discuss.

---

## 开发约定 / Development Conventions

请尽量遵循仓库现有风格，新代码被合并的几率会更高：

Please follow the repository's existing conventions to increase the chance of your PR being merged:

- **分支命名 / Branch naming**：`feat/<desc>`、`fix/<desc>`、`docs/<desc>`、`refactor/<desc>`。
- **提交信息 / Commit messages**：使用 Conventional Commits，参考仓库历史：
  - `feat(algorithms): Add <Method-Name> and register in experiment.py`
  - `fix: rename moudle -> module, ...`
  - `docs: update README`
- **代码注释 / Comments**：中英双语注释；算法文件头部保留出处（论文链接）与核心思想说明。
- **命名规范 / Naming**：统一风格（全小写下划线或全驼峰），避免混用；优先可读性与可维护性。
- **数据集与大文件 / Datasets & large files**：不要提交数据集、模型权重或二进制产物，遵守 [.gitignore](.gitignore)。
- **改动前自测 / Self-test before PR**：涉及核心训练逻辑的改动，请在本地跑通相关任务的冒烟测试（tabular / image / text 对应 `main_Tabular_CLF.py`、`main_IMG_CLF.py`、`main_SENT_CLF.py`）再提 PR。
- **新算法 / New algorithms**：尽量复用 `module/` 与 `tool/` 下的基础设施，避免重复造轮子。

- **Branch naming**: `feat/<desc>`, `fix/<desc>`, `docs/<desc>`, `refactor/<desc>`.
- **Commit messages**: Conventional Commits, as in the repository history (examples above).
- **Comments**: bilingual (Chinese + English); algorithm files keep a header noting the paper link and core idea.
- **Naming**: pick one style (all lowercase with underscores, or camelCase) and stay consistent; favor readability and maintainability.
- **Datasets & large files**: do not commit datasets, checkpoints, or binaries; follow [.gitignore](.gitignore).
- **Self-test before PR**: if you touch core training logic, run the relevant smoke tests locally (tabular / image / text via `main_Tabular_CLF.py`, `main_IMG_CLF.py`, `main_SENT_CLF.py`) before opening the PR.
- **New algorithms**: reuse the infrastructure under `module/` and `tool/` instead of reinventing it.

---

## Pull Request 检查清单 / PR Checklist

提交 PR 前请自查（Before opening a PR, please self-check）：

- [ ] PR 只包含一个明确的改动目标（小而聚焦），描述清楚解决了什么问题 / PR addresses one clear goal, with a description of the problem solved
- [ ] 关联相关 Issue（如有）/ Linked to a related Issue (if any)
- [ ] 代码风格与命名与仓库一致 / Style and naming consistent with the repository
- [ ] 本地已跑通相关冒烟测试 / Relevant smoke tests pass locally
- [ ] 未包含数据集、权重等大文件 / No datasets, weights, or other large files included

---

## 遇到困难？联系我们 / Still Stuck? Contact Us

如果 Fork / PR / 权限等流程问题实在无法解决，可以直接发邮件联系维护者（请附上你的 GitHub 用户名和想做的改动，简单描述即可）：

If you cannot resolve any workflow problem (Fork / PR / permission / etc.) after trying the steps above, email the maintainer directly (please include your GitHub username and a short description of your intended change):

**851132789@qq.com**

邮件标题建议加前缀 `[fairness_fl_code]`。Please prefix your email subject with `[fairness_fl_code]`.

维护者通常会在几天内回复。The maintainer usually replies within a few days.
