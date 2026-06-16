# Fairness Federated Learning Framework

> A production-grade fairness-aware federated learning research framework | 28+ Algorithms | 3 Task Types | 10 Datasets | Full-Stack Engineering

---

## Table of Contents

- [Overview](#overview)
- [Engineering Optimizations](#engineering-optimizations)
- [Getting Started Guide](#getting-started-guide)
- [Project Structure](#project-structure)
- [Algorithms](#algorithms)
- [Datasets](#datasets)
- [Advanced Usage](#advanced-usage)
- [Citation](#citation)

---

## Overview

You're doing **fairness-aware federated learning** research and need to run extensive benchmarking experiments:

- 10+ baselines vs. your method
- 10 datasets across tabular / image / text modalities
- 4 data partitioning strategies (Uniform, Dirichlet 0.1/0.5/1.0), 3 client scales (20/30/40)
- Each experiment repeated 3× for Mean ± STD
- Compute from mixed sources: local RTX 4090/5090, rented P4 machines, shared lab servers

**This framework is built for exactly that.** It's not a "just-get-it-running" script — it's a battle-tested, production-grade experiment system shaped by real-world large-scale experimentation.

### At a Glance

| Dimension | Count |
|-----------|-------|
| Total Algorithms | 28+ (23 active + backups) |
| Fairness-aware FL | 13 |
| Standard FL Baselines | 6 |
| One-shot FL | 3 |
| Task Types | Tabular / Image / Text Classification |
| Datasets | 10 |
| Model Architectures | BERT / CNN / ANN / Logistic Regression |

---

## Engineering Optimizations

Every optimization below was born from real pain during large-scale experiments. No fluff.

### 1. AMP (Automatic Mixed Precision) — Hardware-Aware

- **Auto-detects GPU Compute Capability**: CC >= 7.0 (V100/T4/RTX 2080/A100/4090/5090, etc.) → enables FP16; older GPUs (P4/K80) → auto-disabled
- **All 24 algorithms unified** — AMP is resolved once at the `Experiment()` entry layer via `param_dict['use_amp']` (supports `True` / `False` / `"auto"`), regardless of which `main_*.py` entry point or `run_experiments.py` is used
- Uses PyTorch's built-in `torch.cuda.amp` — **zero extra dependencies**
- Correctly handles gradient accumulation + GradScaler interaction
- 1.5×–2× training speedup, 30%–50% VRAM savings

> Code: [`tool/amp_utils.py`](tool/amp_utils.py), [`experiment.py`](experiment.py) L502

### 2. Image Cache Stacked Tensors — 2× Memory Efficiency

- Old: each sample stored as a separate Python dict → list of 5000 dicts in a `.pt` file
- New: `torch.stack` into one big tensor (e.g., `[5000, 3, 224, 224]`), eliminating Python dict overhead
- ~5× cache load speedup, ~50% RAM reduction
- Covers all image datasets: CelebA / UTKFace / FairFace / LFWA+
- Old-format caches auto-invalidated, auto-rebuilt

> Code: [`moudle/dataset.py`](moudle/dataset.py) `_load_shards_stacked()`

### 3. Checkpoint & Resume — Per-Round, Per-Repeat Precision

- Auto-saves checkpoint after every communication round: model weights, optimizer state, RNG seeds, client selection history
- Three-level naming: `Experiment_NO` + `repeat{N}` + `round_X`
  ```
  save_path/checkpoint_repeat0_round_5.pt
  save_path/checkpoint_repeat0_round_10.pt
  ```
- Resume mode auto-scans completed rounds and picks up exactly where it left off — **never skips incomplete repeats**
- `clean_old_checkpoints()` for automatic cleanup of stale checkpoints

> Code: [`tool/checkpoint.py`](tool/checkpoint.py)

### 4. `torch.compile` Model Acceleration

- **Opt-in** (default off) via `param_dict['use_compile'] = True`
- BERT text tasks: auto-uses `mode="reduce-overhead"` to minimize recompilation from dynamic seq_len
- CNN image tasks: default mode
- Small tabular models: auto-skipped (compilation overhead > benefit)
- Auto-detects PyTorch >= 2.0 availability; graceful fallback if unavailable

> Code: [`moudle/experiment_setup.py`](moudle/experiment_setup.py) `Experiment_Create_model()`

### 5. Memory-Aware DataLoader — OOM Prevention

- Runtime memory monitoring: >85% usage → auto `num_workers=0` to avoid multi-process fork MemoryError
- Smart `pin_memory` (enabled for GPU training, disabled for CPU)
- Image cache building uses `ThreadPoolExecutor` (shared memory) instead of `ProcessPoolExecutor` (fork explosion)
- `get_dataloader_config()` for one-shot optimal DataLoader setup

> Code: [`tool/memory_utils.py`](tool/memory_utils.py)

### 6. Distributed Task Queue — Multi-Machine Collaboration

- Uses a **GitHub repository** as the coordination hub (no OSS/NAS needed)
- Per-task lock files + heartbeat mechanism to prevent race conditions
- Heartbeat timeout auto-reclamation (if a P4 instance gets evicted, another machine takes over)
- Contributor registration + hardware reporting + experiment reproducibility tracking (git hash, Python version, dependency list)
- Worker script with background heartbeat, cloud checkpointing, SIGTERM graceful shutdown

> Code: [`tool/task_queue.py`](tool/task_queue.py), [`tool/worker.py`](tool/worker.py)

### 7. Cloud Storage Abstraction — Multi-Backend

- Unified interface supporting 4 backends: local filesystem / Tencent COS / Alibaba OSS / AWS S3
- Configurable via config file or environment variables; credentials in `.gitignore`
- Automatic sync of datasets and checkpoints between local and cloud

> Code: [`tool/cloud_storage.py`](tool/cloud_storage.py)

### 8. GPU+CPU Hybrid Parallel Scheduler

- `run_experiments.py`: one script to launch the entire experiment matrix
- Auto-detects available GPUs, schedules in tabular → image → text order
- Mixed GPU + CPU parallel slots
- Repeat experiments (Mean ± STD) automatically managed
- Completed experiments auto-skipped (log-based detection)

> Code: [`run_experiments.py`](run_experiments.py)

### 9. Email Notifications

- Experiment completion / failure notifications via QQ Mail SMTP
- Configuration file excluded from version control

> Code: [`tool/notification.py`](tool/notification.py)

### 10. Contributor Identity & Stats

- Each collaborator configures their academic identity (name, email, affiliation, Google Scholar, ORCID, OpenReview)
- Experiments auto-record executor identity, hardware info, and duration
- `contrib_stats.py` provides a compute contribution leaderboard

> Code: [`tool/user_config.py`](tool/user_config.py), [`tool/contrib_stats.py`](tool/contrib_stats.py)

### 11. Universal Experiment Entry — One Framework, All Scenarios

- 4 `main_*.py` entry points + `run_experiments.py` batch runner + unified `Experiment()` constructor
- Regardless of how you launch, AMP / compile / checkpoint / memory optimizations all apply automatically
- New algorithms only need to implement a standard interface to plug into the entire system

---

## Getting Started Guide

> Assumes you just `git clone`'d this repo, with an NVIDIA GPU and Anaconda/Miniconda installed.

### Step 1: Environment Setup

```bash
# Clone the repo
git clone https://github.com/NOVAflyyy/fairness_fl_code.git
cd fairness_fl_code

# Create Conda environment (recommended)
conda env create -f environment.yml
conda activate FL
```

If Conda is too slow, use pip instead:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers numpy pandas scikit-learn scipy tqdm Pillow mat73 matplotlib openpyxl pyarrow requests psutil PyGithub
```

### Step 2: Set Your Identity (Optional)

Edit `tool/user_config.py` with your name and email:

```python
USER_NAME = "Your Name"
USER_EMAIL = "yourname@example.edu"
USER_AFFILIATION = "Your University"
USER_GITHUB = "your_github_id"
```

### Step 3: Download Datasets

```bash
# Check which datasets are missing
python setup_data.py --check

# Download all missing datasets
python setup_data.py

# Or download specific ones
python setup_data.py --datasets celeba
```

> Tabular datasets (ADULT / COMPAS / DRUG / DUTCH) and text datasets (moji / bios) are already bundled in the repo — no extra download needed.

### Step 4: Run Your First Experiment

```bash
# Tabular classification (fastest — good for verifying your setup)
python main_Tabular_CLF.py --dataset ADULT --algorithm FedAvg

# Image classification
python main_IMG_CLF.py --dataset CelebA --algorithm FedAvg

# Text classification
python main_SENT_CLF.py --dataset moji --algorithm FedAvg
```

### Step 5: Understand the Output

Results are saved to the `result_path/` directory. Each output file name encodes the full experiment configuration.

Logs are in `log_path/` — you'll see entries like:

```
INFO     : Communication Round 5/100 completed, avg loss: 0.4231
INFO     : Evaluation - Accuracy: 0.8523, DEO: 0.0312, SPD: 0.0287
INFO     : Checkpoint saved at round 5
```

### Step 6: Run Batch Experiments

When you need to run the full experiment matrix (multiple algorithms × datasets × partitioning strategies):

```bash
# Edit run_experiments.py to configure your experiment matrix
# Then run:
python run_experiments.py
```

`run_experiments.py` auto-detects your GPUs, schedules tasks in parallel, and automatically skips already-completed experiments.

---

## Project Structure

```
fairness_fl_code/
│
├── main.py                      # General entry (defaults to SENT_CLF)
├── main_Tabular_CLF.py          # Tabular classification entry
├── main_IMG_CLF.py              # Image classification entry
├── main_SENT_CLF.py             # Text classification entry
├── experiment.py                # Core experiment class (AMP/checkpoint/compile unified entry)
├── run_experiments.py           # GPU+CPU hybrid parallel batch scheduler
├── setup_data.py                # Dataset download/pack tool
├── environment.yml              # Conda environment definition
├── README.md                    # This file (English)
├── README_CN.md                 # Chinese version
├── REFERENCES.md                # Paper reference list
│
├── algorithm/                   # 28+ federated learning algorithm implementations
│   ├── FederatedAverage.py      # FedAvg
│   ├── PDFFed.py                # PDFFed (core algorithm)
│   ├── FairFed.py / FedFair.py  # Fair FL algorithms
│   ├── PraFFL.py / LoGoFair.py  # Fair FL algorithms
│   ├── DOSFL.py / OSFL.py       # One-shot FL algorithms
│   ├── Scaffold.py              # SCAFFOLD
│   ├── FederatedProximal.py     # FedProx
│   ├── ...                      # More algorithms
│   ├── abandon/                 # Deprecated experimental algorithms
│   └── backup/                  # Backup algorithms
│
├── hypothesis/                  # Model definitions
│   ├── BERTCLASSIFIER.py        # BERT classifier (text)
│   ├── CNNCLASSIFIER.py         # RegularCNN (image)
│   ├── ANNCLASSIFIER.py         # MLP (tabular)
│   ├── LogisticRegression.py    # Logistic regression (tabular)
│   └── generator.py             # Generator (for distillation experiments)
│
├── moudle/                      # Core modules
│   ├── experiment_setup.py      # Experiment config automation (model creation/dataset detection/compile)
│   ├── dataset.py               # 10 dataset classes + image cache stack optimization
│   └── dataloader.py            # Federated learning DataLoader factory
│
├── tool/                        # Utility toolkit
│   ├── amp_utils.py             # AMP mixed precision (24 algorithms unified)
│   ├── checkpoint.py            # Checkpoint/resume (three-level naming/auto-cleanup)
│   ├── memory_utils.py          # Memory-aware DataLoader configuration
│   ├── cloud_storage.py         # Cloud storage abstraction (local/COS/OSS/S3)
│   ├── task_queue.py            # Distributed experiment task queue
│   ├── worker.py                # Distributed worker (heartbeat/cloud CKPT/graceful shutdown)
│   ├── notification.py          # Email notifications
│   ├── user_config.py           # Contributor identity configuration
│   ├── contrib_stats.py         # Contribution statistics & leaderboard
│   ├── logger.py                # Logging configuration
│   ├── utils.py                 # General utilities (Harmonic Mean, etc.)
│   ├── cleanup.py               # Cleanup utilities
│   ├── estimate_resources.py    # Resource estimation
│   └── config_checker.py        # Configuration checker
│
├── dataset/                     # Dataset directory
│   ├── ADULT/                   # Tabular data (bundled)
│   ├── COMPAS/                  # Tabular data (bundled)
│   ├── DRUG/                    # Tabular data (bundled)
│   ├── DUTCH/                   # Tabular data (bundled)
│   ├── moji/                    # Text data (bundled)
│   ├── bios/                    # Text data (bundled)
│   ├── celeba/                  # CelebA labels (bundled) + images (downloadable)
│   ├── UTKFace/                 # Images (downloadable)
│   ├── FairFace/                # Images (downloadable)
│   ├── LFWAPlus/                # Images (downloadable)
│   └── cache/                   # Image cache (auto-generated, gitignored)
│
├── save_path/                   # Model checkpoints (gitignored)
├── log_path/                    # Experiment logs (gitignored)
├── result_path/                 # Experiment results (gitignored)
│
├── ablation/                    # Ablation study scripts
├── patch_experiment/            # Experiment patches
├── plot_quadrantal_diagram.py   # Quadrant diagram plotter
└── DataScalability.py           # Data scalability experiment
```

---

## Algorithms

### Fairness-Aware Federated Learning

| Algorithm | File | Core Idea |
|-----------|------|-----------|
| **PDFFed** | `PDFFed.py` | Probability distribution-driven fair FL |
| FairFed | `FairFed.py` | Global fairness-aware aggregation reweighting |
| FedFair | `FedFair.py` | Local fairness-constrained training |
| FedFB | `FedFB.py` | FairBatch federated extension |
| FedMix | `FedMix.py` | Multi-objective fairness-accuracy optimization |
| NaiveMix | `NaiveMix.py` | Naive fairness mixing |
| FL_FairBatch | `FL_FairBatch.py` | FairBatch for federated learning |
| mFairFL | `mFairFL.py` | Multi-objective fair FL |
| Simple_mFairFL | `Simple_mFairFL.py` | Simplified mFairFL |
| FedRenyi | `FederatedRenyi.py` | Renyi differential privacy fair FL |
| PraFFL | `PraFFL.py` | Preference-driven fair FL (Hypernetwork) |
| FedFACT | `FedFACT.py` | Fairness-aware cost-sensitive training |
| LoGoFair | `LoGoFair.py` | Local-global fairness joint optimization |
| CoBoosting | `CoBoosting.py` | Collaborative fairness boosting |
| ProxProbability | `ProxProbability.py` | Probabilistic proximal fair FL |

### Standard Federated Learning

| Algorithm | File | Core Idea |
|-----------|------|-----------|
| FedAvg | `FederatedAverage.py` | Weighted average aggregation |
| FedProx | `FederatedProximal.py` | Proximal regularization for local updates |
| SCAFFOLD | `Scaffold.py` | Control variates to correct client drift |
| FedNova | `FederatedNova.py` | Normalized heterogeneous updates |
| FedProto | `FederatedProto.py` | Prototype contrastive FL |
| FedRep | `FederatedRep.py` | Representation-classifier decoupled training |

### One-Shot Federated Learning

| Algorithm | File | Core Idea |
|-----------|------|-----------|
| DOSFL | `DOSFL.py` | Distilled one-shot FL |
| OSFL | `OSFL.py` | One-shot FL baseline |
| SeparateTraining | `SeparateTraining.py` | Per-client independent training (non-FL baseline) |

> Full paper references: [REFERENCES.md](REFERENCES.md).

---

## Datasets

### Tabular Classification (Tabular_CLF)

| Dataset | Task | Sensitive Attribute | Size | Source |
|---------|------|---------------------|------|--------|
| ADULT | Income prediction | Gender, Race | ~48K samples | UCI |
| COMPAS | Recidivism prediction | Race, Gender | ~7K samples | ProPublica |
| DRUG | Drug use | Ethnicity, Gender | ~1.9K samples | UCI |
| DUTCH | Occupation prediction | Gender | ~60K samples | CBS |

> Tabular datasets are bundled in the repo — ready to use after `git clone`.

### Image Classification (IMG_CLF)

| Dataset | Task | Sensitive Attribute | Image Size | Size |
|---------|------|---------------------|------------|------|
| CelebA | Smile detection | Gender | 64×64 | ~1.4 GB |
| UTKFace | Gender classification | Race | 64×64 | ~1.5 GB |
| FairFace | Race classification | Gender | 224×224 | ~500 MB |
| LFWA+ | Smile detection | Gender | 64×64 | ~100 MB |

> Label files are bundled in the repo; images must be downloaded via `python setup_data.py`.

### Text Classification (SENT_CLF)

| Dataset | Task | Sensitive Attribute | Model | Size |
|---------|------|---------------------|-------|------|
| moji | Sentiment analysis | Sentiment bias | BERT-base | ~100K samples |
| bios | Occupation classification | Gender | BERT-base | ~400K samples |

> Text datasets are bundled in the repo — ready to use after `git clone`.

---

## Advanced Usage

### Full Parameter Configuration

```bash
# Each main_*.py supports rich command-line arguments
python main_Tabular_CLF.py \
    --dataset ADULT \
    --algorithm PDFFed \
    --lr 0.001 \
    --batch_size 128 \
    --communication_rounds 100 \
    --clients 20 \
    --partition Dirichlet05 \
    --exp_repeat_times 3 \
    --cuda 0 \
    --resume \
    --use_amp auto \
    --use_compile
```

### Batch Experiment Matrix

Edit the configuration in `run_experiments.py`:

```python
ALGORITHMS = ["PDFFed", "FairFed", "FedFair", "LoGoFair", "FedAvg"]
TABULAR_DATASETS = ["COMPAS", "DRUG", "DUTCH", "ADULT"]
SPLITS = ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"]
CLIENTS = ["20Clients", "30Clients", "40Clients"]
EXP_REPEAT_TIMES = 3
```

Then launch with one command:

```bash
python run_experiments.py
```

### Multi-Machine Distributed Experiments

```bash
# Register this machine as a Worker — auto-claims tasks from the queue
python tool/worker.py --type all --status

# Low-end GPU Worker (P4, etc. — AMP auto-disabled)
python tool/worker.py --type image --small_gpu
```

> Requires configuring GitHub repo info in `tool/task_queue.py`.

### AMP Configuration

```bash
# auto: auto-detect GPU capability (recommended)
python main_IMG_CLF.py --use_amp auto

# Force enable
python main_IMG_CLF.py --use_amp true

# Force disable
python main_IMG_CLF.py --use_amp false
```

### Checkpoint & Resume

```bash
# First run
python main_Tabular_CLF.py --resume

# If interrupted, run the same command again to resume from checkpoint
python main_Tabular_CLF.py --resume
```

---

## Citation

```bibtex
@misc{fairness_fl_code,
  title   = {Fairness Federated Learning Framework},
  author  = {Zhiyong Ma},
  year    = {2025},
  url     = {https://github.com/NOVAflyyy/fairness_fl_code}
}
```

---

## License

This project is for academic research purposes. See [LICENSE](LICENSE) if available.

---

*Last updated: 2025*

> Chinese documentation: [README_CN.md](README_CN.md)
