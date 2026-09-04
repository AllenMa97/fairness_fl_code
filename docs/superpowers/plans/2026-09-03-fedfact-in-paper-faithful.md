# FedFACT-In Paper-Faithful Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current FedFACT approximation with a binary-text-classification FedFACT-In implementation that follows the paper's calibrated loss, persistent personalized probability ensemble, global/local signed fairness constraints, communication boundary, specialized evaluation, and resumable state.

**Architecture:** Put the mathematical, device-independent primitives and canonical serializable state in `algorithm/fedfact_core.py`; keep round orchestration, two-model client optimization, streaming aggregation, and checkpoint handoff in `algorithm/FedFACT.py`; evaluate final per-client ensembles through `algorithm/fedfact_evaluation.py`. The algorithm consumes the typed data/checkpoint contracts from the prerequisite infrastructure plan, requires all clients every round, keeps inactive personal models as CPU state dictionaries, and transmits only the unified model.

**Tech Stack:** Python 3.11, PyTorch 2.5, NumPy, the repository's `BERTCLASSIFIER` and `BERTCLF_Optimizer`, `unittest`, and the experiment-state infrastructure specified in `docs/superpowers/specs/2026-09-03-paper-faithful-baselines-and-experiment-state-design.md`.

---

## Scope, sources, and fixed choices

Implement **FedFACT-In**, not FedFACT-Post.

Sources of truth:

- Zhang et al., *FedFACT: Federated Counterfactual Fairness-Aware Collaborative Training*, Equations (1)-(4) and Algorithm 1: <https://arxiv.org/abs/2506.03777>
- Official code pinned at commit `26e72f74b077820f1d44856d28c20525b49241b9`: <https://github.com/liizhang/FedFACT/tree/26e72f74b077820f1d44856d28c20525b49241b9>
- Local design: `docs/superpowers/specs/2026-09-03-paper-faithful-baselines-and-experiment-state-design.md`.

The paper's rendered Equation (3) has an ambiguous sign/mark in some PDF extraction. Resolve it with the signed Lagrangian and the official `get_cal_Matrix` probability factors. Do not copy the official implementation's elementwise dual clipping, logit averaging, or global-only evaluation: those conflict with the paper/design.

These choices are requirements, not open extension points:

1. Accept only binary labels `{0,1}`, binary protected groups `{0,1}`, and task `SENT_CLF`.
2. `fairness_metric` is exactly `DP` or `EO`. EO applies constraints for both `Y=0` and `Y=1`.
3. Require `FL_fraction == 1.0` and `FL_drop_rate == 0.0`; paper-faithful FedFACT-In is all-client.
4. Each round creates unified `theta_k` from server `theta`; persistent `phi_k` comes from algorithm state. Train both on the same local batches. Upload/aggregate only `theta_k`.
5. Ensemble probabilities, never logits:
   `p_k = w_k softmax(theta_k(x)) + (1-w_k) softmax(phi_k(x))`.
6. DP signed disparity is `P(pred=1|A=1)-P(pred=1|A=0)`. EO contains that disparity conditioned on each `Y=y`.
7. Positive/negative constraint residuals are `d-xi` and `-d-xi`. Maintain separate global `lambda` and per-client `mu_k`, each projected onto its own nonnegative L1 ball.
8. Update global `lambda` from confusion/support totals over **all current client ensembles**, not from any average of `mu_k`.
9. Calibration loss is only `-sum_i M[a,y,i] log_softmax_i`, averaged over samples. Never add ordinary cross entropy.
10. Apply one scalar `kappa=max(0, calibration_epsilon-min(raw_M))` across both protected-group matrices, making every entry at least `calibration_epsilon`.
11. Fail before model/optimizer creation when DP lacks either protected group in a client, or EO lacks any `(protected,label)` cell. Apply the same support rule to each client test split. No smoothing.
12. The special evaluator uses client identity, final server `theta`, final persistent `phi_k`, and final `w_k`. It reports both aggregate/global and per-client-derived local fairness.
13. State tensors for duals/weights/support are CPU `float64`. Model loss uses `float32` under autocast. One scaler steps both optimizers, then updates once.
14. Final reporting is explicitly the final-state ensemble. It does not claim to be the paper's theoretical time-average predictor.
15. Communication is one model download and one unified-model upload per client-round; the persistent personal model is private and not transmitted.

## Prerequisite API

Land the infrastructure pull request before starting this one. This plan consumes, rather than reimplements:

```python
from module.experiment_setup import FederatedDataBundle
from module.partition import DatasetView, extract_dataset_view
from tool.checkpoint import CheckpointState
from tool.experiment_state import AlgorithmRunResult

# FederatedDataBundle:
# training_dataloaders, client_dataset_list, testing_dataloader,
# client_testing_dataloaders, client_testing_dataset_list,
# partition_fingerprint, partition_metadata

# FedFACT keeps its existing positional arguments and adds:
resume_state: CheckpointState | None = None

# Experiment runner evaluator callable type:
FedFACTEvaluator = Callable[
    [torch.nn.Module, dict, FederatedDataBundle, dict], dict
]
```

`save_checkpoint(param_dict, iter_t, global_model, algorithm_state=state,
amp_scaler=scaler)` derives `next_round=iter_t+1` and phase from
`communication_round_I`. The runner restores RNG after constructing all objects,
calls a final-round checkpoint after normal algorithm return, and passes
`CheckpointState.algorithm_state` and scaler state back into FedFACT. Do not
write a second checkpoint format.

## Target file map

Create:

- `algorithm/fedfact_core.py` — configuration, support counts, calibrated matrices/loss, ensembles, confusion/disparity, exponentiated weight update, L1 projection, dual update, state initialization/validation.
- `algorithm/fedfact_evaluation.py` — client-indexed final ensemble evaluator.
- `json/algorithm/FedFACT.json` — explicit paper-faithful defaults.
- `tests/fedfact_test_utils.py` — deterministic tiny text fixtures.
- `tests/test_fedfact_core.py` — hand-calculated math and support tests.
- `tests/test_fedfact_training.py` — round semantics and aggregation-boundary tests.
- `tests/test_fedfact_evaluation.py` — specialized evaluator tests.
- `tests/test_fedfact_resume.py` — uninterrupted-versus-resume equality.
- `tests/test_fedfact_smoke.py` — CPU toy and opt-in BERT/CUDA/AMP smoke tests.

Modify:

- `algorithm/FedFACT.py` — replace the approximation with FedFACT-In orchestration.
- `experiment.py` — exact-name evaluator dispatch and 2-model-copy communication accounting.
- `main_SENT_CLF.py` — force/validate paper-faithful participation and load explicit defaults.
- `main_IMG_CLF.py` and `main_Tabular_CLF.py` — reject exact `FedFACT` with an actionable text-only message.
- `tool/smoke_test_registry.md` — register both smoke commands.
- `README.md`, `README_CN.md`, and `REFERENCES.md` — document/cite the corrected variant.

Do not change unrelated algorithms or generic evaluators.

---

## Task 1: Create the worktree and verify the prerequisite

**Files:**
- Inspect: `docs/superpowers/plans/2026-09-03-repeats-resume-dirichlet.md`
- Inspect: `module/experiment_setup.py`
- Inspect: `module/partition.py`
- Inspect: `tool/checkpoint.py`
- Inspect: `tool/experiment_state.py`
- Inspect: `experiment.py`

- [ ] **Step 1: Create the isolated Ronnie worktree**

```bash
ssh ronnie@10.74.201.81
cd /home/ronnie/fairness_fl_code
git fetch origin
git worktree add /home/ronnie/.config/superpowers/worktrees/fairness_fl_code/fedfact-in-paper-faithful -b fix/fedfact-in-paper-faithful
cd /home/ronnie/.config/superpowers/worktrees/fairness_fl_code/fedfact-in-paper-faithful
```

Expected: Git creates branch `fix/fedfact-in-paper-faithful`.

- [ ] **Step 2: Verify dependency names rather than adding compatibility branches**

```bash
/home/ronnie/anaconda3/envs/FL/bin/python - <<'PY'
import inspect
from module.experiment_setup import FederatedDataBundle
from module.partition import DatasetView, extract_dataset_view
from tool.checkpoint import CheckpointState, load_checkpoint, restore_rng_state, save_checkpoint
from tool.experiment_state import AlgorithmRunResult

assert {
    "training_dataloaders", "client_dataset_list", "testing_dataloader",
    "client_testing_dataloaders", "client_testing_dataset_list",
    "partition_fingerprint", "partition_metadata",
} <= set(FederatedDataBundle.__dataclass_fields__)
assert {"next_round", "phase", "algorithm_state", "amp_scaler_state"} <= set(CheckpointState.__dataclass_fields__)
assert {
    "global_model", "total_gpu_seconds", "total_communication_cost",
    "algorithm_state", "amp_scaler_state", "client_selection_history",
} <= set(AlgorithmRunResult.__dataclass_fields__)
print("FedFACT prerequisite contracts present")
PY
```

Expected: `FedFACT prerequisite contracts present`. If it fails, integrate the prerequisite branch; do not compensate inside FedFACT.

- [ ] **Step 3: Record the clean baseline**

```bash
git status --short
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

Expected: clean status and the prerequisite suite green.

---

## Task 2: Add deterministic tiny text fixtures

**Files:**
- Create: `tests/fedfact_test_utils.py`

- [ ] **Step 1: Add the exact fixture module**

```python
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tool.checkpoint import build_experiment_config_hash


class TinyTextDataset(Dataset):
    def __init__(self, rows: list[tuple[float, int, int]]):
        self.rows = rows
        self.labels = np.asarray([row[1] for row in rows], dtype=np.int64)
        self.protected = np.asarray([row[2] for row in rows], dtype=np.int64)
        self.sample_ids = np.arange(len(rows), dtype=np.int64)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        feature, label, protected = self.rows[index]
        return {
            "input_ids": torch.tensor([feature], dtype=torch.float32),
            "attention_mask": torch.tensor([1], dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
            "protected": torch.tensor(protected, dtype=torch.long),
        }


class TinyTextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = nn.Linear(1, 2)

    def forward(self, input_ids, attention_mask=None):
        features = input_ids.float().reshape(input_ids.shape[0], 1)
        return features, self.out(features)


def balanced_rows(offset=0.0):
    return [
        (-2.0 + offset, 0, 0), (-1.0 + offset, 1, 0),
        (1.0 + offset, 0, 1), (2.0 + offset, 1, 1),
    ]


def make_datasets_and_loaders(batch_size=4):
    datasets = [TinyTextDataset(balanced_rows()), TinyTextDataset(balanced_rows(0.25))]
    loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False) for ds in datasets]
    return datasets, loaders


def seeded_model(seed=17):
    torch.manual_seed(seed)
    return TinyTextClassifier()


def cpu_state_dict(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def assert_state_dict_equal(testcase, left, right):
    testcase.assertEqual(set(left), set(right))
    for name in left:
        testcase.assertTrue(torch.equal(left[name], right[name]), name)


def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fedfact_params(root: Path, rounds=1):
    params = {
        "task": "SENT_CLF",
        "algorithm": "FedFACT",
        "dataset": "toy",
        "dataset_name": "toy",
        "hypothesis": "TinyTextClassifier",
        "split_strategy": "Uniform",
        "num_clients_K": 2,
        "communication_round_I": rounds,
        "algorithm_epoch_T": 1,
        "FL_fraction": 1.0,
        "FL_drop_rate": 0.0,
        "batch_size": 4,
        "learning_rate": 0.05,
        "optimize_method": "SGD",
        "fairness_metric": "DP",
        "global_constraint": 0.10,
        "local_constraint": 0.10,
        "dual_learning_rate": 0.5,
        "dual_bound": 5.0,
        "dual_init": 0.1,
        "ensemble_learning_rate": 0.3,
        "ensemble_weight_init": 0.5,
        "calibration_epsilon": 0.001,
        "device": "cpu",
        "use_amp": False,
        "checkpoint_save_freq": 1,
        "checkpoint_keep_latest": 1,
        "model_path": str(root / "models"),
        "result_path": str(root / "result.json"),
        "log_path": str(root / "run.log"),
        "checkpoint_dir": str(root / "checkpoints"),
        "base_seed": 77,
        "repeat_idx": 0,
        "repeat_seed": 77,
        "partition_fingerprint": "fedfact-test-partition",
        "partition_metadata": {},
        "resume": False,
        "parallel_repeats": 1,
    }
    params["experiment_config_hash"] = build_experiment_config_hash(params)
    return params
```

Keep fixtures deterministic: no shuffling and no dropout.

- [ ] **Step 2: Syntax-check and commit**

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m py_compile tests/fedfact_test_utils.py
git add tests/fedfact_test_utils.py
git commit -m "test: add deterministic FedFACT fixtures"
```

Expected: compilation succeeds and one fixture-only commit is created.

---

## Task 3: Specify configuration, support, projection, and canonical state

**Files:**
- Create: `tests/test_fedfact_core.py`
- Create: `algorithm/fedfact_core.py`

- [ ] **Step 1: Write failing tests**

Create the first test classes in `tests/test_fedfact_core.py`:

```python
import unittest
import numpy as np
import torch

from algorithm.fedfact_core import (
    FedFACTConfig, SupportError, build_support_statistics,
    initialize_fedfact_state, project_nonnegative_l1_ball,
    validate_fedfact_state,
)
from tests.fedfact_test_utils import TinyTextDataset, make_datasets_and_loaders, seeded_model


class ConfigSupportStateTest(unittest.TestCase):
    def test_configuration_is_explicit_and_all_client(self):
        raw = {
            "task": "SENT_CLF", "num_clients_K": 2, "FL_fraction": 1.0,
            "FL_drop_rate": 0.0, "fairness_metric": "EO",
            "global_constraint": .1, "local_constraint": .2,
            "dual_learning_rate": .3, "dual_bound": 5.0, "dual_init": .1,
            "ensemble_learning_rate": .4, "ensemble_weight_init": .5,
            "calibration_epsilon": .001,
        }
        self.assertEqual(FedFACTConfig.from_param_dict(raw).num_constraints, 2)
        with self.assertRaisesRegex(ValueError, "FL_fraction == 1.0"):
            FedFACTConfig.from_param_dict(dict(raw, FL_fraction=.5))
        with self.assertRaisesRegex(ValueError, "FL_drop_rate == 0.0"):
            FedFACTConfig.from_param_dict(dict(raw, FL_drop_rate=.1))
        with self.assertRaisesRegex(ValueError, "DP or EO"):
            FedFACTConfig.from_param_dict(dict(raw, fairness_metric="DEO"))
        with self.assertRaisesRegex(ValueError, "SENT_CLF"):
            FedFACTConfig.from_param_dict(dict(raw, task="IMG_CLF"))

    def test_support_axes_are_client_protected_label(self):
        datasets, _ = make_datasets_and_loaders()
        stats = build_support_statistics(datasets, metric="EO")
        self.assertEqual(tuple(stats.counts.shape), (2, 2, 2))
        torch.testing.assert_close(stats.counts[0], torch.ones((2, 2), dtype=torch.float64))
        torch.testing.assert_close(stats.client_totals, torch.tensor([4., 4.], dtype=torch.float64))

    def test_missing_dp_group_and_eo_cell_fail_closed(self):
        no_group_one = TinyTextDataset([(-1., 0, 0), (1., 1, 0)])
        with self.assertRaisesRegex(SupportError, r"client 0.*protected=1"):
            build_support_statistics([no_group_one], metric="DP")
        no_a1_y1 = TinyTextDataset([(-2., 0, 0), (-1., 1, 0), (1., 0, 1)])
        with self.assertRaisesRegex(SupportError, r"client 0.*protected=1.*label=1"):
            build_support_statistics([no_a1_y1], metric="EO")

    def test_projection_is_over_the_whole_nonnegative_l1_set(self):
        actual = project_nonnegative_l1_ball(
            torch.tensor([.8, .6, -.2], dtype=torch.float64), bound=1.0
        )
        torch.testing.assert_close(actual, torch.tensor([.6, .4, 0.], dtype=torch.float64))
        self.assertLessEqual(actual.sum().item(), 1.0)

    def test_state_has_independent_cpu_personal_models_and_shapes(self):
        model = seeded_model()
        datasets, _ = make_datasets_and_loaders()
        stats = build_support_statistics(datasets, metric="EO")
        state = initialize_fedfact_state(
            model, 2, "EO", stats, dual_init=.1, ensemble_weight_init=.5
        )
        self.assertEqual(state["schema_version"], 1)
        self.assertEqual(state["variant"], "fedfact_in")
        self.assertEqual(tuple(state["global_dual"].shape), (2, 2))
        self.assertEqual(tuple(state["local_duals"].shape), (2, 2, 2))
        self.assertEqual(tuple(state["ensemble_weights"].shape), (2,))
        self.assertIsNot(state["personal_model_states"][0], state["personal_model_states"][1])
        self.assertTrue(all(
            tensor.device.type == "cpu"
            for client_state in state["personal_model_states"]
            for tensor in client_state.values()
        ))
        validate_fedfact_state(state, model, 2, "EO", stats, dual_bound=5.0)
```

- [ ] **Step 2: Prove RED**

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_fedfact_core.ConfigSupportStateTest -v
```

Expected: import failure for `algorithm.fedfact_core`.

- [ ] **Step 3: Implement configuration and support statistics**

In `algorithm/fedfact_core.py`, define:

```python
import math
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class FedFACTConfig:
    task: str
    num_clients: int
    fraction: float
    drop_rate: float
    fairness_metric: Literal["DP", "EO"]
    global_constraint: float
    local_constraint: float
    dual_learning_rate: float
    dual_bound: float
    dual_init: float
    ensemble_learning_rate: float
    ensemble_weight_init: float
    calibration_epsilon: float

    @property
    def num_constraints(self):
        return 1 if self.fairness_metric == "DP" else 2

    @classmethod
    def from_param_dict(cls, params):
        metric = str(params["fairness_metric"]).upper()
        config = cls(
            task=str(params["task"]),
            num_clients=int(params["num_clients_K"]),
            fraction=float(params["FL_fraction"]),
            drop_rate=float(params["FL_drop_rate"]),
            fairness_metric=metric,
            global_constraint=float(params["global_constraint"]),
            local_constraint=float(params["local_constraint"]),
            dual_learning_rate=float(params["dual_learning_rate"]),
            dual_bound=float(params["dual_bound"]),
            dual_init=float(params["dual_init"]),
            ensemble_learning_rate=float(params["ensemble_learning_rate"]),
            ensemble_weight_init=float(params["ensemble_weight_init"]),
            calibration_epsilon=float(params["calibration_epsilon"]),
        )
        numeric = (
            config.fraction, config.drop_rate, config.global_constraint,
            config.local_constraint, config.dual_learning_rate,
            config.dual_bound, config.dual_init,
            config.ensemble_learning_rate, config.ensemble_weight_init,
            config.calibration_epsilon,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("FedFACT configuration values must be finite")
        if config.task != "SENT_CLF":
            raise ValueError("paper-faithful FedFACT only supports SENT_CLF")
        if config.num_clients <= 0:
            raise ValueError("num_clients_K must be positive")
        if config.fraction != 1.0:
            raise ValueError("paper-faithful FedFACT requires FL_fraction == 1.0")
        if config.drop_rate != 0.0:
            raise ValueError("paper-faithful FedFACT requires FL_drop_rate == 0.0")
        if metric not in {"DP", "EO"}:
            raise ValueError("FedFACT fairness_metric must be DP or EO")
        if config.global_constraint < 0 or config.local_constraint < 0:
            raise ValueError("FedFACT global/local constraints must be nonnegative")
        if config.dual_learning_rate <= 0 or config.dual_bound <= 0:
            raise ValueError("FedFACT dual learning rate/bound must be positive")
        if config.dual_init < 0:
            raise ValueError("FedFACT dual_init must be nonnegative")
        units = 1 if metric == "DP" else 2
        if 2 * units * config.dual_init > config.dual_bound:
            raise ValueError("FedFACT initial dual is outside the L1 bound")
        if config.ensemble_learning_rate <= 0:
            raise ValueError("FedFACT ensemble_learning_rate must be positive")
        if not 0 < config.ensemble_weight_init < 1:
            raise ValueError("FedFACT ensemble_weight_init must be in (0,1)")
        if config.calibration_epsilon <= 0:
            raise ValueError("FedFACT calibration_epsilon must be positive")
        return config
```

Use the prerequisite extractor rather than dataset-specific probing:

```python
@dataclass(frozen=True)
class SupportStatistics:
    counts: torch.Tensor          # CPU float64 [client, protected, label]
    client_totals: torch.Tensor   # CPU float64 [client]

def build_support_statistics(datasets, metric):
    counts = torch.zeros((len(datasets), 2, 2), dtype=torch.float64)
    for client_id, dataset in enumerate(datasets):
        view = extract_dataset_view(dataset)
        for label, protected in zip(view.labels.tolist(), view.protected.tolist()):
            if label not in (0, 1) or protected not in (0, 1):
                raise SupportError(
                    f"FedFACT requires binary values; client {client_id} has "
                    f"label={label!r}, protected={protected!r}"
                )
            counts[client_id, int(protected), int(label)] += 1
        for protected in range(2):
            if counts[client_id, protected].sum() == 0:
                raise SupportError(
                    f"FedFACT DP support missing: client {client_id}, protected={protected}"
                )
            if metric == "EO":
                for label in range(2):
                    if counts[client_id, protected, label] == 0:
                        raise SupportError(
                            f"FedFACT EO support missing: client {client_id}, "
                            f"protected={protected}, label={label}"
                        )
    return SupportStatistics(counts, counts.sum(dim=(1, 2)))
```

The validation must run immediately on entry to `FedFACT`, before deep-copying/moving a model or constructing optimizers. Partition metadata may be used for logging, but the canonical counts come from `client_dataset_list` through `extract_dataset_view`.

- [ ] **Step 4: Implement Euclidean L1 projection and state**

Implement Duchi-style projection:

```python
def project_nonnegative_l1_ball(values, bound):
    shape = values.shape
    flat = values.to(torch.float64).flatten().clamp_min(0)
    if flat.sum().item() <= bound:
        return flat.reshape(shape)
    ordered = torch.sort(flat, descending=True).values
    cssv = torch.cumsum(ordered, 0) - bound
    positions = torch.arange(1, flat.numel() + 1, dtype=torch.float64, device=flat.device)
    active = ordered - cssv / positions > 0
    rho = torch.nonzero(active, as_tuple=False)[-1, 0]
    threshold = cssv[rho] / positions[rho]
    return (flat - threshold).clamp_min(0).reshape(shape)
```

Canonical `algorithm_state` schema:

```python
{
    "schema_version": 1,
    "variant": "fedfact_in",
    "fairness_metric": metric,
    "personal_model_states": [cpu_clone(global_model.state_dict()) for client in clients],
    "global_dual": float64_tensor[num_constraints, 2],
    "local_duals": float64_tensor[num_clients, num_constraints, 2],
    "ensemble_weights": float64_tensor[num_clients],
    "support_counts": stats.counts.clone(),
    "client_sample_counts": stats.client_totals.clone(),
}
```

Initialize every positive/negative dual coordinate to `dual_init` and each weight to `ensemble_weight_init`. `validate_fedfact_state` checks schema/variant/metric, all shapes and dtypes, model key/shape compatibility, CPU placement, finite nonnegative duals, each global/client L1 norm `<= dual_bound`, weights strictly in `(0,1)`, and exact support-count equality. It returns a deep CPU clone so caller mutation cannot alter a loaded `CheckpointState`.

- [ ] **Step 5: Run and commit**

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_fedfact_core.ConfigSupportStateTest -v
git add algorithm/fedfact_core.py tests/test_fedfact_core.py
git commit -m "feat: define FedFACT state and support contracts"
```

Expected: 5 tests pass.


---

## Task 4: Lock down calibration matrices and the loss with hand calculations

**Files:**
- Modify: `tests/test_fedfact_core.py`
- Modify: `algorithm/fedfact_core.py`

- [ ] **Step 1: Add exact DP and EO matrix fixtures**

Append this class:

```python
from algorithm.fedfact_core import build_calibration_matrices, calibrated_loss


class CalibrationMatrixAndLossTest(unittest.TestCase):
    def setUp(self):
        # counts[k,a,y]; every global (a,y) count is four and every client group has four.
        self.counts = torch.tensor([
            [[3., 1.], [1., 3.]],
            [[1., 3.], [3., 1.]],
        ], dtype=torch.float64)

    def test_dp_matrix_matches_get_cal_matrix_probability_factors(self):
        matrices = build_calibration_matrices(
            client_id=0,
            support_counts=self.counts,
            metric="DP",
            global_dual=torch.tensor([[.4, .1]], dtype=torch.float64),
            local_dual=torch.tensor([[.2, .05]], dtype=torch.float64),
            epsilon=.001,
        )
        expected = torch.tensor([
            [[2.201, 2.401], [1.201, 3.401]],
            [[2.201, .001], [1.201, 1.001]],
        ], dtype=torch.float64)
        torch.testing.assert_close(matrices, expected, rtol=0, atol=1e-12)

    def test_eo_matrix_matches_both_label_conditioned_terms(self):
        # Constraint axis order is natural label order [y=0, y=1].
        matrices = build_calibration_matrices(
            client_id=0,
            support_counts=self.counts,
            metric="EO",
            global_dual=torch.tensor([[.2, 0.], [.1, 0.]], dtype=torch.float64),
            local_dual=torch.tensor([[.05, 0.], [.02, 0.]], dtype=torch.float64),
            epsilon=.001,
        )
        expected = torch.tensor([
            [[2.601, 2.6676666666666664], [1.601, 3.321]],
            [[2.601, .001], [1.601, 2.094333333333333]],
        ], dtype=torch.float64)
        torch.testing.assert_close(matrices, expected, rtol=0, atol=1e-12)

    def test_calibrated_loss_is_only_selected_matrix_row_times_log_softmax(self):
        matrix = torch.tensor([
            [[2.201, 2.401], [1.201, 3.401]],
            [[2.201, .001], [1.201, 1.001]],
        ], dtype=torch.float64)
        logits = torch.log(torch.tensor([[.8, .2], [.25, .75]], dtype=torch.float32))
        labels = torch.tensor([0, 1])
        protected = torch.tensor([1, 0])
        expected = -(
            2.201 * np.log(.8) + .001 * np.log(.2)
            + 1.201 * np.log(.25) + 3.401 * np.log(.75)
        ) / 2
        actual = calibrated_loss(logits, labels, protected, matrix)
        self.assertAlmostEqual(actual.item(), expected, places=6)

        actual.backward()
        self.assertIsNotNone(logits.grad)
```

Before the backward call, construct `logits` as a leaf:

```python
logits = torch.log(
    torch.tensor([[.8, .2], [.25, .75]], dtype=torch.float32)
).detach().requires_grad_(True)
```

This test would fail if implementation adds CE, weights probabilities rather than log-probabilities, chooses the wrong protected matrix/label row, or detaches rows via `.item()`.

- [ ] **Step 2: Prove RED**

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest   tests.test_fedfact_core.CalibrationMatrixAndLossTest -v
```

Expected: import errors for `build_calibration_matrices` and `calibrated_loss`.

- [ ] **Step 3: Implement the matrix literally from the probability factors**

Use `counts[k,a,y]`, `N=counts.sum()`, protected sign `2*a-1`, and identity base:

```python
def build_calibration_matrices(
    client_id, support_counts, metric, global_dual, local_dual, epsilon
):
    counts = support_counts.to(torch.float64)
    total = counts.sum()
    matrices = torch.eye(2, dtype=torch.float64).repeat(2, 1, 1)
    for a in (0, 1):
        sign = 2 * a - 1
        n_a_k = counts[client_id, a].sum()
        p_a_k = n_a_k / total
        if metric == "DP":
            p_k_given_a = n_a_k / counts[:, a, :].sum()
            d_global = sign * p_k_given_a
            d_local = float(sign)
            correction = (
                (global_dual[0, 0] - global_dual[0, 1]) * d_global
                + (local_dual[0, 0] - local_dual[0, 1]) * d_local
            ) / p_a_k
            matrices[a, :, 1] -= correction
        else:
            for y in (0, 1):
                p_a_y = counts[:, a, y].sum() / total
                p_a_y_k = counts[client_id, a, y] / total
                d_global = sign * p_a_k / p_a_y
                d_local = sign * p_a_k / p_a_y_k
                correction = (
                    (global_dual[y, 0] - global_dual[y, 1]) * d_global
                    + (local_dual[y, 0] - local_dual[y, 1]) * d_local
                ) / p_a_k
                matrices[a, y, 1] -= correction
    kappa = max(0.0, epsilon - matrices.min().item())
    return matrices + kappa
```

Validate all shapes, metric, client index, positive epsilon, finite duals, and support denominators at function entry. Return CPU `float64`; move/cast at the training call site.

Implement the vectorized objective:

```python
def calibrated_loss(logits, labels, protected, matrices):
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("FedFACT requires two-logit model output")
    labels = labels.reshape(-1).long()
    protected = protected.reshape(-1).long()
    matrix = matrices.to(device=logits.device, dtype=torch.float32)
    selected = matrix[protected, labels, :]
    log_probabilities = torch.log_softmax(logits.float(), dim=1)
    return -(selected * log_probabilities).sum(dim=1).mean()
```

Do not normalize each selected matrix row and do not append CE; Equation (4) is the complete objective.

- [ ] **Step 4: Run focused and full core tests**

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest   tests.test_fedfact_core.CalibrationMatrixAndLossTest -v
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_fedfact_core -v
```

Expected: 8 core tests pass.

- [ ] **Step 5: Commit**

```bash
git add algorithm/fedfact_core.py tests/test_fedfact_core.py
git commit -m "feat: implement FedFACT calibrated objective"
```

---

## Task 5: Implement probability ensembles, exponentiated weights, confusion, and signed dual ascent

**Files:**
- Modify: `tests/test_fedfact_core.py`
- Modify: `algorithm/fedfact_core.py`

- [ ] **Step 1: Add failing behavioral tests**

Append:

```python
from algorithm.fedfact_core import (
    ensemble_probabilities, update_ensemble_weight,
    confusion_from_predictions, disparity_from_confusion, update_dual,
)


class EnsembleAndDualTest(unittest.TestCase):
    def test_probability_mixture_has_a_different_decision_than_logit_mixture(self):
        theta = torch.tensor([[0., -10.]])
        phi = torch.tensor([[0., .5]])
        probability_mix = ensemble_probabilities(theta, phi, weight=.1)
        probability_prediction = probability_mix.argmax(1).item()
        logit_prediction = (.1 * theta + .9 * phi).argmax(1).item()
        self.assertEqual(probability_prediction, 1)
        self.assertEqual(logit_prediction, 0)

    def test_weight_moves_toward_the_lower_loss_unified_model(self):
        actual = update_ensemble_weight(
            weight=torch.tensor(.5, dtype=torch.float64),
            theta_loss=.2,
            phi_loss=.8,
            learning_rate=.3,
        )
        self.assertAlmostEqual(actual.item(), .5448788923735801, places=12)
        self.assertGreater(actual.item(), .5)

    def test_dp_and_eo_signed_disparities_use_positive_prediction_rates(self):
        predictions = torch.tensor([0, 1, 1, 1, 0, 0, 1, 1])
        labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
        protected = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        confusion = confusion_from_predictions(predictions, labels, protected)
        self.assertEqual(tuple(confusion.shape), (2, 2, 2))
        torch.testing.assert_close(
            disparity_from_confusion(confusion, "DP"),
            torch.tensor([-.25], dtype=torch.float64),
        )
        torch.testing.assert_close(
            disparity_from_confusion(confusion, "EO"),
            torch.tensor([-.5, 0.], dtype=torch.float64),
        )

    def test_dual_has_distinct_positive_negative_residuals_and_l1_projection(self):
        current = torch.zeros((1, 2), dtype=torch.float64)
        positive = update_dual(current, torch.tensor([.3]), tolerance=.1,
                               learning_rate=.5, bound=1.)
        torch.testing.assert_close(positive, torch.tensor([[.1, 0.]], dtype=torch.float64))
        projected = update_dual(
            torch.tensor([[.8, .6]], dtype=torch.float64),
            torch.tensor([.1]), tolerance=0., learning_rate=1., bound=1.,
        )
        torch.testing.assert_close(projected, torch.tensor([[.7, .3]], dtype=torch.float64))

    def test_global_disparity_is_from_summed_confusions_not_mean_local_duals(self):
        client_zero = torch.tensor([
            [[90., 10.], [0., 0.]],
            [[10., 90.], [0., 0.]],
        ])
        client_one = torch.tensor([
            [[0., 0.], [1., 9.]],
            [[0., 0.], [9., 1.]],
        ])
        global_d = disparity_from_confusion(client_zero + client_one, "DP")
        local_ds = torch.cat([
            disparity_from_confusion(client_zero, "DP"),
            disparity_from_confusion(client_one, "DP"),
        ])
        torch.testing.assert_close(
            local_ds, torch.tensor([.8, -.8], dtype=torch.float64)
        )
        self.assertAlmostEqual(local_ds.mean().item(), 0.)
        self.assertAlmostEqual(global_d.item(), 72 / 110)
        # Dual values are deliberately unrelated; averaging them must not enter this path.
        local_duals = torch.tensor([[[4., 0.]], [[0., 4.]]], dtype=torch.float64)
        next_global = update_dual(
            torch.zeros((1, 2), dtype=torch.float64), global_d,
            tolerance=.1, learning_rate=.5, bound=5.,
        )
        self.assertFalse(torch.equal(next_global, local_duals.mean(0)))
        torch.testing.assert_close(
            next_global,
            torch.tensor([[.5 * (72 / 110 - .1), 0.]], dtype=torch.float64),
        )
```

The first two confusion tensors deliberately have zero true-label rows that are
irrelevant for DP. `disparity_from_confusion(confusion, "DP")` validates only
group denominators; EO validates all four group/label denominators.

- [ ] **Step 2: Prove RED**

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest   tests.test_fedfact_core.EnsembleAndDualTest -v
```

Expected: missing-symbol import failure.

- [ ] **Step 3: Implement stable ensemble/weight primitives**

```python
def ensemble_probabilities(theta_logits, phi_logits, weight):
    if theta_logits.shape != phi_logits.shape or theta_logits.shape[-1] != 2:
        raise ValueError("FedFACT ensemble requires matching two-logit tensors")
    w = torch.as_tensor(weight, device=theta_logits.device, dtype=torch.float32)
    if not bool((w > 0) & (w < 1)):
        raise ValueError("FedFACT ensemble weight must be in (0,1)")
    return w * torch.softmax(theta_logits.float(), 1) + (
        1 - w
    ) * torch.softmax(phi_logits.float(), 1)


def update_ensemble_weight(weight, theta_loss, phi_loss, learning_rate):
    w = torch.as_tensor(weight, dtype=torch.float64)
    theta_loss = torch.as_tensor(theta_loss, dtype=torch.float64)
    phi_loss = torch.as_tensor(phi_loss, dtype=torch.float64)
    log_odds = torch.log(w) - torch.log1p(-w)
    new_weight = torch.sigmoid(log_odds + learning_rate * (phi_loss - theta_loss))
    return new_weight.clamp(torch.finfo(torch.float64).eps, 1 - torch.finfo(torch.float64).eps)
```

This is algebraically identical to
`w exp(-eta L_theta)/(w exp(-eta L_theta)+(1-w) exp(-eta L_phi))`
but avoids underflow.

- [ ] **Step 4: Implement confusion, disparity, and dual updates**

```python
def confusion_from_predictions(predictions, labels, protected):
    confusion = torch.zeros((2, 2, 2), dtype=torch.float64)
    flat = (
        protected.detach().cpu().long() * 4
        + labels.detach().cpu().long() * 2
        + predictions.detach().cpu().long()
    )
    confusion += torch.bincount(flat, minlength=8).reshape(2, 2, 2)
    return confusion


def disparity_from_confusion(confusion, metric):
    confusion = confusion.to(torch.float64)
    if metric == "DP":
        denominator = confusion.sum(dim=(1, 2))
        if (denominator == 0).any():
            raise SupportError("DP disparity requires both protected groups")
        rate = confusion[:, :, 1].sum(1) / denominator
        return (rate[1] - rate[0]).reshape(1)
    denominator = confusion.sum(dim=2)
    if (denominator == 0).any():
        raise SupportError("EO disparity requires every protected/label cell")
    rate = confusion[:, :, 1] / denominator
    return rate[1, :] - rate[0, :]


def update_dual(dual, disparity, tolerance, learning_rate, bound):
    disparity = torch.as_tensor(disparity, dtype=torch.float64).reshape(-1)
    residual = torch.stack((disparity - tolerance, -disparity - tolerance), dim=1)
    if residual.shape != dual.shape:
        raise ValueError("dual/disparity shape mismatch")
    return project_nonnegative_l1_ball(
        dual.to(torch.float64) + learning_rate * residual, bound
    )
```

Validate binary values before `bincount`, all finite inputs, metric, and exact shapes. Projection covers the flattened **whole** global dual, and separately the flattened **whole** dual of each client. Never project each coordinate or EO row independently.

- [ ] **Step 5: Run the skewed pooled-disparity and full core tests**

The fixture above has client disparities `[.8, -.8]`, whose unweighted mean is
zero, but its pooled all-client disparity is `72/110`. This prevents an
implementation from passing by averaging local disparities or local duals.

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest   tests.test_fedfact_core.EnsembleAndDualTest -v
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_fedfact_core -v
```

Expected: all 13 core tests pass.

- [ ] **Step 6: Commit**

```bash
git add algorithm/fedfact_core.py tests/test_fedfact_core.py
git commit -m "feat: add FedFACT ensemble and signed dual updates"
```


---

## Task 6: Specify one-client audit and two-model training

**Files:**
- Create: `tests/test_fedfact_training.py`
- Modify: `algorithm/FedFACT.py`

- [ ] **Step 1: Write tests for logits, the pre-update ensemble audit, and both optimizers**

Create `tests/test_fedfact_training.py`:

```python
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from algorithm.FedFACT import (
    _audit_current_ensemble,
    _forward_text_logits,
    _train_theta_and_phi,
    StreamingModelAverage,
)
from algorithm.fedfact_core import (
    build_calibration_matrices, build_support_statistics,
)
from tests.fedfact_test_utils import (
    TinyTextDataset, cpu_state_dict, make_datasets_and_loaders, seeded_model,
)


class ClientRoundPrimitiveTest(unittest.TestCase):
    def setUp(self):
        self.datasets, self.loaders = make_datasets_and_loaders(batch_size=2)
        self.stats = build_support_statistics(self.datasets, "DP")
        self.matrix = build_calibration_matrices(
            0, self.stats.counts, "DP",
            torch.tensor([[.1, .1]], dtype=torch.float64),
            torch.tensor([[.1, .1]], dtype=torch.float64),
            .001,
        )

    def test_forward_adapter_extracts_exactly_two_logits(self):
        model = seeded_model()
        batch = next(iter(self.loaders[0]))
        logits = _forward_text_logits(model, batch, torch.device("cpu"))
        self.assertEqual(tuple(logits.shape), (2, 2))

    def test_audit_uses_current_probability_ensemble_and_sample_mean_losses(self):
        theta, phi = seeded_model(3), seeded_model(4)
        audit = _audit_current_ensemble(
            theta, phi, self.loaders[0], self.matrix, torch.tensor(.25),
            torch.device("cpu"), use_amp=False,
        )
        self.assertEqual(audit.sample_count, 4)
        self.assertEqual(tuple(audit.confusion.shape), (2, 2, 2))
        self.assertGreater(audit.theta_loss, 0)
        self.assertGreater(audit.phi_loss, 0)
        manual_predictions = []
        with torch.no_grad():
            for batch in self.loaders[0]:
                theta_logits = _forward_text_logits(theta, batch, torch.device("cpu"))
                phi_logits = _forward_text_logits(phi, batch, torch.device("cpu"))
                probabilities = .25 * theta_logits.softmax(1) + .75 * phi_logits.softmax(1)
                manual_predictions.append(probabilities.argmax(1))
        expected_confusion = confusion_from_predictions(
            torch.cat(manual_predictions),
            torch.tensor(self.datasets[0].labels),
            torch.tensor(self.datasets[0].protected),
        )
        torch.testing.assert_close(audit.confusion, expected_confusion)

    def test_same_batches_update_both_theta_and_phi(self):
        theta, phi = seeded_model(7), seeded_model(8)
        theta_before, phi_before = cpu_state_dict(theta), cpu_state_dict(phi)
        batch_trace = []
        _train_theta_and_phi(
            theta, phi, self.loaders[0], self.matrix,
            epochs=1, param_dict={"optimize_method": "sgd", "learning_rate": .05},
            device=torch.device("cpu"), use_amp=False, scaler=None,
            batch_trace=batch_trace,
        )
        self.assertEqual(batch_trace, [[-2., -1.], [1., 2.]])
        self.assertTrue(any(
            not torch.equal(theta_before[name], theta.state_dict()[name])
            for name in theta_before
        ))
        self.assertTrue(any(
            not torch.equal(phi_before[name], phi.state_dict()[name])
            for name in phi_before
        ))

    def test_streaming_average_accepts_only_unified_state(self):
        base = seeded_model(11)
        one, three = copy.deepcopy(base), copy.deepcopy(base)
        with torch.no_grad():
            for parameter in one.parameters():
                parameter.fill_(1)
            for parameter in three.parameters():
                parameter.fill_(3)
        average = StreamingModelAverage(base.state_dict(), total_weight=4)
        average.add(cpu_state_dict(one), sample_weight=1)
        average.add(cpu_state_dict(three), sample_weight=3)
        result = average.finish()
        for name, tensor in base.state_dict().items():
            if tensor.is_floating_point():
                torch.testing.assert_close(result[name], torch.full_like(tensor, 2.5))
```

The optional `batch_trace` is a test-only observer receiving the flattened `input_ids` once per shared batch, before either forward. It must never alter loader iteration.

- [ ] **Step 2: Prove RED against the current approximation**

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_fedfact_training.ClientRoundPrimitiveTest -v
```

Expected: import failures because these helpers do not exist.

- [ ] **Step 3: Replace model extraction and audit code**

Rewrite `algorithm/FedFACT.py`; remove task branches for image/tabular, `ClientParallelExecutor`, per-client `model.pt` files, generic evaluation, and the old cost/dual helpers.

Add:

```python
@dataclass(frozen=True)
class ClientAudit:
    theta_loss: float
    phi_loss: float
    confusion: torch.Tensor
    sample_count: int


def _forward_text_logits(model, batch, device):
    result = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("FedFACT SENT_CLF model must return (features, logits)")
    logits = result[1]
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("FedFACT requires a two-logit classifier")
    return logits


def _audit_current_ensemble(theta, phi, loader, matrices, weight, device, use_amp):
    theta.eval()
    phi.eval()
    theta_loss_sum = phi_loss_sum = 0.0
    sample_count = 0
    predictions, labels, protected = [], [], []
    with torch.no_grad():
        for batch in loader:
            y = batch["labels"].to(device).reshape(-1).long()
            a = batch["protected"].to(device).reshape(-1).long()
            with autocast_context(device, use_amp):
                theta_logits = _forward_text_logits(theta, batch, device)
                phi_logits = _forward_text_logits(phi, batch, device)
                theta_loss = calibrated_loss(theta_logits, y, a, matrices)
                phi_loss = calibrated_loss(phi_logits, y, a, matrices)
                probabilities = ensemble_probabilities(theta_logits, phi_logits, weight)
            n = y.numel()
            theta_loss_sum += theta_loss.item() * n
            phi_loss_sum += phi_loss.item() * n
            sample_count += n
            predictions.append(probabilities.argmax(1).cpu())
            labels.append(y.cpu())
            protected.append(a.cpu())
    if sample_count == 0:
        raise SupportError("FedFACT client loader is empty")
    pred = torch.cat(predictions)
    return ClientAudit(
        theta_loss_sum / sample_count,
        phi_loss_sum / sample_count,
        confusion_from_predictions(pred, torch.cat(labels), torch.cat(protected)),
        sample_count,
    )
```

The audit happens while both models are in `eval()`, before either is trained. It is the `h_k^t` used for both local/global dual updates. It uses `w_k^t`; `w_k^{t+1}` is only stored after the audit.

- [ ] **Step 4: Train both models on each identical batch**

Create two fresh `BERTCLF_Optimizer` wrappers per client-round using `optimize_method` and `learning_rate`, set their named parameters, and implement:

```python
def _step_both(theta_optimizer, phi_optimizer, scaler):
    if scaler is None:
        theta_optimizer.step()
        phi_optimizer.step()
    else:
        scaler.step(theta_optimizer)
        scaler.step(phi_optimizer)
        scaler.update()


def _train_theta_and_phi(
    theta, phi, loader, matrices, epochs, param_dict,
    device, use_amp, scaler, batch_trace=None,
):
    theta.train()
    phi.train()
    theta_optimizer = _make_optimizer(theta, param_dict)
    phi_optimizer = _make_optimizer(phi, param_dict)
    for _ in range(epochs):
        for batch in loader:
            if batch_trace is not None:
                batch_trace.append(batch["input_ids"].reshape(-1).tolist())
            theta_optimizer.zero_grad()
            phi_optimizer.zero_grad()
            y = batch["labels"].to(device).reshape(-1).long()
            a = batch["protected"].to(device).reshape(-1).long()
            with autocast_context(device, use_amp):
                theta_loss = calibrated_loss(
                    _forward_text_logits(theta, batch, device), y, a, matrices
                )
                phi_loss = calibrated_loss(
                    _forward_text_logits(phi, batch, device), y, a, matrices
                )
            if scaler is None:
                theta_loss.backward()
                phi_loss.backward()
            else:
                scaler.scale(theta_loss).backward()
                scaler.scale(phi_loss).backward()
            _step_both(theta_optimizer, phi_optimizer, scaler)
```

Do not call the repository helper `scaler_step` twice; it calls `scaler.update()` twice. Optimizer state is intentionally client-round-local, so continuous and resumed execution both recreate it at round boundaries.

Use no hidden effective-batch-size rule. `algorithm_epoch_T` means full passes through the loader, and each minibatch is one simultaneous `theta/phi` update as in Algorithm 1.

- [ ] **Step 5: Implement streaming sample-weighted model aggregation**

`StreamingModelAverage` keeps one CPU floating accumulator. `add(state_dict, sample_weight)` validates keys/shapes and adds `sample_weight/total_weight * tensor.cpu()` for floating values. Non-floating buffers must exactly equal the reference and are copied unchanged. `finish()` rejects total added weight different from `total_weight` and returns one state dictionary. Its API accepts no personal state, preventing accidental `phi` aggregation.

- [ ] **Step 6: Run and commit**

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_fedfact_training.ClientRoundPrimitiveTest -v
git add algorithm/FedFACT.py tests/test_fedfact_training.py
git commit -m "feat: train FedFACT unified and personal models"
```

Expected: all 4 tests pass.

---

## Task 7: Implement the complete FedFACT-In round and state handoff

**Files:**
- Modify: `tests/test_fedfact_training.py`
- Modify: `algorithm/FedFACT.py`

- [ ] **Step 1: Add orchestration tests before implementation**

Append tests using `tempfile.TemporaryDirectory` and the infrastructure `AlgorithmRunResult`:

```python
from algorithm.FedFACT import FedFACT
from tool.experiment_state import AlgorithmRunResult
from tests.fedfact_test_utils import (
    assert_state_dict_equal, fedfact_params, seed_everything,
)


class FedFACTOrchestrationTest(unittest.TestCase):
    def test_support_failure_precedes_optimizer_or_model_copy(self):
        invalid = TinyTextDataset([(-1., 0, 0), (1., 1, 0)])
        with tempfile.TemporaryDirectory() as raw:
            params = fedfact_params(Path(raw))
            with patch("algorithm.FedFACT._make_optimizer") as optimizer:
                with self.assertRaisesRegex(Exception, r"client 0.*protected=1"):
                    FedFACT(
                        "cpu", seeded_model(), 1, 1, 1, 1.0, 0.0,
                        [torch.utils.data.DataLoader(invalid, batch_size=2)],
                        invalid, [invalid], dict(params, num_clients_K=1),
                        None, 0,
                    )
            optimizer.assert_not_called()

    def test_round_updates_persistent_phi_and_server_theta_and_records_all_clients(self):
        datasets, loaders = make_datasets_and_loaders(batch_size=2)
        with tempfile.TemporaryDirectory() as raw:
            params = fedfact_params(Path(raw), rounds=1)
            initial = seeded_model(19)
            initial_state = cpu_state_dict(initial)
            seed_everything()
            result = FedFACT(
                "cpu", initial, 1, 2, 1, 1.0, 0.0,
                loaders, datasets[0], datasets, params, None, 0,
            )
            self.assertIsInstance(result, AlgorithmRunResult)
            self.assertEqual(result.client_selection_history, [[0, 1]])
            self.assertEqual(len(result.algorithm_state["personal_model_states"]), 2)
            self.assertTrue(any(
                not torch.equal(initial_state[name], result.global_model.state_dict()[name])
                for name in initial_state
            ))
            for personal in result.algorithm_state["personal_model_states"]:
                self.assertTrue(any(
                    not torch.equal(initial_state[name], personal[name])
                    for name in initial_state
                ))
            self.assertFalse(torch.equal(
                result.algorithm_state["personal_model_states"][0]["out.weight"],
                result.algorithm_state["personal_model_states"][1]["out.weight"],
            ))

    def test_round_uses_old_weight_for_confusion_then_stores_new_weight(self):
        datasets, loaders = make_datasets_and_loaders(batch_size=4)
        with tempfile.TemporaryDirectory() as raw:
            params = fedfact_params(Path(raw), rounds=1)
            observed_weights = []
            real_audit = _audit_current_ensemble
            def recording_audit(*args, **kwargs):
                observed_weights.append(float(torch.as_tensor(args[4])))
                return real_audit(*args, **kwargs)
            with patch("algorithm.FedFACT._audit_current_ensemble", side_effect=recording_audit):
                result = FedFACT(
                    "cpu", seeded_model(23), 1, 2, 1, 1.0, 0.0,
                    loaders, datasets[0], datasets, params, None, 0,
                )
            self.assertEqual(observed_weights, [.5, .5])
            self.assertTrue(torch.isfinite(result.algorithm_state["ensemble_weights"]).all())
```

Pass a harmless training dataset as the legacy `training_dataset`; support must always use the complete `client_dataset_list`.

- [ ] **Step 2: Prove RED**

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest   tests.test_fedfact_training.FedFACTOrchestrationTest -v
```

Expected: the current `FedFACT` return/state semantics fail.

- [ ] **Step 3: Use the exact compatible signature and resume rules**

Keep all legacy positional arguments, then add the typed keyword:

```python
def FedFACT(
    device, global_model,
    algorithm_epoch_T, num_clients_K, communication_round_I,
    FL_fraction, FL_drop_rate,
    training_dataloaders, training_dataset, client_dataset_list,
    param_dict, testing_dataloader, testing_dataset_len,
    start_round=0,
    resume_state: CheckpointState | None = None,
):
```

At entry:

1. Construct `FedFACTConfig` using a copy of `param_dict` overwritten with the positional client/fraction/drop values; reject mismatches rather than trusting duplicate values.
2. Check loader/dataset/client counts.
3. Build support and fail closed.
4. Only then resolve `device`, AMP, state, models, and optimizers.
5. Fresh run requires `start_round == 0` and initializes state.
6. Resume requires non-null `resume_state.algorithm_state`, validates it, uses `resume_state.next_round`, restores scaler from `resume_state.amp_scaler_state`, and carries counters/history forward. If supplied `start_round` is neither 0 nor `next_round`, raise.
7. If `resume_state.phase == "evaluate"`, return immediately without training, preserving model/state/counters/history.
8. The runner, not FedFACT, restores the checkpoint RNG after object construction.

- [ ] **Step 4: Implement Algorithm 1 ordering**

For every round and every client in `range(num_clients_K)`:

```text
theta_k <- deep copy of this round's server theta on CPU, then active device
phi_k   <- model shell loaded from persistent CPU personal state, then active device
M_k     <- calibration(global_dual_t, local_dual_k_t, fixed support)
audit   <- losses and confusion of theta_t/phi_k_t with w_k_t
w_k_next <- exponentiated loss update(audit theta loss, audit phi loss)
mu_k_next <- signed projected ascent(disparity(audit confusion), local_constraint)
train theta_k and phi_k together using M_k on the same batches
persist phi_k state to CPU
stream theta_k state into sample-count-weighted server accumulator
discard both active models and optimizers; empty CUDA cache
```

After all clients:

```text
global_confusion <- sum client audit confusions
lambda_next <- signed projected ascent(
    disparity(global_confusion), global_constraint
)
server theta_next <- streaming unified average
state <- personal states, lambda_next, every mu_k_next, every w_k_next
history <- append list(range(K))
checkpoint round boundary
```

Use the pre-training `w_k^t` for audit/dual, exactly matching `h_k^t` in Algorithm 1. Store `w_k^{t+1}` for the next round/evaluation. Do not recompute confusion from a bare personal or bare server model.

The global update must literally be:

```python
global_confusion = torch.stack(round_confusions).sum(dim=0)
global_disparity = disparity_from_confusion(global_confusion, config.fairness_metric)
state["global_dual"] = update_dual(
    state["global_dual"], global_disparity,
    config.global_constraint, config.dual_learning_rate, config.dual_bound,
)
```

There must be no reference to `state["local_duals"].mean` in this code path.

- [ ] **Step 5: Implement resources, accounting, and checkpoint output**

Keep server model/state on CPU between clients. Only active `theta_k` and `phi_k`, their optimizers, and current batch live on GPU. Synchronize CUDA immediately before/after client timing; CPU runs add zero GPU seconds.

Model size is `sum(parameter.numel()*parameter.element_size()) / 2**20`. Add `2 * num_clients_K * model_MiB` per completed round. Do not count private `phi` as communication.

At configured round boundaries call:

```python
save_checkpoint(
    param_dict, iter_t, global_model,
    algorithm_state=state,
    amp_scaler=scaler,
    total_gpu_seconds=total_gpu_seconds,
    total_runtime_seconds=resume_runtime_seconds + (time.monotonic() - run_started),
    total_communication_cost=total_communication_cost,
    client_selection_history=client_selection_history,
)
clean_old_checkpoints(param_dict, keep_latest=1)
```

Return exactly:

```python
AlgorithmRunResult(
    global_model=global_model.cpu(),
    total_gpu_seconds=total_gpu_seconds,
    total_communication_cost=total_communication_cost,
    algorithm_state=deep_cpu_clone(state),
    amp_scaler_state=None if scaler is None else scaler.state_dict(),
    client_selection_history=client_selection_history,
)
```

Do not write `save_path/global_FedFACT.pt`; final artifact policy belongs to the runner.

- [ ] **Step 6: Run tests and inspect forbidden legacy paths**

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_fedfact_training -v
! grep -nE 'ClientParallelExecutor|client_selection|logits_ens|aggregated_mu|model\.pt|CrossEntropyLoss|BCELoss' algorithm/FedFACT.py
```

Expected: 7 tests pass and grep has no matches.

- [ ] **Step 7: Commit**

```bash
git add algorithm/FedFACT.py tests/test_fedfact_training.py
git commit -m "feat: orchestrate paper-faithful FedFACT-In rounds"
```


---

## Task 8: Add the client-indexed personalized evaluator

**Files:**
- Create: `tests/test_fedfact_evaluation.py`
- Create: `algorithm/fedfact_evaluation.py`

- [ ] **Step 1: Write a fixture where the generic global model gives the wrong answer**

Create `tests/test_fedfact_evaluation.py`:

```python
import copy
import tempfile
import unittest
from pathlib import Path

import torch

from algorithm.fedfact_core import build_support_statistics, initialize_fedfact_state
from algorithm.fedfact_evaluation import evaluate_fedfact
from module.experiment_setup import FederatedDataBundle
from tests.fedfact_test_utils import (
    TinyTextDataset, fedfact_params, make_datasets_and_loaders, seeded_model,
)


def make_bundle(datasets, loaders):
    return FederatedDataBundle(
        training_dataloaders=loaders,
        client_dataset_list=datasets,
        testing_dataloader=None,
        client_testing_dataloaders=loaders,
        client_testing_dataset_list=datasets,
        partition_fingerprint="fedfact-evaluator-fixture",
        partition_metadata={},
    )


class FedFACTEvaluatorTest(unittest.TestCase):
    def setUp(self):
        self.datasets, self.loaders = make_datasets_and_loaders(batch_size=2)
        self.bundle = make_bundle(self.datasets, self.loaders)
        self.global_model = seeded_model(41)
        with torch.no_grad():
            self.global_model.out.weight.zero_()
            self.global_model.out.bias.copy_(torch.tensor([2., -2.]))
        stats = build_support_statistics(self.datasets, "DP")
        self.state = initialize_fedfact_state(
            self.global_model, 2, "DP", stats, dual_init=.1,
            ensemble_weight_init=.5,
        )
        personal = copy.deepcopy(self.global_model)
        with torch.no_grad():
            # Personal classifier predicts class one exactly when feature > 0.
            personal.out.weight.copy_(torch.tensor([[-1.], [1.]]))
            personal.out.bias.zero_()
        personal_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in personal.state_dict().items()
        }
        self.state["personal_model_states"] = [
            copy.deepcopy(personal_state), copy.deepcopy(personal_state)
        ]

    def test_metrics_change_with_private_model_weight_and_are_not_global_only(self):
        with tempfile.TemporaryDirectory() as raw:
            params = fedfact_params(Path(raw))
            personal_dominant = copy.deepcopy(self.state)
            personal_dominant["ensemble_weights"] = torch.tensor([.01, .01], dtype=torch.float64)
            result = evaluate_fedfact(
                self.global_model, params, self.bundle, personal_dominant
            )
            self.assertLess(result["SPD"], -.9)
            self.assertGreater(result["global_fairness"], .9)
            self.assertEqual(len(result["local_fairness_by_client"]), 2)

            global_dominant = copy.deepcopy(self.state)
            global_dominant["ensemble_weights"] = torch.tensor([.99, .99], dtype=torch.float64)
            changed = evaluate_fedfact(
                self.global_model, params, self.bundle, global_dominant
            )
            self.assertAlmostEqual(changed["SPD"], 0.)
            self.assertAlmostEqual(changed["global_fairness"], 0.)
            self.assertNotEqual(result["SPD"], changed["SPD"])

    def test_client_id_selects_the_matching_private_model(self):
        with tempfile.TemporaryDirectory() as raw:
            params = fedfact_params(Path(raw))
            state = copy.deepcopy(self.state)
            reversed_personal = copy.deepcopy(self.global_model)
            with torch.no_grad():
                reversed_personal.out.weight.copy_(torch.tensor([[1.], [-1.]]))
                reversed_personal.out.bias.zero_()
            state["personal_model_states"][1] = {
                name: tensor.detach().cpu().clone()
                for name, tensor in reversed_personal.state_dict().items()
            }
            state["ensemble_weights"] = torch.tensor([.01, .01], dtype=torch.float64)
            result = evaluate_fedfact(self.global_model, params, self.bundle, state)
            self.assertGreater(result["local_signed_disparity_by_client"][0][0], 0.9)
            self.assertLess(result["local_signed_disparity_by_client"][1][0], -0.9)

    def test_missing_test_support_raises_instead_of_reporting_zero(self):
        bad = TinyTextDataset([(-1., 0, 0), (1., 1, 0)])
        bad_bundle = make_bundle([bad, self.datasets[1]], [
            torch.utils.data.DataLoader(bad, batch_size=2), self.loaders[1]
        ])
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(Exception, r"client 0.*protected=1"):
                evaluate_fedfact(
                    self.global_model, fedfact_params(Path(raw)), bad_bundle, self.state
                )
```

The expected local signed disparity sign uses core convention `A=1 minus A=0`; the reported `SPD` uses legacy table convention `A=0 minus A=1`.

- [ ] **Step 2: Prove RED**

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_fedfact_evaluation -v
```

Expected: import failure for `algorithm.fedfact_evaluation`.

- [ ] **Step 3: Implement sequential client ensemble inference**

Implement the required hook signature exactly:

```python
def evaluate_fedfact(global_model, param_dict, data_bundle, algorithm_state):
```

Its sequence is:

1. Build/validate `FedFACTConfig`.
2. Build training support from `data_bundle.client_dataset_list`; validate state against it.
3. Build test support from `data_bundle.client_testing_dataset_list`, enforcing DP/EO support before inference.
4. Check that testing loaders and personal states both have exactly `num_clients` entries.
5. Keep final server `theta` in eval mode on the configured device. For each `client_id`, deep-copy a model shell, load `personal_model_states[client_id]`, move only that `phi_k` to the device, and mix probabilities using `ensemble_weights[client_id]`.
6. Create one `[protected,true,pred]` confusion tensor per client; release `phi_k` before continuing.
7. Sum client confusions for global metrics. Do not concatenate generic global-test predictions and do not call `FL_fairness_and_accuracy_test`.

Create a private `_metrics_from_confusions` and return:

```python
{
    "ACC": accuracy,
    "DEO": abs(global_eo_disparity_y1) or None if DP support cannot estimate it,
    "SPD": -global_dp_signed_disparity,
    "FR": 1 - DEO or None,
    "HM": harmonic_mean(ACC, FR) or None,
    "fairness_metric": config.fairness_metric,
    "global_signed_disparity": [float(value) for value in selected_global_disparity],
    "global_fairness": max(abs(selected_global_disparity)),
    "local_signed_disparity_by_client": [
        [float(value) for value in disparity] for disparity in local_disparities
    ],
    "local_fairness_by_client": [
        max(abs(disparity)) for disparity in local_disparities
    ],
    "mean_local_fairness": mean(local_fairness),
    "max_local_fairness": max(local_fairness),
    "global_constraint_violation":
        max(0, global_fairness - config.global_constraint),
    "mean_local_constraint_violation":
        mean(max(0, value - config.local_constraint) for value in local_fairness),
    "max_local_constraint_violation":
        max(max(0, value - config.local_constraint) for value in local_fairness),
}
```

For selected DP, only group support is required. Compute `DEO` only if both global `(A,Y=1)` denominators are nonzero; otherwise return JSON `null` through Python `None`. For selected EO, all relevant denominators have already been validated. `ACC`, `SPD`, and the selected global/local fairness values are always finite.

- [ ] **Step 4: Run tests and verify the evaluator has no generic fallback**

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_fedfact_evaluation -v
! grep -n 'FL_fairness_and_accuracy_test' algorithm/fedfact_evaluation.py
```

Expected: 3 tests pass and grep has no match.

- [ ] **Step 5: Commit**

```bash
git add algorithm/fedfact_evaluation.py tests/test_fedfact_evaluation.py
git commit -m "feat: evaluate FedFACT personalized ensembles"
```

---

## Task 9: Wire exact evaluator dispatch and communication semantics

**Files:**
- Modify: `tests/test_fedfact_evaluation.py`
- Modify: `experiment.py`

- [ ] **Step 1: Add dispatch and communication tests**

Append:

```python
from unittest.mock import patch
from algorithm.FedFACT import FedFACT
from experiment import calculate_communication_cost


class FedFACTExperimentIntegrationTest(unittest.TestCase):
    def test_communication_counts_one_download_and_one_upload(self):
        model = seeded_model()
        params = {
            "communication_round_I": 3,
            "num_clients_K": 2,
            "FL_fraction": 1.0,
            "task": "SENT_CLF",
            "emb_dim": 1,
        }
        model_mib = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / (1024 ** 2)
        self.assertAlmostEqual(
            calculate_communication_cost("FedFACT", params, model),
            round(3 * 2 * 2 * model_mib, 3),
        )

    def test_exact_fedfact_registration_passes_special_evaluator(self):
        # Exercise the small dispatch helper extracted from Experiment, avoiding data/model setup.
        from experiment import get_fedfact_registration
        registration = get_fedfact_registration("FedFACT")
        self.assertIs(registration.algorithm_function, FedFACT)
        self.assertIs(registration.evaluator_function, evaluate_fedfact)
        with self.assertRaisesRegex(ValueError, "Unknown algorithm"):
            get_fedfact_registration("FedFACT-Post")
```

As part of GREEN, extract only the FedFACT name-to-callable decision into
`get_fedfact_registration`; do not rewrite unrelated runner logic.

- [ ] **Step 2: Prove RED**

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest   tests.test_fedfact_evaluation.FedFACTExperimentIntegrationTest -v
```

Expected: 3-copy cost mismatch and missing registration helper.

- [ ] **Step 3: Register the exact variant**

Add:

```python
@dataclass(frozen=True)
class AlgorithmRegistration:
    algorithm_function: Callable
    evaluator_function: Callable | None = None


def get_fedfact_registration(name):
    if name != "FedFACT":
        raise ValueError(
            f"Unknown algorithm: {name}; only paper-faithful FedFACT-In is registered"
        )
    return AlgorithmRegistration(FedFACT, evaluate_fedfact)
```

Leave every non-FedFACT dispatch branch untouched. Replace only the FedFACT
substring branch with:

```python
elif str(param_dict["algorithm"]).startswith("FedFACT"):
    registration = get_fedfact_registration(param_dict["algorithm"])
    Experiment_FL(
        registration.algorithm_function,
        param_dict,
        evaluator_function=registration.evaluator_function,
    )
```

This uses the prerequisite runner's exact
`Experiment_FL(algorithm_function, param_dict, evaluator_function=None)`
signature. Exact `FedFACT` receives `evaluate_fedfact`; a suffix such as
`FedFACT-Post` reaches the helper and fails closed.

Change the communication branch to:

```python
elif algorithm_name == "FedFACT":
    cost = I * K * 2 * model_MB
```

Use `K`, not `K*fraction`, because configuration validation requires all clients. Update the nearby comment to “unified model download + unified update upload; personal model remains private.”

- [ ] **Step 4: Run focused and infrastructure integration suites**

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_fedfact_evaluation -v
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest   tests.test_repeat_runner tests.test_checkpoint -v
```

Expected: FedFACT tests and prerequisite runner/state tests pass; the runner calls `evaluate_fedfact(global_model, param_dict, data_bundle, algorithm_state)` and persists list-valued diagnostic fields while aggregating only scalar numeric metrics.

- [ ] **Step 5: Commit**

```bash
git add experiment.py tests/test_fedfact_evaluation.py
git commit -m "feat: route FedFACT through personalized evaluation"
```


---

## Task 10: Make paper-faithful configuration explicit at every entrypoint

**Files:**
- Create: `json/algorithm/FedFACT.json`
- Create: `tests/test_fedfact_config.py`
- Modify: `algorithm/fedfact_core.py`
- Modify: `main_SENT_CLF.py`
- Modify: `main_IMG_CLF.py`
- Modify: `main_Tabular_CLF.py`

- [ ] **Step 1: Write failing configuration tests**

Create `tests/test_fedfact_config.py`:

```python
import json
import unittest
from pathlib import Path

from algorithm.fedfact_core import validate_fedfact_entrypoint
from main_SENT_CLF import fedfact_fraction_list, merge_fedfact_cli_overrides


class FedFACTConfigurationTest(unittest.TestCase):
    def test_json_names_metric_tolerances_and_all_algorithm_parameters(self):
        config = json.loads(Path("json/algorithm/FedFACT.json").read_text())
        self.assertEqual(config, {
            "fairness_metric": "DP",
            "global_constraint": 0.01,
            "local_constraint": 0.01,
            "dual_learning_rate": 0.03,
            "dual_bound": 5.0,
            "dual_init": 0.1,
            "ensemble_learning_rate": 0.3,
            "ensemble_weight_init": 0.5,
            "calibration_epsilon": 0.001,
            "FL_fraction": 1.0,
            "FL_drop_rate": 0.0,
            "checkpoint_keep_latest": 1,
        })
        self.assertNotIn("fairness_level", config)
        self.assertNotIn("eta_d", config)
        self.assertNotIn("eta_w", config)
        self.assertNotIn("w_init", config)

    def test_fedfact_fraction_is_all_clients_but_other_defaults_are_unchanged(self):
        self.assertEqual(fedfact_fraction_list("FedFACT"), [1.0])
        self.assertEqual(fedfact_fraction_list("FedAvg"), [.1])

    def test_non_null_cli_values_override_json_values(self):
        merged = merge_fedfact_cli_overrides(
            {"fairness_metric": "DP", "global_constraint": .01},
            {"fairness_metric": "EO", "global_constraint": .02,
             "local_constraint": None},
        )
        self.assertEqual(merged["fairness_metric"], "EO")
        self.assertEqual(merged["global_constraint"], .02)
        self.assertNotIn("local_constraint", merged)

    def test_image_and_tabular_entrypoints_are_rejected(self):
        validate_fedfact_entrypoint("FedFACT", "SENT_CLF")
        with self.assertRaisesRegex(ValueError, "only supports SENT_CLF"):
            validate_fedfact_entrypoint("FedFACT", "IMG_CLF")
        with self.assertRaisesRegex(ValueError, "only supports SENT_CLF"):
            validate_fedfact_entrypoint("FedFACT", "Tabular_CLF")
```

- [ ] **Step 2: Prove RED**

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_fedfact_config -v
```

Expected: missing JSON/helpers.

- [ ] **Step 3: Add the exact JSON**

```json
{
  "fairness_metric": "DP",
  "global_constraint": 0.01,
  "local_constraint": 0.01,
  "dual_learning_rate": 0.03,
  "dual_bound": 5.0,
  "dual_init": 0.1,
  "ensemble_learning_rate": 0.3,
  "ensemble_weight_init": 0.5,
  "calibration_epsilon": 0.001,
  "FL_fraction": 1.0,
  "FL_drop_rate": 0.0,
  "checkpoint_keep_latest": 1
}
```

These are the paper's reported `xi_g=xi_k=.01`, the official implementation's initial dual/weight/calibration epsilon, and the pinned implementation's learning-rate/bound choices. Every experiment records them in its canonical config hash.

- [ ] **Step 4: Expose CLI overrides without restoring the old ambiguous key**

Add parser arguments in `main_SENT_CLF.Argparse`, each defaulting to `None`:

```python
parser.add_argument("-fairness_metric", choices=["DP", "EO"], default=None)
parser.add_argument("-global_constraint", type=float, default=None)
parser.add_argument("-local_constraint", type=float, default=None)
parser.add_argument("-dual_learning_rate", type=float, default=None)
parser.add_argument("-dual_bound", type=float, default=None)
parser.add_argument("-dual_init", type=float, default=None)
parser.add_argument("-ensemble_learning_rate", type=float, default=None)
parser.add_argument("-ensemble_weight_init", type=float, default=None)
parser.add_argument("-calibration_epsilon", type=float, default=None)
```

Before reading algorithm JSON, retain those non-null command-line values; after JSON load, apply them via:

```python
FEDFACT_CLI_KEYS = (
    "fairness_metric", "global_constraint", "local_constraint",
    "dual_learning_rate", "dual_bound", "dual_init",
    "ensemble_learning_rate", "ensemble_weight_init",
    "calibration_epsilon",
)

def merge_fedfact_cli_overrides(base, parsed):
    return {**base, **{
        key: parsed[key] for key in FEDFACT_CLI_KEYS
        if parsed.get(key) is not None
    }}
```

Use `fraction_list = fedfact_fraction_list(algorithm)`, where exact `FedFACT` returns `[1.0]` and every other algorithm returns the existing `[0.1]`. Force `FL_drop_rate_list=[0.0]` as already configured. A user attempting to mutate the finalized dictionary later still gets a core validation error.

- [ ] **Step 5: Fail early from unsupported task entrypoints**

Add to core:

```python
def validate_fedfact_entrypoint(algorithm, task):
    if algorithm == "FedFACT" and task != "SENT_CLF":
        raise ValueError(
            "paper-faithful FedFACT-In currently only supports SENT_CLF "
            "with a two-logit BERT-compatible classifier"
        )
```

Call it at the beginning of each entrypoint's `main` function, before
dataset/model construction. It is a no-op for other algorithms. Do not pretend
the sigmoid image/tabular interfaces implement Equation (4).

- [ ] **Step 6: Run and commit**

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_fedfact_config -v
git add json/algorithm/FedFACT.json tests/test_fedfact_config.py   algorithm/fedfact_core.py main_SENT_CLF.py main_IMG_CLF.py main_Tabular_CLF.py
git commit -m "feat: expose explicit FedFACT paper parameters"
```

Expected: 4 tests pass.

---

## Task 11: Prove exact round-boundary resume including private state

**Files:**
- Create: `tests/test_fedfact_resume.py`
- Verify: `algorithm/FedFACT.py`
- Verify: `experiment.py`
- Verify: `tool/checkpoint.py`

- [ ] **Step 1: Write a real runner-level interrupted/resumed regression**

Create `tests/test_fedfact_resume.py` with the imports and helpers below:

```python
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from algorithm.FedFACT import FedFACT
from algorithm.fedfact_evaluation import evaluate_fedfact
from experiment import _run_single_repeat
from module.experiment_setup import FederatedDataBundle
from tool.checkpoint import (
    build_experiment_config_hash, get_repeat_state_dir,
    load_checkpoint, save_checkpoint,
)
from tests.fedfact_test_utils import (
    TinyTextClassifier, TinyTextDataset, balanced_rows,
    fedfact_params, make_datasets_and_loaders,
)


PARTITION_FINGERPRINT = "fedfact-resume-partition"


def assert_nested_equal(testcase, left, right, path="state"):
    if torch.is_tensor(left):
        testcase.assertIsInstance(right, torch.Tensor, path)
        testcase.assertTrue(torch.equal(left, right), path)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right, err_msg=path)
    elif isinstance(left, dict):
        testcase.assertEqual(set(left), set(right), path)
        for key in left:
            assert_nested_equal(testcase, left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        testcase.assertEqual(type(left), type(right), path)
        testcase.assertEqual(len(left), len(right), path)
        for index, (a, b) in enumerate(zip(left, right)):
            assert_nested_equal(testcase, a, b, f"{path}[{index}]")
    else:
        testcase.assertEqual(left, right, path)


class InjectedRoundBoundaryCrash(RuntimeError):
    pass


class FedFACTResumeTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def params(self, name, resume):
        params = fedfact_params(self.root / name, rounds=2)
        params.update({
            "base_seed": 2026,
            "resume": resume,
            "exp_repeat_times": 1,
            "parallel_repeats": 1,
            "final_artifact_policy": "metrics_only",
        })
        params["experiment_config_hash"] = build_experiment_config_hash(params)
        return params

    @staticmethod
    def dataset_factory(params):
        del params
        train = TinyTextDataset(balanced_rows() + balanced_rows(.25))
        test = TinyTextDataset(balanced_rows() + balanced_rows(.25))
        return train, None, test

    @staticmethod
    def dataloader_factory(params, training_dataset, validation_dataset,
                           testing_dataset, split_strategy):
        del params, training_dataset, validation_dataset, testing_dataset, split_strategy
        datasets, loaders = make_datasets_and_loaders(batch_size=2)
        return FederatedDataBundle(
            training_dataloaders=loaders,
            client_dataset_list=datasets,
            testing_dataloader=loaders[0],
            client_testing_dataloaders=loaders,
            client_testing_dataset_list=datasets,
            partition_fingerprint=PARTITION_FINGERPRINT,
            partition_metadata={},
        )

    @staticmethod
    def model_factory(params):
        del params
        return TinyTextClassifier()

    def run_once(self, params):
        with mock.patch(
            "experiment.Experiment_Create_dataset", side_effect=self.dataset_factory
        ), mock.patch(
            "experiment.Experiment_Create_dataloader", side_effect=self.dataloader_factory
        ), mock.patch(
            "experiment.Experiment_Create_model", side_effect=self.model_factory
        ):
            return _run_single_repeat(0, FedFACT, evaluate_fedfact, params)

    @staticmethod
    def checkpoint_params(params):
        return dict(
            params,
            repeat_idx=0,
            repeat_seed=2026,
            partition_fingerprint=PARTITION_FINGERPRINT,
            partition_metadata={},
        )
```

- [ ] **Step 2: Add the continuous-versus-crash/resume assertion**

Append:

```python
    def test_two_round_continuous_equals_round_one_checkpoint_plus_resume(self):
        continuous_params = self.params("continuous", resume=False)
        continuous_result = self.run_once(continuous_params)

        resumed_params = self.params("resumed", resume=True)
        real_save = save_checkpoint

        def save_then_crash(*args, **kwargs):
            path = real_save(*args, **kwargs)
            if int(args[1]) == 0:
                raise InjectedRoundBoundaryCrash("planned FedFACT boundary crash")
            return path

        with mock.patch(
            "algorithm.FedFACT.save_checkpoint", side_effect=save_then_crash
        ):
            with self.assertRaisesRegex(
                InjectedRoundBoundaryCrash, "planned FedFACT boundary crash"
            ):
                self.run_once(resumed_params)

        boundary = load_checkpoint(self.checkpoint_params(resumed_params))
        self.assertEqual(boundary.next_round, 1)
        self.assertEqual(boundary.phase, "train")
        self.assertEqual(boundary.client_selection_history, [[0, 1]])
        self.assertEqual(len(boundary.algorithm_state["personal_model_states"]), 2)
        self.assertIn("global_dual", boundary.algorithm_state)
        self.assertIn("local_duals", boundary.algorithm_state)
        self.assertIn("ensemble_weights", boundary.algorithm_state)
        self.assertIsNone(boundary.amp_scaler_state)

        resumed_result = self.run_once(resumed_params)
        continuous = load_checkpoint(self.checkpoint_params(continuous_params))
        resumed = load_checkpoint(self.checkpoint_params(resumed_params))

        self.assertEqual(continuous.phase, "evaluate")
        self.assertEqual(resumed.phase, "evaluate")
        assert_nested_equal(
            self, continuous.global_model_state, resumed.global_model_state,
            "global_model_state",
        )
        assert_nested_equal(
            self, continuous.algorithm_state, resumed.algorithm_state,
            "algorithm_state",
        )
        assert_nested_equal(
            self, continuous.amp_scaler_state, resumed.amp_scaler_state,
            "amp_scaler_state",
        )
        assert_nested_equal(
            self, continuous.rng_state, resumed.rng_state, "rng_state"
        )
        self.assertEqual(
            continuous.client_selection_history,
            resumed.client_selection_history,
        )
        self.assertEqual(
            continuous.client_selection_history, [[0, 1], [0, 1]]
        )
        self.assertEqual(
            continuous.total_communication_cost,
            resumed.total_communication_cost,
        )
        self.assertEqual(continuous.total_gpu_seconds, 0.0)
        self.assertEqual(resumed.total_gpu_seconds, 0.0)
        assert_nested_equal(
            self, continuous_result.metrics, resumed_result.metrics, "metrics"
        )
```

This uses the real atomic checkpoint and the prerequisite runner, so it checks
model, every `phi_k`, both dual families, weights, RNG, scaler, counters,
history, and final specialized metrics. Do not compare wall-clock runtime.

- [ ] **Step 3: Add evaluation-phase recovery**

Append:

```python
    def test_final_round_checkpoint_without_metrics_resumes_at_evaluation(self):
        params = self.params("evaluate_only", resume=True)
        first_result = self.run_once(params)
        state_params = self.checkpoint_params(params)
        checkpoint = load_checkpoint(state_params)
        self.assertEqual(checkpoint.phase, "evaluate")

        metrics_path = get_repeat_state_dir(state_params) / "metrics.json"
        metrics_path.unlink()

        forbidden = mock.Mock(side_effect=AssertionError("training was rerun"))
        forbidden.__name__ = "FedFACT"
        with mock.patch(
            "experiment.Experiment_Create_dataset", side_effect=self.dataset_factory
        ), mock.patch(
            "experiment.Experiment_Create_dataloader", side_effect=self.dataloader_factory
        ), mock.patch(
            "experiment.Experiment_Create_model", side_effect=self.model_factory
        ):
            recovered = _run_single_repeat(
                0, forbidden, evaluate_fedfact, params
            )
        forbidden.assert_not_called()
        assert_nested_equal(self, first_result.metrics, recovered.metrics, "metrics")


if __name__ == "__main__":
    unittest.main()
```

The prerequisite runner short-circuits training for `phase=="evaluate"` and
evaluates the checkpoint's `algorithm_state`. This reproduces a crash after
the last training checkpoint but before atomic `metrics.json` completion.

- [ ] **Step 4: Prove RED**

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_fedfact_resume -v
```

Expected before complete FedFACT state handoff: a model, private-state, history,
RNG, or metric equality assertion fails.

- [ ] **Step 5: Make only state-boundary corrections**

If RED exposes a defect, constrain changes to these requirements:

- never reinitialize/overwrite personal states, duals, or weights on resume;
- load scaler state exactly once;
- begin at `resume_state.next_round`;
- carry previous history/GPU seconds/communication cost, then append/add;
- checkpoint after server aggregation and global/local dual/weight updates;
- return deep CPU clones so later mutation cannot alter stored state.

The CPU tiny-model equality remains bitwise; do not add a tolerance.

- [ ] **Step 6: Run and commit**

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_fedfact_resume -v
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest   tests.test_checkpoint tests.test_repeat_runner tests.test_fedfact_core   tests.test_fedfact_training tests.test_fedfact_evaluation   tests.test_fedfact_config tests.test_fedfact_resume -v
git add tests/test_fedfact_resume.py algorithm/FedFACT.py
git commit -m "test: prove exact FedFACT round resume"
```

Expected: both resume tests and all prerequisite/FedFACT tests pass.

---

## Task 12: Add CPU and real BERT/AMP smoke coverage

**Files:**
- Create: `tests/test_fedfact_smoke.py`
- Modify: `tool/smoke_test_registry.md`

- [ ] **Step 1: Add the always-on CPU smoke**

Create `tests/test_fedfact_smoke.py` with a one-round, two-client test using the tiny fixtures. Before calling `FedFACT`, enrich parameters exactly as checkpoint infrastructure requires:

```python
params.update({
    "dataset_name": "toy",
    "dataset": "toy",
    "hypothesis": "TinyTextClassifier",
    "split_strategy": "Uniform",
    "base_seed": 77,
    "repeat_idx": 0,
    "repeat_seed": 77,
    "partition_fingerprint": "fedfact-smoke-partition",
    "partition_metadata": {},
    "resume": False,
    "parallel_repeats": 1,
})
params["experiment_config_hash"] = build_experiment_config_hash(params)
```

The test calls FedFACT, then its special evaluator through a `FederatedDataBundle`, and asserts:

- `client_selection_history == [[0,1]]`;
- global/personal parameters are finite;
- global/local duals are finite, nonnegative, and each relevant flattened L1 norm is at most `dual_bound`;
- weights are strictly in `(0,1)`;
- `ACC`, `SPD`, `global_fairness`, `mean_local_fairness`, and `max_local_fairness` are finite;
- latest checkpoint contains both personal states, duals, weights, support, history, RNG, and scaler state `None`.

- [ ] **Step 2: Run the CPU smoke RED, then GREEN**

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest   tests.test_fedfact_smoke.FedFACTSmokeTest.test_cpu_toy_one_round -v
```

Expected before completing checkpoint/evaluator integration: a state assertion fails. Correct the owning module, not the test. Expected after correction: pass.

- [ ] **Step 3: Add opt-in BERT smoke tests**

In the same file add these imports, dataset, and smoke class. Each client has
exactly one sample for every `(protected,label)` cell.

```python
import os
import resource
import shutil

from hypothesis.BERTCLASSIFIER import BertClassifier
from tool.checkpoint import load_checkpoint


class BalancedTokenDataset(torch.utils.data.Dataset):
    def __init__(self, client_id):
        self.labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
        self.protected = np.asarray([0, 0, 1, 1], dtype=np.int64)
        self.sample_ids = np.asarray([
            f"client-{client_id}-sample-{index}" for index in range(4)
        ])
        self.tokens = []
        for label, protected in zip(self.labels, self.protected):
            word = 1000 + 100 * client_id + 10 * int(protected) + int(label)
            self.tokens.append(torch.tensor(
                [101, word, 102, 0, 0, 0, 0, 0], dtype=torch.long
            ))

    def __len__(self):
        return 4

    def __getitem__(self, index):
        tokens = self.tokens[index]
        return {
            "input_ids": tokens,
            "attention_mask": (tokens != 0).long(),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
            "protected": torch.tensor(self.protected[index], dtype=torch.long),
        }


@unittest.skipUnless(
    os.environ.get("RUN_FEDFACT_BERT_SMOKE") == "1" and torch.cuda.is_available(),
    "set RUN_FEDFACT_BERT_SMOKE=1 on a CUDA host with cached bert-base-uncased",
)
class FedFACTBertSmokeTest(unittest.TestCase):
    def run_mode(self, use_amp):
        mode = "amp_on" if use_amp else "amp_off"
        root = Path(os.environ.get(
            "FEDFACT_SMOKE_ROOT", "/tmp/fedfact-in-bert-smoke"
        )) / mode
        if root.exists():
            shutil.rmtree(root)
        datasets = [BalancedTokenDataset(0), BalancedTokenDataset(1)]
        loaders = [
            torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
            for dataset in datasets
        ]
        bundle = FederatedDataBundle(
            training_dataloaders=loaders,
            client_dataset_list=datasets,
            testing_dataloader=loaders[0],
            client_testing_dataloaders=loaders,
            client_testing_dataset_list=datasets,
            partition_fingerprint=f"fedfact-bert-{mode}",
            partition_metadata={},
        )
        params = fedfact_params(root, rounds=1)
        params.update({
            "hypothesis": "BertClassifier",
            "batch_size": 2,
            "device": "cuda",
            "use_amp": use_amp,
            "fairness_metric": "EO",
            "partition_fingerprint": bundle.partition_fingerprint,
        })
        params["experiment_config_hash"] = build_experiment_config_hash(params)
        model = BertClassifier(n_classes=2)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        result = FedFACT(
            "cuda", model, 1, 2, 1, 1.0, 0.0,
            loaders, datasets[0], datasets, params, loaders[0], 4,
        )
        metrics = evaluate_fedfact(result.global_model, params, bundle,
                                   result.algorithm_state)
        checkpoint = load_checkpoint(params)
        cuda_mib = torch.cuda.max_memory_allocated() / 2**20
        rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        checkpoint_mib = checkpoint.path.stat().st_size / 2**20
        print(
            f"FEDFACT_BERT_SMOKE mode={mode} cuda_mib={cuda_mib:.1f} "
            f"rss_mib={rss_mib:.1f} checkpoint_mib={checkpoint_mib:.1f} "
            f"ACC={metrics['ACC']:.6f} "
            f"global={metrics['global_fairness']:.6f} "
            f"max_local={metrics['max_local_fairness']:.6f}"
        )
        return result, metrics, checkpoint.path, cuda_mib, rss_mib

    def test_bert_one_round_amp_off(self):
        result, metrics, path, cuda_mib, rss_mib = self.run_mode(False)
        self.assertIsNone(result.amp_scaler_state)
        self.assertLess(cuda_mib, 12 * 1024)
        self.assertLess(rss_mib, 24 * 1024)
        self.assertLess(path.stat().st_size / 2**20, 2 * 1024)

    def test_bert_one_round_amp_on(self):
        result, metrics, path, cuda_mib, rss_mib = self.run_mode(True)
        self.assertIsNotNone(result.amp_scaler_state)
        self.assertLess(cuda_mib, 12 * 1024)
        self.assertLess(rss_mib, 24 * 1024)
        self.assertLess(path.stat().st_size / 2**20, 2 * 1024)
```

On Linux, `ru_maxrss` is KiB. Print a single `FEDFACT_BERT_SMOKE` line containing mode, CUDA MiB, RSS MiB, checkpoint MiB, ACC, global fairness, and max local fairness. Do not download a model during the smoke; use `TRANSFORMERS_OFFLINE=1`.

- [ ] **Step 4: Register exact commands**

Add two rows to `tool/smoke_test_registry.md`:

```markdown
| FedFACT-In CPU | `python -m unittest tests.test_fedfact_smoke.FedFACTSmokeTest.test_cpu_toy_one_round -v` | Always | State/math/evaluator/checkpoint |
| FedFACT-In BERT AMP | `RUN_FEDFACT_BERT_SMOKE=1 TRANSFORMERS_OFFLINE=1 python -m unittest tests.test_fedfact_smoke.FedFACTBertSmokeTest -v` | Ronnie CUDA | AMP off/on, memory, checkpoint bytes |
```

- [ ] **Step 5: Run CPU smoke and commit**

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest   tests.test_fedfact_smoke.FedFACTSmokeTest.test_cpu_toy_one_round -v
git add tests/test_fedfact_smoke.py tool/smoke_test_registry.md
git commit -m "test: add FedFACT CPU and BERT smoke coverage"
```

Expected: CPU smoke passes; BERT tests are skipped unless explicitly enabled.

---

## Task 13: Document the exact scientific meaning

**Files:**
- Modify: `README.md`
- Modify: `README_CN.md`
- Modify: `REFERENCES.md`

- [ ] **Step 1: Add the English method note**

Under the algorithm section in `README.md`, state:

- `FedFACT` means FedFACT-In at the pinned paper/code version, not post-processing.
- Only binary `SENT_CLF` two-logit models are accepted.
- Every client participates; `FL_fraction=1`, `FL_drop_rate=0`.
- `fairness_metric=DP|EO`, `global_constraint`, and `local_constraint` are explicit.
- EO contains both label-conditioned constraints.
- Predictions/dual updates/evaluation use a probability ensemble of final server `theta` and client-private persistent `phi_k`.
- Only `theta_k` is transmitted/aggregated.
- Missing support fails closed.
- Results include selected global fairness, mean/max local fairness, and constraint violations.
- Checkpoints retain personal models, positive/negative global/local duals, weights, scaler, RNG, counters, and history; only the latest resumable state is retained.
- Final evaluation is final-state, not the theoretical time-average classifier.

Add the reproducible command:

```bash
CUDA_VISIBLE_DEVICES=0 /home/ronnie/anaconda3/envs/FL/bin/python main_SENT_CLF.py   -algorithm FedFACT -dataset moji -split_strategy Dirichlet1   -num_clients_K 2 -algorithm_epoch_T 1 -communication_round_I 1   -fairness_metric DP -global_constraint 0.01 -local_constraint 0.01   -parallel_repeats 1 -checkpoint_keep_latest 1 -resume
```

Do not include a fraction flag; the entrypoint fixes it at one.

- [ ] **Step 2: Add the equivalent concise Chinese note**

Mirror every semantic point and the same command in `README_CN.md`. Use “统一模型 `theta` / 个性模型 `phi_k` / 概率集成 / 全客户端参与 / 缺失支持直接报错” consistently.

- [ ] **Step 3: Pin citations**

Add to `REFERENCES.md`:

```markdown
- **FedFACT-In** — Zhang et al., “FedFACT: Federated Counterfactual
  Fairness-Aware Collaborative Training,” arXiv:2506.03777, 2025.
  Paper: <https://arxiv.org/abs/2506.03777>.
  Official implementation (audited commit
  `26e72f74b077820f1d44856d28c20525b49241b9`):
  <https://github.com/liizhang/FedFACT/tree/26e72f74b077820f1d44856d28c20525b49241b9>.
```

- [ ] **Step 4: Verify terms and commit**

```bash
grep -nE 'FedFACT-In|global_constraint|local_constraint|probability|全客户端|概率集成'   README.md README_CN.md REFERENCES.md
git add README.md README_CN.md REFERENCES.md
git commit -m "docs: document paper-faithful FedFACT-In"
```

Expected: both READMEs and references contain the required semantics.

---

## Task 14: Full verification and paper-faithfulness audit

**Files:**
- Verify all files listed in the target map

- [ ] **Step 1: Run formatting/static checks**

```bash
git diff --check "$(git merge-base origin/main HEAD)" HEAD
/home/ronnie/anaconda3/envs/FL/bin/python -m py_compile   algorithm/fedfact_core.py algorithm/FedFACT.py algorithm/fedfact_evaluation.py   tests/fedfact_test_utils.py tests/test_fedfact_core.py   tests/test_fedfact_training.py tests/test_fedfact_evaluation.py   tests/test_fedfact_config.py tests/test_fedfact_resume.py tests/test_fedfact_smoke.py
```

Expected: no output except normal command headers; exit 0.

- [ ] **Step 2: Run the complete unit suite**

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

Expected: all tests pass; only opt-in BERT/CUDA tests may be skipped.

- [ ] **Step 3: Run focused acceptance tests with unbuffered output**

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -u -m unittest   tests.test_fedfact_core   tests.test_fedfact_training   tests.test_fedfact_evaluation   tests.test_fedfact_config   tests.test_fedfact_resume   tests.test_fedfact_smoke.FedFACTSmokeTest -v
```

Expected: matrix/loss, dual projection/direction, weight direction, probability-vs-logit counterexample, two-model training, theta-only aggregation, pooled global dual, evaluator sensitivity, support failure, resume equality, and CPU smoke all pass.

- [ ] **Step 4: Audit source invariants mechanically**

```bash
grep -n 'log_softmax' algorithm/fedfact_core.py
grep -n 'softmax' algorithm/fedfact_core.py algorithm/fedfact_evaluation.py
grep -n 'global_confusion' algorithm/FedFACT.py
grep -n 'personal_model_states\|global_dual\|local_duals\|ensemble_weights'   algorithm/FedFACT.py algorithm/fedfact_core.py
! grep -R -nE 'fairness_level|eta_d|eta_w|w_init|logits_ens|aggregated_mu|ClientParallelExecutor'   algorithm/FedFACT.py algorithm/fedfact_core.py algorithm/fedfact_evaluation.py   json/algorithm/FedFACT.json
! grep -nE 'CrossEntropyLoss|BCELoss|FL_fairness_and_accuracy_test'   algorithm/FedFACT.py algorithm/fedfact_evaluation.py
```

Expected: positive greps show the intended paths; negative greps return no matches.

- [ ] **Step 5: Run Ronnie BERT AMP off/on smoke**

```bash
cd /home/ronnie/.config/superpowers/worktrees/fairness_fl_code/fedfact-in-paper-faithful
CUDA_VISIBLE_DEVICES=0 RUN_FEDFACT_BERT_SMOKE=1 TRANSFORMERS_OFFLINE=1   /home/ronnie/anaconda3/envs/FL/bin/python -u -m unittest   tests.test_fedfact_smoke.FedFACTBertSmokeTest -v 2>&1 |   tee /tmp/fedfact-in-bert-smoke.log
grep 'FEDFACT_BERT_SMOKE' /tmp/fedfact-in-bert-smoke.log
```

Expected: AMP-off and AMP-on pass; two resource lines stay below 12 GiB peak CUDA, 24 GiB peak RSS, and 2 GiB checkpoint size. If the cached model is absent, provision `bert-base-uncased` before this offline verification; do not let the measured command use the network.

- [ ] **Step 6: Inspect one real checkpoint payload**

```bash
/home/ronnie/anaconda3/envs/FL/bin/python - <<'PY'
from pathlib import Path
import torch

paths = sorted(Path("/tmp/fedfact-in-bert-smoke/amp_on/models").glob(
    "**/checkpoint_latest.pt"
))
assert paths, "BERT smoke produced no checkpoint"
payload = torch.load(paths[-1], map_location="cpu", weights_only=False)
state = payload["algorithm_state"]
assert payload["phase"] == "evaluate"
assert state["variant"] == "fedfact_in"
assert len(state["personal_model_states"]) == 2
assert state["global_dual"].shape[-1] == 2
assert state["local_duals"].shape[0] == 2
assert state["ensemble_weights"].shape == (2,)
assert payload["amp_scaler_state"] is not None
assert payload["rng_state"]["torch_cpu"].numel() > 0
assert payload["client_selection_history"] == [[0, 1]]
print(paths[-1], paths[-1].stat().st_size)
PY
```

Expected: path and nonzero byte size print; every assertion passes. If the
smoke command sets `FEDFACT_SMOKE_ROOT`, use that exact absolute root in this
inspection command as well.

- [ ] **Step 7: Perform the final Equation/Algorithm checklist**

Review the diff beside paper Equations (1)-(4), Algorithm 1, and official `get_cal_Matrix`. Check each item and paste the evidence (test name plus source line) into the pull-request description:

1. DP/EO `D_global` and `D_local` support probabilities and both subtraction signs.
2. Strictly positive common `kappa`.
3. Equation (4) with no extra CE.
4. `h_k^t` probability ensemble with `w_k^t`.
5. Exponentiated loss update direction.
6. Separate positive/negative global and local duals; signed residuals and whole-vector nonnegative L1 projection.
7. Global disparity from summed all-client ensemble confusions.
8. Same local batches update `theta_k` and persistent `phi_k`.
9. Only `theta_k` enters the server average and communication cost.
10. Client-indexed personalized evaluation and global/mean-local/max-local reporting.
11. Full state/scaler/RNG/counter/history checkpoint and exact resume.
12. DP/EO support failure before training and evaluation.

Expected: every item maps to at least one focused test and one implementation location.

- [ ] **Step 8: Inspect scope and commit any verification-only corrections**

```bash
git status --short
git diff --stat "$(git merge-base origin/main HEAD)" HEAD
git log --oneline --decorate -12
```

Expected: only target-map files changed. If verification required a correction, rerun Steps 1-7, then commit that focused correction. Otherwise the worktree is clean.

- [ ] **Step 9: Request review**

Request review specifically for:

- matrix probability factors and EO axis order;
- paper-versus-official choices (L1 projection and evaluation);
- pre-training `w_t` confusion order;
- private state and theta-only communication boundary;
- final-state evaluator labeling;
- checkpoint memory footprint and resume equality.

Use `superpowers:requesting-code-review` before opening the pull request. After review fixes and a repeated full verification, push `fix/fedfact-in-paper-faithful` and open PR 3 against the infrastructure branch (or `main` after PR 1 is merged).
