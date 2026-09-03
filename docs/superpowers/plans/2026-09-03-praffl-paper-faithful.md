# Paper-Faithful PraFFL for BERT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current PraFFL approximation with a paper-faithful BERT adaptation that communicates only the encoder, keeps one persistent private hypernetwork per client, performs the two prescribed local phases, resumes exactly at round boundaries, and reports both the legacy comparison metrics and PraFFL Pareto/hypervolume metrics.

**Architecture:** Split the implementation into a pure PraFFL core, a serial one-active-client trainer/orchestrator, and an algorithm-specific evaluator. The existing `BertClassifier` receives explicit `encode` and `classify` boundaries; generated heads are applied with `torch.nn.functional.linear`, while the post-infrastructure runner owns repeat construction, checkpoint loading, RNG restoration, and atomic final metrics.

**Tech Stack:** Python 3.11, PyTorch, Hugging Face Transformers, NumPy, standard-library `unittest`, the repository's AMP/checkpoint/experiment-state utilities, and no new runtime dependency.

---

## Execution prerequisite and source contract

Implement this plan only after the deterministic-partition/repeat/resume pull request described in `docs/superpowers/specs/2026-09-03-paper-faithful-baselines-and-experiment-state-design.md` has landed. Work in a dedicated branch/worktree based on that commit; the commands below assume:

```bash
cd /home/ronnie/.config/superpowers/worktrees/fairness_fl_code/praffl-paper-faithful
```

The prerequisite pull request provides these exact interfaces; do not duplicate or rename them in this change:

```python
from module.experiment_setup import FederatedDataBundle
from tool.checkpoint import CheckpointState, clean_old_checkpoints, save_checkpoint
from tool.experiment_state import AlgorithmRunResult
```

`FederatedDataBundle` has `training_dataloaders`, `client_dataset_list`, `testing_dataloader`, `client_testing_dataloaders`, `client_testing_dataset_list`, `partition_fingerprint`, and `partition_metadata`. `Experiment_FL` accepts `evaluator_function`; its evaluator contract is:

```python
def evaluate_praffl(global_model, param_dict, data_bundle, algorithm_state) -> dict:
    raise RuntimeError("This signature is documented here and implemented in Task 5")
```

The post-prerequisite runner signature is `Experiment_FL(algorithm_function, param_dict, evaluator_function=None)`. `_run_single_repeat(repeat_idx, algorithm_function, evaluator_function, param_dict)` seeds the repeat, then creates its `FederatedDataBundle` and model, and passes `resume_state` only when the algorithm signature declares it.

PraFFL keeps its existing positional algorithm arguments, adds `resume_state: CheckpointState | None = None`, and returns:

```python
AlgorithmRunResult(
    global_model=global_model,
    total_gpu_seconds=total_gpu_seconds,
    total_communication_cost=total_communication_cost,
    algorithm_state=algorithm_state,
    amp_scaler_state=amp_scaler_state,
    client_selection_history=client_selection_history,
)
```

The checkpoint call used in this plan is the prerequisite API, not the legacy keyword-dictionary API:

```python
save_checkpoint(
    param_dict,
    iter_t,
    global_model,
    algorithm_state=algorithm_state,
    amp_scaler=scaler,
    total_gpu_seconds=total_gpu_seconds,
    total_runtime_seconds=total_runtime_seconds,
    total_communication_cost=total_communication_cost,
    client_selection_history=client_selection_history,
    extra_state={"phase": "train"},
)
```

The runner constructs all models/loaders/algorithm objects before calling `restore_rng_state(checkpoint_state)`. Therefore PraFFL object construction must use a forked RNG and must not consume the restored training RNG. Checkpoints are round-boundary checkpoints; optimizers are intentionally recreated for each client phase, matching the author implementation, so optimizer state is not persistent between rounds. The single AMP scaler is persistent and is checkpointed.

`save_checkpoint` sets `next_round = iter_t + 1` and infers the top-level phase from `param_dict['communication_round_I']`: the final round is saved as `phase='evaluate'`, and earlier rounds as `phase='train'`. PraFFL must not put a competing phase value inside `algorithm_state` or `extra_state`. The exact `CheckpointState` properties consumed below are `algorithm_state`, `amp_scaler_state`, `total_gpu_seconds`, `total_runtime_seconds`, `total_communication_cost`, `client_selection_history`, `next_round`, and `phase`.

## Paper-to-code decisions

The implementation follows equations 3, 6, and 11–15 in the [PraFFL paper](https://arxiv.org/abs/2404.08973) and uses the author repository at commit [`3949240`](https://github.com/rG223/PraFFL/tree/3949240) to resolve tensor shapes and evaluation flow.

The implementation review should keep these pinned references open: author hypernetwork/functional-head helpers in [`utils.py` lines 98–151](https://github.com/rG223/PraFFL/blob/3949240/utils.py#L98-L151), scalarization helpers in [`utils.py` lines 458–494](https://github.com/rG223/PraFFL/blob/3949240/utils.py#L458-L494), private-client training in [`DP_server.py` lines 225–383](https://github.com/rG223/PraFFL/blob/3949240/DP_server.py#L225-L383), encoder-only aggregation in [`DP_server.py` lines 1468–1550](https://github.com/rG223/PraFFL/blob/3949240/DP_server.py#L1468-L1550), and Pareto evaluation in [`DP_server.py` lines 2304–2387](https://github.com/rG223/PraFFL/blob/3949240/DP_server.py#L2304-L2387). The paper overrides the repository's off-by-one phase boundary and first-client aggregation defect.

| Concern | Required decision |
|---|---|
| Communicated parameters | `global_model.bert` only; `global_model.out` is shape metadata/legacy compatibility and is never locally optimized or averaged by PraFFL. |
| Personalized parameters | One CPU-resident two-input hypernetwork state per client; only the selected client's copy is moved to the accelerator. |
| Communicated phase | Exactly `tau_c` complete local epochs, fixed preference `(0.5, 0.5)`, generated head detached/frozen, cross-entropy updates encoder only. |
| Personalized phase | Exactly `tau_p` complete local epochs, encoder/dropout in evaluation mode and feature detached, one encoder call per data batch, `Dirichlet([1, 1])` preference batch, hypernetwork only. |
| Fairness training loss | Squared covariance/correlation surrogate over both binary logits: center the protected attribute and each logit over examples, square their products, average over examples, then sum over classes. |
| Scalarization | `(1 / gamma) * logsumexp(gamma * [L_acc / lambda_acc, L_fair / lambda_fair])`; sampled training preferences are strictly positive. |
| Aggregation | Uniform arithmetic mean of selected encoder state dictionaries, as equation 14 states. Do not copy the author code's erroneous first-client weighting. |
| Evaluation fairness | Paper DP disparity `max_g |P(y_hat=1 | a=g) - P(y_hat=1)|`; the comparison row additionally retains repository-compatible ACC/DEO/signed SPD at the named report preference. |
| Evaluation scope | Every private hypernetwork is evaluated locally on its own client test split and globally on the common test loader; objectives are `(1 - ACC, DP disparity)`. |
| Hypervolume | Exact two-objective minimization hypervolume with reference point `(1, 1)`, implemented locally because `pymoo` is not installed. |
| Preferences | Training uses `Dirichlet([1, 1])`; evaluation uses a deterministic inclusive line grid with 1000 points by default. |
| Unsupported inputs | PraFFL in this pull request accepts `SENT_CLF` plus a binary BERT linear head only and raises a diagnostic error for image/tabular/multiclass configurations. |

The paper's original tabular networks and exact dataset hyperparameters remain outside this BERT adaptation. Do not change FedAvg, FedFACT, LoGoFair, generic monitoring, or generic fairness helpers in this pull request.

## File structure

- Create `algorithm/praffl_core.py`: configuration, hypernetwork, functional heads, paper losses, CPU state helpers, and uniform encoder averaging.
- Create `algorithm/praffl_training.py`: the two isolated optimization phases and one-client training wrapper.
- Replace `algorithm/PraFFL.py`: strict two-phase client training, private-state lifecycle, serial round orchestration, communication accounting, and checkpoint state.
- Modify `hypothesis/BERTCLASSIFIER.py`: explicit, backward-compatible representation/head boundary.
- Create `tool/praffl_evaluation.py`: deterministic grid evaluation, DP metrics, Pareto filtering, two-dimensional hypervolume, and `evaluate_praffl`.
- Modify `experiment.py`: route PraFFL through its evaluator and count encoder communication only.
- Modify `main_SENT_CLF.py`: expose validated PraFFL controls.
- Create `tests/test_praffl_core.py`: exact math, gradients, configuration, and aggregation tests.
- Create `tests/test_praffl_bert_interface.py`: BERT boundary regression without downloading a model.
- Create `tests/test_praffl_training.py`: phase freezing, feature reuse, private persistence, serial resource policy, and communication tests.
- Create `tests/test_praffl_evaluation.py`: grid, DP disparity, Pareto/hypervolume, report preference, and evaluator-state tests.
- Create `tests/test_praffl_resume.py`: continuous-versus-resumed two-round equivalence.
- Create `tests/test_praffl_wiring.py`: CLI, evaluator routing, and formula accounting tests.
- Modify `README.md` and `README_CN.md`: document PraFFL semantics, outputs, and launch flags.

### Task 1: Lock down the paper math and state-dictionary behavior

**Files:**
- Create: `tests/test_praffl_core.py`
- Create: `algorithm/praffl_core.py`

- [ ] **Step 1: Create the failing core test module**

Create `tests/test_praffl_core.py` with:

```python
import math
import unittest
from collections import OrderedDict

import torch

from algorithm.praffl_core import (
    HyperNetwork,
    PraFFLConfig,
    demographic_parity_surrogate,
    functional_linear_heads,
    smooth_tchebycheff,
    uniform_average_state_dicts,
)


class PraFFLCoreTest(unittest.TestCase):
    def test_hypernetwork_accepts_two_preferences_and_gradients_reach_it(self):
        hypernetwork = HyperNetwork(
            preference_dim=2,
            feature_dim=3,
            num_classes=2,
            hidden_dim=5,
        )
        preferences = torch.tensor([[0.5, 0.5], [0.2, 0.8]])
        features = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])

        weight, bias = hypernetwork(preferences)
        logits = functional_linear_heads(features, weight, bias)
        logits.square().mean().backward()

        self.assertEqual(weight.shape, (2, 2, 3))
        self.assertEqual(bias.shape, (2, 2))
        self.assertEqual(logits.shape, (2, 2, 2))
        self.assertTrue(all(parameter.grad is not None for parameter in hypernetwork.parameters()))

    def test_dp_surrogate_matches_hand_calculation_and_is_differentiable(self):
        logits = torch.tensor(
            [[[1.0, 3.0], [5.0, 7.0]]],
            requires_grad=True,
        )
        protected = torch.tensor([0.0, 1.0])

        loss = demographic_parity_surrogate(logits, protected)
        loss.sum().backward()

        self.assertTrue(torch.allclose(loss, torch.tensor([2.0])))
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_inverse_weighted_smooth_tchebycheff_matches_formula(self):
        accuracy_loss = torch.tensor([2.0])
        fairness_loss = torch.tensor([6.0])
        preference = torch.tensor([[0.25, 0.75]])

        actual = smooth_tchebycheff(
            accuracy_loss,
            fairness_loss,
            preference,
            gamma=2.0,
        )
        expected = torch.logsumexp(torch.tensor([[16.0, 16.0]]), dim=1) / 2.0

        self.assertTrue(torch.allclose(actual, expected))

    def test_scalarization_rejects_zero_training_preferences(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            smooth_tchebycheff(
                torch.ones(1),
                torch.ones(1),
                torch.tensor([[0.0, 1.0]]),
                gamma=1.0,
            )

    def test_uniform_average_does_not_apply_dataset_size_weights(self):
        states = [
            OrderedDict(weight=torch.tensor([0.0, 2.0]), counter=torch.tensor(3)),
            OrderedDict(weight=torch.tensor([4.0, 6.0]), counter=torch.tensor(3)),
        ]

        averaged = uniform_average_state_dicts(states)

        self.assertTrue(torch.equal(averaged["weight"], torch.tensor([2.0, 4.0])))
        self.assertEqual(averaged["counter"].item(), 3)

    def test_non_float_buffers_must_match_before_aggregation(self):
        states = [
            OrderedDict(counter=torch.tensor(1)),
            OrderedDict(counter=torch.tensor(2)),
        ]
        with self.assertRaisesRegex(ValueError, "non-floating tensor"):
            uniform_average_state_dicts(states)

    def test_config_splits_total_epochs_and_validates_explicit_sum(self):
        config = PraFFLConfig.from_param_dict(
            {"learning_rate": 5e-5, "optimize_method": "adam"},
            algorithm_epoch_T=5,
        )
        self.assertEqual((config.tau_c, config.tau_p), (2, 3))
        self.assertEqual(config.report_preference, (0.5, 0.5))
        self.assertEqual(config.preference_points, 1000)

        with self.assertRaisesRegex(ValueError, "must equal algorithm_epoch_T"):
            PraFFLConfig.from_param_dict(
                {
                    "learning_rate": 5e-5,
                    "praffl_tau_c": 2,
                    "praffl_tau_p": 2,
                },
                algorithm_epoch_T=5,
            )

    def test_config_rejects_one_total_epoch(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            PraFFLConfig.from_param_dict(
                {"learning_rate": 5e-5},
                algorithm_epoch_T=1,
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify the new module is absent**

Run:

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_praffl_core -v
```

Expected: `ERROR` with `ModuleNotFoundError: No module named 'algorithm.praffl_core'`.

- [ ] **Step 3: Create the core module with the exact public API**

Create `algorithm/praffl_core.py` with:

```python
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


PRAFFL_STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PraFFLConfig:
    tau_c: int
    tau_p: int
    preference_batch_size: int
    hypernetwork_hidden_dim: int
    encoder_learning_rate: float
    hypernetwork_learning_rate: float
    optimizer_name: str
    smooth_gamma: float
    report_preference: tuple[float, float]
    preference_points: int
    preference_chunk_size: int
    hypernetwork_seed_offset: int

    @classmethod
    def from_param_dict(cls, param_dict: Mapping[str, object], algorithm_epoch_T: int) -> "PraFFLConfig":
        if algorithm_epoch_T < 2:
            raise ValueError("PraFFL algorithm_epoch_T must be at least 2")
        raw_tau_c = param_dict.get("praffl_tau_c")
        raw_tau_p = param_dict.get("praffl_tau_p")
        if raw_tau_c is None and raw_tau_p is None:
            tau_c = algorithm_epoch_T // 2
            tau_p = algorithm_epoch_T - tau_c
        elif raw_tau_c is None:
            tau_p = int(raw_tau_p)
            tau_c = algorithm_epoch_T - tau_p
        elif raw_tau_p is None:
            tau_c = int(raw_tau_c)
            tau_p = algorithm_epoch_T - tau_c
        else:
            tau_c = int(raw_tau_c)
            tau_p = int(raw_tau_p)
        if tau_c < 1 or tau_p < 1:
            raise ValueError("PraFFL tau_c and tau_p must each be at least 1")
        if tau_c + tau_p != algorithm_epoch_T:
            raise ValueError("PraFFL tau_c + tau_p must equal algorithm_epoch_T")

        raw_report = param_dict.get("praffl_report_preference", (0.5, 0.5))
        report = tuple(float(value) for value in raw_report)
        if len(report) != 2 or min(report) < 0.0 or not math.isclose(sum(report), 1.0):
            raise ValueError("praffl_report_preference must contain two non-negative values summing to 1")

        config = cls(
            tau_c=tau_c,
            tau_p=tau_p,
            preference_batch_size=int(param_dict.get("praffl_preference_batch_size", 8)),
            hypernetwork_hidden_dim=int(param_dict.get("praffl_hypernetwork_hidden_dim", 256)),
            encoder_learning_rate=float(param_dict.get("learning_rate", 5e-5)),
            hypernetwork_learning_rate=float(param_dict.get("praffl_hypernetwork_learning_rate", 1e-3)),
            optimizer_name=str(param_dict.get("optimize_method", "adam")).lower(),
            smooth_gamma=float(param_dict.get("praffl_smooth_gamma", 1.0)),
            report_preference=(report[0], report[1]),
            preference_points=int(param_dict.get("praffl_preference_points", 1000)),
            preference_chunk_size=int(param_dict.get("praffl_preference_chunk_size", 128)),
            hypernetwork_seed_offset=int(param_dict.get("praffl_hypernetwork_seed_offset", 1701)),
        )
        positive_fields = {
            "praffl_preference_batch_size": config.preference_batch_size,
            "praffl_hypernetwork_hidden_dim": config.hypernetwork_hidden_dim,
            "learning_rate": config.encoder_learning_rate,
            "praffl_hypernetwork_learning_rate": config.hypernetwork_learning_rate,
            "praffl_smooth_gamma": config.smooth_gamma,
            "praffl_preference_points": config.preference_points,
            "praffl_preference_chunk_size": config.preference_chunk_size,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if config.preference_points < 2:
            raise ValueError("praffl_preference_points must be at least 2")
        if config.optimizer_name not in {"adam", "sgd"}:
            raise ValueError("PraFFL optimize_method must be 'adam' or 'sgd'")
        return config


class HyperNetwork(nn.Module):
    def __init__(self, preference_dim: int, feature_dim: int, num_classes: int, hidden_dim: int):
        super().__init__()
        if preference_dim != 2:
            raise ValueError("PraFFL requires a two-dimensional preference")
        self.preference_dim = preference_dim
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        output_dim = num_classes * feature_dim + num_classes
        self.network = nn.Sequential(
            nn.Linear(preference_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, preferences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if preferences.ndim != 2 or preferences.shape[1] != self.preference_dim:
            raise ValueError("preferences must have shape [num_preferences, 2]")
        generated = self.network(preferences)
        split = self.num_classes * self.feature_dim
        weight = generated[:, :split].reshape(-1, self.num_classes, self.feature_dim)
        bias = generated[:, split:].reshape(-1, self.num_classes)
        return weight, bias


def functional_linear_heads(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if features.ndim != 2 or weight.ndim != 3 or bias.ndim != 2:
        raise ValueError("features, weight, and bias must have ranks 2, 3, and 2")
    return torch.stack(
        [
            F.linear(features, head_weight, head_bias)
            for head_weight, head_bias in zip(weight, bias)
        ],
        dim=0,
    )


def preference_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [num_preferences, batch, classes]")
    count, batch_size, num_classes = logits.shape
    expanded_labels = labels.unsqueeze(0).expand(count, batch_size).reshape(-1)
    losses = F.cross_entropy(logits.reshape(-1, num_classes), expanded_labels, reduction="none")
    return losses.reshape(count, batch_size).mean(dim=1)


def demographic_parity_surrogate(logits: torch.Tensor, protected: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or logits.shape[1] != protected.numel():
        raise ValueError("logits and protected batch dimensions must match")
    sensitive = protected.to(dtype=logits.dtype)
    centered_sensitive = sensitive - sensitive.mean()
    centered_logits = logits - logits.mean(dim=1, keepdim=True)
    products = centered_sensitive[None, :, None] * centered_logits
    return products.square().mean(dim=1).sum(dim=1)


def smooth_tchebycheff(
    accuracy_loss: torch.Tensor,
    fairness_loss: torch.Tensor,
    preferences: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    if torch.any(preferences <= 0):
        raise ValueError("training preferences must be strictly positive")
    objectives = torch.stack(
        (accuracy_loss / preferences[:, 0], fairness_loss / preferences[:, 1]),
        dim=1,
    )
    return torch.logsumexp(gamma * objectives, dim=1) / gamma


def clone_state_dict_to_cpu(module: nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, tensor.detach().cpu().clone())
        for name, tensor in module.state_dict().items()
    )


def uniform_average_state_dicts(
    states: Sequence[Mapping[str, torch.Tensor]],
) -> OrderedDict[str, torch.Tensor]:
    if not states:
        raise ValueError("cannot average an empty state list")
    reference_keys = tuple(states[0].keys())
    if any(tuple(state.keys()) != reference_keys for state in states[1:]):
        raise ValueError("all state dictionaries must have identical ordered keys")
    averaged: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name in reference_keys:
        tensors = [state[name].detach().cpu() for state in states]
        if tensors[0].is_floating_point() or tensors[0].is_complex():
            averaged[name] = torch.stack(tensors, dim=0).mean(dim=0)
        else:
            if any(not torch.equal(tensors[0], tensor) for tensor in tensors[1:]):
                raise ValueError(f"non-floating tensor {name!r} differs across clients")
            averaged[name] = tensors[0].clone()
    return averaged
```

The hand calculation in the surrogate test is: protected values center to `(-0.5, 0.5)`; each logit column centers to `(-2, 2)`; all four products have squared value `1`; the class-wise means sum to `2`. This prevents accidentally replacing equation 3 with argmax DP.

- [ ] **Step 4: Run the focused tests**

Run:

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_praffl_core -v
```

Expected: 8 tests pass.

- [ ] **Step 5: Commit the core**

Run:

```bash
git add algorithm/praffl_core.py tests/test_praffl_core.py
git commit -m "feat: add paper-faithful PraFFL core"
```

Expected: one commit containing only the new core and its tests.

### Task 2: Expose a stable BERT encoder/head boundary

**Files:**
- Create: `tests/test_praffl_bert_interface.py`
- Modify: `hypothesis/BERTCLASSIFIER.py`

- [ ] **Step 1: Add a download-free regression test**

Create `tests/test_praffl_bert_interface.py` with:

```python
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hypothesis.BERTCLASSIFIER import BertClassifier


class FakeBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=3)

    def forward(self, input_ids, attention_mask, return_dict=False):
        del attention_mask, return_dict
        sequence = input_ids.float().unsqueeze(-1).expand(-1, -1, 3)
        pooled = sequence.mean(dim=1)
        return sequence, pooled


class BertClassifierBoundaryTest(unittest.TestCase):
    @patch("hypothesis.BERTCLASSIFIER.BertModel.from_pretrained", return_value=FakeBert())
    def test_forward_delegates_to_encode_and_classify_without_changing_legacy_api(self, _factory):
        model = BertClassifier(n_classes=2, pooled_output_flag=False)
        model.eval()
        input_ids = torch.tensor([[1, 2], [3, 4]])
        attention_mask = torch.ones_like(input_ids)

        encoded = model.encode(input_ids, attention_mask)
        classified = model.classify(encoded)
        legacy_encoded = model.only_PLM_forward(input_ids, attention_mask)
        legacy_feature, legacy_logits = model.only_clf_forward(encoded)
        forward_feature, forward_logits = model(input_ids, attention_mask)

        self.assertTrue(torch.equal(encoded, legacy_encoded))
        self.assertTrue(torch.equal(encoded, legacy_feature))
        self.assertTrue(torch.equal(encoded, forward_feature))
        self.assertTrue(torch.equal(classified, legacy_logits))
        self.assertTrue(torch.equal(classified, forward_logits))

    @patch("hypothesis.BERTCLASSIFIER.BertModel.from_pretrained", return_value=FakeBert())
    def test_pooled_flag_selects_pooled_feature(self, _factory):
        model = BertClassifier(n_classes=2, pooled_output_flag=True)
        input_ids = torch.tensor([[1, 3]])
        encoded = model.encode(input_ids, torch.ones_like(input_ids))
        self.assertTrue(torch.equal(encoded, torch.tensor([[2.0, 2.0, 2.0]])))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and observe the missing methods**

Run:

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_praffl_bert_interface -v
```

Expected: 2 errors mentioning that `BertClassifier` has no `encode` method.

- [ ] **Step 3: Deduplicate the model interface**

In `hypothesis/BERTCLASSIFIER.py`, keep `__init__` and `latent_forward` compatible and replace `only_PLM_forward`, `only_clf_forward`, and `forward` with this block, adding `encode` and `classify` immediately before them:

```python
    def encode(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        if self.pooled_output_flag:
            return outputs[1]
        return outputs[0][:, 0, :]

    def classify(self, feature):
        return self.out(self.drop(feature))

    def only_PLM_forward(self, input_ids, attention_mask):
        return self.encode(input_ids, attention_mask)

    def only_clf_forward(self, feature):
        return feature, self.classify(feature)

    def forward(self, input_ids, attention_mask):
        feature = self.encode(input_ids, attention_mask)
        return feature, self.classify(feature)
```

Do not change `latent_forward`; other algorithms may depend on its `inputs_embeds` interface.

- [ ] **Step 4: Run the boundary and existing test suites**

Run:

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_praffl_bert_interface -v
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest discover -s tests -v
```

Expected: the 2 boundary tests and all pre-existing tests pass.

- [ ] **Step 5: Commit the interface**

Run:

```bash
git add hypothesis/BERTCLASSIFIER.py tests/test_praffl_bert_interface.py
git commit -m "refactor: expose BERT representation boundary"
```

Expected: one commit with the backward-compatible model refactor and tests.

### Task 3: Implement the two strict local optimization phases

**Files:**
- Create: `algorithm/praffl_training.py`
- Create: `tests/test_praffl_training.py`
- Test: `tests/test_praffl_core.py`

- [ ] **Step 1: Create phase-isolation regression tests**

Create `tests/test_praffl_training.py` with:

```python
import unittest

import torch

from algorithm.praffl_core import HyperNetwork, PraFFLConfig, clone_state_dict_to_cpu
from algorithm.praffl_training import (
    make_optimizer,
    train_communicated_phase,
    train_personalized_phase,
)


class TinyBertClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = torch.nn.Linear(3, 4)
        self.drop = torch.nn.Identity()
        self.out = torch.nn.Linear(4, 2)
        self.encode_calls = 0

    def encode(self, input_ids, attention_mask):
        del attention_mask
        self.encode_calls += 1
        return self.bert(input_ids.float())


class RecordingHyperNetwork(HyperNetwork):
    def __init__(self):
        super().__init__(2, 4, 2, 6)
        self.seen_preferences = []

    def forward(self, preferences):
        self.seen_preferences.append(preferences.detach().cpu().clone())
        return super().forward(preferences)


def make_config(tau_c=1, tau_p=1):
    return PraFFLConfig.from_param_dict(
        {
            "learning_rate": 0.05,
            "optimize_method": "sgd",
            "praffl_tau_c": tau_c,
            "praffl_tau_p": tau_p,
            "praffl_preference_batch_size": 3,
            "praffl_hypernetwork_hidden_dim": 6,
            "praffl_hypernetwork_learning_rate": 0.05,
        },
        algorithm_epoch_T=tau_c + tau_p,
    )


def make_batches():
    return [
        {
            "input_ids": torch.tensor(
                [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5], [1.0, 1.0, 0.0], [0.5, 0.0, 1.0]]
            ),
            "attention_mask": torch.ones(4, 3, dtype=torch.long),
            "labels": torch.tensor([0, 1, 0, 1]),
            "protected": torch.tensor([0, 0, 1, 1]),
        },
        {
            "input_ids": torch.tensor(
                [[0.5, 1.0, 0.0], [1.0, -0.5, 1.0], [0.0, 0.5, 1.0], [1.0, 1.0, 1.0]]
            ),
            "attention_mask": torch.ones(4, 3, dtype=torch.long),
            "labels": torch.tensor([1, 0, 1, 0]),
            "protected": torch.tensor([0, 1, 0, 1]),
        },
    ]


class PraFFLTrainingPhaseTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.model = TinyBertClassifier()
        self.hypernetwork = RecordingHyperNetwork()
        self.config = make_config()
        self.device = torch.device("cpu")

    def test_communicated_phase_uses_balanced_frozen_head_and_updates_only_encoder(self):
        encoder_before = clone_state_dict_to_cpu(self.model.bert)
        classifier_before = clone_state_dict_to_cpu(self.model.out)
        hypernetwork_before = clone_state_dict_to_cpu(self.hypernetwork)
        optimizer = make_optimizer(
            self.model.bert.parameters(), self.config.optimizer_name, self.config.encoder_learning_rate
        )

        losses = train_communicated_phase(
            self.model,
            self.hypernetwork,
            make_batches(),
            epochs=self.config.tau_c,
            optimizer=optimizer,
            device=self.device,
            use_amp=False,
            scaler=None,
        )

        self.assertEqual(len(losses), 2)
        self.assertTrue(
            any(not torch.equal(encoder_before[name], value) for name, value in self.model.bert.state_dict().items())
        )
        self.assertTrue(
            all(torch.equal(classifier_before[name], value) for name, value in self.model.out.state_dict().items())
        )
        self.assertTrue(
            all(torch.equal(hypernetwork_before[name], value) for name, value in self.hypernetwork.state_dict().items())
        )
        self.assertTrue(
            all(torch.equal(preference, torch.tensor([[0.5, 0.5]])) for preference in self.hypernetwork.seen_preferences)
        )
        self.assertTrue(all(parameter.grad is None for parameter in self.hypernetwork.parameters()))

    def test_personalized_phase_updates_only_hypernetwork_and_encodes_once_per_batch(self):
        encoder_before = clone_state_dict_to_cpu(self.model.bert)
        classifier_before = clone_state_dict_to_cpu(self.model.out)
        hypernetwork_before = clone_state_dict_to_cpu(self.hypernetwork)
        optimizer = make_optimizer(
            self.hypernetwork.parameters(), "adam", self.config.hypernetwork_learning_rate
        )

        def fixed_preferences(count, device, dtype):
            self.assertEqual(count, 3)
            return torch.tensor(
                [[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]],
                device=device,
                dtype=dtype,
            )

        losses = train_personalized_phase(
            self.model,
            self.hypernetwork,
            make_batches(),
            epochs=self.config.tau_p,
            preference_batch_size=self.config.preference_batch_size,
            smooth_gamma=self.config.smooth_gamma,
            optimizer=optimizer,
            device=self.device,
            use_amp=False,
            scaler=None,
            preference_sampler=fixed_preferences,
        )

        self.assertEqual(len(losses), 2)
        self.assertEqual(self.model.encode_calls, 2)
        self.assertTrue(
            all(torch.equal(encoder_before[name], value) for name, value in self.model.bert.state_dict().items())
        )
        self.assertTrue(
            all(torch.equal(classifier_before[name], value) for name, value in self.model.out.state_dict().items())
        )
        self.assertTrue(
            any(not torch.equal(hypernetwork_before[name], value) for name, value in self.hypernetwork.state_dict().items())
        )
        self.assertTrue(all(parameter.grad is None for parameter in self.model.bert.parameters()))

    def test_default_sampler_draws_two_dimensional_dirichlet_preferences(self):
        optimizer = make_optimizer(self.hypernetwork.parameters(), "adam", 0.01)
        train_personalized_phase(
            self.model,
            self.hypernetwork,
            make_batches()[:1],
            epochs=1,
            preference_batch_size=7,
            smooth_gamma=1.0,
            optimizer=optimizer,
            device=self.device,
            use_amp=False,
            scaler=None,
        )
        sampled = self.hypernetwork.seen_preferences[-1]
        self.assertEqual(sampled.shape, (7, 2))
        self.assertTrue(torch.all(sampled > 0))
        self.assertTrue(torch.allclose(sampled.sum(dim=1), torch.ones(7)))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the phase tests and verify the module is missing**

Run:

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_praffl_training -v
```

Expected: `ERROR` with `ModuleNotFoundError: No module named 'algorithm.praffl_training'`.

- [ ] **Step 3: Create the isolated training implementation**

Create `algorithm/praffl_training.py` with:

```python
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import torch

from algorithm.praffl_core import (
    HyperNetwork,
    PraFFLConfig,
    clone_state_dict_to_cpu,
    demographic_parity_surrogate,
    functional_linear_heads,
    preference_cross_entropy,
    smooth_tchebycheff,
)
from tool.amp_utils import autocast_context, scale_backward, scaler_step


PreferenceSampler = Callable[[int, torch.device, torch.dtype], torch.Tensor]


@dataclass(frozen=True)
class ClientTrainResult:
    encoder_state: dict[str, torch.Tensor]
    hypernetwork_state: dict[str, torch.Tensor]
    communicated_losses: tuple[float, ...]
    personalized_losses: tuple[float, ...]
    gpu_seconds: float


def make_optimizer(parameters, optimizer_name: str, learning_rate: float):
    parameters = list(parameters)
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate)
    raise ValueError(f"unsupported PraFFL optimizer {optimizer_name!r}")


def sample_dirichlet_preferences(count: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    concentration = torch.ones(2, device=device, dtype=dtype)
    return torch.distributions.Dirichlet(concentration).sample((count,))


def _move_batch(batch, device: torch.device):
    return (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["labels"].to(device).long(),
        batch["protected"].to(device),
    )


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)
        parameter.grad = None


def _optimizer_step(loss, optimizer, scaler) -> None:
    optimizer.zero_grad(set_to_none=True)
    scale_backward(loss, scaler)
    scaler_step(scaler, optimizer)


def train_communicated_phase(
    model: torch.nn.Module,
    hypernetwork: HyperNetwork,
    dataloader: Iterable[dict[str, torch.Tensor]],
    *,
    epochs: int,
    optimizer,
    device: torch.device,
    use_amp: bool,
    scaler,
) -> list[float]:
    model.bert.train()
    model.drop.train()
    model.out.eval()
    hypernetwork.eval()
    _set_requires_grad(model.bert, True)
    _set_requires_grad(model.out, False)
    _set_requires_grad(hypernetwork, False)
    balanced = torch.tensor([[0.5, 0.5]], device=device)
    with torch.no_grad():
        fixed_weight, fixed_bias = hypernetwork(balanced)
        fixed_weight = fixed_weight.detach()
        fixed_bias = fixed_bias.detach()
    losses: list[float] = []
    for _epoch in range(epochs):
        for batch in dataloader:
            input_ids, attention_mask, labels, _protected = _move_batch(batch, device)
            with autocast_context(device, use_amp):
                features = model.drop(model.encode(input_ids, attention_mask))
                logits = functional_linear_heads(features, fixed_weight, fixed_bias)
                loss = preference_cross_entropy(logits, labels).mean()
            _optimizer_step(loss, optimizer, scaler)
            losses.append(float(loss.detach().cpu()))
    return losses


def train_personalized_phase(
    model: torch.nn.Module,
    hypernetwork: HyperNetwork,
    dataloader: Iterable[dict[str, torch.Tensor]],
    *,
    epochs: int,
    preference_batch_size: int,
    smooth_gamma: float,
    optimizer,
    device: torch.device,
    use_amp: bool,
    scaler,
    preference_sampler: PreferenceSampler = sample_dirichlet_preferences,
) -> list[float]:
    model.bert.eval()
    model.drop.eval()
    model.out.eval()
    hypernetwork.train()
    _set_requires_grad(model.bert, False)
    _set_requires_grad(model.out, False)
    _set_requires_grad(hypernetwork, True)
    losses: list[float] = []
    for _epoch in range(epochs):
        for batch in dataloader:
            input_ids, attention_mask, labels, protected = _move_batch(batch, device)
            with torch.no_grad():
                features = model.drop(model.encode(input_ids, attention_mask)).detach()
            preferences = preference_sampler(
                preference_batch_size,
                device,
                torch.float32,
            )
            with autocast_context(device, use_amp):
                weight, bias = hypernetwork(preferences)
                logits = functional_linear_heads(features, weight, bias)
                accuracy_loss = preference_cross_entropy(logits.float(), labels)
                fairness_loss = demographic_parity_surrogate(logits.float(), protected)
                loss = smooth_tchebycheff(
                    accuracy_loss,
                    fairness_loss,
                    preferences.float(),
                    gamma=smooth_gamma,
                ).mean()
            _optimizer_step(loss, optimizer, scaler)
            losses.append(float(loss.detach().cpu()))
    return losses


def train_praffl_client(
    global_model: torch.nn.Module,
    hypernetwork_template: HyperNetwork,
    private_hypernetwork_state: dict[str, torch.Tensor],
    dataloader,
    config: PraFFLConfig,
    device: torch.device,
    use_amp: bool,
    scaler,
) -> ClientTrainResult:
    local_model = copy.deepcopy(global_model).to(device)
    local_hypernetwork = copy.deepcopy(hypernetwork_template).to(device)
    local_hypernetwork.load_state_dict(private_hypernetwork_state, strict=True)
    encoder_optimizer = make_optimizer(
        local_model.bert.parameters(),
        config.optimizer_name,
        config.encoder_learning_rate,
    )
    hypernetwork_optimizer = make_optimizer(
        local_hypernetwork.parameters(),
        "adam",
        config.hypernetwork_learning_rate,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    communicated_losses = train_communicated_phase(
        local_model,
        local_hypernetwork,
        dataloader,
        epochs=config.tau_c,
        optimizer=encoder_optimizer,
        device=device,
        use_amp=use_amp,
        scaler=scaler,
    )
    personalized_losses = train_personalized_phase(
        local_model,
        local_hypernetwork,
        dataloader,
        epochs=config.tau_p,
        preference_batch_size=config.preference_batch_size,
        smooth_gamma=config.smooth_gamma,
        optimizer=hypernetwork_optimizer,
        device=device,
        use_amp=use_amp,
        scaler=scaler,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = ClientTrainResult(
        encoder_state=clone_state_dict_to_cpu(local_model.bert),
        hypernetwork_state=clone_state_dict_to_cpu(local_hypernetwork),
        communicated_losses=tuple(communicated_losses),
        personalized_losses=tuple(personalized_losses),
        gpu_seconds=elapsed,
    )
    del encoder_optimizer, hypernetwork_optimizer, local_hypernetwork, local_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
```

This code deliberately has no gradient accumulation: `tau_c` and `tau_p` count complete local epochs and each batch produces one optimizer step, as in the paper's local update definitions. It also deliberately applies dropout once before the functional head; `model.out` is never invoked.

- [ ] **Step 4: Run the phase tests**

Run:

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_praffl_training tests.test_praffl_core -v
```

Expected: 11 tests pass; the feature-call assertion proves that three preferences do not cause three encoder passes.

- [ ] **Step 5: Compile and scan the new algorithm modules**

Run:

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m compileall -q algorithm/praffl_core.py algorithm/praffl_training.py
if rg -n '\.data(?:\.|\b)|copy_\(' algorithm/praffl_core.py algorithm/praffl_training.py; then exit 1; fi
```

Expected: exit code 0 and no matching line; generated head tensors stay in the autograd graph.

- [ ] **Step 6: Commit the phases**

Run:

```bash
git add algorithm/praffl_training.py tests/test_praffl_training.py
git commit -m "feat: add PraFFL two-phase client training"
```

Expected: one commit containing phase logic and its isolation tests.

### Task 4: Replace the round orchestrator with private persistent hypernetworks

**Files:**
- Modify: `tests/test_praffl_training.py`
- Replace: `algorithm/PraFFL.py`
- Test: `algorithm/praffl_core.py`
- Test: `algorithm/praffl_training.py`

- [ ] **Step 1: Append orchestration regressions to the training test module**

Add these imports to `tests/test_praffl_training.py`:

```python
import copy
import tempfile
from unittest.mock import patch

from algorithm.PraFFL import PraFFL
from algorithm.praffl_training import ClientTrainResult
```

Then add this test class above the `unittest.main()` guard:

```python
class PraFFLRoundTest(unittest.TestCase):
    def test_round_averages_only_selected_encoders_and_keeps_all_private_heads(self):
        torch.manual_seed(31)
        model = TinyBertClassifier()
        classifier_before = clone_state_dict_to_cpu(model.out)
        encoder_before = clone_state_dict_to_cpu(model.bert)
        recorded_initial_private_states = {}

        def fake_train(
            global_model,
            hypernetwork_template,
            private_hypernetwork_state,
            dataloader,
            config,
            device,
            use_amp,
            scaler,
        ):
            del hypernetwork_template, config, device, use_amp, scaler
            client_id = int(dataloader)
            encoder_state = clone_state_dict_to_cpu(global_model.bert)
            for name, tensor in encoder_state.items():
                if tensor.is_floating_point():
                    tensor.add_(client_id + 1)
            private_state = copy.deepcopy(private_hypernetwork_state)
            recorded_initial_private_states.setdefault(client_id, copy.deepcopy(private_state))
            first_name = next(iter(private_state))
            private_state[first_name] = private_state[first_name] + float(client_id + 1)
            return ClientTrainResult(
                encoder_state=encoder_state,
                hypernetwork_state=private_state,
                communicated_losses=(1.0,),
                personalized_losses=(2.0,),
                gpu_seconds=0.25,
            )

        with tempfile.TemporaryDirectory() as model_path:
            param_dict = {
                "task": "SENT_CLF",
                "learning_rate": 0.01,
                "optimize_method": "sgd",
                "use_amp": False,
                "repeat_seed": 101,
                "model_path": model_path,
                "checkpoint_save_freq": 1,
                "checkpoint_keep_latest": 1,
                "communication_round_I": 1,
                "num_clients_K": 3,
            }
            with (
                patch("algorithm.PraFFL.client_selection", return_value=torch.tensor([0, 2])),
                patch("algorithm.PraFFL.train_praffl_client", side_effect=fake_train),
                patch("algorithm.PraFFL.save_checkpoint") as save_mock,
                patch("algorithm.PraFFL.clean_old_checkpoints") as clean_mock,
            ):
                result = PraFFL(
                    torch.device("cpu"),
                    model,
                    2,
                    3,
                    1,
                    2 / 3,
                    0.0,
                    [0, 1, 2],
                    list(range(6)),
                    [[0, 1], [2, 3], [4, 5]],
                    param_dict,
                    [],
                    0,
                )

        for name, tensor in result.global_model.bert.state_dict().items():
            if tensor.is_floating_point():
                self.assertTrue(torch.allclose(tensor, encoder_before[name] + 2.0))
        self.assertTrue(
            all(torch.equal(classifier_before[name], value) for name, value in result.global_model.out.state_dict().items())
        )
        private_states = result.algorithm_state["client_hypernetworks"]
        self.assertEqual(set(private_states), {0, 1, 2})
        first_name = next(iter(private_states[0]))
        self.assertFalse(torch.equal(private_states[0][first_name], private_states[2][first_name]))
        self.assertTrue(torch.equal(private_states[1][first_name], recorded_initial_private_states[0][first_name]))
        self.assertEqual(result.client_selection_history, [[0, 2]])
        expected_encoder_mb = sum(
            parameter.numel() * parameter.element_size() for parameter in model.bert.parameters()
        ) / (1024 * 1024)
        self.assertAlmostEqual(result.total_communication_cost, 2 * 2 * expected_encoder_mb)
        self.assertAlmostEqual(result.total_gpu_seconds, 0.5)
        save_mock.assert_called_once()
        self.assertEqual(
            set(save_mock.call_args.kwargs["algorithm_state"]["client_hypernetworks"]),
            {0, 1, 2},
        )
        clean_mock.assert_called_once_with(param_dict, keep_latest=1)

    def test_non_bert_or_non_binary_model_fails_before_client_selection(self):
        with self.assertRaisesRegex(ValueError, "binary BERT"):
            PraFFL(
                torch.device("cpu"),
                torch.nn.Linear(2, 2),
                2,
                1,
                1,
                1.0,
                0.0,
                [0],
                [0],
                [[0]],
                {"task": "SENT_CLF", "learning_rate": 0.01},
                [],
                0,
            )
```

- [ ] **Step 2: Run only the new round tests against the legacy implementation**

Run:

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest \
  tests.test_praffl_training.PraFFLRoundTest -v
```

Expected: failures show the legacy implementation averaging a single hypernetwork and writing client model files rather than returning the prerequisite `AlgorithmRunResult`/algorithm state.

- [ ] **Step 3: Replace `algorithm/PraFFL.py` completely**

Replace the file with:

```python
from __future__ import annotations

import copy
import time
from dataclasses import asdict
from typing import Mapping

import torch

from algorithm.client_selection import client_selection
from algorithm.praffl_core import (
    PRAFFL_STATE_SCHEMA_VERSION,
    HyperNetwork,
    PraFFLConfig,
    clone_state_dict_to_cpu,
    uniform_average_state_dicts,
)
from algorithm.praffl_training import train_praffl_client
from tool.amp_utils import get_scaler
from tool.checkpoint import CheckpointState, clean_old_checkpoints, save_checkpoint
from tool.experiment_state import AlgorithmRunResult
from tool.logger import logger


def _validate_model(global_model: torch.nn.Module, task: str) -> tuple[int, int]:
    required = ("bert", "drop", "out", "encode")
    valid = (
        task == "SENT_CLF"
        and all(hasattr(global_model, name) for name in required)
        and isinstance(global_model.out, torch.nn.Linear)
        and global_model.out.bias is not None
        and global_model.out.out_features == 2
    )
    if not valid:
        raise ValueError("PraFFL BERT adaptation requires SENT_CLF with a binary BERT linear head")
    return int(global_model.out.in_features), int(global_model.out.out_features)


def _new_hypernetwork(
    feature_dim: int,
    num_classes: int,
    config: PraFFLConfig,
    repeat_seed: int,
) -> HyperNetwork:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(repeat_seed + config.hypernetwork_seed_offset)
        return HyperNetwork(
            preference_dim=2,
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=config.hypernetwork_hidden_dim,
        )


def _clone_tensor_mapping(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def _state_snapshot(
    config: PraFFLConfig,
    template: HyperNetwork,
    client_hypernetworks: Mapping[int, Mapping[str, torch.Tensor]],
    completed_round: int,
) -> dict:
    return {
        "schema_version": PRAFFL_STATE_SCHEMA_VERSION,
        "completed_round": completed_round,
        "round_boundary": True,
        "config": asdict(config),
        "hypernetwork_spec": {
            "preference_dim": template.preference_dim,
            "feature_dim": template.feature_dim,
            "num_classes": template.num_classes,
            "hidden_dim": template.hidden_dim,
        },
        "client_hypernetworks": {
            int(client_id): _clone_tensor_mapping(state)
            for client_id, state in sorted(client_hypernetworks.items())
        },
    }


def _initial_private_states(template: HyperNetwork, num_clients: int) -> dict[int, dict[str, torch.Tensor]]:
    template_state = clone_state_dict_to_cpu(template)
    return {
        client_id: _clone_tensor_mapping(template_state)
        for client_id in range(num_clients)
    }


def _restore_private_states(
    raw_state: Mapping[str, object],
    template: HyperNetwork,
    config: PraFFLConfig,
    num_clients: int,
    start_round: int,
) -> dict[int, dict[str, torch.Tensor]]:
    if int(raw_state.get("schema_version", -1)) != PRAFFL_STATE_SCHEMA_VERSION:
        raise ValueError("incompatible PraFFL algorithm-state schema")
    if int(raw_state.get("completed_round", -2)) != start_round - 1:
        raise ValueError("PraFFL checkpoint round does not match start_round")
    if raw_state.get("round_boundary") is not True:
        raise ValueError("PraFFL only resumes at a communication-round boundary")
    expected_spec = {
        "preference_dim": template.preference_dim,
        "feature_dim": template.feature_dim,
        "num_classes": template.num_classes,
        "hidden_dim": template.hidden_dim,
    }
    if raw_state.get("hypernetwork_spec") != expected_spec:
        raise ValueError("PraFFL checkpoint hypernetwork shape does not match this run")
    if raw_state.get("config") != asdict(config):
        raise ValueError("PraFFL checkpoint configuration does not match this run")
    raw_private = raw_state.get("client_hypernetworks")
    if not isinstance(raw_private, Mapping) or set(raw_private) != set(range(num_clients)):
        raise ValueError("PraFFL checkpoint must contain exactly one private hypernetwork per client")
    restored = {}
    for client_id in range(num_clients):
        candidate = _clone_tensor_mapping(raw_private[client_id])
        template.load_state_dict(candidate, strict=True)
        restored[client_id] = candidate
    return restored


def _encoder_megabytes(global_model: torch.nn.Module) -> float:
    byte_count = sum(
        tensor.numel() * tensor.element_size()
        for tensor in global_model.bert.state_dict().values()
    )
    return byte_count / (1024 * 1024)


def PraFFL(
    device,
    global_model,
    algorithm_epoch_T,
    num_clients_K,
    communication_round_I,
    FL_fraction,
    FL_drop_rate,
    training_dataloaders,
    training_dataset,
    client_dataset_list,
    param_dict,
    testing_dataloader,
    testing_dataset_len,
    start_round=0,
    resume_state: CheckpointState | None = None,
):
    del testing_dataloader, testing_dataset_len
    device = torch.device(device)
    feature_dim, num_classes = _validate_model(global_model, param_dict.get("task", ""))
    if len(training_dataloaders) != num_clients_K or len(client_dataset_list) != num_clients_K:
        raise ValueError("PraFFL requires one training loader and dataset entry per client")
    config = PraFFLConfig.from_param_dict(param_dict, algorithm_epoch_T)
    repeat_seed = int(param_dict.get("repeat_seed", param_dict.get("seed", 0)))
    template = _new_hypernetwork(feature_dim, num_classes, config, repeat_seed)
    use_amp = bool(param_dict.get("use_amp", False))
    scaler = get_scaler(device, use_amp)

    if resume_state is None:
        if start_round != 0:
            raise ValueError("nonzero start_round requires a validated CheckpointState")
        private_states = _initial_private_states(template, num_clients_K)
        selection_history: list[list[int]] = []
        total_gpu_seconds = 0.0
        total_communication_cost = 0.0
        prior_runtime_seconds = 0.0
    else:
        if resume_state.next_round != start_round:
            raise ValueError("CheckpointState.next_round must equal start_round")
        if resume_state.phase not in {"train", "evaluate"}:
            raise ValueError("PraFFL checkpoint phase must be train or evaluate")
        private_states = _restore_private_states(
            resume_state.algorithm_state,
            template,
            config,
            num_clients_K,
            start_round,
        )
        selection_history = [list(selection) for selection in resume_state.client_selection_history]
        total_gpu_seconds = float(resume_state.total_gpu_seconds)
        total_communication_cost = float(resume_state.total_communication_cost)
        prior_runtime_seconds = float(resume_state.total_runtime_seconds)
        if scaler is not None and resume_state.amp_scaler_state is not None:
            scaler.load_state_dict(resume_state.amp_scaler_state)
        if (scaler is None) != (resume_state.amp_scaler_state is None):
            raise ValueError("PraFFL checkpoint AMP state does not match use_amp")

    if resume_state is not None and resume_state.phase == "evaluate":
        if start_round != communication_round_I:
            raise ValueError("evaluate-phase checkpoint must follow the final training round")
        final_state = _state_snapshot(config, template, private_states, start_round - 1)
        return AlgorithmRunResult(
            global_model=global_model,
            total_gpu_seconds=total_gpu_seconds,
            total_communication_cost=total_communication_cost,
            algorithm_state=final_state,
            amp_scaler_state=None if scaler is None else copy.deepcopy(scaler.state_dict()),
            client_selection_history=selection_history,
        )

    training_dataset_size = len(training_dataset)
    client_sizes = [len(dataset) for dataset in client_dataset_list]
    encoder_mb = _encoder_megabytes(global_model)
    checkpoint_frequency = int(param_dict.get("checkpoint_save_freq", 1))
    keep_latest = 1
    run_started = time.perf_counter()

    logger.info("PraFFL training starts with CPU-resident private hypernetworks")
    for round_index in range(start_round, communication_round_I):
        selected = client_selection(
            client_num=num_clients_K,
            fraction=FL_fraction,
            dataset_size=training_dataset_size,
            client_dataset_size_list=client_sizes,
            drop_rate=FL_drop_rate,
            style="FedAvg",
        )
        selected_ids = [int(client_id) for client_id in selected]
        if not selected_ids or len(set(selected_ids)) != len(selected_ids):
            raise ValueError("PraFFL client selection must be non-empty and unique")
        if any(client_id < 0 or client_id >= num_clients_K for client_id in selected_ids):
            raise ValueError("PraFFL client selection contains an invalid client id")

        uploaded_encoder_states = []
        for client_id in selected_ids:
            result = train_praffl_client(
                global_model,
                template,
                private_states[client_id],
                training_dataloaders[client_id],
                config,
                device,
                use_amp,
                scaler,
            )
            uploaded_encoder_states.append(result.encoder_state)
            private_states[client_id] = _clone_tensor_mapping(result.hypernetwork_state)
            total_gpu_seconds += result.gpu_seconds
        global_model.bert.load_state_dict(
            uniform_average_state_dicts(uploaded_encoder_states),
            strict=True,
        )
        selection_history.append(selected_ids)
        total_communication_cost += 2.0 * encoder_mb * len(selected_ids)
        algorithm_state = _state_snapshot(config, template, private_states, round_index)
        logger.info(
            "PraFFL round %s/%s selected=%s communication_mb=%.6f",
            round_index + 1,
            communication_round_I,
            selected_ids,
            total_communication_cost,
        )

        should_checkpoint = checkpoint_frequency > 0 and (
            (round_index + 1) % checkpoint_frequency == 0
            or round_index + 1 == communication_round_I
        )
        if should_checkpoint:
            save_checkpoint(
                param_dict,
                round_index,
                global_model,
                algorithm_state=algorithm_state,
                amp_scaler=scaler,
                total_gpu_seconds=total_gpu_seconds,
                total_runtime_seconds=prior_runtime_seconds + time.perf_counter() - run_started,
                total_communication_cost=total_communication_cost,
                client_selection_history=selection_history,
                extra_state={"algorithm": "PraFFL"},
            )
            clean_old_checkpoints(param_dict, keep_latest=keep_latest)

    final_state = _state_snapshot(config, template, private_states, communication_round_I - 1)
    return AlgorithmRunResult(
        global_model=global_model,
        total_gpu_seconds=total_gpu_seconds,
        total_communication_cost=total_communication_cost,
        algorithm_state=final_state,
        amp_scaler_state=None if scaler is None else copy.deepcopy(scaler.state_dict()),
        client_selection_history=selection_history,
    )
```

Delete every legacy import and behavior not present in this replacement: `ClientParallelExecutor`, per-client model files, the server/global hypernetwork, NumPy object-array aggregation, in-round static-head evaluation, TensorBoard deep-metric calls that evaluate the wrong head, and `save_path/global_PraFFL.pt`.

- [ ] **Step 4: Run the orchestration tests**

Run:

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_praffl_training -v
```

Expected: 5 tests pass. The selected private heads differ, client 1's head remains at the deterministic template, the static classifier is unchanged, and encoder aggregation is the unweighted mean.

- [ ] **Step 5: Verify serial/resource and state invariants statically**

Run:

```bash
if rg -n 'ClientParallelExecutor|hypernetwork_avg|global_PraFFL|model\.pt|\.data(?:\.|\b)|copy_\(' algorithm/PraFFL.py; then exit 1; fi
/home/ronnie/anaconda3/envs/FL/bin/python -m compileall -q algorithm/PraFFL.py
```

Expected: exit code 0 and no matching line. Only the active local model/hypernetwork enters `device`; all persisted private states are cloned onto CPU.

- [ ] **Step 6: Commit the round implementation**

Run:

```bash
git add algorithm/PraFFL.py tests/test_praffl_training.py
git commit -m "feat: persist private PraFFL client heads"
```

Expected: one commit replacing the legacy orchestrator and expanding its tests.

### Task 5: Add deterministic Pareto and hypervolume evaluation

**Files:**
- Create: `tests/test_praffl_evaluation.py`
- Create: `tool/praffl_evaluation.py`
- Test: `algorithm/praffl_core.py`

- [ ] **Step 1: Add exact metric and feature-reuse tests**

Create `tests/test_praffl_evaluation.py` with:

```python
import unittest
from types import SimpleNamespace

import torch

from algorithm.praffl_core import (
    PRAFFL_STATE_SCHEMA_VERSION,
    HyperNetwork,
    clone_state_dict_to_cpu,
)
from tool.praffl_evaluation import (
    PraFFLEvaluationError,
    build_preference_grid,
    evaluate_praffl,
    evaluate_preference_grid,
    hypervolume_2d,
    metrics_from_predictions,
    pareto_front_2d,
)


class CountingBertModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = torch.nn.Linear(3, 4)
        self.drop = torch.nn.Identity()
        self.out = torch.nn.Linear(4, 2)
        self.encode_calls = 0

    def encode(self, input_ids, attention_mask):
        del attention_mask
        self.encode_calls += 1
        return self.bert(input_ids.float())


def make_batch():
    return {
        "input_ids": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        "attention_mask": torch.ones(4, 3, dtype=torch.long),
        "labels": torch.tensor([0, 1, 1, 0]),
        "protected": torch.tensor([0, 0, 1, 1]),
    }


class PraFFLEvaluationMathTest(unittest.TestCase):
    def test_inclusive_preference_grid_is_deterministic(self):
        expected = torch.tensor(
            [[0.0, 1.0], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [1.0, 0.0]]
        )
        self.assertTrue(torch.equal(build_preference_grid(5), expected))

    def test_paper_dp_disparity_is_group_deviation_from_overall_rate(self):
        result = metrics_from_predictions(
            predictions=torch.tensor([[0, 1, 1, 1]]),
            labels=torch.tensor([0, 1, 1, 0]),
            protected=torch.tensor([0, 0, 1, 1]),
            scope="fixture",
        )
        self.assertTrue(torch.allclose(result["ACC"], torch.tensor([0.75], dtype=torch.float64)))
        self.assertTrue(torch.allclose(result["DP"], torch.tensor([0.25], dtype=torch.float64)))
        self.assertTrue(torch.allclose(result["SPD"], torch.tensor([-0.5], dtype=torch.float64)))
        self.assertTrue(torch.allclose(result["DEO"], torch.tensor([0.0], dtype=torch.float64)))

    def test_missing_protected_group_is_diagnostic(self):
        with self.assertRaisesRegex(PraFFLEvaluationError, "client 7.*protected counts"):
            metrics_from_predictions(
                predictions=torch.tensor([[0, 1]]),
                labels=torch.tensor([0, 1]),
                protected=torch.tensor([0, 0]),
                scope="client 7",
            )

    def test_pareto_filter_and_minimization_hypervolume_match_hand_value(self):
        points = [[0.2, 0.8], [0.5, 0.3], [0.6, 0.9], [0.2, 0.8]]
        front = pareto_front_2d(points)
        self.assertEqual(front, [[0.2, 0.8], [0.5, 0.3]])
        self.assertAlmostEqual(hypervolume_2d(front, reference_point=(1.0, 1.0)), 0.41)

    def test_encoder_runs_once_per_batch_not_once_per_preference_chunk(self):
        torch.manual_seed(5)
        model = CountingBertModel()
        hypernetwork = HyperNetwork(2, 4, 2, 6)
        preferences = build_preference_grid(7)
        result = evaluate_preference_grid(
            model,
            hypernetwork,
            [make_batch(), make_batch()],
            preferences,
            device=torch.device("cpu"),
            use_amp=False,
            chunk_size=2,
            scope="feature reuse",
        )
        self.assertEqual(model.encode_calls, 2)
        self.assertEqual(result["ACC"].shape, (7,))
        self.assertEqual(result["DP"].shape, (7,))


class PraFFLEvaluatorHookTest(unittest.TestCase):
    def test_hook_uses_every_private_head_for_local_and_global_fronts(self):
        torch.manual_seed(11)
        model = CountingBertModel()
        template = HyperNetwork(2, 4, 2, 6)
        first_state = clone_state_dict_to_cpu(template)
        second_state = clone_state_dict_to_cpu(template)
        first_key = next(iter(second_state))
        second_state[first_key].add_(0.25)
        algorithm_state = {
            "schema_version": PRAFFL_STATE_SCHEMA_VERSION,
            "completed_round": 0,
            "round_boundary": True,
            "config": {},
            "hypernetwork_spec": {
                "preference_dim": 2,
                "feature_dim": 4,
                "num_classes": 2,
                "hidden_dim": 6,
            },
            "client_hypernetworks": {0: first_state, 1: second_state},
        }
        data_bundle = SimpleNamespace(
            client_testing_dataloaders=[[make_batch()], [make_batch()]],
            testing_dataloader=[make_batch()],
        )
        metrics = evaluate_praffl(
            model,
            {
                "device": "cpu",
                "use_amp": False,
                "num_clients_K": 2,
                "algorithm_epoch_T": 2,
                "learning_rate": 0.01,
                "praffl_tau_c": 1,
                "praffl_tau_p": 1,
                "praffl_hypernetwork_hidden_dim": 6,
                "praffl_preference_points": 3,
                "praffl_preference_chunk_size": 2,
                "praffl_report_preference": [0.5, 0.5],
            },
            data_bundle,
            algorithm_state,
        )

        self.assertEqual(set(metrics), {"ACC", "DEO", "SPD", "report_preference", "praffl"})
        self.assertEqual(set(metrics["praffl"]["local"]["clients"]), {"0", "1"})
        self.assertEqual(set(metrics["praffl"]["global"]["clients"]), {"0", "1"})
        self.assertEqual(len(metrics["praffl"]["preference_grid"]), 3)
        self.assertEqual(metrics["praffl"]["reference_point"], [1.0, 1.0])
        self.assertEqual(model.encode_calls, 4)
        self.assertIsInstance(metrics["ACC"], float)
        self.assertIsInstance(metrics["praffl"]["local"]["mean_hv"], float)
        self.assertIsInstance(metrics["praffl"]["global"]["mean_hv"], float)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the evaluator tests and verify the module is absent**

Run:

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_praffl_evaluation -v
```

Expected: `ERROR` with `ModuleNotFoundError: No module named 'tool.praffl_evaluation'`.

- [ ] **Step 3: Create the complete evaluation module**

Create `tool/praffl_evaluation.py` with:

```python
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from algorithm.praffl_core import (
    PRAFFL_STATE_SCHEMA_VERSION,
    HyperNetwork,
    functional_linear_heads,
)
from tool.amp_utils import autocast_context


class PraFFLEvaluationError(ValueError):
    """Raised when PraFFL metrics would be scientifically undefined."""


def build_preference_grid(num_points: int) -> torch.Tensor:
    if num_points < 2:
        raise ValueError("PraFFL evaluation needs at least two preference points")
    accuracy_weight = torch.linspace(0.0, 1.0, steps=num_points, dtype=torch.float32)
    return torch.stack((accuracy_weight, 1.0 - accuracy_weight), dim=1)


def metrics_from_predictions(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    protected: torch.Tensor,
    *,
    scope: str,
) -> dict[str, torch.Tensor]:
    predictions = predictions.detach().cpu().long()
    labels = labels.detach().cpu().long().reshape(-1)
    protected = protected.detach().cpu().long().reshape(-1)
    if predictions.ndim != 2 or predictions.shape[1] != labels.numel():
        raise ValueError("predictions must have shape [preferences, examples]")
    if labels.numel() == 0:
        raise PraFFLEvaluationError(f"{scope} has no examples")
    if not torch.all((labels == 0) | (labels == 1)):
        raise PraFFLEvaluationError(f"{scope} has non-binary labels")
    if not torch.all((protected == 0) | (protected == 1)):
        raise PraFFLEvaluationError(f"{scope} has protected values outside 0/1")
    group_counts = torch.stack(((protected == 0).sum(), (protected == 1).sum())).to(torch.float64)
    if torch.any(group_counts == 0):
        raise PraFFLEvaluationError(
            f"{scope} protected counts are {group_counts.to(torch.int64).tolist()}; both groups are required"
        )

    accuracy = (predictions == labels.unsqueeze(0)).to(torch.float64).mean(dim=1)
    positive_rates = []
    true_positive_rates = []
    for group in (0, 1):
        group_mask = protected == group
        positive_rates.append(
            predictions[:, group_mask].to(torch.float64).mean(dim=1)
        )
        positive_label_mask = group_mask & (labels == 1)
        if positive_label_mask.any():
            true_positive_rates.append(
                predictions[:, positive_label_mask].to(torch.float64).mean(dim=1)
            )
        else:
            true_positive_rates.append(torch.zeros(predictions.shape[0], dtype=torch.float64))
    rate_0, rate_1 = positive_rates
    tpr_0, tpr_1 = true_positive_rates
    overall_positive = predictions.to(torch.float64).mean(dim=1)
    dp_disparity = torch.maximum(
        (rate_0 - overall_positive).abs(),
        (rate_1 - overall_positive).abs(),
    )
    return {
        "ACC": accuracy,
        "DEO": (tpr_0 - tpr_1).abs(),
        "SPD": rate_0 - rate_1,
        "DP": dp_disparity,
    }


def evaluate_preference_grid(
    model: torch.nn.Module,
    hypernetwork: HyperNetwork,
    dataloader: Iterable[dict[str, torch.Tensor]],
    preferences: torch.Tensor,
    *,
    device: torch.device,
    use_amp: bool,
    chunk_size: int,
    scope: str,
) -> dict[str, torch.Tensor]:
    if preferences.ndim != 2 or preferences.shape[1] != 2:
        raise ValueError("evaluation preferences must have shape [points, 2]")
    if chunk_size < 1:
        raise ValueError("preference chunk size must be positive")
    model.bert.eval()
    model.drop.eval()
    model.out.eval()
    hypernetwork.eval()
    prediction_batches = [[] for _ in range(preferences.shape[0])]
    label_batches = []
    protected_batches = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].detach().cpu().long()
            protected = batch["protected"].detach().cpu().long()
            with autocast_context(device, use_amp):
                features = model.drop(model.encode(input_ids, attention_mask))
            for start in range(0, preferences.shape[0], chunk_size):
                stop = min(start + chunk_size, preferences.shape[0])
                chunk = preferences[start:stop].to(device=device, dtype=features.dtype)
                with autocast_context(device, use_amp):
                    weight, bias = hypernetwork(chunk)
                    logits = functional_linear_heads(features, weight, bias)
                predicted = logits.argmax(dim=2).detach().cpu()
                for offset, row in enumerate(predicted):
                    prediction_batches[start + offset].append(row)
            label_batches.append(labels)
            protected_batches.append(protected)
    if not label_batches:
        raise PraFFLEvaluationError(f"{scope} has no batches")
    predictions = torch.stack(
        [torch.cat(rows, dim=0) for rows in prediction_batches],
        dim=0,
    )
    return metrics_from_predictions(
        predictions,
        torch.cat(label_batches, dim=0),
        torch.cat(protected_batches, dim=0),
        scope=scope,
    )


def pareto_front_2d(points: Sequence[Sequence[float]]) -> list[list[float]]:
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    front = []
    for candidate in unique:
        dominated = any(
            other[0] <= candidate[0]
            and other[1] <= candidate[1]
            and other != candidate
            for other in unique
        )
        if not dominated:
            front.append([candidate[0], candidate[1]])
    return front


def hypervolume_2d(
    pareto_points: Sequence[Sequence[float]],
    *,
    reference_point: tuple[float, float],
) -> float:
    reference_x, reference_y = reference_point
    front = pareto_front_2d(pareto_points)
    current_y = reference_y
    hypervolume = 0.0
    for x_value, y_value in front:
        if x_value < reference_x and y_value < current_y:
            hypervolume += (reference_x - x_value) * (current_y - y_value)
            current_y = y_value
    return float(hypervolume)


def _client_output(
    metrics: Mapping[str, torch.Tensor],
    preferences: torch.Tensor,
    grid_size: int,
) -> dict:
    solutions = []
    objective_points = []
    for index in range(grid_size):
        accuracy = float(metrics["ACC"][index])
        dp_disparity = float(metrics["DP"][index])
        objectives = [1.0 - accuracy, dp_disparity]
        objective_points.append(objectives)
        solutions.append(
            {
                "preference": [float(value) for value in preferences[index]],
                "ACC": accuracy,
                "DP": dp_disparity,
                "objectives": objectives,
            }
        )
    front = pareto_front_2d(objective_points)
    return {
        "solutions": solutions,
        "pareto_front": front,
        "hv": hypervolume_2d(front, reference_point=(1.0, 1.0)),
    }


def _validate_evaluator_state(algorithm_state: Mapping[str, object], num_clients: int) -> tuple[dict, Mapping]:
    if algorithm_state.get("schema_version") != PRAFFL_STATE_SCHEMA_VERSION:
        raise PraFFLEvaluationError("incompatible PraFFL algorithm-state schema")
    spec = algorithm_state.get("hypernetwork_spec")
    private_states = algorithm_state.get("client_hypernetworks")
    if not isinstance(spec, dict):
        raise PraFFLEvaluationError("PraFFL state is missing hypernetwork_spec")
    if not isinstance(private_states, Mapping) or set(private_states) != set(range(num_clients)):
        raise PraFFLEvaluationError("PraFFL state must contain every private client hypernetwork")
    return spec, private_states


def evaluate_praffl(global_model, param_dict, data_bundle, algorithm_state) -> dict:
    num_clients = int(param_dict["num_clients_K"])
    if len(data_bundle.client_testing_dataloaders) != num_clients:
        raise PraFFLEvaluationError("PraFFL needs one client test loader per client")
    spec, private_states = _validate_evaluator_state(algorithm_state, num_clients)
    grid_size = int(param_dict.get("praffl_preference_points", 1000))
    chunk_size = int(param_dict.get("praffl_preference_chunk_size", 128))
    report_raw = param_dict.get("praffl_report_preference", (0.5, 0.5))
    report = torch.tensor(report_raw, dtype=torch.float32).reshape(1, 2)
    if torch.any(report < 0) or not torch.isclose(report.sum(), torch.tensor(1.0)):
        raise PraFFLEvaluationError("praffl_report_preference must be non-negative and sum to 1")
    grid = build_preference_grid(grid_size)
    evaluation_preferences = torch.cat((grid, report), dim=0)
    device = torch.device(param_dict["device"])
    use_amp = bool(param_dict.get("use_amp", False))
    global_model = global_model.to(device)

    local_clients = {}
    global_clients = {}
    local_hv = []
    global_hv = []
    report_metrics = {"ACC": [], "DEO": [], "SPD": []}
    for client_id in range(num_clients):
        hypernetwork = HyperNetwork(
            preference_dim=int(spec["preference_dim"]),
            feature_dim=int(spec["feature_dim"]),
            num_classes=int(spec["num_classes"]),
            hidden_dim=int(spec["hidden_dim"]),
        )
        hypernetwork.load_state_dict(private_states[client_id], strict=True)
        hypernetwork = hypernetwork.to(device)
        local_metrics = evaluate_preference_grid(
            global_model,
            hypernetwork,
            data_bundle.client_testing_dataloaders[client_id],
            evaluation_preferences,
            device=device,
            use_amp=use_amp,
            chunk_size=chunk_size,
            scope=f"client {client_id} local test",
        )
        global_metrics = evaluate_preference_grid(
            global_model,
            hypernetwork,
            data_bundle.testing_dataloader,
            evaluation_preferences,
            device=device,
            use_amp=use_amp,
            chunk_size=chunk_size,
            scope=f"client {client_id} global test",
        )
        local_output = _client_output(local_metrics, grid, grid_size)
        global_output = _client_output(global_metrics, grid, grid_size)
        local_clients[str(client_id)] = local_output
        global_clients[str(client_id)] = global_output
        local_hv.append(local_output["hv"])
        global_hv.append(global_output["hv"])
        for metric_name in report_metrics:
            report_metrics[metric_name].append(float(global_metrics[metric_name][grid_size]))
        del hypernetwork
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = {
        metric_name: float(np.mean(values))
        for metric_name, values in report_metrics.items()
    }
    output["report_preference"] = [float(report[0, 0]), float(report[0, 1])]
    output["praffl"] = {
        "preference_grid": [[float(value) for value in row] for row in grid],
        "reference_point": [1.0, 1.0],
        "report_scope": "mean of private client heads on the common global test loader",
        "local": {
            "mean_hv": float(np.mean(local_hv)),
            "clients": local_clients,
        },
        "global": {
            "mean_hv": float(np.mean(global_hv)),
            "clients": global_clients,
        },
    }
    global_model.cpu()
    return output
```

The strict two-protected-group diagnostic applies to every local and global DP frontier. It prevents an absent group from being silently assigned zero disparity. DEO retains the repository's comparison-table convention: if a protected group has no positive labels, its TPR is treated as zero. The nested return contains only Python numbers/lists/dicts and is therefore safe for the prerequisite atomic repeat-metrics writer.

- [ ] **Step 4: Run the evaluator tests**

Run:

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_praffl_evaluation -v
```

Expected: 6 tests pass, including exact hypervolume `0.41` and exactly four encoder calls in the two-client hook test.

- [ ] **Step 5: Verify JSON serializability explicitly**

Run:

```bash
/home/ronnie/anaconda3/envs/FL/bin/python - <<'PY'
import json
import unittest

suite = unittest.defaultTestLoader.loadTestsFromName(
    "tests.test_praffl_evaluation.PraFFLEvaluatorHookTest"
)
result = unittest.TextTestRunner(verbosity=0).run(suite)
if not result.wasSuccessful():
    raise SystemExit(1)
json.dumps({"reference_point": [1.0, 1.0], "hv": 0.41})
PY
```

Expected: exit code 0 and `Ran 1 test ... OK`; the final `json.dumps` call raises no exception.

- [ ] **Step 6: Commit the evaluator**

Run:

```bash
git add tool/praffl_evaluation.py tests/test_praffl_evaluation.py
git commit -m "feat: evaluate PraFFL Pareto fronts and hypervolume"
```

Expected: one commit containing the algorithm-specific evaluator and tests.

### Task 6: Wire PraFFL into the repeat runner and expose explicit controls

**Files:**
- Create: `tests/test_praffl_wiring.py`
- Modify: `experiment.py`
- Modify: `main_SENT_CLF.py`
- Test: `algorithm/PraFFL.py`
- Test: `tool/praffl_evaluation.py`

- [ ] **Step 1: Add failing CLI, routing, and communication tests**

Create `tests/test_praffl_wiring.py` with:

```python
import sys
import unittest
from unittest.mock import patch

import torch

import experiment
from algorithm.PraFFL import PraFFL
from main_SENT_CLF import Argparse
from tool.praffl_evaluation import evaluate_praffl


class ModelWithLargeUnusedHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = torch.nn.Linear(2, 2)
        self.bert.register_buffer("communicated_buffer", torch.ones(3))
        self.out = torch.nn.Linear(2, 200)


class PraFFLWiringTest(unittest.TestCase):
    def test_formula_counts_exact_selected_encoder_upload_and_download_only(self):
        model = ModelWithLargeUnusedHead()
        parameters = {
            "communication_round_I": 3,
            "num_clients_K": 4,
            "FL_fraction": 0.75,
            "FL_drop_rate": 0.5,
            "task": "SENT_CLF",
            "praffl_hypernetwork_hidden_dim": 5000,
        }
        encoder_mb = sum(
            tensor.numel() * tensor.element_size()
            for tensor in model.bert.state_dict().values()
        ) / (1024 * 1024)
        selected_before_drop = 3
        selected_after_drop = 2
        expected = 3 * selected_after_drop * 2 * encoder_mb

        actual = experiment.calculate_communication_cost("PraFFL", parameters, model)

        self.assertEqual(selected_before_drop, 3)
        self.assertAlmostEqual(actual, round(expected, 3))

    def test_praffl_dispatch_supplies_algorithm_specific_evaluator(self):
        with patch("experiment.Experiment_FL") as runner:
            experiment._run_praffl_experiment({"algorithm": "PraFFL"})
        runner.assert_called_once_with(
            PraFFL,
            {"algorithm": "PraFFL"},
            evaluator_function=evaluate_praffl,
        )

    def test_cli_parses_named_praffl_controls(self):
        argv = [
            "main_SENT_CLF.py",
            "-praffl_tau_c", "3",
            "-praffl_tau_p", "4",
            "-praffl_preference_batch_size", "9",
            "-praffl_hypernetwork_hidden_dim", "64",
            "-praffl_hypernetwork_learning_rate", "0.002",
            "-praffl_smooth_gamma", "2.5",
            "-praffl_report_preference", "0.4", "0.6",
            "-praffl_preference_points", "101",
            "-praffl_preference_chunk_size", "16",
        ]
        with patch.object(sys, "argv", argv):
            parsed = Argparse()
        self.assertEqual(parsed["praffl_tau_c"], 3)
        self.assertEqual(parsed["praffl_tau_p"], 4)
        self.assertEqual(parsed["praffl_preference_batch_size"], 9)
        self.assertEqual(parsed["praffl_hypernetwork_hidden_dim"], 64)
        self.assertAlmostEqual(parsed["praffl_hypernetwork_learning_rate"], 0.002)
        self.assertAlmostEqual(parsed["praffl_smooth_gamma"], 2.5)
        self.assertEqual(parsed["praffl_report_preference"], [0.4, 0.6])
        self.assertEqual(parsed["praffl_preference_points"], 101)
        self.assertEqual(parsed["praffl_preference_chunk_size"], 16)


if __name__ == "__main__":
    unittest.main()
```

The formula test includes a persistent encoder buffer, a much larger unused static head, and an exaggerated hypernetwork setting. Only the encoder state-dictionary bytes may affect the answer. With `K=4`, fraction `0.75`, and drop rate `0.5`, client selection chooses three then drops one, so two upload/download pairs are counted each round.

- [ ] **Step 2: Run the wiring tests and verify all three regressions fail**

Run:

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_praffl_wiring -v
```

Expected: failures report the missing PraFFL CLI flags, missing `_run_praffl_experiment`, and a communication value polluted by the static head/hypernetwork.

- [ ] **Step 3: Correct PraFFL communication accounting**

In `experiment.py`, replace only the `algorithm_name == "PraFFL"` branch of `calculate_communication_cost` with:

```python
    elif algorithm_name == "PraFFL":
        if task != "SENT_CLF" or not hasattr(global_model, "bert"):
            raise ValueError("PraFFL communication accounting requires a SENT_CLF BERT encoder")
        selected_count = max(int(fraction * K), 1)
        if float(param_dict.get("FL_drop_rate", 0.0)) != 0.0:
            selected_count -= max(
                int(selected_count * float(param_dict["FL_drop_rate"])),
                1,
            )
        if selected_count < 1:
            raise ValueError("PraFFL FL_drop_rate leaves no selected clients")
        encoder_mb = sum(
            tensor.numel() * tensor.element_size()
            for tensor in global_model.bert.state_dict().values()
        ) / (1024 * 1024)
        cost = I * selected_count * 2 * encoder_mb
```

Keep the function's existing final rounding behavior. Do not modify formulas for unrelated algorithms in this task.

- [ ] **Step 4: Add and use the evaluator-specific dispatch helper**

Add this import next to the existing algorithm/tool imports in `experiment.py`:

```python
from tool.praffl_evaluation import evaluate_praffl
```

Add this helper immediately before `Experiment`:

```python
def _run_praffl_experiment(param_dict):
    return Experiment_FL(
        PraFFL,
        param_dict,
        evaluator_function=evaluate_praffl,
    )
```

Replace the post-prerequisite PraFFL dispatch branch with:

```python
    elif "PraFFL" in param_dict["algorithm"]:
        logger.info("~~~~~~ Algorithm: PraFFL ~~~~~~")
        _run_praffl_experiment(param_dict)
```

The generic static-head evaluator must not be called before or after `evaluate_praffl`. The repeat runner writes the entire returned nested metrics dictionary with `save_repeat_metrics`; top-level `ACC`, `DEO`, and `SPD` continue to feed the generic mean/std summary.

- [ ] **Step 5: Add the PraFFL arguments**

In `main_SENT_CLF.py`, add these parser definitions immediately after `-algorithm_epoch_T`:

```python
    parser.add_argument("-praffl_tau_c", type=int, default=None,
                        help="PraFFL communicated-encoder local epochs; tau_c + tau_p must equal algorithm_epoch_T")
    parser.add_argument("-praffl_tau_p", type=int, default=None,
                        help="PraFFL private-hypernetwork local epochs; tau_c + tau_p must equal algorithm_epoch_T")
    parser.add_argument("-praffl_preference_batch_size", type=int, default=8,
                        help="Dirichlet preferences sampled per personalized batch")
    parser.add_argument("-praffl_hypernetwork_hidden_dim", type=int, default=256,
                        help="Width of the two-layer PraFFL private hypernetwork")
    parser.add_argument("-praffl_hypernetwork_learning_rate", type=float, default=1e-3,
                        help="Adam learning rate for private hypernetworks")
    parser.add_argument("-praffl_smooth_gamma", type=float, default=1.0,
                        help="Smooth Tchebycheff log-sum-exp gamma")
    parser.add_argument("-praffl_report_preference", type=float, nargs=2, default=[0.5, 0.5],
                        metavar=("ACCURACY_WEIGHT", "FAIRNESS_WEIGHT"),
                        help="Named preference used for comparison-table ACC/DEO/SPD")
    parser.add_argument("-praffl_preference_points", type=int, default=1000,
                        help="Number of deterministic preferences in each Pareto sweep")
    parser.add_argument("-praffl_preference_chunk_size", type=int, default=128,
                        help="Preference heads evaluated at once after each encoder forward")
```

Do not add compatibility aliases for the legacy ambiguous `pref_bs`, scalar `tau_p`, `hypernet_lr`, or `hypernet_hidden` settings. Failing closed avoids silently interpreting an old configuration as the corrected method.

- [ ] **Step 6: Run the wiring and focused PraFFL suites**

Run:

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest \
  tests.test_praffl_wiring \
  tests.test_praffl_training \
  tests.test_praffl_evaluation -v
```

Expected: 14 tests pass and the dispatch mock sees `evaluate_praffl` exactly once.

- [ ] **Step 7: Commit runner and CLI wiring**

Run:

```bash
git add experiment.py main_SENT_CLF.py tests/test_praffl_wiring.py
git commit -m "feat: route PraFFL metrics through private heads"
```

Expected: one commit containing only experiment/CLI integration and its tests.

### Task 7: Prove exact round-boundary resume for PraFFL state

**Files:**
- Create: `tests/test_praffl_resume.py`
- Test: `algorithm/PraFFL.py`
- Test: `tool/praffl_evaluation.py`
- Test: `tool/checkpoint.py`

- [ ] **Step 1: Create a continuous-versus-crash/resume integration test**

Create `tests/test_praffl_resume.py` with:

```python
import copy
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from algorithm.PraFFL import PraFFL
from module.experiment_setup import FederatedDataBundle
from tool.praffl_evaluation import evaluate_praffl


class TinyResumeBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = torch.nn.Linear(3, 4)
        self.drop = torch.nn.Identity()
        self.out = torch.nn.Linear(4, 2)

    def encode(self, input_ids, attention_mask):
        del attention_mask
        return self.bert(input_ids.float())


def client_batch(offset):
    return {
        "input_ids": torch.tensor(
            [
                [1.0 + offset, 0.0, 0.5],
                [0.0, 1.0 + offset, -0.5],
                [1.0, 1.0, offset],
                [0.5, 0.0, 1.0 + offset],
            ]
        ),
        "attention_mask": torch.ones(4, 3, dtype=torch.long),
        "labels": torch.tensor([0, 1, 0, 1]),
        "protected": torch.tensor([0, 0, 1, 1]),
    }


def set_all_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def assert_nested_close(test_case, left, right, path="root"):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0.0, atol=1e-7, msg=path)
    elif isinstance(left, dict):
        test_case.assertEqual(set(left), set(right), path)
        for key in left:
            assert_nested_close(test_case, left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        test_case.assertEqual(len(left), len(right), path)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            assert_nested_close(test_case, left_item, right_item, f"{path}[{index}]")
    elif isinstance(left, float):
        test_case.assertAlmostEqual(left, right, places=7, msg=path)
    else:
        test_case.assertEqual(left, right, path)


class PlannedCrash(RuntimeError):
    pass


class PraFFLResumeTest(unittest.TestCase):
    def test_two_round_run_matches_round_one_checkpoint_plus_resume(self):
        set_all_rng(404)
        initial_model = TinyResumeBert()
        training_loaders = [[client_batch(0.0)], [client_batch(0.25)]]
        client_datasets = [[0, 1, 2, 3], [4, 5, 6, 7]]
        training_dataset = list(range(8))
        param_dict = {
            "task": "SENT_CLF",
            "device": "cpu",
            "learning_rate": 0.02,
            "optimize_method": "sgd",
            "use_amp": False,
            "repeat_seed": 7404,
            "model_path": "/tmp/praffl-resume-test",
            "checkpoint_save_freq": 1,
            "communication_round_I": 2,
            "num_clients_K": 2,
            "FL_fraction": 1.0,
            "FL_drop_rate": 0.0,
            "algorithm_epoch_T": 2,
            "praffl_tau_c": 1,
            "praffl_tau_p": 1,
            "praffl_preference_batch_size": 3,
            "praffl_hypernetwork_hidden_dim": 6,
            "praffl_hypernetwork_learning_rate": 0.01,
            "praffl_preference_points": 5,
            "praffl_preference_chunk_size": 2,
            "praffl_report_preference": [0.5, 0.5],
        }

        continuous_model = copy.deepcopy(initial_model)
        set_all_rng(909)
        with (
            patch("algorithm.PraFFL.save_checkpoint"),
            patch("algorithm.PraFFL.clean_old_checkpoints"),
        ):
            continuous = PraFFL(
                torch.device("cpu"), continuous_model, 2, 2, 2, 1.0, 0.0,
                training_loaders, training_dataset, client_datasets, param_dict, [], 0,
            )

        captured = {}

        def capture_checkpoint(
            checkpoint_param_dict,
            iter_t,
            checkpoint_model,
            **kwargs,
        ):
            self.assertIs(checkpoint_param_dict, param_dict)
            self.assertEqual(iter_t, 0)
            captured["model_state"] = copy.deepcopy(checkpoint_model.state_dict())
            captured["algorithm_state"] = copy.deepcopy(kwargs["algorithm_state"])
            captured["amp_scaler_state"] = None
            captured["total_gpu_seconds"] = kwargs["total_gpu_seconds"]
            captured["total_runtime_seconds"] = kwargs["total_runtime_seconds"]
            captured["total_communication_cost"] = kwargs["total_communication_cost"]
            captured["client_selection_history"] = copy.deepcopy(kwargs["client_selection_history"])
            captured["python_rng"] = random.getstate()
            captured["numpy_rng"] = np.random.get_state()
            captured["torch_rng"] = torch.get_rng_state().clone()
            raise PlannedCrash("stop after durable round one")

        resumed_model = copy.deepcopy(initial_model)
        set_all_rng(909)
        with (
            patch("algorithm.PraFFL.save_checkpoint", side_effect=capture_checkpoint),
            patch("algorithm.PraFFL.clean_old_checkpoints"),
        ):
            with self.assertRaisesRegex(PlannedCrash, "durable round one"):
                PraFFL(
                    torch.device("cpu"), resumed_model, 2, 2, 2, 1.0, 0.0,
                    training_loaders, training_dataset, client_datasets, param_dict, [], 0,
                )

        resumed_model.load_state_dict(captured["model_state"])
        checkpoint_state = SimpleNamespace(
            next_round=1,
            phase="train",
            algorithm_state=captured["algorithm_state"],
            amp_scaler_state=captured["amp_scaler_state"],
            total_gpu_seconds=captured["total_gpu_seconds"],
            total_runtime_seconds=captured["total_runtime_seconds"],
            total_communication_cost=captured["total_communication_cost"],
            client_selection_history=captured["client_selection_history"],
        )
        random.setstate(captured["python_rng"])
        np.random.set_state(captured["numpy_rng"])
        torch.set_rng_state(captured["torch_rng"])
        with (
            patch("algorithm.PraFFL.save_checkpoint"),
            patch("algorithm.PraFFL.clean_old_checkpoints"),
        ):
            resumed = PraFFL(
                torch.device("cpu"), resumed_model, 2, 2, 2, 1.0, 0.0,
                training_loaders, training_dataset, client_datasets, param_dict, [], 0,
                start_round=1,
                resume_state=checkpoint_state,
            )

        for name, tensor in continuous.global_model.state_dict().items():
            torch.testing.assert_close(tensor, resumed.global_model.state_dict()[name], rtol=0.0, atol=1e-7)
        assert_nested_close(self, continuous.algorithm_state, resumed.algorithm_state)
        self.assertEqual(continuous.client_selection_history, resumed.client_selection_history)
        self.assertAlmostEqual(
            continuous.total_communication_cost,
            resumed.total_communication_cost,
            places=12,
        )

        data_bundle = FederatedDataBundle(
            training_dataloaders=training_loaders,
            client_dataset_list=client_datasets,
            testing_dataloader=[client_batch(0.1)],
            client_testing_dataloaders=[[client_batch(0.0)], [client_batch(0.25)]],
            client_testing_dataset_list=client_datasets,
            partition_fingerprint="praffl-resume-fixture",
            partition_metadata={"fixture": True},
        )
        continuous_metrics = evaluate_praffl(
            continuous.global_model,
            param_dict,
            data_bundle,
            continuous.algorithm_state,
        )
        resumed_metrics = evaluate_praffl(
            resumed.global_model,
            param_dict,
            data_bundle,
            resumed.algorithm_state,
        )
        assert_nested_close(self, continuous_metrics, resumed_metrics)


if __name__ == "__main__":
    unittest.main()
```

This test deliberately raises from the checkpoint writer after it has captured round 1, reproducing a process loss immediately after durable state publication. It restores Python, NumPy, and Torch RNG exactly where the prerequisite runner restores them. Hypernetwork construction occurs after restoration inside `PraFFL`; the test therefore also catches any missing `torch.random.fork_rng` protection.

- [ ] **Step 2: Run the resume test**

Run:

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_praffl_resume -v
```

Expected after Tasks 1–6: 1 test passes. If it fails, use the first differing tensor/path printed by `assert_nested_close`; do not loosen `atol=1e-7` or omit private states.

- [ ] **Step 3: Add a final-round evaluate-phase regression**

Append this method to `PraFFLResumeTest`:

```python
    def test_evaluate_phase_checkpoint_does_not_train_again(self):
        model = TinyResumeBert()
        param_dict = {
            "task": "SENT_CLF",
            "learning_rate": 0.01,
            "optimize_method": "sgd",
            "use_amp": False,
            "repeat_seed": 12,
            "communication_round_I": 1,
            "praffl_tau_c": 1,
            "praffl_tau_p": 1,
            "praffl_hypernetwork_hidden_dim": 6,
        }
        with (
            patch("algorithm.PraFFL.save_checkpoint"),
            patch("algorithm.PraFFL.clean_old_checkpoints"),
        ):
            trained = PraFFL(
                torch.device("cpu"), model, 2, 2, 1, 1.0, 0.0,
                [[client_batch(0.0)], [client_batch(0.2)]],
                list(range(8)),
                [list(range(4)), list(range(4, 8))],
                param_dict, [], 0,
            )
        evaluate_state = SimpleNamespace(
            next_round=1,
            phase="evaluate",
            algorithm_state=copy.deepcopy(trained.algorithm_state),
            amp_scaler_state=None,
            total_gpu_seconds=trained.total_gpu_seconds,
            total_runtime_seconds=0.0,
            total_communication_cost=trained.total_communication_cost,
            client_selection_history=copy.deepcopy(trained.client_selection_history),
        )
        model_before = copy.deepcopy(trained.global_model.state_dict())
        with patch("algorithm.PraFFL.train_praffl_client") as train_mock:
            restored = PraFFL(
                torch.device("cpu"), trained.global_model, 2, 2, 1, 1.0, 0.0,
                [[client_batch(0.0)], [client_batch(0.2)]],
                list(range(8)),
                [list(range(4)), list(range(4, 8))],
                param_dict, [], 0,
                start_round=1,
                resume_state=evaluate_state,
            )
        train_mock.assert_not_called()
        for name, tensor in model_before.items():
            torch.testing.assert_close(tensor, restored.global_model.state_dict()[name], rtol=0.0, atol=0.0)
```

- [ ] **Step 4: Run both resume tests and checkpoint tests from the prerequisite pull request**

Run:

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest \
  tests.test_praffl_resume \
  tests.test_checkpoint \
  tests.test_repeat_runner -v
```

Expected: all tests pass. The exact prerequisite module names are `tests.test_checkpoint` and `tests.test_repeat_runner`; if they do not exist, the prerequisite pull request is not present and this PraFFL worktree was based on the wrong commit.

- [ ] **Step 5: Commit the exact-resume coverage**

Run:

```bash
git add tests/test_praffl_resume.py
git commit -m "test: prove exact PraFFL round resume"
```

Expected: one test-only commit; production code should already satisfy both cases.

### Task 8: Document the corrected method and metric contract

**Files:**
- Modify: `README.md`
- Modify: `README_CN.md`

- [ ] **Step 1: Add the English method section**

Add this section under the algorithm documentation in `README.md`:

```markdown
### PraFFL (paper-faithful BERT adaptation)

PraFFL follows [Preference-aware Fair Federated Learning](https://arxiv.org/abs/2404.08973). For `SENT_CLF`, only the BERT encoder is communicated. Client `k` keeps a private persistent two-input hypernetwork that maps `(accuracy_weight, fairness_weight)` to the weight and bias of the binary linear head; generated tensors are applied functionally so the private update remains differentiable.

Each selected client executes `praffl_tau_c` communicated epochs and then `praffl_tau_p` personalized epochs. The first phase fixes preference `(0.5, 0.5)`, freezes the generated head, and updates the encoder with cross-entropy. The second phase freezes/detaches one encoder feature per batch, samples `Dirichlet([1, 1])` preferences, and updates only the private hypernetwork with the differentiable DP covariance surrogate and inverse-weighted smooth Tchebycheff loss. `praffl_tau_c + praffl_tau_p` must equal `algorithm_epoch_T`.

Final repeat metrics preserve top-level `ACC`, `DEO`, and signed `SPD` at `praffl_report_preference` (default `0.5 0.5`). These values are the mean across all private heads on the common global test loader. `metrics.json` also records `praffl.local` and `praffl.global`, each with every client's preference solutions, nondominated objective points `(1 - ACC, DP disparity)`, per-client hypervolume, and mean hypervolume. Hypervolume uses minimization reference point `(1, 1)`. A local/global split without both protected groups raises a diagnostic evaluation error rather than being reported as zero disparity.

Key controls:

| Flag | Default | Meaning |
|---|---:|---|
| `-praffl_tau_c` | half of `algorithm_epoch_T` | communicated encoder epochs |
| `-praffl_tau_p` | remaining epochs | private hypernetwork epochs |
| `-praffl_preference_batch_size` | 8 | Dirichlet preferences per personalized batch |
| `-praffl_hypernetwork_hidden_dim` | 256 | private hypernetwork width |
| `-praffl_hypernetwork_learning_rate` | 0.001 | private Adam learning rate |
| `-praffl_smooth_gamma` | 1.0 | log-sum-exp smoothness |
| `-praffl_report_preference` | `0.5 0.5` | comparison-table preference |
| `-praffl_preference_points` | 1000 | deterministic Pareto-grid size |
| `-praffl_preference_chunk_size` | 128 | heads evaluated per feature chunk |

PraFFL is serial on one GPU: the global model plus one selected client's local copy/private hypernetwork are active, inactive private hypernetworks remain CPU state dictionaries, and only the latest resumable checkpoint is retained.
```

- [ ] **Step 2: Add the matching Chinese method section**

Add this section at the corresponding location in `README_CN.md`:

```markdown
### PraFFL（论文一致的 BERT 适配）

PraFFL 按照论文 [Preference-aware Fair Federated Learning](https://arxiv.org/abs/2404.08973) 实现。对 `SENT_CLF`，服务器只通信 BERT 编码器。客户端 `k` 持久保存私有的二维偏好超网络，将 `(准确率权重, 公平性权重)` 生成为二分类线性头的权重和偏置；生成参数通过函数式线性层使用，因此梯度不会被参数复制操作截断。

每个入选客户端先执行 `praffl_tau_c` 个通信阶段 epoch，再执行 `praffl_tau_p` 个个性化阶段 epoch。通信阶段固定偏好 `(0.5, 0.5)`，冻结生成头，只用交叉熵更新编码器。个性化阶段冻结并分离每个 batch 只计算一次的编码器特征，从 `Dirichlet([1, 1])` 采样偏好，只用可微 DP 协方差替代目标和逆偏好加权的平滑 Tchebycheff 损失更新私有超网络。`praffl_tau_c + praffl_tau_p` 必须等于 `algorithm_epoch_T`。

每次 repeat 的最终指标保留 `praffl_report_preference`（默认 `0.5 0.5`）下顶层的 `ACC`、`DEO` 和带符号 `SPD`；这些值是在公共全局测试集上对所有私有头结果求均值。`metrics.json` 还保存 `praffl.local` 和 `praffl.global`：每个客户端的偏好解、非支配目标点 `(1 - ACC, DP disparity)`、客户端 hypervolume 及平均 hypervolume。Hypervolume 使用最小化参考点 `(1, 1)`。若某个本地或全局评估切分缺少任一保护组，评估会报告明确错误，不会把缺失组静默当作零差异。

关键参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `-praffl_tau_c` | `algorithm_epoch_T` 的前半部分 | 通信编码器 epoch 数 |
| `-praffl_tau_p` | 剩余部分 | 私有超网络 epoch 数 |
| `-praffl_preference_batch_size` | 8 | 每个个性化 batch 的 Dirichlet 偏好数 |
| `-praffl_hypernetwork_hidden_dim` | 256 | 私有超网络宽度 |
| `-praffl_hypernetwork_learning_rate` | 0.001 | 私有 Adam 学习率 |
| `-praffl_smooth_gamma` | 1.0 | log-sum-exp 平滑系数 |
| `-praffl_report_preference` | `0.5 0.5` | 横向比较表偏好 |
| `-praffl_preference_points` | 1000 | 确定性 Pareto 网格点数 |
| `-praffl_preference_chunk_size` | 128 | 每份编码器特征同时评估的头数 |

PraFFL 在单张 GPU 上串行运行：仅全局模型、当前客户端的本地副本和私有超网络进入显存；未激活客户端的超网络以 CPU state dictionary 保存；断点只保留最新的一份可恢复状态。
```

- [ ] **Step 3: Verify documentation names match the code**

Run:

```bash
for name in \
  praffl_tau_c \
  praffl_tau_p \
  praffl_preference_batch_size \
  praffl_hypernetwork_hidden_dim \
  praffl_hypernetwork_learning_rate \
  praffl_smooth_gamma \
  praffl_report_preference \
  praffl_preference_points \
  praffl_preference_chunk_size; do
  rg -q "$name" main_SENT_CLF.py README.md README_CN.md || exit 1
done
```

Expected: exit code 0 with no output.

- [ ] **Step 4: Commit documentation**

Run:

```bash
git add README.md README_CN.md
git commit -m "docs: describe paper-faithful PraFFL"
```

Expected: one documentation-only commit.

### Task 9: Run the complete local verification gate

**Files:**
- Test: `tests/test_praffl_core.py`
- Test: `tests/test_praffl_bert_interface.py`
- Test: `tests/test_praffl_training.py`
- Test: `tests/test_praffl_evaluation.py`
- Test: `tests/test_praffl_resume.py`
- Test: `tests/test_praffl_wiring.py`
- Verify: `algorithm/PraFFL.py`
- Verify: `algorithm/praffl_core.py`
- Verify: `algorithm/praffl_training.py`
- Verify: `tool/praffl_evaluation.py`

- [ ] **Step 1: Run every PraFFL test in one fresh process**

Run:

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest \
  tests.test_praffl_core \
  tests.test_praffl_bert_interface \
  tests.test_praffl_training \
  tests.test_praffl_evaluation \
  tests.test_praffl_resume \
  tests.test_praffl_wiring -v
```

Expected: 26 tests pass with no skip on CPU.

- [ ] **Step 2: Run the whole repository unit suite**

Run:

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m unittest discover -s tests -v
```

Expected: all tests pass; CUDA-only tests may report `skipped` on a CPU host, and no test reports `FAIL` or `ERROR`.

- [ ] **Step 3: Compile every changed Python module**

Run:

```bash
/home/ronnie/anaconda3/envs/FL/bin/python -m compileall -q \
  algorithm/PraFFL.py \
  algorithm/praffl_core.py \
  algorithm/praffl_training.py \
  hypothesis/BERTCLASSIFIER.py \
  tool/praffl_evaluation.py \
  experiment.py \
  main_SENT_CLF.py
```

Expected: exit code 0 with no output.

- [ ] **Step 4: Run paper-fidelity source guards**

Run:

```bash
if rg -n '\.data(?:\.|\b)|copy_\(|ClientParallelExecutor|global_PraFFL|hypernetwork_avg|client_.*/model\.pt' \
  algorithm/PraFFL.py algorithm/praffl_core.py algorithm/praffl_training.py tool/praffl_evaluation.py; then
  exit 1
fi
rg -n 'F\.linear|Dirichlet|smooth_tchebycheff|uniform_average_state_dicts|client_hypernetworks' \
  algorithm/PraFFL.py algorithm/praffl_core.py algorithm/praffl_training.py tool/praffl_evaluation.py
```

Expected: the first scan has no output. The second scan finds functional heads, Dirichlet sampling, inverse-weighted smooth scalarization, uniform encoder averaging, and plural private client states.

- [ ] **Step 5: Inspect the final diff and commit state**

Run:

```bash
git diff --check
git status --short
git log --oneline --decorate -8
```

Expected: `git diff --check` has no output; `git status --short` is empty; the log contains the eight focused commits from Tasks 1–8.

### Task 10: Run Ronnie AMP-off and AMP-on BERT smoke gates

**Files:**
- Verify: `save_path/moji/Dirichlet1/PraFFL/BERTCLASSIFIER/2Clients/`
- Create outside Git: `artifacts/praffl-smoke/amp-off.txt`
- Create outside Git: `artifacts/praffl-smoke/amp-on.txt`

- [ ] **Step 1: Confirm the one-process GPU environment**

Run on Ronnie from the dedicated worktree:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
pgrep -af 'main_SENT_CLF.py.*PraFFL' || true
mkdir -p artifacts/praffl-smoke
```

Expected: GPU 0 is listed and no other PraFFL experiment process is running. Record the total GPU memory printed here for the 80% gate in Step 5.

- [ ] **Step 2: Run one round with AMP disabled and record peak memory**

Run:

```bash
/usr/bin/time -v /home/ronnie/anaconda3/envs/FL/bin/python - <<'PY' 2>&1 | tee artifacts/praffl-smoke/amp-off.txt
import runpy
import sys

import torch

torch.cuda.reset_peak_memory_stats(0)
sys.argv = [
    "main_SENT_CLF.py",
    "-mode", "train",
    "-algorithm", "PraFFL",
    "-dataset", "moji",
    "-split_strategy", "Dirichlet1",
    "-communication_round_I", "1",
    "-algorithm_epoch_T", "2",
    "-num_clients_K", "2",
    "-system_data_count", "64",
    "-batch_size", "8",
    "-test_batch_size", "16",
    "-exp_repeat_times", "1",
    "-parallel_repeats", "1",
    "-checkpoint_save_freq", "1",
    "-checkpoint_keep_latest", "1",
    "-praffl_tau_c", "1",
    "-praffl_tau_p", "1",
    "-praffl_preference_batch_size", "2",
    "-praffl_preference_points", "9",
    "-praffl_preference_chunk_size", "3",
    "-use_amp", "false",
    "-cuda", "0",
]
runpy.run_path("main_SENT_CLF.py", run_name="__main__")
print(f"PRAFFL_PEAK_CUDA_BYTES={torch.cuda.max_memory_allocated(0)}")
PY
```

Expected: exit code 0; the log contains final `ACC`, `DEO`, `SPD`, local/global mean hypervolume, `PRAFFL_PEAK_CUDA_BYTES=...`, and `/usr/bin/time`'s `Maximum resident set size`.

- [ ] **Step 3: Validate AMP-off artifacts and isolate them from the AMP-on run**

Run:

```bash
find save_path/moji/Dirichlet1/PraFFL/BERTCLASSIFIER/2Clients -type f \
  \( -name '*.pt' -o -name 'metrics.json' \) -printf '%s %p\n' \
  | tee artifacts/praffl-smoke/amp-off-files.txt
test "$(find save_path/moji/Dirichlet1/PraFFL/BERTCLASSIFIER/2Clients -path '*/checkpoints/*.pt' -type f | wc -l)" -eq 1
mv save_path/moji/Dirichlet1/PraFFL save_path/moji/Dirichlet1/PraFFL-amp-off-smoke
```

Expected: exactly one checkpoint is listed before the move, its byte size is recorded, and at least one atomic `metrics.json` is listed. No `client_*/model.pt` or client hypernetwork file exists; all private hypernetworks are inside the one checkpoint.

- [ ] **Step 4: Run the identical smoke with AMP enabled**

Run:

```bash
/usr/bin/time -v /home/ronnie/anaconda3/envs/FL/bin/python - <<'PY' 2>&1 | tee artifacts/praffl-smoke/amp-on.txt
import runpy
import sys

import torch

torch.cuda.reset_peak_memory_stats(0)
sys.argv = [
    "main_SENT_CLF.py",
    "-mode", "train",
    "-algorithm", "PraFFL",
    "-dataset", "moji",
    "-split_strategy", "Dirichlet1",
    "-communication_round_I", "1",
    "-algorithm_epoch_T", "2",
    "-num_clients_K", "2",
    "-system_data_count", "64",
    "-batch_size", "8",
    "-test_batch_size", "16",
    "-exp_repeat_times", "1",
    "-parallel_repeats", "1",
    "-checkpoint_save_freq", "1",
    "-checkpoint_keep_latest", "1",
    "-praffl_tau_c", "1",
    "-praffl_tau_p", "1",
    "-praffl_preference_batch_size", "2",
    "-praffl_preference_points", "9",
    "-praffl_preference_chunk_size", "3",
    "-use_amp", "true",
    "-cuda", "0",
]
runpy.run_path("main_SENT_CLF.py", run_name="__main__")
print(f"PRAFFL_PEAK_CUDA_BYTES={torch.cuda.max_memory_allocated(0)}")
PY
find save_path/moji/Dirichlet1/PraFFL/BERTCLASSIFIER/2Clients -type f \
  \( -name '*.pt' -o -name 'metrics.json' \) -printf '%s %p\n' \
  | tee artifacts/praffl-smoke/amp-on-files.txt
test "$(find save_path/moji/Dirichlet1/PraFFL/BERTCLASSIFIER/2Clients -path '*/checkpoints/*.pt' -type f | wc -l)" -eq 1
```

Expected: exit code 0; the same final metric keys appear, AMP is logged as enabled, peak CUDA bytes and maximum RSS are recorded, and exactly one latest checkpoint remains.

- [ ] **Step 5: Apply the resource and result gate before a full matrix**

Run:

```bash
/home/ronnie/anaconda3/envs/FL/bin/python - <<'PY'
import json
import pathlib
import re
import subprocess

root = pathlib.Path("artifacts/praffl-smoke")
total_mib = int(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"],
    text=True,
).strip())
for mode in ("amp-off", "amp-on"):
    text = (root / f"{mode}.txt").read_text()
    cuda_bytes = int(re.search(r"PRAFFL_PEAK_CUDA_BYTES=(\d+)", text).group(1))
    rss_kib = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", text).group(1))
    if cuda_bytes >= 0.8 * total_mib * 1024 * 1024:
        raise SystemExit(f"{mode}: CUDA peak exceeds 80% of GPU memory")
    if rss_kib <= 0:
        raise SystemExit(f"{mode}: maximum RSS was not recorded")
    file_lines = (root / f"{mode}-files.txt").read_text().splitlines()
    checkpoint_lines = [line for line in file_lines if "/checkpoints/" in line and line.endswith(".pt")]
    metric_lines = [line for line in file_lines if line.endswith("metrics.json")]
    if len(checkpoint_lines) != 1 or not metric_lines:
        raise SystemExit(f"{mode}: expected one checkpoint and at least one metrics.json")
    metric_path = pathlib.Path(metric_lines[-1].split(" ", 1)[1])
    if mode == "amp-off":
        metric_path = pathlib.Path(str(metric_path).replace("/PraFFL/", "/PraFFL-amp-off-smoke/"))
    metrics = json.loads(metric_path.read_text())
    for key in ("ACC", "DEO", "SPD", "report_preference", "praffl"):
        if key not in metrics:
            raise SystemExit(f"{mode}: metrics missing {key}")
    for scope in ("local", "global"):
        if "mean_hv" not in metrics["praffl"][scope]:
            raise SystemExit(f"{mode}: {scope} mean_hv missing")
print("PraFFL Ronnie smoke resource/result gate passed")
PY
```

Expected: `PraFFL Ronnie smoke resource/result gate passed`. Attach the four text artifacts to the pull request or copy their peak CUDA bytes, maximum RSS, checkpoint byte size, and metric keys into the pull-request description.

- [ ] **Step 6: Run the representative three-repeat job only after the gate passes**

Run:

```bash
/home/ronnie/anaconda3/envs/FL/bin/python main_SENT_CLF.py \
  -mode train \
  -algorithm PraFFL \
  -dataset moji \
  -split_strategy Dirichlet1 \
  -communication_round_I 5 \
  -algorithm_epoch_T 2 \
  -num_clients_K 20 \
  -system_data_count 2000 \
  -batch_size 32 \
  -test_batch_size 128 \
  -exp_repeat_times 3 \
  -parallel_repeats 1 \
  -checkpoint_save_freq 1 \
  -checkpoint_keep_latest 1 \
  -praffl_tau_c 1 \
  -praffl_tau_p 1 \
  -praffl_preference_batch_size 8 \
  -praffl_preference_points 1000 \
  -praffl_preference_chunk_size 128 \
  -use_amp auto \
  -cuda 0
```

Expected: three distinct repeat seeds run serially; each repeat writes atomic metrics containing top-level comparison metrics and both PraFFL Pareto/HV scopes; only the encoder contributes to reported communication; no more than one active client job appears in process/GPU monitoring.

- [ ] **Step 7: Record the validation commit without adding smoke artifacts**

Run:

```bash
git status --short
git diff --check
```

Expected: generated `artifacts/`, logs, caches, results, and checkpoints are ignored or remain untracked and are not staged; tracked source/documentation files are clean. Do not commit machine-specific smoke output.

## Final acceptance checklist

- [ ] The communicated object is exactly `global_model.bert`; the static `out` layer and all hypernetworks are excluded from averaging and communication bytes.
- [ ] Each selected client completes `tau_c` encoder epochs followed by `tau_p` private epochs, with no off-by-one boundary.
- [ ] The balanced generated head is detached in the communicated phase; the encoder is detached in the personalized phase.
- [ ] Functional head application preserves private-hypernetwork gradients, and no parameter-copy shortcut exists in PraFFL files.
- [ ] Training fairness is equation-3 squared centered covariance over both logits; scalarization is inverse-preference smooth Tchebycheff.
- [ ] All clients have durable CPU private states, selected states diverge through local training, inactive states remain unchanged, and no server average touches them.
- [ ] Evaluation emits the named report-preference ACC/DEO/SPD plus deterministic local/global solutions, nondominated fronts, and two-dimensional hypervolume.
- [ ] Each evaluator batch performs one encoder call per private client and reuses its feature across every preference chunk.
- [ ] Round checkpoints contain every private hypernetwork plus scaler/RNG/counters/history through the prerequisite schema; final-round checkpoints enter the prerequisite `evaluate` phase.
- [ ] Continuous and resumed runs match encoder, private states, client history, communication, and final nested metrics within absolute tolerance `1e-7`.
- [ ] Existing tests, CPU PraFFL tests, and Ronnie AMP-off/AMP-on BERT smokes pass before the three-repeat run.
