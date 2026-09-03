# Deterministic Repeats, Exact Resume, and Label-Dirichlet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every three-repeat experiment deterministic and paired across algorithms, resume exactly at communication-round boundaries, and replace the legacy quantity-skew `Dirichlet*` behavior with a finite, versioned label-conditioned Dirichlet partition shared by training and client-level test allocation.

**Architecture:** Put partition semantics, fingerprints, validation, repair, and the algorithm-independent cache in a focused `module/partition.py`; keep `module/dataloader.py` responsible only for turning a validated artifact into PyTorch datasets/loaders. Put typed algorithm results and repeat results in `tool/experiment_state.py`, make `tool/checkpoint.py` the fail-closed persistence boundary, and make both serial and multiprocessing execution call the same repeat worker in `experiment.py`. Existing algorithms may continue returning the legacy three-tuple, while stateful algorithms opt into the new `resume_state` and `AlgorithmRunResult` contract without changing their mathematics.

**Tech Stack:** Python 3.11, NumPy local `Generator(PCG64)`, PyTorch `Dataset`/`Subset`/`DataLoader`, standard-library `dataclasses`/`hashlib`/`json`/`tempfile`, and `unittest`.

---

## Scope and compatibility boundary

This is the first, independently reviewable infrastructure PR from the design spec. It does not change PraFFL or FedFACT-In objectives, updates, aggregation, or evaluation; their later PRs consume the interfaces defined here. It does not reinterpret old results: `Dirichlet01`, `Dirichlet05`, and `Dirichlet1` now mean label-skew schema v2, while old `split_indices.json` files are accessible only through explicitly named `LegacyQuantityDirichlet01`, `LegacyQuantityDirichlet05`, `LegacyQuantityDirichlet1`, or `LegacyQuantityDirichlet8` modes.

The exact-resume protocol is integrated minimally into FedAvg as the stateless reference path so the two-round equivalence acceptance test exercises a real FL loop. No FedAvg loss, client weighting, or aggregation formula changes. Stateful PraFFL/FedFACT checkpoint payloads and their algorithm-specific evaluators are added only in their respective plans.

Run every command below from the repository root. On Ronnie, replace `python` with `/home/ronnie/anaconda3/envs/FL/bin/python` while keeping every argument unchanged.

## File structure

- Create `module/partition.py`: dataset views, canonical fingerprints, label-conditioned draws, deterministic repair, train/test allocation, cache publication/loading, integrity checks, and legacy read-only loading.
- Modify `module/dataloader.py:1-398`: delete the inline quantity-skew branch and build loaders only from validated partition indices; make worker/batch order deterministic.
- Modify `module/experiment_setup.py:323-346`: return a `FederatedDataBundle` containing global and client-level test loaders plus partition identity.
- Modify `tool/seed_manager.py:24-72`: retain `base_seed + 1000 * repeat_idx` and seed Python workers from the derived worker seed.
- Create `tool/experiment_state.py`: typed `AlgorithmRunResult`, `RepeatResult`, legacy-result normalization, and aggregate calculations.
- Modify `tool/checkpoint.py:1-189`: canonical config hashing, checkpoint schema v2, complete RNG/scaler/counter/algorithm state, atomic latest-only persistence, fail-closed validation, structured repeat metrics, and completion/final-artifact policy.
- Modify `experiment.py:347-682,689-1138`: one repeat worker for serial/parallel modes, seed-before-construction ordering, opt-in resume, evaluator hooks, and aggregation of fresh plus previously completed repeats.
- Modify `algorithm/FederatedAverage.py:114-321`: minimal new result/resume/checkpoint adapter, including full selection history and scaler state, without changing FedAvg training math.
- Modify `main_SENT_CLF.py:114-195,263-355`, `main_IMG_CLF.py:87-167,235-319`, and `main_Tabular_CLF.py:87-190,266-360`: expose common state/partition flags and stop treating free-form log text as resume state.
- Create `tool/experiment_cli.py`: one shared argument group for repeat, partition, checkpoint, and artifact controls used by all three entry points.
- Create `tests/test_partition.py`: pure partition, repair, fingerprint, cache, corruption, and legacy isolation regressions.
- Create `tests/test_dataloader_partition.py`: bundle construction, paired partitions, train/test allocation, and deterministic loader integration.
- Create `tests/test_checkpoint.py`: schema/hash/RNG/scaler/counter/mismatch/atomic-completion regressions.
- Create `tests/test_repeat_runner.py`: construction ordering, three seeds, opt-in resume, mixed completion, evaluation-stage recovery, serial/parallel parity, and two-round exact-resume tests.
- Create `tests/test_experiment_state_cli.py`: common CLI defaults and GPU serial-repeat guard.
- Modify `README.md:449-528` and `README_CN.md:447-523`: document the new split meaning, cache, repeat seeds, resume contract, and artifact policy.

### Task 1: Lock the dataset-view and fingerprint contract

**Files:**
- Create: `tests/test_partition.py`
- Create: `module/partition.py`

- [ ] **Step 1: Write failing dataset-view and fingerprint tests**

Create the reusable fixture and the first test class exactly as follows:

```python
import json
import os
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from module.partition import (
    DatasetView,
    PartitionDataError,
    dataset_fingerprint,
    extract_dataset_view,
)


class ToyDataset(Dataset):
    def __init__(self, sample_ids, labels, protected):
        self.sample_ids = list(sample_ids)
        self.labels = np.asarray(labels)
        self.protected = np.asarray(protected)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "x": torch.tensor(float(index)),
            "labels": torch.tensor(self.labels[index]),
            "protected": torch.tensor(self.protected[index]),
        }


class DatasetViewTest(unittest.TestCase):
    def test_extracts_subset_in_subset_order(self):
        source = ToyDataset(["a", "b", "c", "d"], [0, 1, 0, 1], [1, 0, 1, 0])
        view = extract_dataset_view(Subset(source, [3, 1, 2]))
        self.assertEqual(view.sample_ids, ("d", "b", "c"))
        np.testing.assert_array_equal(view.labels, np.array([1, 1, 0]))
        np.testing.assert_array_equal(view.protected, np.array([0, 0, 1]))

    def test_legacy_y_and_s1_are_supported(self):
        dataset = ToyDataset(["a", "b"], [0, 1], [1, 0])
        del dataset.labels
        dataset.y = np.array([0, 1])
        del dataset.protected
        dataset.s1 = np.array([1, 0])
        view = extract_dataset_view(dataset)
        np.testing.assert_array_equal(view.labels, np.array([0, 1]))
        np.testing.assert_array_equal(view.protected, np.array([1, 0]))

    def test_rejects_length_mismatch_and_non_scalar_labels(self):
        bad_length = ToyDataset(["a"], [0, 1], [0, 1])
        with self.assertRaisesRegex(PartitionDataError, "sample identity length"):
            extract_dataset_view(bad_length)
        bad_labels = ToyDataset(["a", "b"], [[0], [1]], [0, 1])
        with self.assertRaisesRegex(PartitionDataError, "one-dimensional"):
            extract_dataset_view(bad_labels)

    def test_order_and_label_changes_change_fingerprint(self):
        first = ToyDataset(["a", "b", "c"], [0, 1, 0], [0, 1, 1])
        reordered = ToyDataset(["b", "a", "c"], [1, 0, 0], [1, 0, 1])
        relabeled = ToyDataset(["a", "b", "c"], [1, 0, 0], [0, 1, 1])
        fp = dataset_fingerprint(first, dataset_name="toy", split="train", system_data_count=3)
        self.assertNotEqual(fp["sample_order_sha256"], dataset_fingerprint(
            reordered, dataset_name="toy", split="train", system_data_count=3
        )["sample_order_sha256"])
        self.assertNotEqual(fp["ordered_labels_sha256"], dataset_fingerprint(
            relabeled, dataset_name="toy", split="train", system_data_count=3
        )["ordered_labels_sha256"])
```

The production adapter must never enumerate `dataset[index]` merely to discover labels or identities. Tests use `sample_ids`; production adapters also recognize, in order, `texts`, `img_names`, `X`, and `input_ids`. A cached text dataset uses `_stacked_cache["input_ids"]`, `_stacked_cache["labels"]`, and `_stacked_cache["protected"]` instead of trusting its dictionary-backed `__len__`.

- [ ] **Step 2: Run the focused tests and verify the missing-module failure**

Run:

```bash
python -m unittest tests.test_partition.DatasetViewTest -v
```

Expected: import fails with `ModuleNotFoundError: No module named 'module.partition'`.

- [ ] **Step 3: Add the dataset-view implementation**

Start `module/partition.py` with these public types and adapters:

```python
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import math
import os
import tempfile

import numpy as np
import torch
from torch.utils.data import Subset


PARTITION_SCHEMA_VERSION = 2
PARTITIONER_NAME = "label_dirichlet_v2"


class PartitionDataError(ValueError):
    pass


class PartitionCacheError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetView:
    sample_ids: tuple[str, ...]
    labels: np.ndarray
    protected: np.ndarray


def _as_one_dimensional(values: Any, field: str) -> np.ndarray:
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if array.ndim != 1:
        raise PartitionDataError(f"{field} must be one-dimensional, got shape {array.shape}")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise PartitionDataError(f"{field} contains NaN or infinity")
    return array


def _row_identities(dataset: Any) -> tuple[str, ...]:
    if hasattr(dataset, "sample_ids"):
        return tuple(str(value) for value in dataset.sample_ids)
    if hasattr(dataset, "texts"):
        return tuple(str(value) for value in dataset.texts)
    if hasattr(dataset, "img_names"):
        return tuple(Path(str(value)).as_posix() for value in dataset.img_names)
    values = getattr(dataset, "X", None)
    if values is None:
        values = getattr(dataset, "input_ids", None)
    if values is None and hasattr(dataset, "_stacked_cache"):
        values = dataset._stacked_cache.get("input_ids")
    if values is None:
        raise PartitionDataError(
            f"{type(dataset).__name__} exposes no stable sample identities; "
            "provide sample_ids, texts, img_names, X, or input_ids"
        )
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    rows = np.asarray(values)
    return tuple(
        sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in rows
    )


def extract_dataset_view(dataset: Any) -> DatasetView:
    if isinstance(dataset, Subset):
        parent = extract_dataset_view(dataset.dataset)
        indices = np.asarray(dataset.indices, dtype=np.int64)
        return DatasetView(
            sample_ids=tuple(parent.sample_ids[index] for index in indices),
            labels=parent.labels[indices],
            protected=parent.protected[indices],
        )

    labels = getattr(dataset, "labels", None)
    if labels is None:
        labels = getattr(dataset, "targets", None)
    if labels is None:
        labels = getattr(dataset, "y", None)
    if labels is None and hasattr(dataset, "_stacked_cache"):
        labels = dataset._stacked_cache.get("labels")
    protected = getattr(dataset, "protected", None)
    if protected is None:
        protected = getattr(dataset, "s1", None)
    if protected is None and hasattr(dataset, "_stacked_cache"):
        protected = dataset._stacked_cache.get("protected")
    if labels is None or protected is None:
        raise PartitionDataError("dataset must expose labels and protected attributes")

    labels_array = _as_one_dimensional(labels, "labels")
    protected_array = _as_one_dimensional(protected, "protected")
    sample_ids = _row_identities(dataset)
    size = len(labels_array)
    if len(protected_array) != size:
        raise PartitionDataError(
            f"protected length {len(protected_array)} does not match labels length {size}"
        )
    if len(sample_ids) != size:
        raise PartitionDataError(
            f"sample identity length {len(sample_ids)} does not match labels length {size}"
        )
    declared_size = size if hasattr(dataset, "_stacked_cache") else len(dataset)
    if declared_size != size:
        raise PartitionDataError(
            f"dataset length {declared_size} does not match ordered labels length {size}"
        )
    return DatasetView(sample_ids, labels_array.copy(), protected_array.copy())


def _digest_strings(values: Sequence[str]) -> str:
    digest = sha256()
    for value in values:
        encoded = value.encode("utf-8", errors="surrogatepass")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _digest_scalars(values: np.ndarray) -> str:
    return _digest_strings(
        [json.dumps(value.item() if hasattr(value, "item") else value,
                    ensure_ascii=False, sort_keys=True, allow_nan=False)
         for value in values]
    )


def dataset_fingerprint(dataset: Any, *, dataset_name: str, split: str,
                        system_data_count: int | None) -> dict[str, Any]:
    view = extract_dataset_view(dataset)
    result = {
        "dataset_name": dataset_name,
        "split": split,
        "size": len(view.labels),
        "system_data_count": system_data_count,
        "sample_order_sha256": _digest_strings(view.sample_ids),
        "ordered_labels_sha256": _digest_scalars(view.labels),
        "ordered_protected_sha256": _digest_scalars(view.protected),
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result["dataset_sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
    return result
```

- [ ] **Step 4: Run the focused tests**

Run:

```bash
python -m unittest tests.test_partition.DatasetViewTest -v
```

Expected: four tests pass.

- [ ] **Step 5: Commit the dataset identity boundary**

Run:

```bash
git add module/partition.py tests/test_partition.py
git commit -m "feat: fingerprint ordered federated datasets"
```

Expected: one commit containing only the dataset-view/fingerprint implementation and its tests.

### Task 2: Replace quantity skew with finite label-conditioned Dirichlet

**Files:**
- Modify: `tests/test_partition.py`
- Modify: `module/partition.py`

- [ ] **Step 1: Add failing pure-partitioner tests**

Append these imports and tests to `tests/test_partition.py`:

```python
from module.partition import (
    PartitionSpec,
    build_label_dirichlet_partition,
    validate_indices,
)


class ScriptedGenerator:
    def __init__(self, profiles):
        self.profiles = iter(profiles)
        self.dirichlet_calls = 0

    def dirichlet(self, concentration):
        self.dirichlet_calls += 1
        return np.asarray(next(self.profiles), dtype=np.float64)

    def shuffle(self, values):
        values[:] = values[::-1]


class LabelDirichletTest(unittest.TestCase):
    def test_samples_one_profile_per_label_and_reuses_it_for_test(self):
        train_labels = np.array([0] * 6 + [1] * 6)
        test_labels = np.array([0] * 3 + [1] * 3)
        rng = ScriptedGenerator([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
        spec = PartitionSpec(
            strategy="Dirichlet05", alpha=0.5, num_clients=3, seed=7,
            min_samples_per_client=1, max_retries=1,
            repair_policy="minimum_move_v1",
        )
        result = build_label_dirichlet_partition(
            train_labels, test_labels, spec, rng=rng
        )
        self.assertEqual(rng.dirichlet_calls, 2)
        np.testing.assert_allclose(
            result.class_client_profile,
            np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]]),
        )
        self.assertEqual(sum(map(len, result.train_indices.values())), 12)
        self.assertEqual(sum(map(len, result.test_indices.values())), 6)

    def test_same_seed_is_independent_of_global_rng_state(self):
        labels = np.repeat(np.arange(3), 30)
        spec = PartitionSpec("Dirichlet1", 1.0, 5, 42, 1, 20, "minimum_move_v1")
        random.seed(1)
        np.random.seed(1)
        first = build_label_dirichlet_partition(labels, labels, spec)
        random.seed(999)
        np.random.seed(999)
        second = build_label_dirichlet_partition(labels, labels, spec)
        for client_id in range(spec.num_clients):
            np.testing.assert_array_equal(
                first.train_indices[client_id], second.train_indices[client_id]
            )

    def test_repair_is_minimum_move_and_explicit(self):
        labels = np.array([0] * 50 + [1] * 50)
        rng = ScriptedGenerator([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        spec = PartitionSpec("Dirichlet01", 0.1, 4, 11, 3, 1, "minimum_move_v1")
        result = build_label_dirichlet_partition(labels, labels[:20], spec, rng=rng)
        self.assertTrue(result.repaired)
        self.assertEqual(len(result.repair_moves), 9)
        self.assertEqual(min(map(len, result.train_indices.values())), 3)
        self.assertEqual(result.partitioner, "label_dirichlet_repaired_v2")

    def test_impossible_minimum_fails_before_sampling(self):
        spec = PartitionSpec("Dirichlet01", 0.1, 40, 42, 51, 2, "minimum_move_v1")
        with self.assertRaisesRegex(PartitionDataError, "requires at least 2040"):
            build_label_dirichlet_partition(np.zeros(2000), np.zeros(100), spec)

    def test_alpha_point_one_forty_clients_finishes_with_valid_coverage(self):
        labels = np.tile(np.array([0, 1]), 1000)
        spec = PartitionSpec("Dirichlet01", 0.1, 40, 42, 1, 2, "minimum_move_v1")
        result = build_label_dirichlet_partition(labels, labels[:400], spec)
        validate_indices(result.train_indices, dataset_size=2000,
                         num_clients=40, min_size=1)
        self.assertLessEqual(result.attempts, 2)
```

- [ ] **Step 2: Run the tests and verify missing symbols**

Run:

```bash
python -m unittest tests.test_partition.LabelDirichletTest -v
```

Expected: import fails because `PartitionSpec`, `build_label_dirichlet_partition`, and `validate_indices` are not defined.

- [ ] **Step 3: Implement exact spec parsing and index validation**

Add these types and functions to `module/partition.py`:

```python
DIRICHLET_ALPHAS = {
    "Dirichlet01": 0.1,
    "Dirichlet05": 0.5,
    "Dirichlet1": 1.0,
}


@dataclass(frozen=True)
class PartitionSpec:
    strategy: str
    alpha: float | None
    num_clients: int
    seed: int
    min_samples_per_client: int
    max_retries: int
    repair_policy: str
    schema_version: int = PARTITION_SCHEMA_VERSION

    @classmethod
    def from_params(cls, param_dict: Mapping[str, Any], repeat_idx: int) -> "PartitionSpec":
        strategy = str(param_dict["split_strategy"])
        alpha = DIRICHLET_ALPHAS.get(strategy)
        if strategy != "Uniform" and alpha is None:
            raise PartitionDataError(f"unsupported versioned split strategy: {strategy}")
        seed = int(param_dict.get("base_seed", 42)) + 1000 * int(repeat_idx)
        return cls(
            strategy=strategy,
            alpha=alpha,
            num_clients=int(param_dict["num_clients_K"]),
            seed=seed,
            min_samples_per_client=int(param_dict.get("partition_min_size", 1)),
            max_retries=int(param_dict.get("partition_max_retries", 100)),
            repair_policy=str(param_dict.get("partition_repair_policy", "minimum_move_v1")),
        )


@dataclass(frozen=True)
class PartitionResult:
    train_indices: dict[int, np.ndarray]
    test_indices: dict[int, np.ndarray]
    class_values: tuple[Any, ...]
    class_client_profile: np.ndarray
    attempts: int
    repaired: bool
    repair_moves: tuple[dict[str, Any], ...]
    partitioner: str


def validate_indices(indices: Mapping[int, np.ndarray], *, dataset_size: int,
                     num_clients: int, min_size: int = 0) -> None:
    if set(indices) != set(range(num_clients)):
        raise PartitionCacheError("client keys must be exactly 0..num_clients-1")
    arrays = []
    for client_id in range(num_clients):
        array = np.asarray(indices[client_id])
        if not np.issubdtype(array.dtype, np.integer):
            raise PartitionCacheError(f"client {client_id} indices are not integers")
        array = array.astype(np.int64, copy=False)
        if len(array) < min_size:
            raise PartitionCacheError(
                f"client {client_id} has {len(array)} samples; minimum is {min_size}"
            )
        if len(array) and (array.min() < 0 or array.max() >= dataset_size):
            raise PartitionCacheError(f"client {client_id} has out-of-bounds indices")
        arrays.append(array)
    flat = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int64)
    if len(flat) != dataset_size:
        raise PartitionCacheError(
            f"partition covers {len(flat)} entries, expected {dataset_size}"
        )
    if len(np.unique(flat)) != dataset_size:
        raise PartitionCacheError("partition contains duplicate or missing indices")
```

- [ ] **Step 4: Implement per-label draws, finite retries, deterministic repair, and test allocation**

Use local generators only. Add the following implementation to `module/partition.py`:

```python
def _split_class_indices(class_indices: np.ndarray, profile: np.ndarray) -> list[np.ndarray]:
    boundaries = (np.cumsum(profile)[:-1] * len(class_indices)).astype(np.int64)
    return list(np.split(class_indices, boundaries))


def _largest_remainder_counts(size: int, profile: np.ndarray) -> np.ndarray:
    exact = profile * size
    counts = np.floor(exact).astype(np.int64)
    remainder = size - int(counts.sum())
    order = sorted(range(len(profile)), key=lambda client: (-(exact[client] - counts[client]), client))
    for client_id in order[:remainder]:
        counts[client_id] += 1
    return counts


def _allocate_with_profile(labels: np.ndarray, class_values: tuple[Any, ...],
                           profile: np.ndarray, rng: Any) -> dict[int, np.ndarray]:
    clients: list[list[int]] = [[] for _ in range(profile.shape[1])]
    for class_row, class_value in enumerate(class_values):
        class_indices = np.flatnonzero(labels == class_value).astype(np.int64)
        rng.shuffle(class_indices)
        counts = _largest_remainder_counts(len(class_indices), profile[class_row])
        offset = 0
        for client_id, count in enumerate(counts):
            clients[client_id].extend(class_indices[offset:offset + count].tolist())
            offset += count
    for values in clients:
        rng.shuffle(values)
    return {client_id: np.asarray(values, dtype=np.int64)
            for client_id, values in enumerate(clients)}


def _repair_minimum_move(clients: list[list[int]], labels: np.ndarray,
                         class_values: tuple[Any, ...], profile: np.ndarray,
                         minimum: int) -> tuple[dict[str, Any], ...]:
    class_row = {value: row for row, value in enumerate(class_values)}
    moves: list[dict[str, Any]] = []
    while True:
        recipients = [client for client, values in enumerate(clients) if len(values) < minimum]
        if not recipients:
            return tuple(moves)
        recipient = recipients[0]
        label_order = sorted(
            class_values,
            key=lambda value: (-profile[class_row[value], recipient], repr(value)),
        )
        chosen = None
        for value in label_order:
            donors = [
                donor for donor, values in enumerate(clients)
                if len(values) > minimum and any(labels[index] == value for index in values)
            ]
            if donors:
                donor = sorted(donors, key=lambda item: (-len(clients[item]), item))[0]
                index = min(index for index in clients[donor] if labels[index] == value)
                chosen = donor, index, value
                break
        if chosen is None:
            raise PartitionDataError("minimum-move repair found no donor with surplus")
        donor, index, value = chosen
        clients[donor].remove(index)
        clients[recipient].append(index)
        moves.append({
            "from_client": donor,
            "to_client": recipient,
            "index": int(index),
            "label": value.item() if hasattr(value, "item") else value,
        })


def build_label_dirichlet_partition(train_labels: Any, test_labels: Any,
                                    spec: PartitionSpec, rng: Any | None = None) -> PartitionResult:
    train = _as_one_dimensional(train_labels, "train labels")
    test = _as_one_dimensional(test_labels, "test labels")
    if spec.alpha is None or not math.isfinite(spec.alpha) or spec.alpha <= 0:
        raise PartitionDataError("Dirichlet alpha must be positive and finite")
    if spec.num_clients <= 0 or spec.min_samples_per_client < 0 or spec.max_retries <= 0:
        raise PartitionDataError("client count, minimum size, and retry count are invalid")
    required = spec.num_clients * spec.min_samples_per_client
    if len(train) < required:
        raise PartitionDataError(
            f"minimum size requires at least {required} training samples, got {len(train)}"
        )
    class_values = tuple(np.unique(train).tolist())
    unseen = sorted(set(np.unique(test).tolist()) - set(class_values), key=repr)
    if unseen:
        raise PartitionDataError(f"test labels absent from training labels: {unseen}")
    generator = rng if rng is not None else np.random.Generator(np.random.PCG64(spec.seed))
    best_clients = None
    best_profile = None
    best_deficit = None
    used_attempt = 0
    for attempt in range(1, spec.max_retries + 1):
        clients: list[list[int]] = [[] for _ in range(spec.num_clients)]
        rows = []
        for class_value in class_values:
            class_indices = np.flatnonzero(train == class_value).astype(np.int64)
            generator.shuffle(class_indices)
            profile = np.asarray(
                generator.dirichlet(np.full(spec.num_clients, spec.alpha)),
                dtype=np.float64,
            )
            profile = profile / profile.sum()
            rows.append(profile)
            for client_id, chunk in enumerate(_split_class_indices(class_indices, profile)):
                clients[client_id].extend(chunk.tolist())
        deficit = sum(max(0, spec.min_samples_per_client - len(values)) for values in clients)
        if best_deficit is None or deficit < best_deficit:
            best_clients = [list(values) for values in clients]
            best_profile = np.stack(rows, axis=0)
            best_deficit = deficit
            used_attempt = attempt
        if deficit == 0:
            break
    assert best_clients is not None and best_profile is not None
    moves: tuple[dict[str, Any], ...] = ()
    if best_deficit:
        if spec.repair_policy != "minimum_move_v1":
            raise PartitionDataError(f"unsupported repair policy: {spec.repair_policy}")
        moves = _repair_minimum_move(
            best_clients, train, class_values, best_profile, spec.min_samples_per_client
        )
    for values in best_clients:
        generator.shuffle(values)
    train_indices = {
        client_id: np.asarray(values, dtype=np.int64)
        for client_id, values in enumerate(best_clients)
    }
    test_rng = np.random.Generator(np.random.PCG64(spec.seed ^ 0x9E3779B97F4A7C15))
    test_indices = _allocate_with_profile(test, class_values, best_profile, test_rng)
    validate_indices(train_indices, dataset_size=len(train), num_clients=spec.num_clients,
                     min_size=spec.min_samples_per_client)
    validate_indices(test_indices, dataset_size=len(test), num_clients=spec.num_clients)
    repaired = bool(moves)
    return PartitionResult(
        train_indices=train_indices,
        test_indices=test_indices,
        class_values=class_values,
        class_client_profile=best_profile,
        attempts=used_attempt,
        repaired=repaired,
        repair_moves=moves,
        partitioner="label_dirichlet_repaired_v2" if repaired else PARTITIONER_NAME,
    )
```

This repair moves exactly one sample for each unit of client deficit; therefore its move count is minimal. Label choice maximizes the undersized client's sampled class preference, while donor size, client id, and sample index provide deterministic tie-breaking.

- [ ] **Step 5: Run the pure partition tests under a hard hang guard**

Run:

```bash
timeout 20s python -m unittest tests.test_partition.LabelDirichletTest -v
```

Expected: five tests pass and the command exits 0 well before the 20-second guard; the former alpha `0.1`/40-client configuration cannot loop indefinitely.

- [ ] **Step 6: Commit the versioned label-skew implementation**

Run:

```bash
git add module/partition.py tests/test_partition.py
git commit -m "feat: add finite label-dirichlet partitions"
```

Expected: one commit with the pure partitioner and no loader/checkpoint edits.

### Task 3: Add the shared, validated partition artifact cache

**Files:**
- Modify: `tests/test_partition.py`
- Modify: `module/partition.py`
- Modify: `tool/checkpoint.py:102-148`

- [ ] **Step 1: Add failing cache, metadata, corruption, and legacy-isolation tests**

Append a `PartitionCacheTest` using `TemporaryDirectory` and these test methods:

```python
from module.partition import (
    build_or_load_partition,
    load_partition_artifact,
    partition_fingerprint,
)


class PartitionCacheTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_root = Path(self.temp_dir.name) / "partition_cache"
        self.train = ToyDataset(
            [f"train-{index}" for index in range(120)],
            np.tile([0, 1], 60), np.tile([0, 1, 1, 0], 30),
        )
        self.test = ToyDataset(
            [f"test-{index}" for index in range(40)],
            np.tile([0, 1], 20), np.tile([0, 1, 1, 0], 10),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_params(self, algorithm="FedAvg", alpha_name="Dirichlet05", base_seed=42):
        return {
            "algorithm": algorithm, "hypothesis": "Tiny", "dataset_name": "toy",
            "task": "Tabular_CLF", "split_strategy": alpha_name,
            "num_clients_K": 4, "base_seed": base_seed,
            "partition_min_size": 2, "partition_max_retries": 3,
            "partition_repair_policy": "minimum_move_v1",
            "partition_cache_root": str(self.cache_root), "system_data_count": 120,
        }

    def test_algorithm_and_model_do_not_affect_cache_identity(self):
        first = build_or_load_partition(self.make_params("FedAvg"), self.train, self.test, 0)
        second = build_or_load_partition(self.make_params("PraFFL"), self.train, self.test, 0)
        self.assertEqual(first.fingerprint, second.fingerprint)
        for client_id in range(4):
            np.testing.assert_array_equal(first.train_indices[client_id], second.train_indices[client_id])

    def test_alpha_seed_and_data_order_each_change_identity(self):
        base = build_or_load_partition(self.make_params(), self.train, self.test, 0)
        alpha = build_or_load_partition(self.make_params(alpha_name="Dirichlet1"), self.train, self.test, 0)
        seed = build_or_load_partition(self.make_params(base_seed=43), self.train, self.test, 0)
        order = np.arange(len(self.train))[::-1]
        reordered = ToyDataset(
            [self.train.sample_ids[index] for index in order],
            self.train.labels[order], self.train.protected[order],
        )
        data = build_or_load_partition(self.make_params(), reordered, self.test, 0)
        self.assertEqual(len({base.fingerprint, alpha.fingerprint, seed.fingerprint, data.fingerprint}), 4)

    def test_metadata_records_profile_repair_and_fairness_cells(self):
        artifact = build_or_load_partition(self.make_params(), self.train, self.test, 0)
        self.assertIn(artifact.metadata["partitioner"],
                      {"label_dirichlet_v2", "label_dirichlet_repaired_v2"})
        self.assertIn("repair_count", artifact.metadata)
        self.assertIn("joint_counts", artifact.metadata["train_stats"]["0"])
        self.assertEqual(artifact.metadata["indices_sha256"], artifact.indices_sha256)

    def test_duplicate_missing_out_of_bounds_and_digest_corruption_fail_closed(self):
        artifact = build_or_load_partition(self.make_params(), self.train, self.test, 0)
        npz_path = artifact.cache_dir / "indices.npz"
        arrays = dict(np.load(npz_path, allow_pickle=False))
        arrays["train_0"] = np.append(arrays["train_0"], arrays["train_1"][0])
        with open(npz_path, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        with self.assertRaises(PartitionCacheError):
            load_partition_artifact(artifact.cache_dir, artifact.fingerprint,
                                    self.train, self.test, artifact.spec)

    def test_new_dirichlet_never_reads_legacy_algorithm_local_json(self):
        params = self.make_params()
        model_path = Path(self.temp_dir.name) / "model"
        params["model_path"] = str(model_path)
        legacy_dir = model_path / "split_info"
        legacy_dir.mkdir(parents=True)
        with open(legacy_dir / "split_indices.json", "w", encoding="utf-8") as stream:
            json.dump({"split_strategy": "Dirichlet05", "num_clients": 4,
                       "indices": {"0": list(range(120)), "1": [], "2": [], "3": []}}, stream)
        artifact = build_or_load_partition(params, self.train, self.test, 0)
        self.assertTrue(all(len(artifact.train_indices[index]) >= 2 for index in range(4)))
```

- [ ] **Step 2: Run the cache tests and verify missing API failures**

Run:

```bash
python -m unittest tests.test_partition.PartitionCacheTest -v
```

Expected: import fails because the artifact cache API is not defined.

- [ ] **Step 3: Implement canonical cache identity and typed artifact fields**

Add these definitions to `module/partition.py`:

```python
def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def partition_fingerprint(spec: PartitionSpec, train_fingerprint: Mapping[str, Any],
                          test_fingerprint: Mapping[str, Any]) -> str:
    identity = {
        "schema_version": spec.schema_version,
        "partitioner": "uniform_v2" if spec.strategy == "Uniform" else PARTITIONER_NAME,
        "strategy": spec.strategy,
        "alpha": None if spec.alpha is None else format(spec.alpha, ".17g"),
        "num_clients": spec.num_clients,
        "partition_seed": spec.seed,
        "min_samples_per_client": spec.min_samples_per_client,
        "max_retries": spec.max_retries,
        "repair_policy": spec.repair_policy,
        "rng": "PCG64",
        "train_dataset": dict(train_fingerprint),
        "test_dataset": dict(test_fingerprint),
    }
    return sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PartitionArtifact:
    fingerprint: str
    indices_sha256: str
    cache_dir: Path
    spec: PartitionSpec
    train_indices: dict[int, np.ndarray]
    test_indices: dict[int, np.ndarray]
    metadata: dict[str, Any]
```

The cache directory is exactly `<partition_cache_root>/v2/<dataset_name>/<fingerprint>/`. Algorithm, hypothesis, classifier, learning rate, model path, and result path never enter `partition_fingerprint`.

- [ ] **Step 4: Implement statistics, digest validation, and atomic publication**

Add concrete helpers with these signatures and rules:

```python
def _scalar_key(value: Any) -> str:
    raw = value.item() if hasattr(value, "item") else value
    return json.dumps(raw, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _client_stats(indices: Mapping[int, np.ndarray], view: DatasetView) -> dict[str, Any]:
    result = {}
    for client_id, client_indices in indices.items():
        labels = view.labels[client_indices]
        protected = view.protected[client_indices]
        label_counts = {_scalar_key(value): int(np.sum(labels == value)) for value in np.unique(labels)}
        protected_counts = {
            _scalar_key(value): int(np.sum(protected == value)) for value in np.unique(protected)
        }
        joint_counts = {}
        for label, sensitive in zip(labels, protected):
            key = f"label={_scalar_key(label)}|protected={_scalar_key(sensitive)}"
            joint_counts[key] = joint_counts.get(key, 0) + 1
        result[str(client_id)] = {
            "size": len(client_indices),
            "label_counts": label_counts,
            "protected_counts": protected_counts,
            "joint_counts": joint_counts,
        }
    return result


def _indices_bytes(train_indices: Mapping[int, np.ndarray],
                   test_indices: Mapping[int, np.ndarray]) -> bytes:
    chunks = []
    for split, mapping in (("train", train_indices), ("test", test_indices)):
        for client_id in sorted(mapping):
            array = np.asarray(mapping[client_id], dtype="<i8")
            chunks.extend([
                split.encode("ascii"), client_id.to_bytes(8, "big"),
                len(array).to_bytes(8, "big"), array.tobytes(),
            ])
    return b"".join(chunks)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
```

Publish `indices.npz`, then `metadata.json`, then an empty `READY` file. A reader requires all three, recomputes the dataset fingerprints, partition fingerprint, index digest, coverage/uniqueness/bounds/minimum, and both train/test statistics. It raises `PartitionCacheError` for any disagreement rather than drawing a replacement under the same key.

- [ ] **Step 5: Implement uniform v2, build/load orchestration, and read-only legacy routing**

Implement `Uniform` with `Generator(PCG64(spec.seed)).permutation`, balanced lengths differing by at most one, and the same validation/cache schema. Implement:

```python
def build_or_load_partition(param_dict: Mapping[str, Any], train_dataset: Any,
                            test_dataset: Any, repeat_idx: int) -> PartitionArtifact:
    spec = PartitionSpec.from_params(param_dict, repeat_idx)
    train_fp = dataset_fingerprint(
        train_dataset, dataset_name=str(param_dict["dataset_name"]), split="train",
        system_data_count=param_dict.get("system_data_count"),
    )
    test_fp = dataset_fingerprint(
        test_dataset, dataset_name=str(param_dict["dataset_name"]), split="test",
        system_data_count=param_dict.get("system_data_count"),
    )
    fingerprint = partition_fingerprint(spec, train_fp, test_fp)
    cache_dir = (Path(param_dict.get("partition_cache_root", "./partition_cache")) /
                 "v2" / str(param_dict["dataset_name"]) / fingerprint)
    if (cache_dir / "READY").exists():
        return load_partition_artifact(
            cache_dir, fingerprint, train_dataset, test_dataset, spec
        )
    return _build_and_publish_partition(
        cache_dir, fingerprint, spec, train_dataset, test_dataset, train_fp, test_fp
    )
```

For a `LegacyQuantityDirichlet*` strategy, route before `PartitionSpec.from_params` to `load_legacy_quantity_partition`. That function maps the explicit legacy name back to the stored old name, calls the existing `load_split_indices`, validates integer coverage/uniqueness/bounds, returns train subsets only, and never writes into `partition_cache/v2`. If the old file is absent or invalid, raise a diagnostic error; do not synthesize a new quantity-skew split. Keep `save_split_indices`/`load_split_indices` in `tool/checkpoint.py` solely for this read-only path and add a deprecation docstring naming the four allowed legacy aliases.

- [ ] **Step 6: Run partition tests and inspect the published artifact**

Run:

```bash
python -m unittest tests.test_partition -v
python - <<'PY'
from pathlib import Path
for path in Path("partition_cache").glob("v2/*/*"):
    assert (path / "metadata.json").exists()
    assert (path / "indices.npz").exists()
    assert (path / "READY").exists()
print("partition artifacts are complete")
PY
```

Expected: all partition tests pass. The inspection prints `partition artifacts are complete` when tests were intentionally pointed at the repository cache; normal unit fixtures use temporary directories and may leave the glob empty.

- [ ] **Step 7: Commit the shared cache**

Run:

```bash
git add module/partition.py tool/checkpoint.py tests/test_partition.py
git commit -m "feat: cache validated partitions across algorithms"
```

Expected: one commit containing the partition artifact boundary and legacy isolation.

### Task 4: Build deterministic train and client-test loaders from one artifact

**Files:**
- Create: `tests/test_dataloader_partition.py`
- Modify: `module/dataloader.py:1-398`
- Modify: `module/experiment_setup.py:323-346`
- Modify: `tool/seed_manager.py:68-72`

- [ ] **Step 1: Write failing data-bundle integration tests**

Create `tests/test_dataloader_partition.py` with this fixture and contract:

```python
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from module.experiment_setup import Experiment_Create_dataloader, FederatedDataBundle


class LoaderToyDataset(Dataset):
    def __init__(self, prefix, size):
        self.sample_ids = [f"{prefix}-{index}" for index in range(size)]
        self.labels = np.tile([0, 1], size // 2)
        self.protected = np.tile([0, 1, 1, 0], size // 4)
        self.X = torch.arange(size, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "X": self.X[index],
            "labels": torch.tensor(self.labels[index]),
            "protected": torch.tensor(self.protected[index]),
        }


class FederatedDataBundleTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.train = LoaderToyDataset("train", 120)
        self.test = LoaderToyDataset("test", 40)
        self.params = {
            "algorithm": "FedAvg", "dataset_name": "toy", "task": "Tabular_CLF",
            "split_strategy": "Dirichlet05", "num_clients_K": 4,
            "batch_size": 8, "test_batch_size": 10, "base_seed": 42,
            "partition_min_size": 2, "partition_max_retries": 3,
            "partition_repair_policy": "minimum_move_v1",
            "partition_cache_root": self.temp_dir.name,
            "dataloader_num_workers": 0, "system_data_count": 120,
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def build(self, repeat_idx=0, algorithm="FedAvg"):
        params = dict(self.params, repeat_idx=repeat_idx, algorithm=algorithm)
        return Experiment_Create_dataloader(
            params, self.train, None, self.test, params["split_strategy"]
        )

    def test_returns_global_and_client_level_test_loaders(self):
        bundle = self.build()
        self.assertIsInstance(bundle, FederatedDataBundle)
        self.assertEqual(len(bundle.training_dataloaders), 4)
        self.assertEqual(len(bundle.client_dataset_list), 4)
        self.assertEqual(len(bundle.client_testing_dataloaders), 4)
        self.assertEqual(len(bundle.client_testing_dataset_list), 4)
        self.assertEqual(sum(map(len, bundle.client_dataset_list)), len(self.train))
        self.assertEqual(sum(map(len, bundle.client_testing_dataset_list)), len(self.test))
        self.assertEqual(len(bundle.testing_dataloader.dataset), len(self.test))

    def test_same_repeat_is_paired_across_algorithms(self):
        first = self.build(0, "FedAvg")
        second = self.build(0, "FedFACT")
        self.assertEqual(first.partition_fingerprint, second.partition_fingerprint)
        for left, right in zip(first.client_dataset_list, second.client_dataset_list):
            self.assertEqual(list(left.indices), list(right.indices))

    def test_next_repeat_has_distinct_partition_seed_and_fingerprint(self):
        first = self.build(0)
        second = self.build(1)
        self.assertNotEqual(first.partition_fingerprint, second.partition_fingerprint)
        self.assertEqual(first.partition_metadata["partition_seed"], 42)
        self.assertEqual(second.partition_metadata["partition_seed"], 1042)

    def test_global_and_client_test_loaders_do_not_shuffle(self):
        bundle = self.build()
        global_order = torch.cat([batch["X"][:, 0] for batch in bundle.testing_dataloader]).tolist()
        self.assertEqual(global_order, list(range(40)))
        for loader, subset in zip(bundle.client_testing_dataloaders,
                                  bundle.client_testing_dataset_list):
            observed = torch.cat([batch["X"][:, 0] for batch in loader]).tolist()
            expected = [float(index) for index in subset.indices]
            self.assertEqual(observed, expected)
```

- [ ] **Step 2: Run the integration test and verify the missing bundle failure**

Run:

```bash
python -m unittest tests.test_dataloader_partition -v
```

Expected: import fails because `FederatedDataBundle` does not exist.

- [ ] **Step 3: Fix deterministic worker seeding**

Replace `tool/seed_manager.py:68-72` with:

```python
def seed_worker(worker_id: int):
    """Seed Python and NumPy from the worker seed selected by PyTorch."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
```

Do not set a module-global seed in `module/dataloader.py`. The repeat runner owns global RNG setup; partitioning owns a separate local PCG64 generator.

- [ ] **Step 4: Replace inline splitting with artifact-to-loader construction**

In `module/dataloader.py`, import `build_or_load_partition` and `seed_worker`, remove the existing `if "Dirichlet"` loop and the new-split calls to `save_split_indices`, and add:

```python
from dataclasses import dataclass
from typing import Any

from module.partition import PartitionArtifact, build_or_load_partition
from tool.seed_manager import seed_worker


def _loader(dataset, *, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )


def loaders_from_partition(param_dict, training_dataset, testing_dataset,
                           artifact: PartitionArtifact):
    config = get_global_dataloader_config()
    requested_workers = param_dict.get("dataloader_num_workers")
    num_workers = int(config["num_workers"] if requested_workers is None else requested_workers)
    pin_memory = bool(config["pin_memory"])
    client_train = [
        Subset(training_dataset, artifact.train_indices[client_id].tolist())
        for client_id in range(artifact.spec.num_clients)
    ]
    client_test = [
        Subset(testing_dataset, artifact.test_indices[client_id].tolist())
        for client_id in range(artifact.spec.num_clients)
    ]
    train_loaders = [
        _loader(dataset, batch_size=int(param_dict["batch_size"]), shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory)
        for dataset in client_train
    ]
    client_test_loaders = [
        _loader(dataset, batch_size=int(param_dict["test_batch_size"]), shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory)
        for dataset in client_test
    ]
    global_test_loader = _loader(
        testing_dataset, batch_size=int(param_dict["test_batch_size"]), shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loaders, client_train, global_test_loader, client_test_loaders, client_test
```

Keep `get_FL_dataloader` as a compatibility wrapper for callers outside `Experiment_Create_dataloader`, but route new `Dirichlet*` and `Uniform` calls through a supplied `PartitionArtifact`; it must not draw or persist indices itself. Explicit legacy aliases may call `load_legacy_quantity_partition`.

- [ ] **Step 5: Return the typed bundle from experiment setup**

Add before `Experiment_Create_dataloader` in `module/experiment_setup.py`:

```python
from dataclasses import dataclass
from typing import Any

from module.dataloader import loaders_from_partition
from module.partition import build_or_load_partition


@dataclass
class FederatedDataBundle:
    training_dataloaders: list
    client_dataset_list: list
    testing_dataloader: Any
    client_testing_dataloaders: list
    client_testing_dataset_list: list
    partition_fingerprint: str
    partition_metadata: dict
```

Replace `Experiment_Create_dataloader` with:

```python
def Experiment_Create_dataloader(param_dict, training_dataset, validation_dataset,
                                 testing_dataset, split_strategy="Uniform"):
    del validation_dataset
    if split_strategy != param_dict["split_strategy"]:
        raise ValueError("split_strategy argument must match param_dict['split_strategy']")
    repeat_idx = int(param_dict.get("repeat_idx", 0))
    artifact = build_or_load_partition(
        param_dict, training_dataset, testing_dataset, repeat_idx
    )
    values = loaders_from_partition(
        param_dict, training_dataset, testing_dataset, artifact
    )
    param_dict["partition_fingerprint"] = artifact.fingerprint
    param_dict["partition_metadata"] = artifact.metadata
    return FederatedDataBundle(
        training_dataloaders=values[0],
        client_dataset_list=values[1],
        testing_dataloader=values[2],
        client_testing_dataloaders=values[3],
        client_testing_dataset_list=values[4],
        partition_fingerprint=artifact.fingerprint,
        partition_metadata=artifact.metadata,
    )
```

The artifact metadata is the shared support-count source for later FedFACT validation: `train_stats[str(client_id)]` and `test_stats[str(client_id)]` contain `size`, `label_counts`, `protected_counts`, and `joint_counts`; do not rescan image/text payloads in algorithms.

- [ ] **Step 6: Run loader and partition suites**

Run:

```bash
python -m unittest tests.test_partition tests.test_dataloader_partition -v
```

Expected: all tests pass; repeat 0 reports partition seed 42 and repeat 1 reports 1042.

- [ ] **Step 7: Commit loader integration**

Run:

```bash
git add module/dataloader.py module/experiment_setup.py tool/seed_manager.py tests/test_dataloader_partition.py
git commit -m "feat: build repeat-scoped federated loaders"
```

Expected: one commit containing only loader construction and its integration tests.

### Task 5: Define typed algorithm and repeat results

**Files:**
- Create: `tool/experiment_state.py`
- Create: `tests/test_repeat_runner.py`

- [ ] **Step 1: Write failing result-normalization tests**

Start `tests/test_repeat_runner.py` with:

```python
import json
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from tool.experiment_state import (
    AlgorithmRunResult,
    RepeatResult,
    aggregate_repeat_results,
    normalize_algorithm_result,
)
from tool.checkpoint import (
    build_experiment_config_hash,
    load_checkpoint,
    save_checkpoint,
    save_repeat_metrics,
)


class ExperimentStateTest(unittest.TestCase):
    def test_normalizes_legacy_three_tuple(self):
        model = torch.nn.Linear(1, 1)
        result = normalize_algorithm_result((model, 1.5, 2.5))
        self.assertIs(result.global_model, model)
        self.assertEqual(result.total_gpu_seconds, 1.5)
        self.assertEqual(result.total_communication_cost, 2.5)
        self.assertEqual(result.algorithm_state, {})
        self.assertIsNone(result.amp_scaler_state)

    def test_rejects_unstructured_return_values(self):
        with self.assertRaisesRegex(TypeError, "AlgorithmRunResult or a three-item tuple"):
            normalize_algorithm_result((torch.nn.Linear(1, 1), 1.0))

    def test_aggregate_requires_each_repeat_exactly_once(self):
        rows = [
            RepeatResult(index, 42 + 1000 * index, f"fp-{index}",
                         {"ACC": 0.5 + index * 0.1, "DEO": 0.2}, 1.0, 2.0)
            for index in range(3)
        ]
        aggregate = aggregate_repeat_results(rows, expected_repeats=3)
        self.assertEqual(aggregate["repeat_seeds"], [42, 1042, 2042])
        self.assertAlmostEqual(aggregate["metrics"]["ACC"]["mean"], 0.6)
        with self.assertRaisesRegex(ValueError, "repeat indices"):
            aggregate_repeat_results(rows[:2], expected_repeats=3)
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run:

```bash
python -m unittest tests.test_repeat_runner.ExperimentStateTest -v
```

Expected: import fails with `ModuleNotFoundError: No module named 'tool.experiment_state'`.

- [ ] **Step 3: Implement the stable result protocol**

Create `tool/experiment_state.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class AlgorithmRunResult:
    global_model: Any
    total_gpu_seconds: float
    total_communication_cost: float
    algorithm_state: dict[str, Any] = field(default_factory=dict)
    amp_scaler_state: dict[str, Any] | None = None
    client_selection_history: list[list[int]] = field(default_factory=list)


@dataclass(frozen=True)
class RepeatResult:
    repeat_idx: int
    repeat_seed: int
    partition_fingerprint: str
    metrics: dict[str, Any]
    total_gpu_seconds: float
    total_communication_cost: float


def normalize_algorithm_result(value: Any) -> AlgorithmRunResult:
    if isinstance(value, AlgorithmRunResult):
        return value
    if isinstance(value, tuple) and len(value) == 3:
        model, gpu_seconds, communication_cost = value
        return AlgorithmRunResult(model, float(gpu_seconds), float(communication_cost))
    raise TypeError("algorithm must return AlgorithmRunResult or a three-item tuple")


def aggregate_repeat_results(results: Sequence[RepeatResult], expected_repeats: int) -> dict[str, Any]:
    ordered = sorted(results, key=lambda item: item.repeat_idx)
    indices = [item.repeat_idx for item in ordered]
    if indices != list(range(expected_repeats)):
        raise ValueError(f"repeat indices must be 0..{expected_repeats - 1}, got {indices}")
    numeric_keys = sorted(set.intersection(*[
        {key for key, value in item.metrics.items()
         if isinstance(value, (int, float)) and not isinstance(value, bool)}
        for item in ordered
    ]))
    metrics = {}
    for key in numeric_keys:
        values = np.asarray([float(item.metrics[key]) for item in ordered], dtype=np.float64)
        metrics[key] = {
            "values": values.tolist(),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    return {
        "repeat_indices": indices,
        "repeat_seeds": [item.repeat_seed for item in ordered],
        "partition_fingerprints": [item.partition_fingerprint for item in ordered],
        "metrics": metrics,
        "gpu_seconds": [item.total_gpu_seconds for item in ordered],
        "communication_cost": [item.total_communication_cost for item in ordered],
    }
```

- [ ] **Step 4: Run and commit the typed protocol**

Run:

```bash
python -m unittest tests.test_repeat_runner.ExperimentStateTest -v
git add tool/experiment_state.py tests/test_repeat_runner.py
git commit -m "feat: define resumable algorithm result protocol"
```

Expected: three tests pass, followed by one focused commit.

### Task 6: Make checkpoints complete, atomic, latest-only, and fail closed

**Files:**
- Create: `tests/test_checkpoint.py`
- Modify: `tool/checkpoint.py:1-101,151-189`

- [ ] **Step 1: Write failing config-hash and checkpoint-schema tests**

Create `tests/test_checkpoint.py` with:

```python
import os
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tool.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointCompatibilityError,
    CheckpointState,
    build_experiment_config_hash,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)


class FakeScaler:
    def __init__(self, scale):
        self.scale = scale

    def state_dict(self):
        return {"scale": self.scale}


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.params = {
            "model_path": self.temp_dir.name,
            "result_path": str(Path(self.temp_dir.name) / "result.txt"),
            "log_path": str(Path(self.temp_dir.name) / "run.log"),
            "dataset_name": "toy", "task": "Tabular_CLF", "algorithm": "FedAvg",
            "hypothesis": "Tiny", "split_strategy": "Dirichlet05",
            "num_clients_K": 2, "communication_round_I": 2,
            "algorithm_epoch_T": 1, "batch_size": 4, "learning_rate": 0.1,
            "base_seed": 42, "repeat_idx": 0, "repeat_seed": 42,
            "partition_fingerprint": "partition-a", "resume": True,
            "parallel_repeats": 1, "checkpoint_keep_latest": 9,
        }
        self.params["experiment_config_hash"] = build_experiment_config_hash(self.params)
        self.model = torch.nn.Linear(2, 1)

    def tearDown(self):
        self.temp_dir.cleanup()

    def save(self, round_index=0):
        return save_checkpoint(
            self.params, round_index, self.model,
            algorithm_state={"dual": torch.tensor([1.0])},
            amp_scaler=FakeScaler(128.0),
            total_gpu_seconds=2.0,
            total_runtime_seconds=3.0,
            total_communication_cost=4.0,
            client_selection_history=[[1]],
        )

    def test_config_hash_ignores_paths_and_resume_controls(self):
        changed = dict(self.params, model_path="elsewhere", result_path="elsewhere.txt",
                       resume=False, checkpoint_keep_latest=1, parallel_repeats=3)
        self.assertEqual(build_experiment_config_hash(self.params),
                         build_experiment_config_hash(changed))
        changed["learning_rate"] = 0.2
        self.assertNotEqual(build_experiment_config_hash(self.params),
                            build_experiment_config_hash(changed))

    def test_checkpoint_contains_complete_round_boundary_state(self):
        path = self.save(0)
        state = load_checkpoint(
            self.params,
            expected_config_hash=self.params["experiment_config_hash"],
            expected_partition_fingerprint="partition-a",
            expected_repeat_idx=0,
        )
        self.assertIsInstance(state, CheckpointState)
        self.assertEqual(state.schema_version, CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(state.next_round, 1)
        self.assertEqual(state.phase, "train")
        self.assertEqual(state.algorithm_state["dual"].item(), 1.0)
        self.assertEqual(state.amp_scaler_state, {"scale": 128.0})
        self.assertEqual(state.total_gpu_seconds, 2.0)
        self.assertEqual(state.total_runtime_seconds, 3.0)
        self.assertEqual(state.total_communication_cost, 4.0)
        self.assertEqual(state.client_selection_history, [[1]])
        self.assertTrue(Path(path).name == "checkpoint_latest.pt")

    def test_last_round_is_evaluation_phase(self):
        self.save(1)
        state = load_checkpoint(self.params)
        self.assertEqual(state.next_round, 2)
        self.assertEqual(state.phase, "evaluate")

    def test_only_latest_checkpoint_is_retained(self):
        self.save(0)
        self.save(1)
        repeat_dir = Path(self.temp_dir.name) / "experiment_state" / self.params[
            "experiment_config_hash"
        ] / "repeat_00"
        self.assertEqual([path.name for path in repeat_dir.glob("checkpoint*.pt")],
                         ["checkpoint_latest.pt"])
```

- [ ] **Step 2: Add mismatch-before-RNG-mutation and RNG-roundtrip tests**

Append to `CheckpointTest`:

```python
    def test_mismatch_raises_before_rng_is_mutated(self):
        self.save(0)
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        expected = (random.random(), np.random.rand(), torch.rand(1).item())
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        with self.assertRaisesRegex(CheckpointCompatibilityError, "partition fingerprint"):
            load_checkpoint(self.params, expected_partition_fingerprint="wrong")
        observed = (random.random(), np.random.rand(), torch.rand(1).item())
        self.assertEqual(expected, observed)

    def test_rng_restore_round_trips_python_numpy_and_torch(self):
        random.seed(77)
        np.random.seed(77)
        torch.manual_seed(77)
        self.save(0)
        expected = (random.random(), np.random.rand(), torch.rand(2))
        state = load_checkpoint(self.params)
        restore_rng_state(state)
        observed = (random.random(), np.random.rand(), torch.rand(2))
        self.assertEqual(expected[0], observed[0])
        self.assertEqual(expected[1], observed[1])
        torch.testing.assert_close(expected[2], observed[2], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA RNG requires CUDA")
    def test_cuda_rng_state_is_captured_for_every_visible_device(self):
        torch.cuda.manual_seed_all(91)
        self.save(0)
        state = load_checkpoint(self.params)
        self.assertEqual(len(state.rng_state["torch_cuda"]), torch.cuda.device_count())
```

- [ ] **Step 3: Run tests and verify the old permissive loader fails the contract**

Run:

```bash
python -m unittest tests.test_checkpoint.CheckpointTest -v
```

Expected: failures show missing schema/hash/partition/repeat/scaler/CUDA/counter fields, non-atomic per-round file naming, and the old loader mutating RNG during load.

- [ ] **Step 4: Implement canonical config hashing**

In `tool/checkpoint.py`, add:

```python
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import copy
import tempfile


CHECKPOINT_SCHEMA_VERSION = 2
_CONFIG_EXCLUDED_KEYS = {
    "model_path", "result_path", "log_path", "basic_path", "tb_log_dir",
    "resume", "start_exp", "parallel_repeats", "checkpoint_save_freq",
    "checkpoint_keep_latest", "partition_cache_root", "repeat_idx", "repeat_seed",
    "partition_fingerprint", "partition_metadata", "experiment_config_hash",
    "Experiment_NO", "CUDA_VISIBLE_DEVICES",
}


class CheckpointCompatibilityError(RuntimeError):
    pass


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, torch.device):
        return value.type
    raise TypeError(f"non-canonical experiment config value: {type(value).__name__}")


def canonical_experiment_config(param_dict: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in param_dict.items():
        if key in _CONFIG_EXCLUDED_KEYS or key.startswith("_runtime_"):
            continue
        result[str(key)] = (
            str(value).split(":", 1)[0] if key == "device" else _json_value(value)
        )
    return result


def build_experiment_config_hash(param_dict: Mapping[str, Any]) -> str:
    payload = json.dumps(canonical_experiment_config(param_dict), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()
```

`device` is normalized to its device type, so `cuda` and `cuda:0` share configuration identity; `use_amp` remains in the hash and prevents AMP/FP32 resume mixing. Every JSON-serializable algorithm hyperparameter remains included automatically.

- [ ] **Step 5: Implement typed checkpoint payload and atomic latest-only paths**

Use the exact state fields below:

```python
@dataclass(frozen=True)
class CheckpointState:
    schema_version: int
    experiment_config_hash: str
    partition_fingerprint: str
    repeat_idx: int
    repeat_seed: int
    next_round: int
    phase: str
    global_model_state: dict[str, Any]
    algorithm_state: dict[str, Any]
    amp_scaler_state: dict[str, Any] | None
    rng_state: dict[str, Any]
    total_gpu_seconds: float
    total_runtime_seconds: float
    total_communication_cost: float
    client_selection_history: list[list[int]]
    path: Path


def get_repeat_state_dir(param_dict: Mapping[str, Any], repeat_idx: int | None = None) -> Path:
    config_hash = str(param_dict["experiment_config_hash"])
    index = int(param_dict["repeat_idx"] if repeat_idx is None else repeat_idx)
    return Path(param_dict["model_path"]) / "experiment_state" / config_hash / f"repeat_{index:02d}"


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temp_name)
        with open(temp_name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
```

Replace `save_checkpoint` with this compatible signature:

```python
def save_checkpoint(param_dict, iter_t, global_model, *, algorithm_state=None,
                    amp_scaler=None, total_gpu_seconds=0.0,
                    total_runtime_seconds=0.0, total_communication_cost=0.0,
                    client_selection_history=(), extra_state=None, **legacy_kwargs):
    if legacy_kwargs:
        unknown = ", ".join(sorted(legacy_kwargs))
        raise TypeError(f"unsupported checkpoint fields: {unknown}")
    if algorithm_state is not None and extra_state:
        raise ValueError("pass algorithm_state or legacy extra_state, not both")
    stored_algorithm_state = algorithm_state if algorithm_state is not None else (extra_state or {})
    repeat_idx = int(param_dict["repeat_idx"])
    total_rounds = int(param_dict["communication_round_I"])
    next_round = int(iter_t) + 1
    phase = "evaluate" if next_round >= total_rounds else "train"
    scaler_state = None
    if amp_scaler is not None:
        scaler_state = amp_scaler.state_dict() if hasattr(amp_scaler, "state_dict") else amp_scaler
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "experiment_config_hash": str(param_dict["experiment_config_hash"]),
        "partition_fingerprint": str(param_dict["partition_fingerprint"]),
        "repeat_idx": repeat_idx,
        "repeat_seed": int(param_dict["repeat_seed"]),
        "next_round": next_round,
        "phase": phase,
        "global_model_state": copy.deepcopy(global_model.state_dict()),
        "algorithm_state": copy.deepcopy(stored_algorithm_state),
        "amp_scaler_state": copy.deepcopy(scaler_state),
        "rng_state": _capture_rng_state(),
        "total_gpu_seconds": float(total_gpu_seconds),
        "total_runtime_seconds": float(total_runtime_seconds),
        "total_communication_cost": float(total_communication_cost),
        "client_selection_history": [list(map(int, row)) for row in client_selection_history],
    }
    path = get_repeat_state_dir(param_dict) / "checkpoint_latest.pt"
    _atomic_torch_save(payload, path)
    return path
```

The phase is never supplied in `algorithm_state`: `save_checkpoint` derives it from `iter_t + 1` and `communication_round_I`, eliminating a caller-controlled off-by-one.

- [ ] **Step 6: Validate before constructing state or restoring RNG**

Replace `load_checkpoint` and add the explicit restore function:

```python
def load_checkpoint(param_dict, *, expected_config_hash=None,
                    expected_partition_fingerprint=None, expected_repeat_idx=None,
                    target_round=None):
    if target_round is not None:
        raise CheckpointCompatibilityError("schema v2 retains only checkpoint_latest.pt")
    path = get_repeat_state_dir(param_dict, expected_repeat_idx) / "checkpoint_latest.pt"
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "schema_version", "experiment_config_hash", "partition_fingerprint",
        "repeat_idx", "repeat_seed", "next_round", "phase", "global_model_state",
        "algorithm_state", "amp_scaler_state", "rng_state", "total_gpu_seconds",
        "total_runtime_seconds", "total_communication_cost", "client_selection_history",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise CheckpointCompatibilityError(f"checkpoint missing fields: {missing}")
    expected_hash = expected_config_hash or str(param_dict["experiment_config_hash"])
    expected_partition = expected_partition_fingerprint or str(param_dict["partition_fingerprint"])
    expected_index = int(param_dict["repeat_idx"] if expected_repeat_idx is None else expected_repeat_idx)
    comparisons = (
        (payload["schema_version"], CHECKPOINT_SCHEMA_VERSION, "schema version"),
        (payload["experiment_config_hash"], expected_hash, "experiment config hash"),
        (payload["partition_fingerprint"], expected_partition, "partition fingerprint"),
        (payload["repeat_idx"], expected_index, "repeat index"),
        (payload["repeat_seed"], int(param_dict["repeat_seed"]), "repeat seed"),
    )
    for stored, expected, name in comparisons:
        if stored != expected:
            raise CheckpointCompatibilityError(
                f"{name} mismatch: stored={stored!r}, expected={expected!r}"
            )
    if payload["phase"] not in {"train", "evaluate"}:
        raise CheckpointCompatibilityError(f"invalid checkpoint phase: {payload['phase']!r}")
    if not 0 <= int(payload["next_round"]) <= int(param_dict["communication_round_I"]):
        raise CheckpointCompatibilityError("next_round is outside configured round bounds")
    return CheckpointState(path=path, **payload)


def restore_rng_state(state: CheckpointState) -> None:
    cuda_states = state.rng_state["torch_cuda"]
    stored_count = int(state.rng_state["cuda_device_count"])
    current_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if stored_count != current_count:
        raise CheckpointCompatibilityError(
            f"CUDA RNG device count mismatch: stored={stored_count}, current={current_count}"
        )
    random.setstate(state.rng_state["python"])
    np.random.set_state(state.rng_state["numpy"])
    torch.set_rng_state(state.rng_state["torch_cpu"])
    if cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)
```

Retain `check_resume_status` only as a thin deprecated wrapper that returns `load_checkpoint(param_dict)` when `param_dict.get("resume", False)` is true and otherwise returns `None`. Change `clean_old_checkpoints` to remove schema-v1 round files for the current repeat but never delete the schema-v2 `checkpoint_latest.pt`.

- [ ] **Step 7: Run checkpoint tests**

Run:

```bash
python -m unittest tests.test_checkpoint.CheckpointTest -v
```

Expected: seven tests pass on CPU; the CUDA test passes on Ronnie and is skipped only on hosts without CUDA.

- [ ] **Step 8: Commit the checkpoint boundary**

Run:

```bash
git add tool/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: persist exact round-boundary checkpoints"
```

Expected: one commit containing schema-v2 checkpoint persistence and tests.

### Task 7: Make final evaluation the atomic completion boundary

**Files:**
- Modify: `tests/test_checkpoint.py`
- Modify: `tool/checkpoint.py`

- [ ] **Step 1: Add failing repeat-metrics and artifact-policy tests**

Append these imports and methods to `tests/test_checkpoint.py`:

```python
from tool.checkpoint import (
    clear_repeat_artifacts,
    finalize_repeat_artifacts,
    load_repeat_metrics,
    save_aggregate_metrics,
    save_repeat_metrics,
)


class RepeatMetricsTest(CheckpointTest):
    def test_checkpoint_alone_is_not_completion(self):
        self.save(1)
        self.assertIsNone(load_repeat_metrics(
            self.params, 0, self.params["experiment_config_hash"], "partition-a"
        ))

    def test_metrics_become_completion_only_after_atomic_write(self):
        path = save_repeat_metrics(
            self.params, 0, self.params["experiment_config_hash"], "partition-a",
            {"ACC": 0.75, "DEO": 0.2, "SPD": 0.1},
            repeat_seed=42, total_gpu_seconds=2.0, total_communication_cost=4.0,
        )
        loaded = load_repeat_metrics(
            self.params, 0, self.params["experiment_config_hash"], "partition-a"
        )
        self.assertEqual(loaded["metrics"]["ACC"], 0.75)
        self.assertEqual(Path(path).name, "metrics.json")
        self.assertFalse(any(Path(path).parent.glob(".metrics.json.*")))

    def test_metrics_mismatch_fails_closed(self):
        save_repeat_metrics(
            self.params, 0, self.params["experiment_config_hash"], "partition-a",
            {"ACC": 0.75}, repeat_seed=42,
            total_gpu_seconds=0.0, total_communication_cost=0.0,
        )
        with self.assertRaisesRegex(CheckpointCompatibilityError, "partition fingerprint"):
            load_repeat_metrics(
                self.params, 0, self.params["experiment_config_hash"], "partition-b"
            )

    def test_metrics_only_policy_removes_completed_resume_state(self):
        self.save(1)
        checkpoint_path = get_repeat_state_dir(self.params) / "checkpoint_latest.pt"
        finalize_repeat_artifacts(self.params, 0, self.model, policy="metrics_only")
        self.assertFalse(checkpoint_path.exists())

    def test_fresh_run_clears_stale_repeat_state_and_metrics(self):
        self.save(0)
        save_repeat_metrics(
            self.params, 0, self.params["experiment_config_hash"], "partition-a",
            {"ACC": 0.5}, repeat_seed=42,
            total_gpu_seconds=0.0, total_communication_cost=0.0,
        )
        clear_repeat_artifacts(self.params, 0)
        repeat_dir = get_repeat_state_dir(self.params, 0)
        self.assertFalse((repeat_dir / "checkpoint_latest.pt").exists())
        self.assertFalse((repeat_dir / "metrics.json").exists())
```

Also import `get_repeat_state_dir` with the other checkpoint functions.

- [ ] **Step 2: Run the metrics tests and verify missing functions**

Run:

```bash
python -m unittest tests.test_checkpoint.RepeatMetricsTest -v
```

Expected: import fails because the structured metrics API does not exist.

- [ ] **Step 3: Add JSON-safe metrics and atomic completion functions**

Add to `tool/checkpoint.py`:

```python
def _atomic_json_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def save_repeat_metrics(param_dict, repeat_idx, config_hash, partition_fingerprint,
                        metrics, *, repeat_seed, total_gpu_seconds,
                        total_communication_cost):
    path = get_repeat_state_dir(param_dict, repeat_idx) / "metrics.json"
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "experiment_config_hash": config_hash,
        "experiment_config": canonical_experiment_config(param_dict),
        "partition_fingerprint": partition_fingerprint,
        "repeat_idx": int(repeat_idx),
        "repeat_seed": int(repeat_seed),
        "metrics": _json_value(metrics),
        "total_gpu_seconds": float(total_gpu_seconds),
        "total_communication_cost": float(total_communication_cost),
    }
    _atomic_json_save(payload, path)
    return path


def load_repeat_metrics(param_dict, repeat_idx, expected_config_hash,
                        expected_partition_fingerprint):
    path = get_repeat_state_dir(param_dict, repeat_idx) / "metrics.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    checks = (
        (payload.get("schema_version"), CHECKPOINT_SCHEMA_VERSION, "schema version"),
        (payload.get("experiment_config_hash"), expected_config_hash, "experiment config hash"),
        (payload.get("partition_fingerprint"), expected_partition_fingerprint,
         "partition fingerprint"),
        (payload.get("repeat_idx"), int(repeat_idx), "repeat index"),
        (payload.get("repeat_seed"), int(param_dict["repeat_seed"]), "repeat seed"),
    )
    for stored, expected, name in checks:
        if stored != expected:
            raise CheckpointCompatibilityError(
                f"repeat metrics {name} mismatch: stored={stored!r}, expected={expected!r}"
            )
    if not isinstance(payload.get("metrics"), dict):
        raise CheckpointCompatibilityError("repeat metrics payload is missing metrics")
    return payload


def clear_repeat_artifacts(param_dict, repeat_idx):
    repeat_dir = get_repeat_state_dir(param_dict, repeat_idx)
    for name in ("checkpoint_latest.pt", "metrics.json", "final_global_model.pt"):
        path = repeat_dir / name
        if path.exists():
            path.unlink()


def finalize_repeat_artifacts(param_dict, repeat_idx, global_model, policy):
    if policy not in {"metrics_only", "global_model", "full_state"}:
        raise ValueError(f"unsupported final artifact policy: {policy}")
    repeat_dir = get_repeat_state_dir(param_dict, repeat_idx)
    checkpoint = repeat_dir / "checkpoint_latest.pt"
    if policy == "global_model":
        _atomic_torch_save(global_model.state_dict(), repeat_dir / "final_global_model.pt")
    if policy != "full_state" and checkpoint.exists():
        checkpoint.unlink()


def save_aggregate_metrics(param_dict, aggregate):
    path = Path(str(param_dict["result_path"]) + ".json")
    _atomic_json_save({
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "experiment_config_hash": str(param_dict["experiment_config_hash"]),
        "experiment_config": canonical_experiment_config(param_dict),
        "aggregate": _json_value(aggregate),
    }, path)
    return path
```

`metrics.json` is the sole completed-repeat marker. A final-round checkpoint without it means phase `evaluate`, not completion. Write the human-readable `result_path` summary only after the aggregate JSON is safely published.

- [ ] **Step 4: Run all checkpoint tests**

Run:

```bash
python -m unittest tests.test_checkpoint -v
```

Expected: all checkpoint and repeat-metrics tests pass; no temporary metric/checkpoint files remain.

- [ ] **Step 5: Commit atomic completion semantics**

Run:

```bash
git add tool/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: make final metrics the repeat completion marker"
```

Expected: one commit containing repeat completion and artifact retention only.

### Task 8: Route serial and parallel execution through one repeat worker

**Files:**
- Modify: `tests/test_repeat_runner.py`
- Modify: `experiment.py:347-682`

- [ ] **Step 1: Add failing repeat scheduling and construction-order tests**

Append this test class to `tests/test_repeat_runner.py`. Patch the repository factories rather than loading a real corpus:

```python
class RepeatRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.params = {
            "model_path": str(Path(self.temp_dir.name) / "models"),
            "result_path": str(Path(self.temp_dir.name) / "result.txt"),
            "log_path": str(Path(self.temp_dir.name) / "run.log"),
            "partition_cache_root": str(Path(self.temp_dir.name) / "partitions"),
            "dataset_name": "toy", "dataset": "toy", "task": "Tabular_CLF",
            "algorithm": "Toy", "hypothesis": "TinyANN", "model_type": "ANN",
            "split_strategy": "Dirichlet05", "num_clients_K": 2,
            "batch_size": 4, "test_batch_size": 4, "communication_round_I": 2,
            "algorithm_epoch_T": 1, "FL_fraction": 1.0, "FL_drop_rate": 0.0,
            "learning_rate": 0.1, "optimize_method": "sgd", "device": "cpu",
            "use_amp": False, "base_seed": 42, "exp_repeat_times": 3,
            "parallel_repeats": 1, "resume": False, "partition_min_size": 1,
            "partition_max_retries": 2, "partition_repair_policy": "minimum_move_v1",
            "dataloader_num_workers": 0, "final_artifact_policy": "full_state",
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_three_repeats_use_distinct_deterministic_seeds(self):
        from experiment import Experiment_FL
        seen = []

        def fake_worker(repeat_idx, algorithm_function, evaluator_function, param_dict):
            del algorithm_function, evaluator_function
            seed = int(param_dict["base_seed"]) + 1000 * repeat_idx
            seen.append(seed)
            return RepeatResult(repeat_idx, seed, f"partition-{seed}",
                                {"ACC": seed / 10000.0}, 0.0, 0.0)

        with mock.patch("experiment._run_single_repeat", side_effect=fake_worker):
            aggregate = Experiment_FL(lambda: None, self.params)
        self.assertEqual(seen, [42, 1042, 2042])
        self.assertEqual(aggregate["repeat_seeds"], [42, 1042, 2042])

    def test_seed_is_set_before_dataset_loader_and_model_construction(self):
        from experiment import _run_single_repeat
        events = []

        class Bundle:
            partition_fingerprint = "partition-42"
            partition_metadata = {}
            training_dataloaders = []
            client_dataset_list = []
            testing_dataloader = []
            client_testing_dataloaders = []
            client_testing_dataset_list = []

        def dataset_factory(params):
            events.append(("dataset", random.random(), np.random.rand(), torch.rand(1).item()))
            return object(), None, object()

        def loader_factory(params, train, validation, test, split):
            del params, train, validation, test, split
            events.append(("loader", random.random(), np.random.rand(), torch.rand(1).item()))
            return Bundle()

        def model_factory(params):
            del params
            events.append(("model", random.random(), np.random.rand(), torch.rand(1).item()))
            return torch.nn.Linear(1, 1)

        def algorithm(*args, **kwargs):
            return AlgorithmRunResult(args[1], 0.0, 0.0)

        def evaluator(model, params, data_bundle, algorithm_state):
            del model, params, data_bundle, algorithm_state
            return {"ACC": 1.0}

        with mock.patch("experiment.Experiment_Create_dataset", side_effect=dataset_factory), \
             mock.patch("experiment.Experiment_Create_dataloader", side_effect=loader_factory), \
             mock.patch("experiment.Experiment_Create_model", side_effect=model_factory), \
             mock.patch("experiment.calculate_communication_cost", return_value=0.0):
            _run_single_repeat(0, algorithm, evaluator, self.params)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        expected_dataset = (random.random(), np.random.rand(), torch.rand(1).item())
        self.assertEqual(events[0][0], "dataset")
        self.assertEqual(events[0][1:], expected_dataset)
        self.assertEqual([event[0] for event in events], ["dataset", "loader", "model"])

    def test_cuda_repeats_must_be_serial(self):
        from experiment import Experiment_FL
        params = dict(self.params, device="cuda", parallel_repeats=2)
        with self.assertRaisesRegex(ValueError, "CUDA repeats must run serially"):
            Experiment_FL(lambda: None, params)
```

The dataset event is included intentionally: it proves no constructor that could create a sampler, cache worker, or model runs before repeat seeding.

- [ ] **Step 2: Add failing completion and evaluation-stage recovery tests**

Add methods that use temporary schema-v2 artifacts:

```python
    def test_resume_false_ignores_and_replaces_stale_completion(self):
        from experiment import _run_single_repeat
        calls = {"algorithm": 0}

        def algorithm(*args, **kwargs):
            calls["algorithm"] += 1
            return AlgorithmRunResult(args[1], 0.0, 0.0)

        stale = dict(self.params)
        stale["repeat_idx"] = 0
        stale["repeat_seed"] = 42
        stale["experiment_config_hash"] = build_experiment_config_hash(stale)
        stale["partition_fingerprint"] = "partition-42"
        save_repeat_metrics(stale, 0, stale["experiment_config_hash"], "partition-42",
                            {"ACC": -1.0}, repeat_seed=42,
                            total_gpu_seconds=0.0, total_communication_cost=0.0)
        bundle = self._fake_bundle("partition-42")
        with self._patched_factories(bundle):
            result = _run_single_repeat(0, algorithm, self._toy_evaluator, self.params)
        self.assertEqual(calls["algorithm"], 1)
        self.assertEqual(result.metrics["ACC"], 1.0)

    def test_final_round_checkpoint_resumes_at_evaluation(self):
        from experiment import _run_single_repeat
        calls = {"algorithm": 0, "evaluator": 0}
        params = dict(self.params, resume=True)
        bundle = self._fake_bundle("partition-42")

        def algorithm(*args, **kwargs):
            calls["algorithm"] += 1
            return AlgorithmRunResult(args[1], 0.0, 0.0, {"private": 7})

        def evaluator(model, repeat_params, data_bundle, algorithm_state):
            del model, repeat_params, data_bundle
            calls["evaluator"] += 1
            if calls["evaluator"] == 1:
                raise RuntimeError("planned evaluation crash")
            return {"ACC": float(algorithm_state["private"])}

        with self._patched_factories(bundle):
            with self.assertRaisesRegex(RuntimeError, "planned evaluation crash"):
                _run_single_repeat(0, algorithm, evaluator, params)
            result = _run_single_repeat(0, algorithm, evaluator, params)
        self.assertEqual(calls, {"algorithm": 1, "evaluator": 2})
        self.assertEqual(result.metrics["ACC"], 7.0)
```

Define `_fake_bundle`, `_patched_factories`, and `_toy_evaluator` in the class as concrete helpers returning a small `FederatedDataBundle`, deterministic `torch.nn.Linear`, and `{"ACC": 1.0}`. The patched loader must set `params["partition_fingerprint"]` and `params["partition_metadata"]`, matching production behavior.

- [ ] **Step 3: Run the repeat tests and verify current orchestration failures**

Run:

```bash
python -m unittest tests.test_repeat_runner.RepeatRunnerTest -v
```

Expected: failures show the old runner receiving prebuilt model/loaders, unconditional checkpoint loading, completed final-round checkpoints being skipped without metrics, and separate serial/parallel logic.

- [ ] **Step 4: Replace the repeat worker with seed-before-construction orchestration**

In `experiment.py`, import `inspect`, the new result types, and checkpoint functions. Replace `_run_single_repeat` with this control flow and exact signature:

```python
def _run_single_repeat(repeat_idx, algorithm_function, evaluator_function, param_dict):
    repeat_param = dict(param_dict)
    repeat_seed = get_repeat_seed(repeat_idx=repeat_idx,
                                  base_seed=int(param_dict.get("base_seed", 42)))
    repeat_param.update({"repeat_idx": repeat_idx, "repeat_seed": repeat_seed})
    set_all_seeds(repeat_seed)

    training_dataset, validation_dataset, testing_dataset = Experiment_Create_dataset(repeat_param)
    data_bundle = Experiment_Create_dataloader(
        repeat_param, training_dataset, validation_dataset, testing_dataset,
        repeat_param["split_strategy"],
    )
    repeat_param["partition_fingerprint"] = data_bundle.partition_fingerprint
    repeat_param["partition_metadata"] = data_bundle.partition_metadata
    config_hash = str(param_dict["experiment_config_hash"])
    repeat_param["experiment_config_hash"] = config_hash

    if repeat_param.get("resume", False):
        completed = load_repeat_metrics(
            repeat_param, repeat_idx, config_hash, data_bundle.partition_fingerprint
        )
        if completed is not None:
            return RepeatResult(
                repeat_idx, repeat_seed, data_bundle.partition_fingerprint,
                completed["metrics"], float(completed["total_gpu_seconds"]),
                float(completed["total_communication_cost"]),
                resource_usage=completed.get("resource_usage", {}),
            )
    else:
        clear_repeat_artifacts(repeat_param, repeat_idx)

    global_model = Experiment_Create_model(repeat_param)
    resume_state = None
    if repeat_param.get("resume", False):
        resume_state = load_checkpoint(
            repeat_param, expected_config_hash=config_hash,
            expected_partition_fingerprint=data_bundle.partition_fingerprint,
            expected_repeat_idx=repeat_idx,
        )
    if resume_state is not None:
        global_model.load_state_dict(resume_state.global_model_state)
        restore_rng_state(resume_state)

    formula_cost = calculate_communication_cost(
        algorithm_function.__name__, repeat_param, global_model
    )
    wall_start = time.monotonic()
    if resume_state is not None and resume_state.phase == "evaluate":
        run_result = AlgorithmRunResult(
            global_model=global_model,
            total_gpu_seconds=resume_state.total_gpu_seconds,
            total_communication_cost=resume_state.total_communication_cost,
            algorithm_state=resume_state.algorithm_state,
            amp_scaler_state=resume_state.amp_scaler_state,
            client_selection_history=resume_state.client_selection_history,
        )
        prior_runtime = resume_state.total_runtime_seconds
    else:
        signature = inspect.signature(algorithm_function)
        if resume_state is not None and (
            resume_state.algorithm_state or resume_state.amp_scaler_state
        ) and "resume_state" not in signature.parameters:
            raise CheckpointCompatibilityError(
                f"{algorithm_function.__name__} cannot restore algorithm/AMP state"
            )
        kwargs = {"start_round": 0 if resume_state is None else resume_state.next_round}
        if "resume_state" in signature.parameters:
            kwargs["resume_state"] = resume_state
        raw_result = algorithm_function(
            repeat_param["device"], global_model,
            repeat_param["algorithm_epoch_T"], repeat_param["num_clients_K"],
            repeat_param["communication_round_I"], repeat_param["FL_fraction"],
            repeat_param["FL_drop_rate"], data_bundle.training_dataloaders,
            training_dataset, data_bundle.client_dataset_list, repeat_param,
            data_bundle.testing_dataloader, len(testing_dataset), **kwargs,
        )
        run_result = normalize_algorithm_result(raw_result)
        prior_runtime = 0.0 if resume_state is None else resume_state.total_runtime_seconds

    total_runtime = prior_runtime + (time.monotonic() - wall_start)
    save_checkpoint(
        repeat_param, int(repeat_param["communication_round_I"]) - 1,
        run_result.global_model, algorithm_state=run_result.algorithm_state,
        amp_scaler=run_result.amp_scaler_state,
        total_gpu_seconds=run_result.total_gpu_seconds,
        total_runtime_seconds=total_runtime,
        total_communication_cost=run_result.total_communication_cost,
        client_selection_history=run_result.client_selection_history,
    )
    evaluate = evaluator_function or _evaluate_global_model
    metrics = evaluate(
        run_result.global_model, repeat_param, data_bundle, run_result.algorithm_state
    )
    save_repeat_metrics(
        repeat_param, repeat_idx, config_hash, data_bundle.partition_fingerprint,
        metrics, repeat_seed=repeat_seed,
        total_gpu_seconds=run_result.total_gpu_seconds,
        total_communication_cost=run_result.total_communication_cost,
    )
    finalize_repeat_artifacts(
        repeat_param, repeat_idx, run_result.global_model,
        repeat_param.get("final_artifact_policy", "metrics_only"),
    )
    return RepeatResult(
        repeat_idx, repeat_seed, data_bundle.partition_fingerprint, metrics,
        run_result.total_gpu_seconds, run_result.total_communication_cost,
    )
```

Import `get_repeat_seed`/`set_all_seeds` at module scope. Restore checkpoint RNG only after dataset, loaders, and model have been reconstructed; this cancels constructor RNG consumption and resumes at the exact saved boundary. The unconditional checkpoint immediately before evaluation is what makes a crash after the last training round resume directly into evaluation.

- [ ] **Step 5: Extract one generic evaluator hook**

Move the task-specific ACC/DEO/SPD/FR/HM branches out of both old serial and parallel blocks into:

```python
def _evaluate_global_model(global_model, param_dict, data_bundle, algorithm_state):
    del algorithm_state
    loader = data_bundle.testing_dataloader
    size = len(loader.dataset)
    if "SENT_CLF" in param_dict["task"]:
        accuracy, deo, spd = FL_fairness_and_accuracy_test(
            global_model, param_dict, loader, size
        )
    elif "IMG_CLF" in param_dict["task"]:
        accuracy, deo, spd = FL_fairness_and_accuracy_test_4_IMG_CLF(
            global_model, param_dict, loader, size
        )
    elif "Tabular_CLF" in param_dict["task"]:
        accuracy, deo, spd = FL_fairness_and_accuracy_test_4_Tabular_CLF(
            global_model, param_dict, loader, size
        )
    else:
        raise ValueError(f"unsupported evaluation task: {param_dict['task']}")
    metrics = {"ACC": float(accuracy), "DEO": float(deo), "SPD": float(spd)}
    if any(name in param_dict["task"] for name in ("SENT_CLF", "IMG_CLF", "Tabular_CLF")):
        metrics["FR"] = 1.0 - metrics["DEO"]
        metrics["HM"] = float(get_HM_by_two_value(metrics["ACC"], metrics["FR"]))
    return metrics
```

Later algorithm PRs use the same hook signature: `evaluate_praffl(global_model, param_dict, data_bundle, algorithm_state) -> dict` and `evaluate_fedfact(global_model, param_dict, data_bundle, algorithm_state) -> dict`. This infrastructure PR does not import either evaluator.

- [ ] **Step 6: Replace duplicate scheduling with one worker mapping**

Replace `Experiment_FL` with the two-argument public API plus optional evaluator:

```python
def Experiment_FL(algorithm_function, param_dict, evaluator_function=None):
    repeats = int(param_dict.get("exp_repeat_times", 3))
    parallel = max(1, min(int(param_dict.get("parallel_repeats", 1)), repeats))
    if str(param_dict.get("device", "cpu")).startswith("cuda") and parallel != 1:
        raise ValueError("CUDA repeats must run serially; set parallel_repeats=1")
    config_hash = build_experiment_config_hash(param_dict)
    run_param = dict(param_dict, experiment_config_hash=config_hash)
    args = [
        (index, algorithm_function, evaluator_function, run_param)
        for index in range(repeats)
    ]
    if parallel > 1:
        context = mp.get_context("spawn")
        with context.Pool(processes=parallel) as pool:
            results = pool.starmap(_run_single_repeat, args)
    else:
        results = [_run_single_repeat(*arguments) for arguments in args]
    aggregate = aggregate_repeat_results(results, expected_repeats=repeats)
    save_aggregate_metrics(run_param, aggregate)
    _append_human_readable_aggregate(run_param, algorithm_function.__name__, aggregate)
    return aggregate
```

Both paths invoke `_run_single_repeat`; there is no second serial implementation. `_append_human_readable_aggregate` formats values already present in the aggregate JSON and opens `result_path` only after `save_aggregate_metrics` succeeds.

- [ ] **Step 7: Run repeat scheduling tests**

Run:

```bash
python -m unittest tests.test_repeat_runner.RepeatRunnerTest -v
```

Expected: seed ordering, opt-in fresh execution, evaluation-only recovery, CUDA serial enforcement, and the shared scheduling path all pass.

- [ ] **Step 8: Commit the common repeat runner**

Run:

```bash
git add experiment.py tests/test_repeat_runner.py
git commit -m "feat: unify deterministic repeat execution"
```

Expected: one commit containing orchestration only; algorithm math remains untouched.

### Task 9: Adapt FedAvg minimally and prove exact two-round resume

**Files:**
- Modify: `tests/test_repeat_runner.py`
- Modify: `algorithm/FederatedAverage.py:114-321`

- [ ] **Step 1: Add a real FedAvg continuous-versus-resumed regression**

Append a module-level tiny model and dataset, then a test class, to `tests/test_repeat_runner.py`:

```python
from algorithm.FederatedAverage import Fed_AVG
from module.experiment_setup import FederatedDataBundle
from tool.checkpoint import get_repeat_state_dir


class TinyANN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, values):
        logits = self.linear(values.float())
        return torch.sigmoid(logits), values.float()


class TinyFederatedDataset(torch.utils.data.Dataset):
    def __init__(self, prefix, size):
        self.sample_ids = [f"{prefix}-{index}" for index in range(size)]
        self.labels = np.tile([0, 1], size // 2)
        self.protected = np.tile([0, 1, 1, 0], size // 4)
        self.X = torch.arange(size, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "X": self.X[index],
            "labels": torch.tensor(self.labels[index]),
            "protected": torch.tensor(self.protected[index]),
        }


class FedAvgResumeTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def params(self, name, resume):
        model_path = self.base / name / "models"
        for client_id in range(2):
            (model_path / f"client_{client_id + 1}").mkdir(parents=True, exist_ok=True)
        return {
            "model_path": str(model_path),
            "result_path": str(self.base / name / "result.txt"),
            "log_path": str(self.base / name / "run.log"),
            "partition_cache_root": str(self.base / "shared_partitions"),
            "dataset_name": "toy", "dataset": "toy", "task": "Tabular_CLF",
            "algorithm": "FedAvg", "hypothesis": "TinyANN", "model_type": "ANN",
            "split_strategy": "Dirichlet1", "num_clients_K": 2,
            "batch_size": 4, "test_batch_size": 4, "communication_round_I": 2,
            "algorithm_epoch_T": 1, "FL_fraction": 1.0, "FL_drop_rate": 0.0,
            "learning_rate": 0.05, "optimize_method": "sgd", "device": "cpu",
            "use_amp": False, "base_seed": 42, "exp_repeat_times": 1,
            "parallel_repeats": 1, "resume": resume, "partition_min_size": 1,
            "partition_max_retries": 2, "partition_repair_policy": "minimum_move_v1",
            "dataloader_num_workers": 0, "client_parallel": False,
            "checkpoint_save_freq": 1, "final_artifact_policy": "full_state",
            "tb_monitor": {"test": False, "gradient": False},
        }

    @staticmethod
    def dataset_factory(params):
        del params
        train = TinyFederatedDataset("train", 16)
        test = TinyFederatedDataset("test", 8)
        return train, None, test

    @staticmethod
    def evaluator(model, params, data_bundle, algorithm_state):
        del params, data_bundle, algorithm_state
        state = model.state_dict()
        return {
            "ACC": float(state["linear.weight"].sum() + state["linear.bias"].sum()),
            "model_values": {
                key: value.detach().cpu().reshape(-1).tolist()
                for key, value in state.items()
            },
        }

    def run_once(self, params):
        from experiment import _run_single_repeat
        with mock.patch("experiment.Experiment_Create_dataset", side_effect=self.dataset_factory), \
             mock.patch("experiment.Experiment_Create_model", side_effect=lambda unused: TinyANN()), \
             mock.patch("experiment.calculate_communication_cost", return_value=0.0), \
             mock.patch("algorithm.FederatedAverage.log_deep_metrics"), \
             mock.patch("algorithm.FederatedAverage.log_system_metrics"), \
             mock.patch("algorithm.FederatedAverage.log_test_metrics"), \
             mock.patch("algorithm.FederatedAverage.flush"):
            params = dict(params, experiment_config_hash=build_experiment_config_hash(params))
            return _run_single_repeat(0, Fed_AVG, self.evaluator, params)

    def test_two_round_continuous_matches_one_round_plus_resume(self):
        continuous_params = self.params("continuous", resume=False)
        continuous = self.run_once(continuous_params)

        resumed_params = self.params("resumed", resume=True)
        real_save = save_checkpoint

        def stop_after_first_checkpoint(*args, **kwargs):
            path = real_save(*args, **kwargs)
            iter_t = kwargs["iter_t"] if "iter_t" in kwargs else args[1]
            if int(iter_t) == 0:
                raise RuntimeError("planned round-boundary crash")
            return path

        with mock.patch("algorithm.FederatedAverage.save_checkpoint",
                        side_effect=stop_after_first_checkpoint):
            with self.assertRaisesRegex(RuntimeError, "planned round-boundary crash"):
                self.run_once(resumed_params)
        resumed = self.run_once(resumed_params)

        self.assertEqual(continuous.metrics, resumed.metrics)
        for key in continuous.metrics["model_values"]:
            np.testing.assert_array_equal(
                continuous.metrics["model_values"][key], resumed.metrics["model_values"][key]
            )
        states = []
        for params in (continuous_params, resumed_params):
            state_params = dict(
                params, repeat_idx=0, repeat_seed=42,
                experiment_config_hash=build_experiment_config_hash(params),
                partition_fingerprint=continuous.partition_fingerprint,
            )
            checkpoint = load_checkpoint(state_params)
            self.assertEqual(checkpoint.next_round, 2)
            states.append(checkpoint)
        continuous_state, resumed_state = states
        self.assertEqual(continuous_state.client_selection_history,
                         resumed_state.client_selection_history)
        self.assertEqual(continuous_state.algorithm_state, resumed_state.algorithm_state)
        for key in continuous_state.global_model_state:
            torch.testing.assert_close(
                continuous_state.global_model_state[key],
                resumed_state.global_model_state[key],
                rtol=0, atol=0,
            )
```

The final assertions compare every tensor, algorithm state, selection history, and metric. Do not compare wall/GPU seconds because they are observations rather than deterministic training state.

- [ ] **Step 2: Run the test and verify resume-state loss**

Run:

```bash
python -m unittest tests.test_repeat_runner.FedAvgResumeTest.test_two_round_continuous_matches_one_round_plus_resume -v
```

Expected: failure shows that FedAvg neither accepts/restores `resume_state` nor preserves full selection/scaler/counter state in `AlgorithmRunResult`.

- [ ] **Step 3: Add the non-mathematical FedAvg resume adapter**

Change the signature to add `resume_state=None` after `start_round=0`, import `AlgorithmRunResult`, and initialize state as follows:

```python
from tool.experiment_state import AlgorithmRunResult


def Fed_AVG(device, global_model, algorithm_epoch_T, num_clients_K,
            communication_round_I, FL_fraction, FL_drop_rate,
            training_dataloaders, training_dataset, client_dataset_list,
            param_dict, testing_dataloader, testing_dataset_len,
            start_round=0, resume_state=None):
    # existing setup remains above/below these state initializers
    use_amp = param_dict.get("use_amp", False)
    scaler = get_scaler(device, use_amp)
    if resume_state is not None and resume_state.amp_scaler_state is not None:
        if scaler is None:
            raise RuntimeError("checkpoint contains AMP scaler state but AMP is disabled")
        scaler.load_state_dict(resume_state.amp_scaler_state)
    total_gpu_seconds = 0.0 if resume_state is None else resume_state.total_gpu_seconds
    total_communication_cost = (
        0.0 if resume_state is None else resume_state.total_communication_cost
    )
    client_selection_history = (
        [] if resume_state is None
        else [list(values) for values in resume_state.client_selection_history]
    )
    runtime_offset = 0.0 if resume_state is None else resume_state.total_runtime_seconds
    runtime_start = time.monotonic()
```

After selecting clients each round, append exactly once:

```python
selected_clients = idxs_users.tolist() if hasattr(idxs_users, "tolist") else list(idxs_users)
client_selection_history.append([int(client_id) for client_id in selected_clients])
```

Accumulate the communication counter using the existing per-round formula. At each current checkpoint call, replace the one-round history payload with:

```python
save_checkpoint(
    param_dict=param_dict,
    iter_t=iter_t,
    global_model=global_model,
    algorithm_state={},
    amp_scaler=scaler,
    total_gpu_seconds=total_gpu_seconds,
    total_runtime_seconds=runtime_offset + (time.monotonic() - runtime_start),
    total_communication_cost=total_communication_cost,
    client_selection_history=client_selection_history,
)
```

Return:

```python
return AlgorithmRunResult(
    global_model=global_model,
    total_gpu_seconds=total_gpu_seconds,
    total_communication_cost=total_communication_cost,
    algorithm_state={},
    amp_scaler_state=None if scaler is None else scaler.state_dict(),
    client_selection_history=client_selection_history,
)
```

Remove the fixed `./save_path/global_FedAvg.pt` write; `finalize_repeat_artifacts` now applies the configured, repeat-specific final artifact policy atomically. Keep local loss, optimizer, client sampling, and weighted aggregation statements byte-for-byte unless a variable name must be threaded into the state adapter.

- [ ] **Step 4: Run the exact-resume test**

Run:

```bash
python -m unittest tests.test_repeat_runner.FedAvgResumeTest.test_two_round_continuous_matches_one_round_plus_resume -v
```

Expected: one test passes; uninterrupted and resumed runs have bit-identical model tensors, selection histories, algorithm state, and metrics.

- [ ] **Step 5: Run all repeat/checkpoint tests**

Run:

```bash
python -m unittest tests.test_checkpoint tests.test_repeat_runner -v
```

Expected: all tests pass on CPU; only the CUDA RNG test is skipped when CUDA is unavailable.

- [ ] **Step 6: Commit the reference adapter**

Run:

```bash
git add algorithm/FederatedAverage.py tests/test_repeat_runner.py
git commit -m "feat: resume FedAvg at exact round boundaries"
```

Expected: one commit with a state-only FedAvg adapter and its equivalence test.

### Task 10: Integrate lazy repeat construction into the experiment dispatcher

**Files:**
- Modify: `experiment.py:689-1138`
- Modify: `tests/test_repeat_runner.py`

- [ ] **Step 1: Add a failing dispatcher regression**

Append to `RepeatRunnerTest`:

```python
    def test_fl_dispatch_does_not_construct_data_or_model_before_repeat_worker(self):
        from experiment import Experiment
        params = dict(self.params, algorithm="FedAvg")
        with mock.patch("experiment.Experiment_Create_dataset") as dataset_factory, \
             mock.patch("experiment.Experiment_Create_dataloader") as loader_factory, \
             mock.patch("experiment.Experiment_Create_model") as model_factory, \
             mock.patch("experiment.Experiment_FL", return_value={}) as repeat_runner, \
             mock.patch("experiment.init_tensorboard_logger", return_value=None):
            Experiment(params)
        dataset_factory.assert_not_called()
        loader_factory.assert_not_called()
        model_factory.assert_not_called()
        repeat_runner.assert_called_once()
        self.assertIs(repeat_runner.call_args.args[0], Fed_AVG)
        self.assertIs(repeat_runner.call_args.args[1], params)
```

- [ ] **Step 2: Run the test and verify eager construction**

Run:

```bash
python -m unittest tests.test_repeat_runner.RepeatRunnerTest.test_fl_dispatch_does_not_construct_data_or_model_before_repeat_worker -v
```

Expected: failure shows `Experiment_Create_dataset`, `Experiment_Create_dataloader`, and `Experiment_Create_model` are called before `Experiment_FL`.

- [ ] **Step 3: Remove eager construction from FL dispatch**

In `Experiment`, keep AMP resolution and logger initialization, but remove lines 729-772 from the FL path. Move their existing behavior into a focused helper used only by Separate/Centralized and the ablation entry point:

```python
def _create_legacy_single_run_inputs(param_dict):
    training_dataset, validation_dataset, testing_dataset = Experiment_Create_dataset(param_dict)
    data_bundle = Experiment_Create_dataloader(
        dict(param_dict, repeat_idx=0), training_dataset, validation_dataset,
        testing_dataset, param_dict["split_strategy"],
    )
    global_model = Experiment_Create_model(param_dict)
    return training_dataset, testing_dataset, data_bundle, global_model
```

For each federated algorithm branch, replace the old call carrying seven prebuilt data/model arguments with the exact public form:

```python
Experiment_FL(ALGORITHM_FUNCTION, param_dict)
```

Here `ALGORITHM_FUNCTION` is the function already named in that branch (`Fed_AVG`, `Fed_Prox`, `Scaffold`, and so on); do not alter branch ordering or substring matching. Later PraFFL and FedFACT plans change only their own branch to:

```python
Experiment_FL(PraFFL, param_dict, evaluator_function=evaluate_praffl)
Experiment_FL(FedFACT, param_dict, evaluator_function=evaluate_fedfact)
```

Use the legacy helper's `data_bundle.training_dataloaders`, `data_bundle.client_dataset_list`, and `data_bundle.testing_dataloader` fields when calling Separate/Centralized. Update `PDFFed_Ablation_Experiment` to unpack the bundle fields rather than a three-tuple, but keep it on its existing single-run behavior because repeat semantics for ablations are outside this PR.

- [ ] **Step 4: Prove every FL branch uses the new public signature**

Run:

```bash
python -m unittest tests.test_repeat_runner.RepeatRunnerTest.test_fl_dispatch_does_not_construct_data_or_model_before_repeat_worker -v
python - <<'PY'
from pathlib import Path
text = Path("experiment.py").read_text(encoding="utf-8")
for line_number, line in enumerate(text.splitlines(), 1):
    if "Experiment_FL(" in line and "def Experiment_FL" not in line:
        assert "training_dataloaders" not in line, (line_number, line)
print("all FL branches defer construction to the repeat worker")
PY
```

Expected: the unit test passes and the static guard prints `all FL branches defer construction to the repeat worker`.

- [ ] **Step 5: Commit dispatcher integration**

Run:

```bash
git add experiment.py tests/test_repeat_runner.py
git commit -m "refactor: defer FL construction to repeat workers"
```

Expected: one commit changing dispatch/wiring only.

### Task 11: Centralize CLI state controls and remove log-text resume decisions

**Files:**
- Create: `tool/experiment_cli.py`
- Create: `tests/test_experiment_state_cli.py`
- Modify: `main_SENT_CLF.py:114-195,263-355`
- Modify: `main_IMG_CLF.py:87-167,235-319`
- Modify: `main_Tabular_CLF.py:87-190,266-360`

- [ ] **Step 1: Write failing common-argument tests**

Create `tests/test_experiment_state_cli.py`:

```python
import argparse
import unittest

from tool.experiment_cli import add_experiment_state_arguments


class ExperimentStateArgumentsTest(unittest.TestCase):
    def parse(self, values):
        parser = argparse.ArgumentParser()
        add_experiment_state_arguments(parser)
        return vars(parser.parse_args(values))

    def test_scientific_defaults(self):
        params = self.parse([])
        self.assertFalse(params["resume"])
        self.assertEqual(params["exp_repeat_times"], 3)
        self.assertEqual(params["parallel_repeats"], 1)
        self.assertEqual(params["use_amp"], "auto")
        self.assertEqual(params["base_seed"], 42)
        self.assertEqual(params["partition_min_size"], 1)
        self.assertEqual(params["partition_max_retries"], 100)
        self.assertEqual(params["partition_repair_policy"], "minimum_move_v1")
        self.assertEqual(params["partition_cache_root"], "./partition_cache")
        self.assertEqual(params["final_artifact_policy"], "metrics_only")
        self.assertEqual(params["checkpoint_keep_latest"], 1)

    def test_resume_is_explicit_and_values_are_configurable(self):
        params = self.parse([
            "-resume", "-base_seed", "9", "-exp_repeat_times", "2",
            "-partition_min_size", "3", "-partition_max_retries", "7",
            "-final_artifact_policy", "full_state", "-use_amp", "false",
        ])
        self.assertTrue(params["resume"])
        self.assertEqual(params["base_seed"], 9)
        self.assertEqual(params["exp_repeat_times"], 2)
        self.assertEqual(params["partition_min_size"], 3)
        self.assertEqual(params["partition_max_retries"], 7)
        self.assertEqual(params["final_artifact_policy"], "full_state")
        self.assertEqual(params["use_amp"], "false")
```

- [ ] **Step 2: Run the tests and verify the missing helper**

Run:

```bash
python -m unittest tests.test_experiment_state_cli -v
```

Expected: import fails with `ModuleNotFoundError: No module named 'tool.experiment_cli'`.

- [ ] **Step 3: Implement the shared argument group**

Create `tool/experiment_cli.py`:

```python
def add_experiment_state_arguments(parser):
    parser.add_argument("-resume", action="store_true",
                        help="Resume only compatible schema-v2 repeat artifacts")
    parser.add_argument("-exp_repeat_times", type=int, default=3)
    parser.add_argument("-parallel_repeats", type=int, default=1)
    parser.add_argument("-base_seed", type=int, default=42)
    parser.add_argument("-use_amp", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("-partition_cache_root", default="./partition_cache")
    parser.add_argument("-partition_min_size", type=int, default=1)
    parser.add_argument("-partition_max_retries", type=int, default=100)
    parser.add_argument("-partition_repair_policy", default="minimum_move_v1",
                        choices=["minimum_move_v1"])
    parser.add_argument("-dataloader_num_workers", type=int, default=None)
    parser.add_argument("-checkpoint_save_freq", type=int, default=1)
    parser.add_argument("-checkpoint_keep_latest", type=int, default=1,
                        help="Compatibility flag; schema v2 always retains one active checkpoint")
    parser.add_argument("-final_artifact_policy", default="metrics_only",
                        choices=["metrics_only", "global_model", "full_state"])
    return parser
```

Call `add_experiment_state_arguments(parser)` once in each of the SENT/IMG/Tabular `Argparse` functions and remove their duplicate definitions of `resume`, `exp_repeat_times`, `parallel_repeats`, checkpoint frequency, and retention.

- [ ] **Step 4: Delete log parsing from the resume control path**

In each main matrix loop, remove the `resume_mode`/`analyze_experiment_log`/`calculate_and_append_summary` decision block. Preserve `start_exp`, but always call `Experiment(param_dict)` for a selected matrix cell. The repeat runner uses schema-v2 `metrics.json` and `checkpoint_latest.pt` only when `param_dict["resume"]` is true; without the flag it clears the matching repeat artifacts and runs fresh.

Keep the old log-analysis helpers temporarily callable for manual log inspection, but add a docstring stating they do not decide training completion. Do not infer completion from test-count strings.

- [ ] **Step 5: Run CLI and source guards**

Run:

```bash
python -m unittest tests.test_experiment_state_cli -v
python - <<'PY'
from pathlib import Path
for name in ("main_SENT_CLF.py", "main_IMG_CLF.py", "main_Tabular_CLF.py"):
    text = Path(name).read_text(encoding="utf-8")
    assert text.count("add_experiment_state_arguments(parser)") == 1, name
    loop = text[text.find("for split_strategy in split_strategy_list:"):]
    assert "analyze_experiment_log(" not in loop, name
print("entry points use structured resume state")
PY
```

Expected: two CLI tests pass and the guard prints `entry points use structured resume state`.

- [ ] **Step 6: Commit the CLI and matrix-loop change**

Run:

```bash
git add tool/experiment_cli.py tests/test_experiment_state_cli.py main_SENT_CLF.py main_IMG_CLF.py main_Tabular_CLF.py
git commit -m "feat: make repeat resume controls explicit"
```

Expected: one commit shared by all three entry points.

### Task 12: Aggregate mixed completed/resumed/fresh repeats and retain resource evidence

**Files:**
- Modify: `tests/test_repeat_runner.py`
- Modify: `tool/experiment_state.py`
- Modify: `tool/checkpoint.py`
- Modify: `experiment.py`

- [ ] **Step 1: Add the mixed-state aggregate regression**

Append to `RepeatRunnerTest`:

```python
    def test_mixed_completed_resumed_and_fresh_repeats_all_enter_aggregate(self):
        from experiment import Experiment_FL
        params = dict(self.params, resume=True)
        config_hash = build_experiment_config_hash(params)
        calls = []

        for repeat_idx in (0, 1):
            state_params = dict(
                params,
                experiment_config_hash=config_hash,
                repeat_idx=repeat_idx,
                repeat_seed=42 + 1000 * repeat_idx,
                partition_fingerprint=f"partition-{repeat_idx}",
            )
            if repeat_idx == 0:
                save_repeat_metrics(
                    state_params, repeat_idx, config_hash, f"partition-{repeat_idx}",
                    {"ACC": 0.1}, repeat_seed=42,
                    total_gpu_seconds=0.0, total_communication_cost=0.0,
                    resource_usage={"peak_cuda_bytes": 0, "peak_rss_bytes": 1,
                                    "checkpoint_bytes": 1},
                )
            else:
                save_checkpoint(
                    state_params, 0, torch.nn.Linear(1, 1), algorithm_state={"step": 1},
                    total_gpu_seconds=0.0, total_runtime_seconds=0.0,
                    total_communication_cost=0.0, client_selection_history=[[0]],
                )

        def algorithm(*args, start_round=0, resume_state=None, **kwargs):
            del kwargs
            repeat_idx = args[10]["repeat_idx"]
            calls.append((repeat_idx, start_round, resume_state is not None))
            return AlgorithmRunResult(
                args[1], 0.0, 0.0,
                {} if resume_state is None else resume_state.algorithm_state,
                None, [] if resume_state is None else resume_state.client_selection_history,
            )

        def bundle_factory(repeat_params, train, validation, test, split):
            del train, validation, test, split
            return self._fake_bundle(f"partition-{repeat_params['repeat_idx']}")

        with mock.patch("experiment.Experiment_Create_dataset",
                        return_value=(object(), None, object())), \
             mock.patch("experiment.Experiment_Create_dataloader",
                        side_effect=bundle_factory), \
             mock.patch("experiment.Experiment_Create_model",
                        side_effect=lambda unused: torch.nn.Linear(1, 1)), \
             mock.patch("experiment.calculate_communication_cost", return_value=0.0):
            aggregate = Experiment_FL(algorithm, params, evaluator_function=self._toy_evaluator)

        self.assertEqual(aggregate["repeat_indices"], [0, 1, 2])
        self.assertEqual(aggregate["repeat_seeds"], [42, 1042, 2042])
        self.assertEqual(calls, [(1, 1, True), (2, 0, False)])
```

The fake bundle helper must set the passed `repeat_params["partition_fingerprint"]` just as the production setup does. Repeat 0 comes from `metrics.json`, repeat 1 resumes round 1, and repeat 2 starts fresh.

- [ ] **Step 2: Add a resource snapshot regression**

Append to `ExperimentStateTest`:

```python
    def test_resource_snapshot_records_peak_cuda_rss_and_checkpoint_bytes(self):
        from tool.experiment_state import capture_resource_snapshot
        with tempfile.NamedTemporaryFile() as stream:
            stream.write(b"checkpoint")
            stream.flush()
            snapshot = capture_resource_snapshot(Path(stream.name))
        self.assertEqual(snapshot["checkpoint_bytes"], 10)
        self.assertGreater(snapshot["peak_rss_bytes"], 0)
        self.assertGreaterEqual(snapshot["peak_cuda_bytes"], 0)
```

- [ ] **Step 3: Run the tests and verify missing resource/mixed-state support**

Run:

```bash
python -m unittest \
  tests.test_repeat_runner.ExperimentStateTest.test_resource_snapshot_records_peak_cuda_rss_and_checkpoint_bytes \
  tests.test_repeat_runner.RepeatRunnerTest.test_mixed_completed_resumed_and_fresh_repeats_all_enter_aggregate -v
```

Expected: the resource import fails first; after adding it, the mixed-state test exposes any repeat that is skipped without contributing metrics.

- [ ] **Step 4: Capture resource evidence before final artifact cleanup**

Add to `tool/experiment_state.py`:

```python
from pathlib import Path
import resource
import sys
import torch


def capture_resource_snapshot(checkpoint_path: Path) -> dict[str, int]:
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        peak_rss_bytes = peak_rss
    else:
        peak_rss_bytes = peak_rss * 1024
    return {
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available() else 0,
        "peak_rss_bytes": peak_rss_bytes,
        "checkpoint_bytes": int(checkpoint_path.stat().st_size),
    }
```

At the beginning of `_run_single_repeat`, after seeding and before dataset creation, call `torch.cuda.reset_peak_memory_stats()` when CUDA is available. Capture the path returned by the pre-evaluation `save_checkpoint`, call `capture_resource_snapshot(checkpoint_path)`, and pass the result into `save_repeat_metrics`. Extend that function's keyword-only signature with `resource_usage=None`, normalize a missing value to three zero counters for compatibility with the earlier focused tests, store it under the same key in `metrics.json`, and make `load_repeat_metrics` require three non-negative integer fields. Add `resource_usage: dict[str, int]` to `RepeatResult` with a default empty dictionary, return the stored value when a completed repeat is loaded, and include per-repeat resource rows in `aggregate_repeat_results`.

- [ ] **Step 5: Run the mixed-state and full runner suites**

Run:

```bash
python -m unittest tests.test_repeat_runner tests.test_checkpoint -v
```

Expected: all tests pass, the mixed run contains indices 0/1/2, and completed metrics retain peak CUDA allocation, peak host RSS, and checkpoint bytes even under `metrics_only` cleanup.

- [ ] **Step 6: Commit mixed-state aggregation and resource evidence**

Run:

```bash
git add tool/experiment_state.py tool/checkpoint.py experiment.py tests/test_repeat_runner.py
git commit -m "feat: aggregate mixed repeat states with resource evidence"
```

Expected: one commit containing the last repeat-runner acceptance behavior.

### Task 13: Document the new scientific contract

**Files:**
- Modify: `README.md:449-528`
- Modify: `README_CN.md:447-523`

- [ ] **Step 1: Replace the split and resume documentation in both languages**

Document all of these exact facts in English and Chinese:

1. `Dirichlet01`, `Dirichlet05`, and `Dirichlet1` are schema-v2 label-conditioned partitions, not quantity-skew partitions.
2. Each repeat uses `base_seed + 1000 * repeat_idx` for training and its paired partition; algorithms share the same partition cache key for the same dataset/spec/repeat.
3. Train and client-test indices share the per-class sampled profile; global testing still covers the full test dataset.
4. Retry is finite and `minimum_move_v1` repair is explicitly recorded as `label_dirichlet_repaired_v2` with move count and label/protected/joint statistics.
5. Old `split_indices.json` is never consumed by the new names; only an explicit `LegacyQuantityDirichlet*` alias reads it.
6. Resume occurs only with `-resume`, validates config/partition/repeat identity, and treats `metrics.json` rather than a last-round checkpoint or log line as completion.
7. `-final_artifact_policy` values are `metrics_only`, `global_model`, and `full_state`; `metrics_only` is the default and retains reproducibility metadata/resource evidence without retaining large personal model states.
8. CUDA repeats require `-parallel_repeats 1`; CPU repeats may use the multiprocessing path.
9. AMP smoke selection uses `-use_amp false` and `-use_amp true`.
10. Historical quantity-skew measurements are not comparable to schema-v2 label-skew measurements and must be reported separately.

- [ ] **Step 2: Add one exact CLI example**

Add this command to both READMEs (translate only the surrounding prose):

```bash
python main_SENT_CLF.py \
  -algorithm FedAvg \
  -dataset moji \
  -split_strategy Dirichlet05 \
  -num_clients_K 2 \
  -communication_round_I 2 \
  -algorithm_epoch_T 1 \
  -exp_repeat_times 3 \
  -parallel_repeats 1 \
  -base_seed 42 \
  -partition_min_size 1 \
  -partition_max_retries 100 \
  -use_amp true \
  -resume
```

- [ ] **Step 3: Verify documentation names against code**

Run:

```bash
python - <<'PY'
from pathlib import Path
required = [
    "Dirichlet01", "label-conditioned", "base_seed + 1000", "-resume",
    "metrics.json", "minimum_move_v1", "LegacyQuantityDirichlet",
    "final_artifact_policy", "use_amp",
]
english = Path("README.md").read_text(encoding="utf-8")
chinese = Path("README_CN.md").read_text(encoding="utf-8")
for token in required:
    assert token in english, ("README.md", token)
for token in ("Dirichlet01", "base_seed + 1000", "-resume", "metrics.json",
              "minimum_move_v1", "LegacyQuantityDirichlet",
              "final_artifact_policy", "use_amp"):
    assert token in chinese, ("README_CN.md", token)
print("repeat/resume/partition documentation is synchronized")
PY
```

Expected: the guard prints `repeat/resume/partition documentation is synchronized`.

- [ ] **Step 4: Commit documentation**

Run:

```bash
git add README.md README_CN.md
git commit -m "docs: explain deterministic partitions and resume"
```

Expected: one documentation-only commit.

### Task 14: Run the full local acceptance suite

**Files:**
- Verify: `module/partition.py`
- Verify: `module/dataloader.py`
- Verify: `module/experiment_setup.py`
- Verify: `tool/seed_manager.py`
- Verify: `tool/experiment_state.py`
- Verify: `tool/checkpoint.py`
- Verify: `experiment.py`
- Verify: `algorithm/FederatedAverage.py`
- Verify: `tests/`

- [ ] **Step 1: Run every unit test**

Run:

```bash
python -m unittest discover -s tests -v
```

Expected: every CPU test passes; CUDA-specific AMP/RNG tests are skipped only on a CPU-only host.

- [ ] **Step 2: Run the former hang regression under a hard deadline**

Run:

```bash
timeout 20s python -m unittest \
  tests.test_partition.LabelDirichletTest.test_alpha_point_one_forty_clients_finishes_with_valid_coverage -v
```

Expected: one test passes and exits 0 before 20 seconds.

- [ ] **Step 3: Compile all touched Python packages and entry points**

Run:

```bash
python -m compileall -q algorithm module tool tests experiment.py \
  main_SENT_CLF.py main_IMG_CLF.py main_Tabular_CLF.py
```

Expected: exit code 0 with no output.

- [ ] **Step 4: Run static scope and stale-API guards**

Run:

```bash
python - <<'PY'
from pathlib import Path

dataloader = Path("module/dataloader.py").read_text(encoding="utf-8")
assert "while min_size" not in dataloader
assert "np.random.dirichlet" not in dataloader
assert "save_split_indices(param_dict" not in dataloader

experiment = Path("experiment.py").read_text(encoding="utf-8")
assert experiment.count("def _run_single_repeat(") == 1
assert "checkpoint['communication_round']" not in experiment
assert "current_round >= total_rounds - 1" not in experiment

checkpoint = Path("tool/checkpoint.py").read_text(encoding="utf-8")
for field in ("experiment_config_hash", "partition_fingerprint", "repeat_idx",
              "repeat_seed", "next_round", "phase", "algorithm_state",
              "amp_scaler_state", "total_runtime_seconds"):
    assert field in checkpoint, field

print("static infrastructure guards pass")
PY

BASE=$(git merge-base HEAD origin/main)
git diff --name-only "$BASE"...HEAD > /tmp/repeats-resume-dirichlet-files.txt
! grep -E '^algorithm/(PraFFL|FedFACT)\.py$' /tmp/repeats-resume-dirichlet-files.txt
```

Expected: the script prints `static infrastructure guards pass`, and `grep` finds neither `algorithm/PraFFL.py` nor `algorithm/FedFACT.py` in this PR.

- [ ] **Step 5: Review commit scope**

Run:

```bash
git status --short
git log --oneline --decorate -12
```

Expected: the worktree is clean and commits are separated by partition, loader, checkpoint, runner, FedAvg adapter, CLI, resource evidence, and documentation responsibilities.

### Task 15: Run Ronnie BERT AMP-off and AMP-on smoke gates

**Files:**
- Verify: `result_path/moji/Dirichlet05/FedAvg/BERTCLASSIFIER/2Clients/1.txt.json`
- Verify: `save_path/moji/Dirichlet05/FedAvg/BERTCLASSIFIER/2Clients/experiment_state/`
- Verify: `partition_cache/v2/moji/`

- [ ] **Step 1: Run the FP32 smoke**

Run on Ronnie's single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 /home/ronnie/anaconda3/envs/FL/bin/python main_SENT_CLF.py \
  -algorithm FedAvg -dataset moji -split_strategy Dirichlet05 \
  -system_data_count 128 -num_clients_K 2 -communication_round_I 2 \
  -algorithm_epoch_T 1 -exp_repeat_times 1 -parallel_repeats 1 \
  -base_seed 42 -partition_min_size 1 -partition_max_retries 10 \
  -use_amp false -final_artifact_policy metrics_only
```

Expected: exit code 0; one repeat metric is atomically written; metadata records repeat seed 42, two-round completion, peak host RSS, checkpoint bytes, and `peak_cuda_bytes`.

- [ ] **Step 2: Run the AMP smoke against the same paired partition**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 /home/ronnie/anaconda3/envs/FL/bin/python main_SENT_CLF.py \
  -algorithm FedAvg -dataset moji -split_strategy Dirichlet05 \
  -system_data_count 128 -num_clients_K 2 -communication_round_I 2 \
  -algorithm_epoch_T 1 -exp_repeat_times 1 -parallel_repeats 1 \
  -base_seed 42 -partition_min_size 1 -partition_max_retries 10 \
  -use_amp true -final_artifact_policy metrics_only
```

Expected: exit code 0 with AMP enabled; it reuses the same partition fingerprint as FP32, while its experiment-config hash and repeat metrics remain separate because `use_amp` differs.

- [ ] **Step 3: Verify resource evidence, partition pairing, completion, and cleanup**

Run:

```bash
/home/ronnie/anaconda3/envs/FL/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("save_path/moji/Dirichlet05/FedAvg/BERTCLASSIFIER/2Clients/experiment_state")
metrics = []
for path in root.glob("*/repeat_00/metrics.json"):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["experiment_config"].get("system_data_count") == 128:
        metrics.append(payload)
assert len(metrics) == 2, len(metrics)
assert {row["experiment_config"]["use_amp"] for row in metrics} == {False, True}
assert len({row["partition_fingerprint"] for row in metrics}) == 1
for row in metrics:
    usage = row["resource_usage"]
    assert usage["peak_rss_bytes"] > 0
    assert usage["peak_cuda_bytes"] > 0
    assert usage["checkpoint_bytes"] > 0
    state_dir = root / row["experiment_config_hash"] / "repeat_00"
    assert not (state_dir / "checkpoint_latest.pt").exists()
print("Ronnie FP32/AMP smoke gates pass")
PY
```

Expected: the script prints `Ronnie FP32/AMP smoke gates pass`; both modes share one partition, and default completed-state cleanup leaves metrics/config/resource evidence without a resumable BERT checkpoint.

- [ ] **Step 4: Hand off the infrastructure interfaces to the two algorithm PRs**

Record these exact contracts in both downstream plans: preserve every existing training positional parameter, append `start_round=0, resume_state: CheckpointState | None = None`, and return `AlgorithmRunResult`. The evaluator signature is `evaluate_algorithm(global_model, param_dict, data_bundle: FederatedDataBundle, algorithm_state: dict) -> dict`. Dispatcher wiring is `Experiment_FL(Algorithm, param_dict, evaluator_function=evaluate_algorithm)`.

`CheckpointState` exposes `algorithm_state`, `amp_scaler_state`, `total_gpu_seconds`, `total_runtime_seconds`, `total_communication_cost`, and `client_selection_history`. `save_checkpoint` derives `phase="evaluate"` automatically when `iter_t + 1 >= communication_round_I`; downstream algorithms never store phase inside their private state.

## Final acceptance checklist

- [ ] Each of three repeats uses seeds `base_seed`, `base_seed + 1000`, and `base_seed + 2000`, with seeding before dataset/loader/model construction.
- [ ] Serial and CPU-parallel modes invoke the same `_run_single_repeat` function; CUDA rejects parallel repeats greater than one.
- [ ] Resume is ignored unless `-resume` is present, and any config/partition/repeat/CUDA-state mismatch raises before partial restoration.
- [ ] A last-round checkpoint has phase `evaluate`; only atomic `metrics.json` marks completion and previously completed metrics re-enter the aggregate.
- [ ] The two-round continuous and round-one-plus-resume FedAvg toy runs match model state, algorithm state, selected clients, and metrics exactly.
- [ ] `Dirichlet01`/`05`/`1` draw once per class using local PCG64, use the same profile for train/client-test allocation, validate every index, and terminate after finite retries plus explicit minimum-move repair.
- [ ] Same data/spec/repeat across algorithms produces identical partition fingerprints/indices; changing data order, labels, alpha, seed, or truncation misses the cache.
- [ ] Legacy quantity-skew files are read only through explicit legacy names and never promoted into schema v2.
- [ ] Metrics retain label/protected/joint partition statistics and peak CUDA/RSS/checkpoint-byte evidence.
- [ ] Existing tests, new CPU tests, compile checks, and Ronnie AMP-off/on BERT smoke gates pass before this PR is merged.
