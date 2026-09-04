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
DIRICHLET_ALPHAS = {
    "Dirichlet01": 0.1,
    "Dirichlet05": 0.5,
    "Dirichlet1": 1.0,
}
LEGACY_QUANTITY_ALIASES = {
    "LegacyQuantityDirichlet01": "Dirichlet01",
    "LegacyQuantityDirichlet05": "Dirichlet05",
    "LegacyQuantityDirichlet1": "Dirichlet1",
    "LegacyQuantityDirichlet8": "Dirichlet8",
}


class PartitionDataError(ValueError):
    pass


class PartitionCacheError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetView:
    sample_ids: tuple[str, ...]
    labels: np.ndarray
    protected: np.ndarray


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
        # Default (thesis protocol): every repeat reuses the single base_seed
        # partition, so all methods and repeats share one data split; repeat
        # seeds only perturb per-round client sampling, model initialization and
        # loader order (paired comparison across methods/repeats is then exact,
        # with partition variance removed).
        # Legacy mode (redraw_partition_per_repeat): each repeat draws its own
        # partition (seed = base_seed + 1000 * repeat_idx), so Mean±STD mixes
        # partition + training randomness; kept only to reproduce/continue runs
        # launched before the fixed-partition default.
        base_seed = int(param_dict.get("base_seed", 42))
        if param_dict.get("redraw_partition_per_repeat", False):
            seed = base_seed + 1000 * int(repeat_idx)
        else:
            seed = base_seed
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


@dataclass(frozen=True)
class PartitionArtifact:
    fingerprint: str
    indices_sha256: str
    cache_dir: Path
    spec: PartitionSpec
    train_indices: dict[int, np.ndarray]
    test_indices: dict[int, np.ndarray]
    metadata: dict[str, Any]


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
    return tuple(sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in rows)


def _declared_length(dataset: Any) -> int:
    if hasattr(dataset, "_stacked_cache"):
        input_ids = dataset._stacked_cache.get("input_ids")
        if input_ids is not None:
            return len(input_ids)
    for field in ("sample_ids", "texts", "img_names", "labels", "targets", "y", "protected", "s1"):
        values = getattr(dataset, field, None)
        if values is not None:
            return len(values)
    for field in ("X", "input_ids"):
        values = getattr(dataset, field, None)
        if values is not None:
            return len(values)
    return len(dataset)


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

    declared_size = _declared_length(dataset)
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
        [
            json.dumps(
                value.item() if hasattr(value, "item") else value,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            for value in values
        ]
    )


def dataset_fingerprint(
    dataset: Any,
    *,
    dataset_name: str,
    split: str,
    system_data_count: int | None,
) -> dict[str, Any]:
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


def validate_indices(
    indices: Mapping[int, np.ndarray],
    *,
    dataset_size: int,
    num_clients: int,
    min_size: int = 0,
) -> None:
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
        raise PartitionCacheError(f"partition covers {len(flat)} entries, expected {dataset_size}")
    if len(np.unique(flat)) != dataset_size:
        raise PartitionCacheError("partition contains duplicate or missing indices")


def _split_class_indices(class_indices: np.ndarray, profile: np.ndarray) -> list[np.ndarray]:
    boundaries = (np.cumsum(profile)[:-1] * len(class_indices)).astype(np.int64)
    return list(np.split(class_indices, boundaries))


def _largest_remainder_counts(size: int, profile: np.ndarray) -> np.ndarray:
    exact = profile * size
    counts = np.floor(exact).astype(np.int64)
    remainder = size - int(counts.sum())
    order = sorted(
        range(len(profile)),
        key=lambda client: (-(exact[client] - counts[client]), client),
    )
    for client_id in order[:remainder]:
        counts[client_id] += 1
    return counts


def _allocate_with_profile(
    labels: np.ndarray,
    class_values: tuple[Any, ...],
    profile: np.ndarray,
    rng: Any,
) -> dict[int, np.ndarray]:
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
    return {
        client_id: np.asarray(values, dtype=np.int64)
        for client_id, values in enumerate(clients)
    }


def _repair_minimum_move(
    clients: list[list[int]],
    labels: np.ndarray,
    class_values: tuple[Any, ...],
    profile: np.ndarray,
    minimum: int,
) -> tuple[dict[str, Any], ...]:
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
                donor
                for donor, values in enumerate(clients)
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
        moves.append(
            {
                "from_client": donor,
                "to_client": recipient,
                "index": int(index),
                "label": value.item() if hasattr(value, "item") else value,
            }
        )


def build_label_dirichlet_partition(
    train_labels: Any,
    test_labels: Any,
    spec: PartitionSpec,
    rng: Any | None = None,
) -> PartitionResult:
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

    assert best_clients is not None and best_profile is not None and best_deficit is not None
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
    validate_indices(
        train_indices,
        dataset_size=len(train),
        num_clients=spec.num_clients,
        min_size=spec.min_samples_per_client,
    )
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


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def partition_fingerprint(
    spec: PartitionSpec,
    train_fingerprint: Mapping[str, Any],
    test_fingerprint: Mapping[str, Any],
) -> str:
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


def _scalar_key(value: Any) -> str:
    raw = value.item() if hasattr(value, "item") else value
    return json.dumps(raw, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _client_stats(indices: Mapping[int, np.ndarray], view: DatasetView) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for client_id, client_indices in indices.items():
        labels = view.labels[client_indices]
        protected = view.protected[client_indices]
        label_counts = {_scalar_key(value): int(np.sum(labels == value)) for value in np.unique(labels)}
        protected_counts = {
            _scalar_key(value): int(np.sum(protected == value)) for value in np.unique(protected)
        }
        joint_counts: dict[str, int] = {}
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


def _indices_bytes(
    train_indices: Mapping[int, np.ndarray],
    test_indices: Mapping[int, np.ndarray],
) -> bytes:
    chunks: list[bytes] = []
    for split, mapping in (("train", train_indices), ("test", test_indices)):
        for client_id in sorted(mapping):
            array = np.asarray(mapping[client_id], dtype="<i8")
            chunks.extend(
                [
                    split.encode("ascii"),
                    client_id.to_bytes(8, "big"),
                    len(array).to_bytes(8, "big"),
                    array.tobytes(),
                ]
            )
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


def _atomic_write_ready(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write("")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _uniform_indices(size: int, num_clients: int, seed: int) -> dict[int, np.ndarray]:
    generator = np.random.Generator(np.random.PCG64(seed))
    order = generator.permutation(size).astype(np.int64)
    base, remainder = divmod(size, num_clients)
    lengths = [base + (1 if client_id < remainder else 0) for client_id in range(num_clients)]
    result: dict[int, np.ndarray] = {}
    offset = 0
    for client_id, length in enumerate(lengths):
        result[client_id] = order[offset:offset + length]
        offset += length
    return result


def _build_partition_result(train_view: DatasetView, test_view: DatasetView, spec: PartitionSpec) -> PartitionResult:
    if spec.strategy == "Uniform":
        train_indices = _uniform_indices(len(train_view.labels), spec.num_clients, spec.seed)
        test_indices = _uniform_indices(len(test_view.labels), spec.num_clients, spec.seed ^ 0x9E3779B97F4A7C15)
        validate_indices(
            train_indices,
            dataset_size=len(train_view.labels),
            num_clients=spec.num_clients,
            min_size=spec.min_samples_per_client,
        )
        validate_indices(test_indices, dataset_size=len(test_view.labels), num_clients=spec.num_clients)
        return PartitionResult(
            train_indices=train_indices,
            test_indices=test_indices,
            class_values=tuple(np.unique(train_view.labels).tolist()),
            class_client_profile=np.empty((0, spec.num_clients), dtype=np.float64),
            attempts=1,
            repaired=False,
            repair_moves=(),
            partitioner="uniform_v2",
        )
    return build_label_dirichlet_partition(train_view.labels, test_view.labels, spec)


def _result_metadata(
    fingerprint: str,
    spec: PartitionSpec,
    train_fp: Mapping[str, Any],
    test_fp: Mapping[str, Any],
    train_view: DatasetView,
    test_view: DatasetView,
    result: PartitionResult,
) -> tuple[str, dict[str, Any]]:
    indices_sha256 = sha256(_indices_bytes(result.train_indices, result.test_indices)).hexdigest()
    metadata = {
        "schema_version": spec.schema_version,
        "fingerprint": fingerprint,
        "partitioner": result.partitioner,
        "strategy": spec.strategy,
        "alpha": None if spec.alpha is None else float(spec.alpha),
        "num_clients": spec.num_clients,
        "partition_seed": spec.seed,
        "min_samples_per_client": spec.min_samples_per_client,
        "max_retries": spec.max_retries,
        "repair_policy": spec.repair_policy,
        "attempts": result.attempts,
        "repaired": result.repaired,
        "repair_count": len(result.repair_moves),
        "repair_moves": list(result.repair_moves),
        "class_values": [value.item() if hasattr(value, "item") else value for value in result.class_values],
        "class_client_profile": result.class_client_profile.tolist(),
        "indices_sha256": indices_sha256,
        "train_dataset": dict(train_fp),
        "test_dataset": dict(test_fp),
        "train_stats": _client_stats(result.train_indices, train_view),
        "test_stats": _client_stats(result.test_indices, test_view),
    }
    return indices_sha256, metadata


def _artifact_from_result(
    cache_dir: Path,
    fingerprint: str,
    spec: PartitionSpec,
    train_fp: Mapping[str, Any],
    test_fp: Mapping[str, Any],
    train_view: DatasetView,
    test_view: DatasetView,
    result: PartitionResult,
) -> PartitionArtifact:
    indices_sha256, metadata = _result_metadata(
        fingerprint,
        spec,
        train_fp,
        test_fp,
        train_view,
        test_view,
        result,
    )
    return PartitionArtifact(
        fingerprint=fingerprint,
        indices_sha256=indices_sha256,
        cache_dir=cache_dir,
        spec=spec,
        train_indices=result.train_indices,
        test_indices=result.test_indices,
        metadata=metadata,
    )


def _indices_arrays(result: PartitionResult) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for split, mapping in (("train", result.train_indices), ("test", result.test_indices)):
        for client_id, values in mapping.items():
            arrays[f"{split}_{client_id}"] = np.asarray(values, dtype=np.int64)
    return arrays


def _load_indices_arrays(path: Path, num_clients: int) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    train_indices: dict[int, np.ndarray] = {}
    test_indices: dict[int, np.ndarray] = {}
    for client_id in range(num_clients):
        train_key = f"train_{client_id}"
        test_key = f"test_{client_id}"
        if train_key not in payload or test_key not in payload:
            raise PartitionCacheError(f"indices archive is missing {train_key} or {test_key}")
        for key, destination in ((train_key, train_indices), (test_key, test_indices)):
            array = payload[key]
            if array.ndim != 1:
                raise PartitionCacheError(f"indices archive array {key} must be one-dimensional")
            if not np.issubdtype(array.dtype, np.integer):
                raise PartitionCacheError(f"indices archive array {key} must have integer dtype")
            destination[client_id] = array.astype(np.int64, copy=False)
    expected_keys = {f"train_{client_id}" for client_id in range(num_clients)} | {
        f"test_{client_id}" for client_id in range(num_clients)
    }
    if set(payload) != expected_keys:
        raise PartitionCacheError("indices archive has unexpected arrays")
    return train_indices, test_indices


def _build_and_publish_partition(
    cache_dir: Path,
    fingerprint: str,
    spec: PartitionSpec,
    train_dataset: Any,
    test_dataset: Any,
    train_fp: Mapping[str, Any],
    test_fp: Mapping[str, Any],
) -> PartitionArtifact:
    train_view = extract_dataset_view(train_dataset)
    test_view = extract_dataset_view(test_dataset)
    result = _build_partition_result(train_view, test_view, spec)
    artifact = _artifact_from_result(
        cache_dir,
        fingerprint,
        spec,
        train_fp,
        test_fp,
        train_view,
        test_view,
        result,
    )
    _atomic_write_npz(cache_dir / "indices.npz", _indices_arrays(result))
    _atomic_write_json(cache_dir / "metadata.json", artifact.metadata)
    _atomic_write_ready(cache_dir / "READY")
    return artifact


def load_partition_artifact(
    cache_dir: Path,
    fingerprint: str,
    train_dataset: Any,
    test_dataset: Any,
    spec: PartitionSpec,
) -> PartitionArtifact:
    cache_dir = Path(cache_dir)
    metadata_path = cache_dir / "metadata.json"
    indices_path = cache_dir / "indices.npz"
    ready_path = cache_dir / "READY"
    if not ready_path.exists() or not metadata_path.exists() or not indices_path.exists():
        raise PartitionCacheError(f"partition artifact is incomplete: {cache_dir}")

    try:
        with open(metadata_path, "r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        if not isinstance(metadata, dict):
            raise PartitionCacheError("metadata must be a JSON object")
        train_metadata = metadata["train_dataset"]
        test_metadata = metadata["test_dataset"]
        if not isinstance(train_metadata, dict) or not isinstance(test_metadata, dict):
            raise PartitionCacheError("metadata is missing dataset fingerprints")

        train_fp = dataset_fingerprint(
            train_dataset,
            dataset_name=str(train_metadata["dataset_name"]),
            split="train",
            system_data_count=train_metadata.get("system_data_count"),
        )
        test_fp = dataset_fingerprint(
            test_dataset,
            dataset_name=str(test_metadata["dataset_name"]),
            split="test",
            system_data_count=test_metadata.get("system_data_count"),
        )
        expected_fingerprint = partition_fingerprint(spec, train_fp, test_fp)
        if fingerprint != expected_fingerprint or metadata.get("fingerprint") != expected_fingerprint:
            raise PartitionCacheError("partition fingerprint mismatch")

        train_indices, test_indices = _load_indices_arrays(indices_path, spec.num_clients)
        train_view = extract_dataset_view(train_dataset)
        test_view = extract_dataset_view(test_dataset)
        validate_indices(
            train_indices,
            dataset_size=len(train_view.labels),
            num_clients=spec.num_clients,
            min_size=spec.min_samples_per_client,
        )
        validate_indices(
            test_indices,
            dataset_size=len(test_view.labels),
            num_clients=spec.num_clients,
        )

        if metadata.get("indices_sha256") != sha256(_indices_bytes(train_indices, test_indices)).hexdigest():
            raise PartitionCacheError("partition indices digest mismatch")

        expected_result = PartitionResult(
            train_indices=train_indices,
            test_indices=test_indices,
            class_values=tuple(metadata.get("class_values", [])),
            class_client_profile=np.asarray(metadata.get("class_client_profile", []), dtype=np.float64),
            attempts=int(metadata.get("attempts", 0)),
            repaired=bool(metadata.get("repaired", False)),
            repair_moves=tuple(dict(move) for move in metadata.get("repair_moves", [])),
            partitioner=str(metadata.get("partitioner")),
        )
        expected_artifact = _artifact_from_result(
            cache_dir,
            expected_fingerprint,
            spec,
            train_fp,
            test_fp,
            train_view,
            test_view,
            expected_result,
        )
        if _canonical_json(expected_artifact.metadata) != _canonical_json(metadata):
            raise PartitionCacheError("partition metadata validation failed")
        return PartitionArtifact(
            fingerprint=expected_fingerprint,
            indices_sha256=expected_artifact.indices_sha256,
            cache_dir=cache_dir,
            spec=spec,
            train_indices=train_indices,
            test_indices=test_indices,
            metadata=metadata,
        )
    except PartitionCacheError as exc:
        raise PartitionCacheError(f"partition cache {cache_dir} is invalid: {exc}") from exc
    except Exception as exc:
        raise PartitionCacheError(f"partition cache {cache_dir} could not be loaded: {exc}") from exc


def load_legacy_quantity_partition(
    param_dict: Mapping[str, Any],
    train_dataset: Any,
    test_dataset: Any,
    repeat_idx: int,
) -> PartitionArtifact:
    del test_dataset
    strategy = str(param_dict["split_strategy"])
    legacy_name = LEGACY_QUANTITY_ALIASES.get(strategy)
    if legacy_name is None:
        raise PartitionDataError(f"unsupported legacy split strategy: {strategy}")
    from tool.checkpoint import load_split_indices

    legacy_params = dict(param_dict)
    legacy_params["split_strategy"] = legacy_name
    train_indices = load_split_indices(legacy_params)
    if train_indices is None:
        raise PartitionCacheError(f"legacy split_indices.json is missing or invalid for {strategy}")
    spec = PartitionSpec(
        strategy=strategy,
        alpha=DIRICHLET_ALPHAS.get(legacy_name),
        num_clients=int(param_dict["num_clients_K"]),
        seed=int(param_dict.get("base_seed", 42)) + 1000 * int(repeat_idx),
        min_samples_per_client=int(param_dict.get("partition_min_size", 1)),
        max_retries=int(param_dict.get("partition_max_retries", 100)),
        repair_policy=str(param_dict.get("partition_repair_policy", "minimum_move_v1")),
    )
    validate_indices(
        train_indices,
        dataset_size=len(extract_dataset_view(train_dataset).labels),
        num_clients=spec.num_clients,
    )
    empty_test = {client_id: np.empty(0, dtype=np.int64) for client_id in range(spec.num_clients)}
    metadata = {
        "schema_version": 1,
        "fingerprint": f"legacy:{strategy}",
        "partitioner": "legacy_quantity_dirichlet_v1",
        "strategy": strategy,
        "indices_sha256": sha256(_indices_bytes(train_indices, empty_test)).hexdigest(),
    }
    return PartitionArtifact(
        fingerprint=metadata["fingerprint"],
        indices_sha256=metadata["indices_sha256"],
        cache_dir=Path(param_dict.get("model_path", ".")) / "split_info",
        spec=spec,
        train_indices={client_id: np.asarray(values, dtype=np.int64) for client_id, values in train_indices.items()},
        test_indices=empty_test,
        metadata=metadata,
    )


def build_or_load_partition(
    param_dict: Mapping[str, Any],
    train_dataset: Any,
    test_dataset: Any,
    repeat_idx: int,
) -> PartitionArtifact:
    strategy = str(param_dict["split_strategy"])
    if strategy in LEGACY_QUANTITY_ALIASES:
        return load_legacy_quantity_partition(param_dict, train_dataset, test_dataset, repeat_idx)

    spec = PartitionSpec.from_params(param_dict, repeat_idx)
    train_fp = dataset_fingerprint(
        train_dataset,
        dataset_name=str(param_dict["dataset_name"]),
        split="train",
        system_data_count=param_dict.get("system_data_count"),
    )
    test_fp = dataset_fingerprint(
        test_dataset,
        dataset_name=str(param_dict["dataset_name"]),
        split="test",
        system_data_count=param_dict.get("system_data_count"),
    )
    fingerprint = partition_fingerprint(spec, train_fp, test_fp)
    cache_dir = (
        Path(param_dict.get("partition_cache_root", "./partition_cache"))
        / "v2"
        / str(param_dict["dataset_name"])
        / fingerprint
    )
    if (cache_dir / "READY").exists():
        return load_partition_artifact(cache_dir, fingerprint, train_dataset, test_dataset, spec)
    return _build_and_publish_partition(
        cache_dir,
        fingerprint,
        spec,
        train_dataset,
        test_dataset,
        train_fp,
        test_fp,
    )
