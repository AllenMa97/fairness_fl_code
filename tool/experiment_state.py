from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
import resource
import sys

import numpy as np
import torch


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
    resource_usage: dict[str, int] = field(default_factory=dict)


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

    numeric_keys = sorted(
        set.intersection(
            *[
                {
                    key
                    for key, value in item.metrics.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
                for item in ordered
            ]
        )
    )
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
        "resource_usage": [dict(item.resource_usage) for item in ordered],
    }


def capture_resource_snapshot(checkpoint_path: Path) -> dict[str, int]:
    """Capture reproducibility-relevant resource evidence before artifact cleanup."""
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    peak_rss_bytes = peak_rss if sys.platform == "darwin" else peak_rss * 1024
    return {
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available() else 0,
        "peak_rss_bytes": peak_rss_bytes,
        "checkpoint_bytes": int(Path(checkpoint_path).stat().st_size),
    }


def capture_training_loader_generator_states(training_dataloaders):
    """Return independent DataLoader generator states in client order.

    Loaders without an explicit generator retain ``None`` so callers can use the
    same stable, serializable list for mixed legacy/new loader configurations.
    """
    states = []
    for loader in training_dataloaders:
        generator = getattr(loader, "generator", None)
        states.append(None if generator is None else generator.get_state().clone())
    return states


def restore_training_loader_generator_states(training_dataloaders, states):
    """Restore DataLoader generator states captured at a round boundary."""
    if states is None:
        return
    if len(training_dataloaders) != len(states):
        raise ValueError("training loader generator state count does not match loaders")
    for loader, state in zip(training_dataloaders, states):
        generator = getattr(loader, "generator", None)
        if state is None:
            continue
        if generator is None:
            raise ValueError("checkpoint has a generator state for a loader without generator")
        generator.set_state(state)
