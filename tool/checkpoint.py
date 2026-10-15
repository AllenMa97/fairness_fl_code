from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, fields
from hashlib import sha256
from pathlib import Path
from typing import Any, ClassVar, Mapping
import copy
import json
import numbers
import os
import random
import tempfile

import numpy as np
import torch

from tool.code_version import code_version
from tool.logger import logger


CHECKPOINT_SCHEMA_VERSION = 2
_LEGACY_PARTITION_FINGERPRINT = "legacy-untyped-partition"
_CONFIG_EXCLUDED_KEYS = {
    "model_path",
    "result_path",
    "log_path",
    "basic_path",
    "tb_log_dir",
    "resume",
    "start_exp",
    "parallel_repeats",
    "checkpoint_save_freq",
    "checkpoint_keep_latest",
    "partition_cache_root",
    "repeat_idx",
    "repeat_seed",
    "partition_fingerprint",
    "partition_metadata",
    "experiment_config_hash",
    "Experiment_NO",
    # 运行环境旋钮：同一实验换卡运行必须命中同一 checkpoint 目录。若把 -cuda 计入
    # 配置哈希，GPU 轮转（批量调度器重启、重试换卡）会让 resume 找不到旧进度而
    # 静默从零开始，断点续跑形同失效。
    "cuda",
    "CUDA_VISIBLE_DEVICES",
    # 编号偏移只影响结果文件名（current_exp 起点），不改变任何实验内容。
    # 若计入哈希，调度器按划分粒度投递（-exp_offset 0/3/6/9）会让同一格子
    # 拿到与整批运行不同的 config hash，旧 checkpoint 全部失配、resume 静默失效。
    "exp_offset",
}


class CheckpointCompatibilityError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointState(MappingABC[str, Any]):
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

    _ALIASES: ClassVar[dict[str, str]] = {
        "communication_round": "next_round",
        "extra_state": "algorithm_state",
    }

    def __getitem__(self, key: str) -> Any:
        if key == "communication_round":
            return self.next_round - 1
        field_name = self._ALIASES.get(key, key)
        if field_name not in {field.name for field in fields(self)}:
            raise KeyError(key)
        return getattr(self, field_name)

    def __iter__(self):
        yield from (field.name for field in fields(self))
        yield from self._ALIASES

    def __len__(self) -> int:
        return len(fields(self)) + len(self._ALIASES)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
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
        if key in _CONFIG_EXCLUDED_KEYS or str(key).startswith("_runtime_"):
            continue
        result[str(key)] = (
            str(value).split(":", 1)[0] if key == "device" else _json_value(value)
        )
    return result


def build_experiment_config_hash(param_dict: Mapping[str, Any]) -> str:
    payload = json.dumps(
        canonical_experiment_config(param_dict),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _legacy_repeat_idx_from_params(
    param_dict: Mapping[str, Any], repeat_idx: int | None = None
) -> int:
    """Resolve historical checkpoint identities. Never use for schema-v2 artifacts."""
    if repeat_idx is not None:
        return int(repeat_idx)
    if "repeat_idx" in param_dict:
        return int(param_dict["repeat_idx"])
    return int(param_dict.get("Experiment_NO", 0))


def _repeat_idx_from_params(param_dict: Mapping[str, Any], repeat_idx: int | None = None) -> int:
    value = repeat_idx if repeat_idx is not None else param_dict.get("repeat_idx")
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < 0:
        raise CheckpointCompatibilityError(
            "schema v2 requires an explicit non-negative 0-based repeat_idx"
        )
    return int(value)


def _repeat_seed_from_params(param_dict: Mapping[str, Any], repeat_idx: int) -> int:
    if "repeat_seed" in param_dict:
        return int(param_dict["repeat_seed"])
    return int(param_dict.get("base_seed", 0)) + 1000 * repeat_idx


def _config_hash_from_params(param_dict: Mapping[str, Any]) -> str:
    return str(param_dict.get("experiment_config_hash") or build_experiment_config_hash(param_dict))


def _partition_fingerprint_from_params(param_dict: Mapping[str, Any]) -> str:
    value = param_dict.get("partition_fingerprint")
    if (
        not isinstance(value, str)
        or not value.strip()
        or value == _LEGACY_PARTITION_FINGERPRINT
    ):
        raise CheckpointCompatibilityError(
            "schema v2 requires a non-empty partition_fingerprint"
        )
    return value


def _require_partition_fingerprint(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value == _LEGACY_PARTITION_FINGERPRINT
    ):
        raise CheckpointCompatibilityError(
            "schema v2 requires a non-empty partition_fingerprint"
        )
    return value


def get_repeat_state_dir(param_dict: Mapping[str, Any], repeat_idx: int | None = None) -> Path:
    config_hash = _config_hash_from_params(param_dict)
    index = _repeat_idx_from_params(param_dict, repeat_idx)
    return Path(param_dict["model_path"]) / "experiment_state" / config_hash / f"repeat_{index:02d}"


def get_checkpoint_path(param_dict: Mapping[str, Any], iter_t: int | None = None):
    repeat_dir = get_repeat_state_dir(param_dict)
    if iter_t is None:
        return str(repeat_dir)
    return str(repeat_dir / "checkpoint_latest.pt")


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


_RNG_STATE_KEYS = frozenset({"python", "numpy", "torch_cpu", "torch_cuda", "cuda_device_count"})


def _current_cuda_device_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def _validate_rng_state(rng_state: Any, *, validate_current_cuda: bool) -> dict[str, Any]:
    if not isinstance(rng_state, MappingABC):
        raise CheckpointCompatibilityError("rng_state must be a mapping")
    missing = sorted(_RNG_STATE_KEYS - set(rng_state))
    if missing:
        raise CheckpointCompatibilityError(f"rng_state missing fields: {missing}")

    python_state = rng_state["python"]
    numpy_state = rng_state["numpy"]
    cpu_state = rng_state["torch_cpu"]
    cuda_states = rng_state["torch_cuda"]
    stored_count = rng_state["cuda_device_count"]
    if not isinstance(python_state, tuple):
        raise CheckpointCompatibilityError("rng_state.python must be a tuple")
    if not isinstance(numpy_state, tuple):
        raise CheckpointCompatibilityError("rng_state.numpy must be a tuple")
    if not isinstance(cpu_state, torch.Tensor):
        raise CheckpointCompatibilityError("rng_state.torch_cpu must be a tensor")
    if not isinstance(cuda_states, list):
        raise CheckpointCompatibilityError("rng_state.torch_cuda must be a list")
    if (
        isinstance(stored_count, bool)
        or not isinstance(stored_count, numbers.Integral)
        or int(stored_count) < 0
    ):
        raise CheckpointCompatibilityError("rng_state.cuda_device_count must be a non-negative integer")
    if len(cuda_states) != int(stored_count):
        raise CheckpointCompatibilityError(
            "CUDA RNG state count does not match rng_state.cuda_device_count"
        )
    if any(
        not isinstance(cuda_state, torch.Tensor)
        or cuda_state.dtype != torch.uint8
        or cuda_state.ndim != 1
        for cuda_state in cuda_states
    ):
        raise CheckpointCompatibilityError(
            "rng_state.torch_cuda entries must be one-dimensional uint8 tensors"
        )

    try:
        random.Random().setstate(python_state)
        np.random.RandomState().set_state(numpy_state)
        torch.Generator(device="cpu").set_state(cpu_state)
    except Exception as exc:
        raise CheckpointCompatibilityError("rng_state contains an invalid RNG state") from exc

    if validate_current_cuda and int(stored_count) != _current_cuda_device_count():
        raise CheckpointCompatibilityError(
            "CUDA RNG device count mismatch: "
            f"stored={int(stored_count)}, current={_current_cuda_device_count()}"
        )
    return dict(rng_state)


def _validate_checkpoint_payload(payload: Any, total_rounds: int) -> Mapping[str, Any]:
    if not isinstance(payload, MappingABC):
        raise CheckpointCompatibilityError("checkpoint payload must be a mapping")
    required = {
        "schema_version",
        "experiment_config_hash",
        "partition_fingerprint",
        "repeat_idx",
        "repeat_seed",
        "next_round",
        "phase",
        "global_model_state",
        "algorithm_state",
        "amp_scaler_state",
        "rng_state",
        "total_gpu_seconds",
        "total_runtime_seconds",
        "total_communication_cost",
        "client_selection_history",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise CheckpointCompatibilityError(f"checkpoint missing fields: {missing}")
    if (
        isinstance(payload["repeat_idx"], bool)
        or not isinstance(payload["repeat_idx"], numbers.Integral)
        or int(payload["repeat_idx"]) < 0
    ):
        raise CheckpointCompatibilityError("checkpoint repeat_idx must be a non-negative integer")
    _require_partition_fingerprint(payload["partition_fingerprint"])
    if (
        isinstance(payload["next_round"], bool)
        or not isinstance(payload["next_round"], numbers.Integral)
        or not 0 <= int(payload["next_round"]) <= total_rounds
    ):
        raise CheckpointCompatibilityError("next_round is outside configured round bounds")
    if not isinstance(payload["phase"], str) or payload["phase"] not in {"train", "evaluate"}:
        raise CheckpointCompatibilityError(f"invalid checkpoint phase: {payload['phase']!r}")
    _validate_rng_state(payload["rng_state"], validate_current_cuda=True)
    return payload


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temp_name)
        _fsync_file_best_effort(Path(temp_name))
        os.replace(temp_name, path)
        _fsync_parent_directory(path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _atomic_json_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.flush()
            try:
                os.fsync(stream.fileno())
            except OSError:  # pragma: no cover - Windows best-effort durability
                pass
        os.replace(temp_name, path)
        _fsync_parent_directory(path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _fsync_file_best_effort(path: Path) -> None:
    """Persist a file to disk where the OS supports it (no-op on failure)."""
    try:
        with open(path, "r+b") as stream:
            os.fsync(stream.fileno())
    except OSError:  # pragma: no cover - e.g. Windows read-handle constraints
        pass


def _fsync_parent_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):  # pragma: no cover - Windows has no dir fsync
        return
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:  # pragma: no cover - directory handle unsupported
        return
    try:
        try:
            os.fsync(directory_fd)
        except OSError:  # pragma: no cover - best-effort durability
            pass
    finally:
        os.close(directory_fd)


def _detach_tensors(value: Any) -> Any:
    """递归地把容器里的非叶张量 detach 成叶子张量。

    PyTorch 不允许 deepcopy 带 grad_fn 的非叶张量（RuntimeError: Only Tensors
    created explicitly by the user (graph leaves) support the deepcopy protocol）。
    算法会把训练中途的张量（如 FedProto 的类原型）放进 algorithm_state，这类张量
    只用于续跑、不需要梯度，统一 detach 后再深拷贝即可。
    """
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, MappingABC):
        return {k: _detach_tensors(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_detach_tensors(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_detach_tensors(v) for v in value)
    return value


def _deepcopy_serializable(value: Any) -> Any:
    """deepcopy 该字段前先把非叶张量 detach，避免带计算图的张量让落盘整体失败。"""
    return copy.deepcopy(_detach_tensors(value))


def save_checkpoint(
    param_dict,
    iter_t,
    global_model,
    *,
    algorithm_state=None,
    amp_scaler=None,
    total_gpu_seconds=0.0,
    total_runtime_seconds=0.0,
    total_communication_cost=0.0,
    client_selection_history=(),
    extra_state=None,
    **legacy_kwargs,
):
    unsupported = sorted(set(legacy_kwargs) - {"start_time"})
    if unsupported:
        raise TypeError(f"unsupported checkpoint fields: {', '.join(unsupported)}")
    if algorithm_state is not None and extra_state:
        raise ValueError("pass algorithm_state or legacy extra_state, not both")

    repeat_idx = _repeat_idx_from_params(param_dict)
    repeat_seed = _repeat_seed_from_params(param_dict, repeat_idx)
    total_rounds = int(param_dict["communication_round_I"])
    next_round = int(iter_t) + 1
    phase = "evaluate" if next_round >= total_rounds else "train"
    scaler_state = None
    if amp_scaler is not None:
        scaler_state = amp_scaler.state_dict() if hasattr(amp_scaler, "state_dict") else amp_scaler

    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "experiment_config_hash": _config_hash_from_params(param_dict),
        "partition_fingerprint": _partition_fingerprint_from_params(param_dict),
        "repeat_idx": repeat_idx,
        "repeat_seed": repeat_seed,
        "next_round": next_round,
        "phase": phase,
        "global_model_state": copy.deepcopy(global_model.state_dict()),
        "algorithm_state": _deepcopy_serializable(algorithm_state if algorithm_state is not None else (extra_state or {})),
        "amp_scaler_state": _deepcopy_serializable(scaler_state),
        "rng_state": _capture_rng_state(),
        "total_gpu_seconds": float(total_gpu_seconds),
        "total_runtime_seconds": float(total_runtime_seconds),
        "total_communication_cost": float(total_communication_cost),
        "client_selection_history": [list(map(int, row)) for row in client_selection_history],
    }
    path = get_repeat_state_dir(param_dict, repeat_idx) / "checkpoint_latest.pt"
    _atomic_torch_save(payload, path)
    logger.info(f"Checkpoint saved at round boundary {next_round}: {path}")
    return path


def load_checkpoint(
    param_dict,
    *,
    expected_config_hash=None,
    expected_partition_fingerprint=None,
    expected_repeat_idx=None,
    target_round=None,
):
    if target_round is not None:
        raise CheckpointCompatibilityError("schema v2 retains only checkpoint_latest.pt")

    repeat_idx = _repeat_idx_from_params(param_dict, expected_repeat_idx)
    expected_partition = _require_partition_fingerprint(
        expected_partition_fingerprint
        if expected_partition_fingerprint is not None
        else _partition_fingerprint_from_params(param_dict)
    )
    path = get_repeat_state_dir(param_dict, repeat_idx) / "checkpoint_latest.pt"
    if not path.exists():
        return None

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointCompatibilityError(f"unable to load checkpoint: {path}") from exc
    payload = _validate_checkpoint_payload(
        payload, total_rounds=int(param_dict["communication_round_I"])
    )

    expected_hash = expected_config_hash or _config_hash_from_params(param_dict)
    expected_seed = _repeat_seed_from_params(param_dict, repeat_idx)
    comparisons = (
        (payload["schema_version"], CHECKPOINT_SCHEMA_VERSION, "schema version"),
        (payload["experiment_config_hash"], expected_hash, "experiment config hash"),
        (payload["partition_fingerprint"], expected_partition, "partition fingerprint"),
        (payload["repeat_idx"], repeat_idx, "repeat index"),
        (payload["repeat_seed"], expected_seed, "repeat seed"),
    )
    for stored, expected, name in comparisons:
        if stored != expected:
            raise CheckpointCompatibilityError(
                f"{name} mismatch: stored={stored!r}, expected={expected!r}"
            )
    state = CheckpointState(path=path, **payload)
    logger.info(f"Successfully loaded checkpoint from {path}")
    return state


def restore_rng_state(state: CheckpointState) -> None:
    try:
        rng_state = state.rng_state
    except AttributeError as exc:
        raise CheckpointCompatibilityError("checkpoint state is missing rng_state") from exc
    rng_state = _validate_rng_state(rng_state, validate_current_cuda=True)
    try:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch_cpu"])
        if rng_state["torch_cuda"]:
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
    except Exception as exc:
        raise CheckpointCompatibilityError("failed to restore rng_state") from exc


_RESOURCE_USAGE_KEYS = frozenset({"peak_cuda_bytes", "peak_rss_bytes", "checkpoint_bytes"})


def _normalize_resource_usage(resource_usage):
    if resource_usage is None:
        return {name: 0 for name in sorted(_RESOURCE_USAGE_KEYS)}
    if not isinstance(resource_usage, MappingABC):
        raise CheckpointCompatibilityError("resource_usage must be a mapping")
    if set(resource_usage) != _RESOURCE_USAGE_KEYS:
        raise CheckpointCompatibilityError(
            "resource_usage must contain exactly peak_cuda_bytes, peak_rss_bytes, checkpoint_bytes"
        )
    normalized = {}
    for name in sorted(_RESOURCE_USAGE_KEYS):
        value = resource_usage[name]
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < 0:
            raise CheckpointCompatibilityError(
                f"resource_usage.{name} must be a non-negative integer"
            )
        normalized[name] = int(value)
    return normalized


def save_repeat_metrics(
    param_dict,
    repeat_idx,
    config_hash,
    partition_fingerprint,
    metrics,
    *,
    repeat_seed,
    total_gpu_seconds,
    total_communication_cost,
    resource_usage=None,
):
    repeat_idx = _repeat_idx_from_params(param_dict, repeat_idx)
    partition_fingerprint = _require_partition_fingerprint(partition_fingerprint)
    normalized_resource_usage = _normalize_resource_usage(resource_usage)
    path = get_repeat_state_dir(param_dict, repeat_idx) / "metrics.json"
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "experiment_config_hash": str(config_hash),
        "experiment_config": canonical_experiment_config(param_dict),
        "partition_fingerprint": str(partition_fingerprint),
        "repeat_idx": int(repeat_idx),
        "repeat_seed": int(repeat_seed),
        "metrics": _json_value(metrics),
        "total_gpu_seconds": float(total_gpu_seconds),
        "total_communication_cost": float(total_communication_cost),
        "resource_usage": normalized_resource_usage,
    }
    _atomic_json_save(payload, path)
    return path


def load_repeat_metrics(param_dict, repeat_idx, expected_config_hash, expected_partition_fingerprint):
    repeat_idx = _repeat_idx_from_params(param_dict, repeat_idx)
    expected_partition_fingerprint = _require_partition_fingerprint(
        expected_partition_fingerprint
    )
    path = get_repeat_state_dir(param_dict, repeat_idx) / "metrics.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    checks = (
        (payload.get("schema_version"), CHECKPOINT_SCHEMA_VERSION, "schema version"),
        (payload.get("experiment_config_hash"), str(expected_config_hash), "experiment config hash"),
        (
            payload.get("partition_fingerprint"),
            str(expected_partition_fingerprint),
            "partition fingerprint",
        ),
        (payload.get("repeat_idx"), int(repeat_idx), "repeat index"),
        (
            payload.get("repeat_seed"),
            _repeat_seed_from_params(param_dict, _repeat_idx_from_params(param_dict, repeat_idx)),
            "repeat seed",
        ),
    )
    for stored, expected, name in checks:
        if stored != expected:
            raise CheckpointCompatibilityError(
                f"repeat metrics {name} mismatch: stored={stored!r}, expected={expected!r}"
            )
    if not isinstance(payload.get("metrics"), dict):
        raise CheckpointCompatibilityError("repeat metrics payload is missing metrics")
    payload["resource_usage"] = _normalize_resource_usage(payload.get("resource_usage"))
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
    checkpoint_path = repeat_dir / "checkpoint_latest.pt"
    if policy == "global_model":
        _atomic_torch_save(global_model.state_dict(), repeat_dir / "final_global_model.pt")
    if policy != "full_state" and checkpoint_path.exists():
        checkpoint_path.unlink()


def _legacy_config_hash_including_cuda(param_dict: Mapping[str, Any]) -> str:
    """按移除 -cuda 之前的旧口径计算配置哈希（仅用于认领历史 experiment_state 目录）。"""
    if "cuda" not in param_dict:
        return build_experiment_config_hash(param_dict)
    legacy_config = canonical_experiment_config(param_dict)
    legacy_config["cuda"] = _json_value(param_dict["cuda"])
    payload = json.dumps(
        legacy_config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


_REPEAT_ARTIFACT_NAMES = ("checkpoint_latest.pt", "metrics.json", "final_global_model.pt")
_HASH_BEARING_ARTIFACTS = ("checkpoint_latest.pt", "metrics.json")


def _rewrite_stored_config_hash(path: Path, config_hash: str) -> None:
    """把产物内记录的旧配置哈希改写为新口径，否则 resume 会以哈希不一致失败。"""
    if path.name == "metrics.json":
        with open(path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
        payload["experiment_config_hash"] = config_hash
        stored_config = payload.get("experiment_config")
        if isinstance(stored_config, dict):
            stored_config.pop("cuda", None)
        _atomic_json_save(payload, path)
        return
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["experiment_config_hash"] = config_hash
    _atomic_torch_save(payload, path)


_LEGACY_CUDA_PROBE_LIMIT = 64


def _legacy_config_hashes(param_dict: Mapping[str, Any]) -> list[str]:
    """列出同一实验（仅 -cuda 取值不同）可能落盘过的全部旧口径目录名。

    修复前 cuda 计入配置哈希，同一实验换卡运行会在 experiment_state 下留下按 GPU
    分布的多个兄弟目录；只反推「当前 cuda」那一份会漏掉换卡前的进度。这里把候选
    cuda 取值枚举一遍供认领时逐个查找，仅用于定位历史目录，正常哈希不受影响。
    """
    new_hash = _config_hash_from_params(param_dict)
    values: list[Any] = [param_dict.get("cuda")]
    values.extend(range(_LEGACY_CUDA_PROBE_LIMIT))
    values.extend(str(index) for index in range(_LEGACY_CUDA_PROBE_LIMIT))
    hashes: list[str] = []
    seen = {new_hash}
    for value in values:
        if value is None:
            continue
        candidate = _legacy_config_hash_including_cuda(dict(param_dict, cuda=value))
        if candidate in seen:
            continue
        seen.add(candidate)
        hashes.append(candidate)
    return hashes


def _migrate_repeat_artifacts(legacy_root: Path, new_root: Path) -> list[Path]:
    """把单个旧口径目录下的 repeat 产物搬到新目录，返回需改写哈希的产物路径。"""
    migrated: list[Path] = []
    for legacy_repeat in sorted(legacy_root.glob("repeat_*")):
        target_repeat = new_root / legacy_repeat.name
        for name in _REPEAT_ARTIFACT_NAMES:
            source = legacy_repeat / name
            if not source.exists():
                continue
            destination = target_repeat / name
            # 目标已存在且写入更晚时保留目标：新旧口径的目录同时存在只会出现在新旧
            # 进程短暂重叠的情形，按写入时间取更靠后的产物可避免用旧进度覆盖新进度。
            if destination.exists() and destination.stat().st_mtime >= source.stat().st_mtime:
                continue
            target_repeat.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            if name in _HASH_BEARING_ARTIFACTS:
                migrated.append(destination)
    return migrated


def _migrate_legacy_state_dir(param_dict: Mapping[str, Any]) -> Path | None:
    new_hash = _config_hash_from_params(param_dict)
    state_root = Path(param_dict["model_path"]) / "experiment_state"
    new_root = state_root / new_hash
    migrated: list[Path] = []
    adopted: list[str] = []
    for legacy_hash in _legacy_config_hashes(param_dict):
        legacy_root = state_root / legacy_hash
        if not legacy_root.is_dir():
            continue
        migrated.extend(_migrate_repeat_artifacts(legacy_root, new_root))
        adopted.append(legacy_hash)
        for directory in (*sorted(legacy_root.glob("repeat_*")), legacy_root):
            try:
                directory.rmdir()
            except OSError:
                pass
    if not adopted:
        return None
    for path in migrated:
        _rewrite_stored_config_hash(path, new_hash)
    logger.info(
        f"Adopted {len(adopted)} legacy experiment_state dir(s) {adopted} -> "
        f"{new_hash} ({len(migrated)} artifacts migrated)"
    )
    return new_root


def adopt_legacy_state_dir(param_dict: Mapping[str, Any]) -> Path | None:
    """把旧口径（配置哈希含 -cuda）的 experiment_state 目录迁到当前口径。

    配置哈希在修复前未排除 CLI 的 -cuda，同一实验换卡运行会落到不同目录（批量调度
    器重试换卡时尤其明显），resume 因此找不到旧 checkpoint 而静默从零重跑。修复后
    哈希不再含 cuda，历史目录不再被命中；本函数按同样口径补回 cuda 键反推旧哈希，
    且覆盖该实验落盘过的所有 GPU 版本，把旧目录的产物搬到当前目录并改写其中记录的
    哈希，使已完成的 repeat 与正在跑的进度都能原地续上。旧目录搬空后本函数自动退化
    为空操作。
    """
    try:
        return _migrate_legacy_state_dir(param_dict)
    except Exception as exc:  # 认领失败不应中断训练，退回“按当前目录继续”的默认行为
        logger.warning(f"legacy experiment_state dir adoption failed: {exc}")
        return None


def save_aggregate_metrics(param_dict, aggregate):
    path = Path(str(param_dict["result_path"]) + ".json")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "code_version": code_version(),
        "experiment_config_hash": _config_hash_from_params(param_dict),
        "experiment_config": canonical_experiment_config(param_dict),
        "aggregate": _json_value(aggregate),
    }
    _atomic_json_save(payload, path)
    return path


def save_split_indices(param_dict, split_indices):
    split_dir = Path(param_dict["model_path"]) / "split_info"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_info = {
        "split_strategy": param_dict.get("split_strategy", ""),
        "num_clients": param_dict.get("num_clients_K", 0),
        "indices": {
            str(key): value.tolist() if isinstance(value, np.ndarray) else list(value)
            for key, value in split_indices.items()
        },
    }
    with open(split_dir / "split_indices.json", "w", encoding="utf-8") as stream:
        json.dump(split_info, stream, indent=2, ensure_ascii=False)
    logger.info("Split indices saved")


def load_split_indices(param_dict):
    split_path = Path(param_dict["model_path"]) / "split_info" / "split_indices.json"
    if not split_path.exists():
        return None
    try:
        with open(split_path, "r", encoding="utf-8") as stream:
            split_info = json.load(stream)
        if split_info["split_strategy"] != param_dict.get("split_strategy", ""):
            logger.warning(
                "Split strategy mismatch: stored=%s, current=%s",
                split_info["split_strategy"],
                param_dict.get("split_strategy"),
            )
            return None
        if split_info["num_clients"] != param_dict.get("num_clients_K", 0):
            logger.warning(
                "Client number mismatch: stored=%s, current=%s",
                split_info["num_clients"],
                param_dict.get("num_clients_K"),
            )
            return None
        logger.info("Successfully loaded split indices")
        return {int(key): np.array(value) for key, value in split_info["indices"].items()}
    except Exception as exc:
        logger.error(f"Failed to load split indices: {exc}")
        return None


def check_resume_status(param_dict):
    if not param_dict.get("resume", False):
        return None
    return load_checkpoint(param_dict)


def clean_old_checkpoints(param_dict, keep_latest=5):
    del keep_latest
    repeat_idx = _repeat_idx_from_params(param_dict)
    checkpoint_dir = Path(param_dict["model_path"]) / "checkpoints"
    if checkpoint_dir.exists():
        prefixes = (
            f"checkpoint_repeat{repeat_idx}_round_",
            "checkpoint_round_",
        )
        for path in checkpoint_dir.iterdir():
            if path.name == "checkpoint_latest.pt":
                continue
            if any(path.name.startswith(prefix) and path.suffix == ".pt" for prefix in prefixes):
                path.unlink()
