from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, fields
from hashlib import sha256
from pathlib import Path
from typing import Any, ClassVar, Mapping
import atexit
import copy
import json
import numbers
import os
import queue
import random
import tempfile
import threading

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
    # 注意：device 不在此集合内。GPU 显存吃紧时会把格子迁到 CPU 跑（CUDA_VISIBLE_DEVICES
    # 指向不存在的卡号），这是同一实验的另一次执行，不是新实验。但历史 checkpoint 的目录名
    # 是按「device 计入哈希」的口径算出来的，直接把 device 从哈希里删掉会让所有历史 GPU 目录
    # 失配、resume 从 round 0 重跑。因此改为在 canonical_experiment_config 中把 device 的
    # 取值固定成常量（见 _DEVICE_HASH_CONSTANT），既保持历史 GPU 哈希不变，也让 CPU 运行命中
    # 同一目录。跨设备续跑所需的 RNG / AMP 兼容处理见下方 _validate_rng_state、
    # restore_rng_state 与 load_checkpoint。
    # 编号偏移只影响结果文件名（current_exp 起点），不改变任何实验内容。
    # 若计入哈希，调度器按划分粒度投递（-exp_offset 0/3/6/9）会让同一格子
    # 拿到与整批运行不同的 config hash，旧 checkpoint 全部失配、resume 静默失效。
    "exp_offset",
}

# 运行环境旋钮，不进配置哈希：客户端并行度只决定"同一轮里同时训练几个客户端"，
# 既不改变算法数学，也不改变聚合结果与随机数流（各客户端仍是独立 deepcopy + 独立
# seed）。把它计入哈希会让同一格在 GPU（并行度高）和 CPU（并行度被强制串行）之间
# 无法互相续跑，调并行度就等于换一个实验，已有断点全部作废。
#
# 历史目录名是按「client_parallel 计入哈希」的旧口径算出来的：_legacy_config_hashes
# 会枚举该键的历史取值重算旧哈希，因此旧断点仍能被 adopt_legacy_state_dir 认领。
_HASH_EXCLUDED_KEYS = frozenset({"client_parallel"})


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


_DEVICE_HASH_CONSTANT = "cuda"


def canonical_experiment_config(param_dict: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in param_dict.items():
        if key in _CONFIG_EXCLUDED_KEYS or str(key).startswith("_runtime_"):
            continue
        if key == "device":
            # 设备只是运行环境旋钮：同一实验从 GPU 迁到 CPU（或换卡）必须命中同一 checkpoint
            # 目录。但历史目录名是按真实 device 取值算出的（GPU 侧即 'cuda'），故这里把它
            # 固定成常量 'cuda' 而不是删键，好让历史 GPU 哈希原样保留、CPU 运行也能续上。
            result[str(key)] = _DEVICE_HASH_CONSTANT
            continue
        result[str(key)] = _json_value(value)
    return result


def build_experiment_config_hash(param_dict: Mapping[str, Any]) -> str:
    config = canonical_experiment_config(param_dict)
    for key in _HASH_EXCLUDED_KEYS:
        config.pop(key, None)
    payload = json.dumps(
        config,
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

    # 跨设备续跑（GPU 存档 -> CPU 复跑）时两侧可见设备数必然不同。这里只在两侧都确实有
    # CUDA 的情况下才要求数目一致；若当前侧没有 CUDA，就跳过多设备 RNG 校验，仅用
    # torch_cpu / numpy / python 三路状态恢复随机性。
    current_count = _current_cuda_device_count()
    if validate_current_cuda and int(stored_count) > 0 and current_count > 0:
        if int(stored_count) != current_count:
            raise CheckpointCompatibilityError(
                "CUDA RNG device count mismatch: "
                f"stored={int(stored_count)}, current={current_count}"
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


# ────────────────────────────────────────────────────────────
# checkpoint 异步落盘 / Asynchronous checkpoint persistence
#
# 保存点真正必须「同步完成」的只有快照本身（state_dict 深拷贝 + RNG 状态）；
# 耗时的 torch.save + fsync（大模型可达数秒）交给单个后台线程，训练轮次不再
# 等待磁盘写完。单工作线程 + 队列天然保证写入顺序，同一路径不会并发写。
# 进程正常退出（atexit）、加载 / 清理 / 收尾产物之前都会 flush，保证可见性。
# ────────────────────────────────────────────────────────────
_CHECKPOINT_WRITE_QUEUE: "queue.Queue" = queue.Queue()
_CHECKPOINT_WRITER: "threading.Thread | None" = None
_CHECKPOINT_WRITER_LOCK = threading.Lock()


def _checkpoint_writer_loop() -> None:
    while True:
        path, payload = _CHECKPOINT_WRITE_QUEUE.get()
        try:
            _atomic_torch_save(payload, path)
            logger.info(f"Checkpoint saved at round boundary {payload['next_round']}: {path}")
        except Exception as exc:  # 后台线程的异常不能静默吞掉，否则断点续跑失效
            logger.error(f"asynchronous checkpoint write failed for {path}: {exc}")
        finally:
            _CHECKPOINT_WRITE_QUEUE.task_done()


def _ensure_checkpoint_writer() -> "threading.Thread":
    global _CHECKPOINT_WRITER
    with _CHECKPOINT_WRITER_LOCK:
        if _CHECKPOINT_WRITER is None or not _CHECKPOINT_WRITER.is_alive():
            _CHECKPOINT_WRITER = threading.Thread(
                target=_checkpoint_writer_loop, name="checkpoint-writer", daemon=True
            )
            _CHECKPOINT_WRITER.start()
        return _CHECKPOINT_WRITER


def flush_checkpoint_writes() -> None:
    """阻塞直到所有已入队的 checkpoint 都落盘完成（无写入线程时为 no-op）。"""
    if _CHECKPOINT_WRITER is None:
        return
    _CHECKPOINT_WRITE_QUEUE.join()


atexit.register(flush_checkpoint_writes)


def _snapshot_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """同步地做一份脱离计算图、位于 CPU 的 state_dict 快照。

    异步落盘会让快照一直存活到后台线程写完：若快照留在 GPU 上，等于整段时间
    多占一份模型显存（大模型可能因此 OOM），后台 torch.save 读 GPU 也会与训练
    抢设备。因此在同步点一次性拷到 CPU；load 侧本就按 map_location="cpu" 读取，
    落盘字节与原实现一致。非张量条目仍走 copy.deepcopy。
    """
    snapshot = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            snapshot[key] = value.detach().to("cpu", copy=True)
        else:
            snapshot[key] = copy.deepcopy(value)
    return snapshot


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
        "global_model_state": _snapshot_state_dict(global_model.state_dict()),
        "algorithm_state": _deepcopy_serializable(algorithm_state if algorithm_state is not None else (extra_state or {})),
        "amp_scaler_state": _deepcopy_serializable(scaler_state),
        "rng_state": _capture_rng_state(),
        "total_gpu_seconds": float(total_gpu_seconds),
        "total_runtime_seconds": float(total_runtime_seconds),
        "total_communication_cost": float(total_communication_cost),
        "client_selection_history": [list(map(int, row)) for row in client_selection_history],
    }
    path = get_repeat_state_dir(param_dict, repeat_idx) / "checkpoint_latest.pt"
    # 快照（state_dict + RNG）已在上面同步取好，这里只把写盘扔给后台线程；
    # 训练轮次不必等待 torch.save + fsync，断点续跑的语义不受影响。
    _ensure_checkpoint_writer()
    _CHECKPOINT_WRITE_QUEUE.put((path, payload))
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

    # 异步落盘可能仍在进行，先确保磁盘上是已排队的最终版本再读。
    flush_checkpoint_writes()
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
    # GPU 存档迁到 CPU 复跑时 AMP 自动退化为 FP32（GradScaler 为 None）。此时若把存下来的
    # scaler 状态原样返回，算法侧 `scaler.load_state_dict(...)` 会碰到 None 而抛异常；
    # 置 None 与"CPU 下 AMP 关闭"的语义一致，也让各算法的 `amp_scaler_state is not None` 守卫生效。
    if int(payload["rng_state"]["cuda_device_count"]) > 0 and _current_cuda_device_count() == 0:
        payload = dict(payload)
        payload["amp_scaler_state"] = None
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
        # GPU 存档在 CPU 上复跑时没有 CUDA 上下文，跳过 CUDA RNG；设备数也对不上时同理。
        if rng_state["torch_cuda"] and torch.cuda.is_available():
            if len(rng_state["torch_cuda"]) == torch.cuda.device_count():
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


# 允许在「同一实验」判定中偏离的键：batch_size 只影响显存与每轮梯度步数以外的算力
# 开销，不改变实验定义（dataset/split/algorithm/K 等），因此同一格子的 3 次 repeat
# 允许由不同 batch_size 的进程分别完成。口径见 find_repeat_metrics_across_batch_sizes。
_BATCH_SIZE_CONFIG_KEYS = ("batch_size", "test_batch_size")


def find_repeat_metrics_across_batch_sizes(
    param_dict,
    repeat_idx,
    expected_repeat_seed,
    expected_partition_fingerprint,
):
    """在同一格子的所有 batch_size 变体目录里认领一个已完成的 repeat 产物。

    动机：显存窗口不同的卡只能用不同 batch_size 跑同一实验，而一个格子的 3 次
    repeat 天然可以分给不同 batch_size 的进程完成。认领后该 repeat 不再重算，聚合时
    把它实际使用的 batch_size 一并写进产物，避免结果文件看起来像单个 batch_size 跑的。

    判定「同一实验」：以当前配置的键为基准（排除 batch_size / test_batch_size），
    要求候选产物记录的值逐键相等。候选里多出来的键（模型/数据集构造阶段补的派生键，
    如 nn_input_size、emb_dim、le_class）不参与比较——认领发生在建模型之前，当前
    配置此刻还没有这些键。多个候选时取 batch_size 最大者，保证结果确定。

    返回 (payload, batch_size)，payload 结构与 load_repeat_metrics 一致；无候选返回 None。
    """
    repeat_idx = _repeat_idx_from_params(param_dict, repeat_idx)
    expected_partition_fingerprint = _require_partition_fingerprint(
        expected_partition_fingerprint
    )
    current = canonical_experiment_config(param_dict)
    reference_keys = [key for key in current if key not in _BATCH_SIZE_CONFIG_KEYS]
    current_signature = {key: current[key] for key in reference_keys}
    root = Path(param_dict["model_path"]) / "experiment_state"
    if not root.is_dir():
        return None

    best = None
    for hash_dir in sorted(root.iterdir()):
        if not hash_dir.is_dir():
            continue
        path = hash_dir / f"repeat_{repeat_idx:02d}" / "metrics.json"
        if not path.is_file():
            continue
        try:
            with open(path, "r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, ValueError):
            continue
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            continue
        if payload.get("repeat_idx") != int(repeat_idx):
            continue
        if payload.get("repeat_seed") != int(expected_repeat_seed):
            continue
        if payload.get("partition_fingerprint") != expected_partition_fingerprint:
            continue
        if not isinstance(payload.get("metrics"), dict):
            continue
        stored = payload.get("experiment_config")
        if not isinstance(stored, dict):
            continue
        stored_signature = {key: stored[key] for key in reference_keys if key in stored}
        if stored_signature != current_signature:
            continue
        try:
            batch_size = int(stored.get("batch_size"))
        except (TypeError, ValueError):
            continue
        if best is None or batch_size > best[1]:
            best = (payload, batch_size)

    if best is None:
        return None
    payload, batch_size = best
    payload["resource_usage"] = _normalize_resource_usage(payload.get("resource_usage"))
    return payload, batch_size


def clear_repeat_artifacts(param_dict, repeat_idx):
    # 先等挂起的异步写盘结束，否则删除后可能被后台线程重新写出来。
    flush_checkpoint_writes()
    repeat_dir = get_repeat_state_dir(param_dict, repeat_idx)
    for name in ("checkpoint_latest.pt", "metrics.json", "final_global_model.pt"):
        path = repeat_dir / name
        if path.exists():
            path.unlink()


def finalize_repeat_artifacts(param_dict, repeat_idx, global_model, policy):
    if policy not in {"metrics_only", "global_model", "full_state"}:
        raise ValueError(f"unsupported final artifact policy: {policy}")
    # 收尾会删除 checkpoint_latest.pt，必须先等异步写盘结束，避免删完又被写回来。
    flush_checkpoint_writes()
    repeat_dir = get_repeat_state_dir(param_dict, repeat_idx)
    checkpoint_path = repeat_dir / "checkpoint_latest.pt"
    if policy == "global_model":
        _atomic_torch_save(global_model.state_dict(), repeat_dir / "final_global_model.pt")
    if policy != "full_state" and checkpoint_path.exists():
        checkpoint_path.unlink()


_MISSING = object()


def _legacy_config_hash_including_cuda(
    param_dict: Mapping[str, Any], *, device: Any = None, include_cuda: bool = True,
    client_parallel: Any = _MISSING,
) -> str:
    """按历史口径计算配置哈希（仅用于认领历史 experiment_state 目录）。

    include_cuda=True 复原「-cuda 计入哈希」的更早期口径；device 不为 None 时按旧口径把
    device 键写回（旧版本同样把真实设备名计入哈希）；client_parallel 不为 _MISSING 时按
    旧口径把客户端并行度写回（该键现已从哈希排除，见 _HASH_EXCLUDED_KEYS）。
    """
    legacy_config = canonical_experiment_config(param_dict)
    if include_cuda and "cuda" in param_dict:
        legacy_config["cuda"] = _json_value(param_dict["cuda"])
    if device is not None:
        legacy_config["device"] = str(device).split(":", 1)[0]
    if client_parallel is not _MISSING:
        legacy_config["client_parallel"] = _json_value(client_parallel)
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
            if "device" in stored_config:
                stored_config["device"] = _DEVICE_HASH_CONSTANT
        _atomic_json_save(payload, path)
        return
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["experiment_config_hash"] = config_hash
    _atomic_torch_save(payload, path)


_LEGACY_CUDA_PROBE_LIMIT = 64


def _legacy_config_hashes(param_dict: Mapping[str, Any]) -> list[str]:
    """列出同一实验可能落盘过的全部旧口径目录名。

    历史上配置哈希经历过三次口径变化：早期把 -cuda 计入哈希（同一实验换卡会散成多个
    兄弟目录），随后短暂把 device 从哈希中删除（GPU 迁 CPU 时又散成第三类目录），再后来
    client_parallel 也被排除（调并行度会散出第四类目录）。只反推其中一种会漏掉另一部分
    进度，这里把 cuda 取值 × device 口径 × client_parallel 取值都枚举一遍供认领时逐个
    查找，仅用于定位历史目录，正常哈希不受影响。
    """
    new_hash = _config_hash_from_params(param_dict)
    values: list[Any] = [param_dict.get("cuda")]
    values.extend(range(_LEGACY_CUDA_PROBE_LIMIT))
    values.extend(str(index) for index in range(_LEGACY_CUDA_PROBE_LIMIT))
    # None 表示「device 不计入哈希」的中间口径；'cuda' / 'cpu' 是 device 计入哈希时的两种取值。
    device_variants: list[Any] = [None, _DEVICE_HASH_CONSTANT, "cpu"]
    cuda_values: list[Any] = [value for value in values if value is not None]
    # client_parallel 计入哈希的旧口径下，历史取值主要是 auto / 1 / 2（个别试点用过 3+）。
    cp_values: list[Any] = [param_dict.get("client_parallel"), 1, 2, 3, 4, "auto"]
    hashes: list[str] = []
    seen = {new_hash}

    def _add(candidate: str) -> None:
        if candidate not in seen:
            seen.add(candidate)
            hashes.append(candidate)

    for client_parallel in cp_values:
        for device in device_variants:
            # device 计入哈希时，-cuda 也可能同时在哈希里，两种口径都要覆盖。
            for value in cuda_values:
                _add(
                    _legacy_config_hash_including_cuda(
                        dict(param_dict, cuda=value), device=device,
                        client_parallel=client_parallel,
                    )
                )
            # cuda 已从哈希排除的口径下，枚举 cuda 取值只会重复算出同一个哈希。
            _add(
                _legacy_config_hash_including_cuda(
                    param_dict, device=device, include_cuda=False,
                    client_parallel=client_parallel,
                )
            )
    return hashes


def _checkpoint_next_round(path: Path) -> int | None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    value = payload.get("next_round") if isinstance(payload, dict) else None
    return int(value) if isinstance(value, int) else None


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
            if destination.exists():
                # 同一实验在旧口径下可能同时留下 GPU 与 CPU 两份产物：两者都合法，取进度
                # （next_round）更大的一份，避免迁移把已经跑到 200 轮的存档换成 100 轮的。
                if name == "checkpoint_latest.pt":
                    stored_round = _checkpoint_next_round(destination)
                    incoming_round = _checkpoint_next_round(source)
                    if stored_round is not None and incoming_round is not None:
                        if stored_round >= incoming_round:
                            # 落后的一份已被取代，直接删掉回收空间（否则旧目录永远搬不空）。
                            source.unlink()
                            continue
                    elif destination.stat().st_mtime >= source.stat().st_mtime:
                        continue
                elif destination.stat().st_mtime >= source.stat().st_mtime:
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
        # 认领会重命名 experiment_state 子目录：必须先等异步写盘结束，
        # 否则后台线程会把临时文件写到已被搬走的旧目录里。
        flush_checkpoint_writes()
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
