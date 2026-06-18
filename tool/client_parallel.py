"""
客户端并行训练模块 — 显存感知的 GPU 并行化

支持两种并行模式：
1. CUDA Stream 并发：同一 GPU 上用多个 CUDA Stream 并发执行多客户端训练
2. 多 GPU 分配：将不同客户端分配到不同 GPU 设备上并行训练

自动检测 GPU 显存，计算最大并行度，显存不足时自动降级为串行。
完全向后兼容，不影响算法的数学逻辑和聚合方式。

用法：
    from tool.client_parallel import ClientParallelExecutor

    executor = ClientParallelExecutor(
        device=device,
        global_model=global_model,
        param_dict=param_dict,
    )

    # 在算法的客户端训练循环处替换：
    # 原始: for id in idxs_users: train_client(id, ...)
    # 替换: executor.run_clients(idxs_users, train_client_fn, ...)
"""

import os
import gc
import copy
import json
import time
import math
import torch
import psutil
import logging
import threading
from typing import Callable, List, Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 资源历史记录文件路径（与 client_parallel.py 同目录）
# ---------------------------------------------------------------------------
_HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.resource_history')
_HISTORY_FILE = os.path.join(_HISTORY_DIR, 'resource_usage_history.json')
_MAX_RECORDS_PER_KEY = 20  # 每个配置键最多保留的历史条数


# ===================================================================
# ResourceMonitor — 使用 CUDA API 真实记录显存峰值
# ===================================================================

class ResourceMonitor:
    """
    跨平台资源监控器（Mac / Win / Linux）。

    - GPU 显存：使用 torch.cuda.max_memory_allocated() 获取真实峰值
    - 系统 RAM：使用 psutil 获取进程 RSS 峰值 + 系统级内存使用
    - 后台线程高频采样，确保捕获瞬时峰值
    """

    def __init__(self, device):
        self.device = device
        self.device_id = _parse_device_id(device)
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        # 采样数据
        self._sample_vram: List[float] = []
        self._sample_rss_mb: List[float] = []
        self._sample_sys_used_mb: List[float] = []
        self._sample_times: List[float] = []
        self._sample_lock = threading.Lock()
        # 进程对象（用于 RSS 采样）
        self._process = psutil.Process()

    def start(self):
        """启动监控：重置 CUDA 峰值统计 + 记录初始 RAM + 启动后台采样线程"""
        self._monitoring = True

        # CUDA 峰值重置
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_id)
            torch.cuda.synchronize(self.device_id)

        # 记录初始 RSS（进程自身内存）
        self._initial_rss_mb = self._process.memory_info().rss / (1024 ** 2)
        self._initial_sys_used_mb = psutil.virtual_memory().used / (1024 ** 2)

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        """
        停止监控，返回本次运行的完整资源统计。

        Returns:
            {
                # GPU 显存（仅 CUDA 可用时）
                'peak_vram_mb': float,           # CUDA API 真实峰值
                'sample_peak_vram_mb': float,    # 采样峰值
                'sample_avg_vram_mb': float,     # 采样均值

                # 系统 RAM（跨平台，Mac/Win/Linux）
                'peak_rss_mb': float,            # 进程 RSS 峰值增量
                'peak_sys_used_mb': float,       # 系统总内存使用峰值增量
                'sample_avg_rss_mb': float,       # 采样平均 RSS

                # 通用
                'sample_count': int,
                'duration_sec': float,           # 由调用方填充
            }
        """
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        result: Dict[str, Any] = {'sample_count': 0}

        # ---- GPU 显存 ----
        if torch.cuda.is_available():
            cuda_peak = torch.cuda.max_memory_allocated(self.device_id) / (1024 ** 2)
            result['peak_vram_mb'] = round(cuda_peak, 2)
        else:
            result['peak_vram_mb'] = 0

        with self._sample_lock:
            result['sample_peak_vram_mb'] = round(
                max(self._sample_vram) if self._sample_vram else 0, 2)
            result['sample_avg_vram_mb'] = round(
                (sum(self._sample_vram) / len(self._sample_vram))
                if self._sample_vram else 0, 2)

            # ---- 系统 RAM ----
            # 进程 RSS 峰值增量（减去初始值，得到本轮训练净增内存）
            rss_values = self._sample_rss_mb
            if rss_values:
                peak_rss = max(rss_values)
                result['peak_rss_mb'] = round(peak_rss - self._initial_rss_mb, 2)
                result['sample_avg_rss_mb'] = round(
                    (sum(rss_values) / len(rss_values)) - self._initial_rss_mb, 2)
            else:
                result['peak_rss_mb'] = 0
                result['sample_avg_rss_mb'] = 0

            # 系统级内存使用峰值增量
            sys_values = self._sample_sys_used_mb
            if sys_values:
                peak_sys = max(sys_values)
                result['peak_sys_used_mb'] = round(peak_sys - self._initial_sys_used_mb, 2)
            else:
                result['peak_sys_used_mb'] = 0

            result['sample_count'] = len(self._sample_vram)

        return result

    def _sample_loop(self):
        """后台线程：每 50ms 采样一次 GPU 显存 + 系统 RAM"""
        while self._monitoring:
            try:
                # GPU 显存
                if torch.cuda.is_available():
                    vram = torch.cuda.memory_allocated(self.device_id) / (1024 ** 2)
                else:
                    vram = 0

                # 进程 RSS
                rss_mb = self._process.memory_info().rss / (1024 ** 2)

                # 系统总内存使用
                sys_used_mb = psutil.virtual_memory().used / (1024 ** 2)

                with self._sample_lock:
                    self._sample_vram.append(vram)
                    self._sample_rss_mb.append(rss_mb)
                    self._sample_sys_used_mb.append(sys_used_mb)
                    self._sample_times.append(time.time())
            except Exception:
                pass
            time.sleep(0.05)


# ===================================================================
# ResourceEstimator — 历史记录 + 迭代学习
# ===================================================================

class ResourceEstimator:
    """
    资源估算器：基于历史真实记录迭代学习，越用越准。

    - 每次训练后记录真实显存峰值、耗时、客户端数等
    - 下次遇到相似配置时，优先查历史记录
    - 无历史记录时退回到公式估算
    - 历史记录持久化到 JSON 文件，跨进程共享
    """

    def __init__(self):
        self._history: Dict[str, dict] = {}
        self._load_history()

    # ---------- 配置键生成 ----------

    @staticmethod
    def make_config_key(param_dict: dict, model) -> str:
        """
        根据实验参数和模型生成配置唯一标识。

        相同的 model 架构 + 数据集 + batch_size + 优化器 + 任务类型
        视为"近似配置"，可复用历史记录。
        """
        model_class = model.__class__.__name__ if model is not None else 'unknown'
        dataset = param_dict.get('dataset_name', 'unknown')
        batch_size = param_dict.get('batch_size', 64)
        optimizer = param_dict.get('optimize_method', 'sgd')
        task = param_dict.get('task', 'unknown')
        use_amp = param_dict.get('use_amp', False)

        # 模型参数量（粗粒度，区分不同规模的模型）
        param_count = 0
        if model is not None:
            try:
                param_count = sum(p.numel() for p in model.parameters())
            except Exception:
                pass

        return (f"{model_class}|{dataset}|bs{batch_size}|{optimizer}"
                f"|{task}|amp{use_amp}|params{param_count}")

    # ---------- 估算 ----------

    def estimate_single_client_vram(self, param_dict: dict, model) -> float:
        """
        估算单个客户端训练所需显存 (MB)。

        优先使用历史记录，无记录时退回公式估算。
        """
        config_key = self.make_config_key(param_dict, model)
        record = self._history.get(config_key)

        if record and record.get('total_runs', 0) > 0:
            # 有历史记录：使用历史平均单客户端显存
            avg_total = record['stats'].get('avg_total_peak_vram_mb', 0)
            avg_clients = record['stats'].get('avg_num_clients', 1)
            if avg_total > 0 and avg_clients > 0:
                estimated = avg_total / avg_clients
                logger.debug(f"[ResourceEstimator] Using history for {config_key}: "
                             f"{estimated:.1f} MB/client (from {avg_clients} clients avg)")
                return estimated

        # 无历史记录：公式估算
        estimated = _estimate_single_client_vram_mb(model, param_dict)
        logger.debug(f"[ResourceEstimator] Using formula for {config_key}: "
                     f"{estimated:.1f} MB/client")
        return estimated

    # ---------- 记录 ----------

    def record_actual_usage(self, param_dict: dict, model,
                             resource_stats: Dict[str, Any],
                             num_clients: int,
                             duration_sec: float):
        """
        记录一次真实运行的资源使用情况。

        Args:
            param_dict: 实验参数
            model: 模型
            resource_stats: ResourceMonitor.stop() 返回的统计
            num_clients: 本轮客户端数量
            duration_sec: 本轮耗时（秒）
        """
        config_key = self.make_config_key(param_dict, model)

        new_record = {
            'timestamp': datetime.now().isoformat(),
            # GPU 显存
            'peak_vram_mb': resource_stats.get('peak_vram_mb', 0),
            'sample_peak_vram_mb': resource_stats.get('sample_peak_vram_mb', 0),
            'sample_avg_vram_mb': resource_stats.get('sample_avg_vram_mb', 0),
            # 系统 RAM
            'peak_rss_mb': resource_stats.get('peak_rss_mb', 0),
            'peak_sys_used_mb': resource_stats.get('peak_sys_used_mb', 0),
            'sample_avg_rss_mb': resource_stats.get('sample_avg_rss_mb', 0),
            # 通用
            'num_clients': num_clients,
            'duration_sec': round(duration_sec, 3),
            'vram_per_client': round(
                resource_stats.get('peak_vram_mb', 0) / max(num_clients, 1), 2),
            'rss_per_client': round(
                resource_stats.get('peak_rss_mb', 0) / max(num_clients, 1), 2),
        }

        if config_key not in self._history:
            self._history[config_key] = {
                'config_key': config_key,
                'records': [],
                'stats': {},
            }

        records = self._history[config_key]['records']
        records.append(new_record)

        # 保留最近 N 条记录
        if len(records) > _MAX_RECORDS_PER_KEY:
            self._history[config_key]['records'] = records[-_MAX_RECORDS_PER_KEY:]

        # 更新聚合统计
        self._update_stats(config_key)
        self._save_history()

        logger.debug(f"[ResourceEstimator] Recorded: {config_key}, "
                     f"peak={new_record['peak_vram_mb']:.1f}MB, "
                     f"clients={num_clients}, duration={duration_sec:.2f}s")

    def _update_stats(self, config_key: str):
        """根据所有记录重新计算聚合统计"""
        records = self._history[config_key]['records']
        if not records:
            return

        n = len(records)
        self._history[config_key]['stats'] = {
            'total_runs': n,
            # GPU 显存
            'avg_total_peak_vram_mb': round(
                sum(r['peak_vram_mb'] for r in records) / n, 2),
            'avg_vram_per_client': round(
                sum(r['vram_per_client'] for r in records) / n, 2),
            'max_peak_vram_mb': round(
                max(r['peak_vram_mb'] for r in records), 2),
            'min_peak_vram_mb': round(
                min(r['peak_vram_mb'] for r in records), 2),
            # 系统 RAM
            'avg_peak_rss_mb': round(
                sum(r['peak_rss_mb'] for r in records) / n, 2),
            'avg_rss_per_client': round(
                sum(r['rss_per_client'] for r in records) / n, 2),
            'max_peak_rss_mb': round(
                max(r['peak_rss_mb'] for r in records), 2),
            # 通用
            'avg_num_clients': round(
                sum(r['num_clients'] for r in records) / n, 1),
            'avg_duration_sec': round(
                sum(r['duration_sec'] for r in records) / n, 3),
            'last_updated': datetime.now().isoformat(),
        }

    # ---------- 持久化 ----------

    def _load_history(self):
        """从 JSON 文件加载历史记录"""
        try:
            if os.path.exists(_HISTORY_FILE):
                with open(_HISTORY_FILE, 'r', encoding='utf-8') as f:
                    self._history = json.load(f)
                logger.debug(f"[ResourceEstimator] Loaded history: "
                             f"{len(self._history)} config keys")
        except Exception as e:
            logger.warning(f"[ResourceEstimator] Failed to load history: {e}")
            self._history = {}

    def _save_history(self):
        """保存历史记录到 JSON 文件"""
        try:
            os.makedirs(_HISTORY_DIR, exist_ok=True)
            with open(_HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"[ResourceEstimator] Failed to save history: {e}")


def _estimate_single_client_vram_mb(model, param_dict: dict) -> float:
    """
    估算单个客户端训练所需的 GPU 显存 (MB)。

    包含：模型参数 + 梯度 + 优化器状态 + 激活值 + batch 数据
    """
    # 模型参数显存
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_mb = param_bytes / (1024 ** 2)

    # 梯度显存（与参数等大）
    gradient_mb = param_mb

    # 优化器状态显存
    opt_method = param_dict.get('optimize_method', 'sgd')
    if opt_method and 'adam' in opt_method.lower():
        optimizer_mb = param_mb * 2.0
    elif opt_method in ('rmsprop', 'adagrad', 'adadelta'):
        optimizer_mb = param_mb * 1.0
    else:
        optimizer_mb = 0.0

    # 激活值显存（经验估算，AMP 开启时减半）
    use_amp = param_dict.get('use_amp', False)
    batch_size = param_dict.get('batch_size', 64)
    activation_factor = 2 if use_amp else 4
    activation_mb = param_mb * activation_factor * (batch_size / 256)

    # batch 数据显存
    task = param_dict.get('task', '')
    if 'SENT_CLF' in task:
        seq_len = param_dict.get('max_len', 128)
        data_mb = batch_size * seq_len * 768 * 4 / (1024 ** 2)
    elif 'IMG_CLF' in task:
        data_mb = batch_size * 3 * 224 * 224 * 4 / (1024 ** 2)
    else:
        data_mb = batch_size * 100 * 4 / (1024 ** 2)

    total_mb = param_mb + gradient_mb + optimizer_mb + activation_mb + data_mb
    # 20% 安全余量
    total_mb *= 1.2

    return total_mb


def _get_available_gpu_memory_mb(device) -> float:
    """获取指定 GPU 设备的可用显存 (MB)"""
    if not torch.cuda.is_available():
        return 0

    try:
        device_id = _parse_device_id(device)
        props = torch.cuda.get_device_properties(device_id)
        total_mb = props.total_mem / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved(device_id) / (1024 ** 2)
        allocated_mb = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
        available_mb = total_mb - max(reserved_mb, allocated_mb)
        return available_mb
    except Exception:
        return 0


def _parse_device_id(device) -> int:
    """从 device 字符串/整数/torch.device 中提取 CUDA 设备 ID"""
    import re
    if isinstance(device, int):
        return device
    if isinstance(device, torch.device):
        return device.index if device.index is not None else 0
    if isinstance(device, str):
        m = re.match(r'^cuda(?::(\d+))?$', device.strip().lower())
        if m:
            return int(m.group(1)) if m.group(1) is not None else 0
    return 0


def _get_all_cuda_devices() -> List[int]:
    """获取所有可用的 CUDA 设备 ID 列表"""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def resolve_parallel_config(param_dict: dict, model=None,
                            estimator: Optional['ResourceEstimator'] = None) -> dict:
    """
    解析客户端并行配置。

    Args:
        param_dict: 实验参数字典
        model: 全局模型（用于估算显存，可选）
        estimator: ResourceEstimator 实例（可选，用于查历史记录）

    Returns:
        dict: {
            'enabled': bool,          # 是否启用并行
            'mode': str,              # 'stream' | 'multi_gpu' | 'auto'
            'max_parallel': int,       # 最大并行客户端数
            'gpu_devices': list,      # 可用 GPU 设备 ID 列表
        }
    """
    client_parallel = param_dict.get('client_parallel', 'auto')

    # 解析配置值
    if isinstance(client_parallel, bool):
        enabled = client_parallel
        mode = 'auto'
    elif isinstance(client_parallel, int):
        enabled = client_parallel > 1
        mode = 'auto'
    elif isinstance(client_parallel, str):
        if client_parallel.lower() in ('off', 'false', '0', '1', 'none'):
            enabled = False
            mode = 'auto'
        else:
            enabled = True
            mode = client_parallel.lower()
    else:
        enabled = False
        mode = 'auto'

    if not enabled:
        return {'enabled': False, 'mode': 'serial', 'max_parallel': 1, 'gpu_devices': []}

    # 获取可用 GPU
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible:
        gpu_devices = [int(x.strip()) for x in cuda_visible.split(',') if x.strip().isdigit()]
    else:
        gpu_devices = _get_all_cuda_devices()

    if not gpu_devices:
        logger.info("[ClientParallel] No GPU available, falling back to serial")
        return {'enabled': False, 'mode': 'serial', 'max_parallel': 1, 'gpu_devices': []}

    # 自动计算最大并行度
    max_parallel = 1
    if model is not None:
        # 优先使用历史记录估算，无记录时退回公式
        if estimator is not None:
            single_client_vram = estimator.estimate_single_client_vram(param_dict, model)
        else:
            single_client_vram = _estimate_single_client_vram_mb(model, param_dict)

        if mode == 'multi_gpu' and len(gpu_devices) > 1:
            # 多 GPU 模式：每个 GPU 上跑一个客户端
            max_parallel = len(gpu_devices)
        elif mode in ('stream', 'auto'):
            # CUDA Stream 模式：同一 GPU 上并发
            # 使用第一个可用 GPU 估算
            primary_device = gpu_devices[0]
            available_vram = _get_available_gpu_memory_mb(primary_device)
            if single_client_vram > 0 and available_vram > 0:
                max_parallel = max(1, int(available_vram / single_client_vram))
            else:
                # 无法估算时，根据 GPU 显存总量做保守估算
                try:
                    props = torch.cuda.get_device_properties(primary_device)
                    total_gb = props.total_mem / (1024 ** 3)
                    if total_gb >= 80:
                        max_parallel = 4
                    elif total_gb >= 48:
                        max_parallel = 3
                    elif total_gb >= 24:
                        max_parallel = 2
                    else:
                        max_parallel = 1
                except Exception:
                    max_parallel = 1

            # 多 GPU 叠加
            if len(gpu_devices) > 1 and mode == 'auto':
                max_parallel = max(max_parallel, len(gpu_devices))
        else:
            max_parallel = 1

    # 限制最大并行度（避免过多并发导致调度开销）
    max_parallel = min(max_parallel, 8)

    actual_mode = 'multi_gpu' if (len(gpu_devices) > 1 and max_parallel > 1) else 'stream'
    if max_parallel <= 1:
        actual_mode = 'serial'

    config = {
        'enabled': max_parallel > 1,
        'mode': actual_mode,
        'max_parallel': max_parallel,
        'gpu_devices': gpu_devices,
    }

    logger.info(f"[ClientParallel] Config: mode={actual_mode}, max_parallel={max_parallel}, "
                f"gpus={gpu_devices}")

    return config


class ClientParallelExecutor:
    """
    客户端并行执行器

    在联邦学习的每个通信轮次中，将选中的客户端训练任务并行化执行。

    支持两种模式：
    - stream: 同一 GPU 上用 CUDA Stream 并发
    - multi_gpu: 不同客户端分配到不同 GPU

    对于在训练中依赖 global_model 的算法（如 PDFFed），
    会自动降级为串行模式，确保正确性。
    """

    def __init__(self, device, global_model, param_dict: dict,
                 needs_global_model_during_training: bool = False):
        """
        Args:
            device: 主设备（如 'cuda' 或 'cuda:0'）
            global_model: 全局模型
            param_dict: 实验参数字典
            needs_global_model_during_training: 算法在客户端训练中是否需要访问 global_model
                （如 PDFFed 用 global_model.only_clf_forward），若为 True 则自动降级串行
        """
        self.device = device
        self.global_model = global_model
        self.param_dict = param_dict
        self.needs_global_model_during_training = needs_global_model_during_training

        # 资源估算器（全局单例，跨轮次共享历史）
        self._estimator = ResourceEstimator()

        self.config = resolve_parallel_config(param_dict, global_model, self._estimator)

        # 如果算法在训练中依赖 global_model，强制串行
        if self.config['enabled'] and needs_global_model_during_training:
            logger.info("[ClientParallel] Algorithm needs global_model during client training, "
                        "forcing serial mode for correctness")
            self.config['enabled'] = False
            self.config['mode'] = 'serial'
            self.config['max_parallel'] = 1

        # CUDA Stream 池
        self._streams: List[torch.cuda.Stream] = []

        # 资源监控器（每轮重新创建）
        self._monitor: Optional[ResourceMonitor] = None

    @property
    def enabled(self) -> bool:
        return self.config['enabled']

    @property
    def max_parallel(self) -> int:
        return self.config['max_parallel']

    def _ensure_streams(self, n: int):
        """确保有足够的 CUDA Stream"""
        while len(self._streams) < n:
            self._streams.append(torch.cuda.Stream())

    def run_clients(self,
                    idxs_users: List[int],
                    train_fn: Callable,
                    **train_kwargs) -> List[Any]:
        """
        并行或串行执行客户端训练，同时监控并记录真实资源消耗。

        Args:
            idxs_users: 选中的客户端 ID 列表
            train_fn: 单客户端训练函数，签名：
                train_fn(client_id, device, global_model_copy, **train_kwargs) -> result
            **train_kwargs: 传递给 train_fn 的额外参数

        Returns:
            List[Any]: 每个客户端的训练结果，顺序与 idxs_users 一致
        """
        # 启动资源监控
        self._monitor = ResourceMonitor(self.device)
        self._monitor.start()
        start_time = time.time()

        try:
            if not self.config['enabled'] or len(idxs_users) <= 1:
                results = self._run_serial(idxs_users, train_fn, **train_kwargs)
            elif self.config['mode'] == 'multi_gpu' and len(self.config['gpu_devices']) > 1:
                results = self._run_multi_gpu(idxs_users, train_fn, **train_kwargs)
            else:
                results = self._run_stream_parallel(idxs_users, train_fn, **train_kwargs)
        finally:
            # 停止监控，获取真实资源统计
            resource_stats = self._monitor.stop()
            duration_sec = time.time() - start_time

            # 记录到历史（异步写入，不阻塞训练）
            try:
                self._estimator.record_actual_usage(
                    param_dict=self.param_dict,
                    model=self.global_model,
                    resource_stats=resource_stats,
                    num_clients=len(idxs_users),
                    duration_sec=duration_sec,
                )
            except Exception as e:
                logger.debug(f"[ClientParallel] Failed to record resource usage: {e}")

        return results

    def _run_serial(self, idxs_users: List[int],
                    train_fn: Callable, **train_kwargs) -> List[Any]:
        """串行执行（原始逻辑，完全不变）"""
        results = []
        for client_id in idxs_users:
            model_copy = copy.deepcopy(self.global_model)
            result = train_fn(client_id, self.device, model_copy, **train_kwargs)
            results.append(result)
            del model_copy
            gc.collect()
        return results

    def _run_stream_parallel(self, idxs_users: List[int],
                             train_fn: Callable, **train_kwargs) -> List[Any]:
        """
        CUDA Stream 并发执行。

        将客户端分成批次，每批最多 max_parallel 个客户端，
        在同一 GPU 上用不同 CUDA Stream 并发执行。
        """
        batch_size = self.config['max_parallel']
        all_results = [None] * len(idxs_users)

        for batch_start in range(0, len(idxs_users), batch_size):
            batch_ids = idxs_users[batch_start:batch_start + batch_size]
            n_clients = len(batch_ids)

            self._ensure_streams(n_clients)

            # 为每个客户端创建模型副本
            model_copies = [copy.deepcopy(self.global_model) for _ in range(n_clients)]

            # 在各自的 Stream 上启动训练
            for i, (client_id, model_copy) in enumerate(zip(batch_ids, model_copies)):
                stream = self._streams[i]
                with torch.cuda.stream(stream):
                    result = train_fn(client_id, self.device, model_copy, **train_kwargs)
                    all_results[batch_start + i] = result

            # 等待所有 Stream 完成
            torch.cuda.synchronize(self.device)

            # 清理
            for model_copy in model_copies:
                del model_copy
            gc.collect()

        return all_results

    def _run_multi_gpu(self, idxs_users: List[int],
                       train_fn: Callable, **train_kwargs) -> List[Any]:
        """
        多 GPU 并行执行。

        将客户端分配到不同 GPU 上并行训练。
        使用 CUDA Stream 在每个 GPU 上并发。
        """
        gpu_devices = self.config['gpu_devices']
        n_gpus = len(gpu_devices)
        all_results = [None] * len(idxs_users)

        # 分批：每批最多 n_gpus 个客户端
        for batch_start in range(0, len(idxs_users), n_gpus):
            batch_ids = idxs_users[batch_start:batch_start + n_gpus]
            n_clients = len(batch_ids)

            # 为每个客户端分配 GPU 和 Stream
            model_copies = []
            streams = []
            devices = []

            for i, client_id in enumerate(batch_ids):
                gpu_id = gpu_devices[i % n_gpus]
                device = torch.device(f'cuda:{gpu_id}')
                devices.append(device)
                model_copy = copy.deepcopy(self.global_model)
                model_copies.append(model_copy)
                streams.append(torch.cuda.Stream(device=device))

            # 在各自的 GPU + Stream 上启动训练
            for i, (client_id, model_copy, device, stream) in enumerate(
                    zip(batch_ids, model_copies, devices, streams)):
                with torch.cuda.stream(stream):
                    result = train_fn(client_id, device, model_copy, **train_kwargs)
                    all_results[batch_start + i] = result

            # 等待所有 GPU 完成
            for device in devices:
                torch.cuda.synchronize(device)

            # 清理
            for model_copy in model_copies:
                del model_copy
            gc.collect()

        return all_results
