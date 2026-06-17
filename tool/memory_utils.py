import os
import platform
import socket
import psutil
import torch


def get_system_info():
    """获取系统硬件信息"""
    info = {
        'has_gpu': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count(),
        'cpu_count': os.cpu_count() or 0,
        'total_memory_gb': psutil.virtual_memory().total / (1024 ** 3),
        'available_memory_gb': psutil.virtual_memory().available / (1024 ** 3),
        'used_memory_gb': psutil.virtual_memory().used / (1024 ** 3),
        'memory_percent': psutil.virtual_memory().percent,
    }
    return info


def get_hardware_profile():
    """
    获取完整的硬件画像（用于贡献者注册和统计）

    Returns:
        dict: 包含CPU、GPU、内存、磁盘、操作系统等详细信息
    """
    profile = {
        'hostname': socket.gethostname(),
        'os': platform.system(),
        'os_version': platform.version(),
        'python_version': platform.python_version(),
        'cpu_count': os.cpu_count() or 0,
        'cpu_model': _get_cpu_model(),
        'total_memory_gb': round(psutil.virtual_memory().total / (1024 ** 3), 2),
        'gpu_info': _get_gpu_info(),
        'disk_free_gb': round(_get_disk_free(), 2),
    }
    return profile


def _get_cpu_model():
    """获取CPU型号"""
    try:
        if platform.system() == 'Linux':
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        return line.split(':')[1].strip()
        elif platform.system() == 'Windows':
            import subprocess
            result = subprocess.run(
                'wmic cpu get name',
                shell=True, capture_output=True, text=True
            )
            lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
            if len(lines) > 1:
                return lines[1]
        elif platform.system() == 'Darwin':
            import subprocess
            result = subprocess.run(
                'sysctl -n machdep.cpu.brand_string',
                shell=True, capture_output=True, text=True
            )
            return result.stdout.strip()
    except Exception:
        pass
    return "Unknown"


def _get_gpu_info():
    """获取GPU详细信息"""
    if not torch.cuda.is_available():
        return []

    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append({
            'id': i,
            'name': props.name,
            'vram_gb': round(props.total_mem / (1024 ** 3), 2),
            'major': props.major,
            'minor': props.minor,
        })
    return gpus


def _get_disk_free():
    """获取工作目录所在磁盘的可用空间（GB）"""
    try:
        usage = psutil.disk_usage(os.getcwd())
        return usage.free / (1024 ** 3)
    except Exception:
        return 0


def print_hardware_profile(profile=None):
    """打印硬件画像"""
    if profile is None:
        profile = get_hardware_profile()

    print(f"\n{'='*50}")
    print(f"[Hardware Profile] {profile['hostname']}")
    print(f"{'='*50}")
    print(f"  OS:         {profile['os']} {profile['os_version']}")
    print(f"  Python:     {profile['python_version']}")
    print(f"  CPU:        {profile['cpu_count']} cores - {profile['cpu_model']}")
    print(f"  Memory:     {profile['total_memory_gb']} GB")
    print(f"  Disk Free:  {profile['disk_free_gb']} GB")

    gpus = profile['gpu_info']
    if gpus:
        for gpu in gpus:
            print(f"  GPU {gpu['id']}:    {gpu['name']} ({gpu['vram_gb']} GB)")
    else:
        print(f"  GPU:        None")

    print(f"{'='*50}")


def should_use_pin_memory(
    min_memory_gb: float = 8.0,
    max_memory_percent: float = 85.0,
    require_gpu: bool = True
) -> bool:
    """
    智能判断是否应该使用 pin_memory
    
    Args:
        min_memory_gb: 最小可用内存要求（GB）
        max_memory_percent: 最大内存使用率限制（%）
        require_gpu: 是否要求必须有GPU
    
    Returns:
        True: 应该使用 pin_memory
        False: 不应该使用 pin_memory
    """
    info = get_system_info()
    
    print(f"[PinMemory Check] System Info:")
    print(f"  GPU Available: {'Yes' if info['has_gpu'] else 'No'} ({info['gpu_count']} devices)")
    print(f"  CPU Cores: {info['cpu_count']}")
    print(f"  Total Memory: {info['total_memory_gb']:.2f} GB")
    print(f"  Used Memory: {info['used_memory_gb']:.2f} GB ({info['memory_percent']:.1f}%)")
    print(f"  Available Memory: {info['available_memory_gb']:.2f} GB")
    
    # 检查GPU
    if require_gpu and not info['has_gpu']:
        print(f"[PinMemory Check] ❌ No GPU available, disabling pin_memory")
        return False
    
    # 检查可用内存
    if info['available_memory_gb'] < min_memory_gb:
        print(f"[PinMemory Check] ❌ Available memory ({info['available_memory_gb']:.2f} GB) below threshold ({min_memory_gb} GB), disabling pin_memory")
        return False
    
    # 检查内存使用率
    if info['memory_percent'] > max_memory_percent:
        print(f"[PinMemory Check] ❌ Memory usage ({info['memory_percent']:.1f}%) above threshold ({max_memory_percent}%), disabling pin_memory")
        return False
    
    print(f"[PinMemory Check] ✅ All checks passed, enabling pin_memory")
    return True


def get_optimal_num_workers(
    max_workers: int = None,
    min_workers: int = 1
) -> int:
    """
    根据系统资源获取最优的 num_workers 值
    
    Args:
        max_workers: 最大工作线程数
        min_workers: 最小工作线程数
    
    Returns:
        推荐的 num_workers 值
    """
    cpu_count = os.cpu_count() or 1
    info = get_system_info()
    
    # 默认使用 CPU 核心数的一半
    optimal = max(min_workers, cpu_count // 2)
    
    # 如果内存紧张，减少 workers
    if info['memory_percent'] > 70:
        optimal = max(min_workers, optimal // 2)
    
    # 如果内存非常紧张，禁用多进程 workers（每个worker子进程会重新import PyTorch，内存翻倍）
    if info['memory_percent'] > 85:
        optimal = 0
        print(f"[Worker Check] Memory very tight ({info['memory_percent']:.1f}%), disabling multi-process workers")
    
    # 应用上限
    if max_workers is not None:
        optimal = min(optimal, max_workers)
    
    print(f"[Worker Check] Recommended num_workers: {optimal} (CPU cores: {cpu_count})")
    return optimal


def get_dataloader_config(
    pin_memory_default: bool = True,
    num_workers_default: int = None
) -> dict:
    """
    获取智能的 DataLoader 配置
    
    Args:
        pin_memory_default: pin_memory 的默认值
        num_workers_default: num_workers 的默认值（None 表示自动计算）
    
    Returns:
        包含 pin_memory 和 num_workers 的配置字典
    """
    pin_memory = should_use_pin_memory() if pin_memory_default else False
    
    if num_workers_default is None:
        num_workers = get_optimal_num_workers()
    else:
        num_workers = num_workers_default
    
    config = {
        'pin_memory': pin_memory,
        'num_workers': num_workers,
        'persistent_workers': pin_memory  # 有 pin_memory 时保持 worker 存活
    }
    
    print(f"\n[DataLoader Config] Final settings:")
    print(f"  pin_memory: {config['pin_memory']}")
    print(f"  num_workers: {config['num_workers']}")
    print(f"  persistent_workers: {config['persistent_workers']}")
    
    return config


def _get_optimizer_memory_factor(optimizer_method: str) -> float:
    """
    根据优化器类型返回优化器状态显存系数（相对于模型参数量）。

    | 优化器    | 系数 | 说明 |
    |----------|------|------|
    | sgd      | 0.0  | 无额外状态 |
    | adam/adamw | 2.0  | momentum + variance |
    | rmsprop  | 1.0  | 均方根缓存 |
    | adagrad  | 1.0  | 累加器缓存 |
    """
    method = optimizer_method.lower() if optimizer_method else 'sgd'
    if 'adam' in method:
        return 2.0
    elif method in ('rmsprop', 'adagrad', 'adadelta'):
        return 1.0
    else:
        return 0.0  # sgd / sparseadam / prox 等


def check_gpu_memory(model, device='cuda', batch_size=64,
                     image_size=None, seq_len=128, use_amp=False,
                     optimizer_method='sgd'):
    """
    GPU 显存预检 — 在训练开始前估算所需显存，避免跑到一半 OOM。
    根据实际使用的优化器类型动态估算显存，不会硬性按 Adam 算。

    Args:
        model: nn.Module 实例
        device: 设备字符串
        batch_size: 训练 batch_size
        image_size: 图像 (C,H,W) 元组，仅 IMG_CLF 需要
        seq_len: 文本序列长度，仅 SENT_CLF 需要
        use_amp: 是否启用 AMP（AMP 可减少约 50% 激活值显存）
        optimizer_method: 优化器类型（'sgd'/'adam'/'adamw'/'rmsprop'/'adagrad' 等）

    Returns:
        dict: {
            'ok': bool,           # 是否有足够显存
            'estimated_mb': float, # 预估总显存需求 (MB)
            'available_mb': float, # 当前可用显存 (MB)
            'utilization': float,  # 预估利用率 (0-1)
            'breakdown': dict,     # 各项明细
            'warning': str,        # 警告信息
        }
    """
    result = {
        'ok': True,
        'estimated_mb': 0,
        'available_mb': 0,
        'utilization': 0.0,
        'breakdown': {},
        'warning': '',
    }

    if not torch.cuda.is_available():
        result['warning'] = 'CUDA not available, skipping GPU memory check'
        return result

    device_id = _parse_cuda_device_id(device)

    # ---- 1. 模型参数显存 ----
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_mb = param_bytes / (1024 ** 2)
    result['breakdown']['model_params_mb'] = round(param_mb, 1)

    # ---- 2. 优化器状态显存（按实际优化器类型估算）----
    opt_factor = _get_optimizer_memory_factor(optimizer_method)
    optimizer_state_mb = param_mb * opt_factor
    result['breakdown']['optimizer_state_mb'] = round(optimizer_state_mb, 1)

    # ---- 3. 梯度显存 ----
    gradient_mb = param_mb
    result['breakdown']['gradients_mb'] = round(gradient_mb, 1)

    # ---- 4. 前向传播激活值显存（粗略估算）----
    # 经验公式：激活值 ≈ 2x ~ 4x 参数量（取决于模型深度和 batch）
    # AMP 开启时减半
    activation_factor = 2 if use_amp else 4
    activation_mb = param_mb * activation_factor * (batch_size / 256)  # 按 batch 归一化
    result['breakdown']['activations_mb'] = round(activation_mb, 1)

    # ---- 5. 单 batch 数据显存 ----
    data_mb = 0
    if image_size:
        c, h, w = image_size
        data_mb = batch_size * c * h * w * 4 / (1024 ** 2)  # FP32
    elif seq_len:
        data_mb = batch_size * seq_len * 768 * 4 / (1024 ** 2)  # BERT hidden=768
    else:
        data_mb = batch_size * 100 * 4 / (1024 ** 2)  # 表格数据估算
    result['breakdown']['batch_data_mb'] = round(data_mb, 1)

    # ---- 总计 ----
    total_mb = param_mb + optimizer_state_mb + gradient_mb + activation_mb + data_mb
    # 加 20% 安全余量
    total_mb *= 1.2
    result['estimated_mb'] = round(total_mb, 1)

    # ---- 可用显存 ----
    try:
        props = torch.cuda.get_device_properties(device_id)
        total_vram_gb = props.total_mem / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 2)
        allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
        available_mb = total_vram_gb * 1024 - max(reserved, allocated)
        result['available_mb'] = round(available_mb, 1)
    except Exception:
        available_mb = total_mb * 2  # 无法获取时假设充足
        result['available_mb'] = round(available_mb, 1)

    utilization = total_mb / (total_mb + available_mb) if (total_mb + available_mb) > 0 else 0
    result['utilization'] = round(utilization, 3)

    # ---- 判断 ----
    if total_mb > available_mb:
        result['ok'] = False
        shortage_pct = ((total_mb - available_mb) / available_mb) * 100
        result['warning'] = (
            f"[GPU Memory] ⚠️ 预估需要 {total_mb:.0f}MB，可用 {available_mb:.0f}MB "
            f"(缺口 {shortage_pct:.0f}%)\n"
            f"  建议：减小 batch_size、开启 AMP(use_amp=true)、或使用更大显存的 GPU"
        )
    elif utilization > 0.85:
        result['ok'] = True
        result['warning'] = (
            f"[GPU Memory] ⚡ 预估占用 {utilization*100:.0f}% 显存 ({total_mb:.0f}MB / {available_mb+total_mb:.0f}MB)\n"
            f"  接近上限，建议开启 AMP 或减小 batch_size"
        )
    else:
        result['warning'] = (
            f"[GPU Memory] ✅ 预估 {total_mb:.0f}MB / {available_mb+total_mb:.0f}MB "
            f"({utilization*100:.0f%})，显存充足"
        )

    print(result['warning'])
    print(f"  明细: {result['breakdown']}")

    return result


def _parse_cuda_device_id(device) -> int:
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
