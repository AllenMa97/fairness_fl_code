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
