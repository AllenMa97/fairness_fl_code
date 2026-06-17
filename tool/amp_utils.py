"""
AMP (Automatic Mixed Precision) 统一控制工具

- torch.cuda.amp 是 PyTorch 内置模块，无需额外 pip install
- compute capability >= 7.0 (V100/T4/2080/A100/4090/5090 等) → 建议启用
- compute capability < 7.0 (P4/K80/CPU) → 禁用
- 通过 param_dict['use_amp'] 控制：True/False/"auto"
"""

import torch
import contextlib
import re


def _parse_cuda_device_id(device) -> int:
    """从 device 参数中提取 CUDA device id。支持 torch.device / "cuda" / "cuda:N" / int"""
    if isinstance(device, torch.device):
        return device.index if device.index is not None else 0
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        m = re.match(r'^cuda(?::(\d+))?$', device.strip().lower())
        if m:
            return int(m.group(1)) if m.group(1) is not None else 0
    return 0


def detect_amp_support(device) -> bool:
    """检测当前 GPU 是否建议启用 AMP（CC >= 7.0，即 Volta 及以上架构）"""
    if not torch.cuda.is_available():
        return False

    device_id = _parse_cuda_device_id(device)
    try:
        cc = torch.cuda.get_device_capability(device_id)
        major, minor = cc[0], cc[1]
        # Compute Capability >= 7.0 → FP16 Tensor Core (Volta/Turing/Ampere/Ada/Hopper/Blackwell)
        return major >= 7
    except Exception:
        # 获取 compute capability 失败 → 保守禁用
        return False


def resolve_amp_config(param_dict) -> bool:
    """
    根据 param_dict['use_amp'] 和当前硬件决定是否启用 AMP。
    
    param_dict['use_amp']:
      - True:  强制启用（⚠️ P4 上可能导致数值不稳定）
      - False:  强制禁用
      - "auto": 根据 GPU compute capability 自动决定（推荐）
      - 未设置:  默认 "auto"
    """
    use_amp_setting = param_dict.get('use_amp', 'auto')

    if isinstance(use_amp_setting, bool):
        return use_amp_setting

    if use_amp_setting != 'auto':
        # 尝试转为 bool
        if isinstance(use_amp_setting, str):
            return use_amp_setting.lower() in ('true', '1', 'yes', 'on')
        return bool(use_amp_setting)

    # auto 模式：检测 GPU
    device = param_dict.get('device', 'cpu')
    return detect_amp_support(device)


@contextlib.contextmanager
def autocast_context(device, enabled: bool = True):
    """
    AMP autocast 上下文管理器。
    包装模型 forward + loss 计算。
    """
    if enabled and torch.cuda.is_available():
        # FP16 autocast：CUDA ops 使用 FP16，数值敏感 ops 保留 FP32
        with torch.cuda.amp.autocast():
            yield
    else:
        # 纯 FP32
        yield


def get_scaler(device, enabled: bool = True):
    """
    获取 GradScaler。AMP 禁用时返回 None。
    """
    if enabled and torch.cuda.is_available():
        return torch.cuda.amp.GradScaler()
    return None


def amp_backward(loss, scaler=None, optimizer=None):
    """
    统一 backward + optimizer step（backward 和 step 合在一起用）。
    AMP 启用时用 scaler；禁用时直接 backward + step。
    """
    if scaler is not None:
        scaler.scale(loss).backward()
        if optimizer is not None:
            scaler.step(optimizer)
            scaler.update()
    else:
        loss.backward()
        if optimizer is not None:
            optimizer.step()


def scale_backward(loss, scaler=None):
    """
    仅做 backward，与 optimizer.step 分离（用于梯度累积场景）。
    AMP 启用时 scaler.scale(loss).backward()；禁用时 loss.backward()。
    """
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()


def scaler_step(scaler, optimizer):
    """
    仅做 optimizer step + scaler update（用于梯度累积场景）。
    AMP 启用时 scaler.step(optimizer) + scaler.update()；禁用时 optimizer.step()。
    """
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()


def clip_grad_norm(scaler, model_or_params, max_norm: float = 1.0):
    """
    AMP 兼容的梯度裁剪。

    AMP 启用时必须先 unscale_ 再裁剪，否则裁剪的是缩放后的梯度值（不正确）。
    禁用时直接调用标准 clip_grad_norm_。

    Args:
        scaler: GradScaler 实例或 None
        model_or_params: nn.Module 或参数迭代器
        max_norm: 最大梯度范数

    Returns:
        梯度的总范数（float）
    """
    if scaler is not None and torch.cuda.is_available():
        # AMP 模式：先 unscale 将梯度还原到 FP32 空间，再裁剪
        params = model_or_params.parameters() if hasattr(model_or_params, 'parameters') else model_or_params
        scaler.unscale_(optimizer=None)
        return torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
    else:
        params = model_or_params.parameters() if hasattr(model_or_params, 'parameters') else model_or_params
        return torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)


def clip_grad_norm_for_optimizer(scaler, optimizer, max_norm: float = 1.0):
    """
    针对特定 optimizer 的 AMP 兼容梯度裁剪。

    Args:
        scaler: GradScaler 实例或 None
        optimizer: 优化器实例
        max_norm: 最大梯度范数

    Returns:
        梯度的总范数（float）
    """
    if scaler is not None and torch.cuda.is_available():
        scaler.unscale_(optimizer)
        return torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=max_norm)
    else:
        return torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=max_norm)
