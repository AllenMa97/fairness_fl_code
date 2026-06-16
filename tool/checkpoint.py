import os
import json
import torch
import random
import numpy as np
from tool.logger import logger


def get_checkpoint_path(param_dict, iter_t=None):
    """获取检查点路径"""
    checkpoint_dir = os.path.join(param_dict['model_path'], 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    if iter_t is not None:
        exp_no = param_dict.get('Experiment_NO', 0)
        return os.path.join(checkpoint_dir, f'checkpoint_repeat{exp_no}_round_{iter_t}.pt')
    return checkpoint_dir


def save_checkpoint(param_dict, iter_t, global_model, **kwargs):
    """保存检查点"""
    checkpoint_path = get_checkpoint_path(param_dict, iter_t)
    
    checkpoint = {
        'communication_round': iter_t,
        'total_gpu_seconds': kwargs.get('total_gpu_seconds', 0),
        'global_model_state': global_model.state_dict(),
        'client_selection_history': kwargs.get('client_selection_history', []),
        'random_seed_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
        'start_time': kwargs.get('start_time', ''),
        'extra_state': kwargs.get('extra_state', {}),
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved at round {iter_t}: {checkpoint_path}")
    
    return checkpoint_path


def load_checkpoint(param_dict, target_round=None):
    """加载检查点"""
    checkpoint_dir = get_checkpoint_path(param_dict)
    
    if not os.path.exists(checkpoint_dir):
        logger.info("No checkpoint directory found, starting from scratch")
        return None
    
    checkpoint_filename = None
    
    if target_round is None:
        # 查找当前 repeat 的最新检查点
        exp_no = param_dict.get('Experiment_NO', 0)
        prefix = f'checkpoint_repeat{exp_no}_round_'
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith(prefix)]
        if not checkpoints:
            # 兼容旧格式 checkpoint_round_X.pt（无 repeat 前缀）
            old_checkpoints = [f for f in os.listdir(checkpoint_dir) 
                             if f.startswith('checkpoint_round_') and not f.startswith('checkpoint_repeat')]
            if not old_checkpoints:
                logger.info("No checkpoints found, starting from scratch")
                return None
            checkpoints = old_checkpoints
        
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].replace('.pt', '')))
        latest_checkpoint = checkpoints[-1]
        target_round = int(latest_checkpoint.split('_')[-1].replace('.pt', ''))
        checkpoint_filename = latest_checkpoint  # 使用实际文件名（兼容新旧格式）
    else:
        exp_no = param_dict.get('Experiment_NO', 0)
        # 先尝试新格式，再回退旧格式
        new_name = f'checkpoint_repeat{exp_no}_round_{target_round}.pt'
        old_name = f'checkpoint_round_{target_round}.pt'
        new_path = os.path.join(checkpoint_dir, new_name)
        old_path = os.path.join(checkpoint_dir, old_name)
        if os.path.exists(new_path):
            checkpoint_filename = new_name
        elif os.path.exists(old_path):
            checkpoint_filename = old_name
        else:
            logger.warning(f"Checkpoint for round {target_round} not found")
            return None
    
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # 恢复随机种子状态
        random.setstate(checkpoint['random_seed_state'])
        np.random.set_state(checkpoint['numpy_random_state'])
        torch.set_rng_state(checkpoint['torch_random_state'])
        
        logger.info(f"Successfully loaded checkpoint from round {checkpoint['communication_round']}")
        return checkpoint
    
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None


def save_split_indices(param_dict, split_indices):
    """保存数据分割索引"""
    split_dir = os.path.join(param_dict['model_path'], 'split_info')
    os.makedirs(split_dir, exist_ok=True)
    
    split_info = {
        'split_strategy': param_dict.get('split_strategy', ''),
        'num_clients': param_dict.get('num_clients_K', 0),
        'indices': {str(k): v.tolist() if isinstance(v, np.ndarray) else list(v) 
                   for k, v in split_indices.items()}
    }
    
    with open(os.path.join(split_dir, 'split_indices.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    logger.info("Split indices saved")


def load_split_indices(param_dict):
    """加载数据分割索引"""
    split_path = os.path.join(param_dict['model_path'], 'split_info', 'split_indices.json')
    
    if not os.path.exists(split_path):
        return None
    
    try:
        with open(split_path, 'r') as f:
            split_info = json.load(f)
        
        # 验证分割策略是否匹配
        if split_info['split_strategy'] != param_dict.get('split_strategy', ''):
            logger.warning(f"Split strategy mismatch: stored={split_info['split_strategy']}, current={param_dict.get('split_strategy')}")
            return None
        
        # 验证客户端数量是否匹配
        if split_info['num_clients'] != param_dict.get('num_clients_K', 0):
            logger.warning(f"Client number mismatch: stored={split_info['num_clients']}, current={param_dict.get('num_clients_K')}")
            return None
        
        # 转换回numpy数组
        split_indices = {int(k): np.array(v) for k, v in split_info['indices'].items()}
        logger.info("Successfully loaded split indices")
        return split_indices
    
    except Exception as e:
        logger.error(f"Failed to load split indices: {e}")
        return None


def check_resume_status(param_dict):
    """检查是否可以从断点恢复"""
    checkpoint = load_checkpoint(param_dict)
    if checkpoint is None:
        return None
    
    current_round = checkpoint['communication_round']
    total_rounds = param_dict.get('communication_round_I', 0)
    
    if current_round >= total_rounds - 1:
        logger.info(f"Experiment already completed at round {current_round + 1}/{total_rounds}")
        return None
    
    logger.info(f"Resuming from round {current_round + 1}/{total_rounds}")
    return checkpoint


def clean_old_checkpoints(param_dict, keep_latest=5):
    """清理旧的检查点，保留最近的N个"""
    checkpoint_dir = get_checkpoint_path(param_dict)
    
    if not os.path.exists(checkpoint_dir):
        return
    
    exp_no = param_dict.get('Experiment_NO', 0)
    prefix = f'checkpoint_repeat{exp_no}_round_'
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith(prefix)]
    if not checkpoints:
        checkpoints = [f for f in os.listdir(checkpoint_dir) 
                      if f.startswith('checkpoint_round_') and not f.startswith('checkpoint_repeat')]
    if len(checkpoints) <= keep_latest:
        return
    
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].replace('.pt', '')))
    to_delete = checkpoints[:-keep_latest]
    
    for checkpoint in to_delete:
        os.remove(os.path.join(checkpoint_dir, checkpoint))
        logger.info(f"Cleaned old checkpoint: {checkpoint}")
