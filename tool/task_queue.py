"""
分布式实验任务队列（v3）

改进点：
  1. 逐任务锁文件：每个任务有独立的锁文件，避免竞态条件
  2. 心跳机制：Worker定期更新锁文件时间戳，超时自动回收
  3. 云端Checkpoint路径管理：支持断点续跑
  4. 用户身份追踪：记录谁跑了哪些实验
  5. 贡献者注册：记录每个贡献者的硬件信息
  6. 实验可复现性：记录git hash、Python版本、依赖信息
  7. 结果自动校验：检查指标合理性
  8. 数据完整性校验：MD5 manifest
  9. 任务优先级：priority字段，默认FIFO

存储结构：
  queue/pending.json       : 待执行任务列表（含priority字段）
  queue/locks/{task_id}.json : 逐任务锁文件（含user_id + machine_id + 心跳）
  queue/done.json          : 已完成任务列表（含user_id + 耗时 + 环境信息）
  contributors.json        : 贡献者注册表（用户名 + 硬件信息）
  checkpoints/{task_id}/    : 云端checkpoint目录
  data_manifest.json       : 数据集MD5校验清单
"""

import os
import json
import time
import socket
import hashlib
import subprocess
from tool.cloud_storage import get_storage

# ============================================================
# 路径常量
# ============================================================
QUEUE_DIR = "queue"
LOCKS_DIR = f"{QUEUE_DIR}/locks"
PENDING_PATH = f"{QUEUE_DIR}/pending.json"
DONE_PATH = f"{QUEUE_DIR}/done.json"
CHECKPOINT_DIR = "checkpoints"
CONTRIBUTORS_PATH = "contributors.json"
DATA_MANIFEST_PATH = "data_manifest.json"

# 超时时间（秒）
DEFAULT_TIMEOUT = 7200       # 2小时无心跳则回收
HEARTBEAT_INTERVAL = 300      # 5分钟心跳一次
MAX_RETRIES = 3               # 最大重试次数


def _get_user_id():
    """获取当前用户标识"""
    try:
        from tool.user_config import USER_NAME
        if USER_NAME:
            return USER_NAME
    except ImportError:
        pass
    return os.environ.get('FL_USER_NAME', os.environ.get('USER', os.environ.get('USERNAME', 'unknown')))


def _get_user_profile():
    """获取用户完整学术身份信息"""
    profile = {}
    fields = [
        ('USER_EMAIL', 'FL_USER_EMAIL', 'email'),
        ('USER_AFFILIATION', '', 'affiliation'),
        ('USER_OPENREVIEW', 'FL_OPENREVIEW', 'openreview'),
        ('USER_GOOGLE_SCHOLAR', 'FL_GOOGLE_SCHOLAR', 'google_scholar'),
        ('USER_ORCID', 'FL_ORCID', 'orcid'),
        ('USER_GITHUB', 'FL_GITHUB', 'github'),
    ]
    for config_key, env_key, json_key in fields:
        try:
            import tool.user_config as _uc
            val = getattr(_uc, config_key, '')
        except ImportError:
            val = ''
        if not val and env_key:
            val = os.environ.get(env_key, '')
        if val:
            profile[json_key] = val
    return profile


def _get_machine_label():
    """获取机器标签"""
    try:
        from tool.user_config import MACHINE_LABEL
        if MACHINE_LABEL:
            return MACHINE_LABEL
    except ImportError:
        pass
    return os.environ.get('FL_MACHINE_LABEL', '')


def _get_machine_id():
    """获取当前机器的唯一标识"""
    hostname = socket.gethostname()
    pid = os.getpid()
    label = _get_machine_label()
    if label:
        return f"{label}_{hostname}_{pid}"
    return f"{hostname}_{pid}"


def _load_json(path):
    """从云存储加载JSON"""
    storage = get_storage()
    if storage.exists(path):
        return storage.read_json(path)
    return []


def _save_json(path, data):
    """保存JSON到云存储"""
    storage = get_storage()
    storage.write_json(path, data)


# ============================================================
# 任务生成（不变）
# ============================================================

def generate_all_tasks(
    algorithms=None,
    tabular_datasets=None,
    img_datasets=None,
    sent_datasets=None,
    splits=None,
    client_counts=None,
    repeat_times=3,
):
    """生成所有实验任务列表"""
    if algorithms is None:
        algorithms = ["LoGoFair", "PraFFL", "FedFACT"]
    if tabular_datasets is None:
        tabular_datasets = ["COMPAS", "DRUG", "DUTCH"]
    if img_datasets is None:
        img_datasets = ["CelebA", "UTKFace", "LFWA+", "FairFace"]
    if sent_datasets is None:
        sent_datasets = ["moji", "AG_NEWS", "IMDB"]
    if splits is None:
        splits = ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"]
    if client_counts is None:
        client_counts = ["20Clients", "30Clients", "40Clients"]

    tasks = []
    task_id = 0

    for task_type, datasets in [
        ("tabular", tabular_datasets),
        ("image", img_datasets),
        ("sent", sent_datasets),
    ]:
        for algo in algorithms:
            for ds in datasets:
                for split in splits:
                    for clients in client_counts:
                        for repeat in range(1, repeat_times + 1):
                            task_id += 1
                            prefix = task_type[:3]  # tab/img/sent
                            tasks.append({
                                "id": f"{prefix}_{task_id:04d}",
                                "type": task_type,
                                "algorithm": algo,
                                "dataset": ds,
                                "split": split,
                                "clients": clients,
                                "repeat": repeat,
                            })

    return tasks


def upload_task_queue(tasks=None):
    """将任务列表上传到云存储"""
    if tasks is None:
        tasks = generate_all_tasks()
    _save_json(PENDING_PATH, tasks)
    print(f"[TaskQueue] Uploaded {len(tasks)} tasks to {PENDING_PATH}")


# ============================================================
# 锁文件操作（核心并发安全机制）
# ============================================================

def _get_lock_path(task_id):
    """获取任务的锁文件路径"""
    return f"{LOCKS_DIR}/{task_id}.json"


def _try_acquire_lock(task_id, machine_id):
    """
    尝试获取任务锁（原子操作）

    策略：
      1. 如果锁文件不存在 → 创建锁文件，获取成功
      2. 如果锁文件存在但已超时 → 回收并重新获取
      3. 如果锁文件存在且未超时 → 获取失败（别人在跑）

    Returns:
        bool: 是否获取成功
    """
    storage = get_storage()
    lock_path = _get_lock_path(task_id)

    # 情况1：锁文件不存在，尝试创建
    if not storage.exists(lock_path):
        lock_data = {
            "user_id": _get_user_id(),
            "machine_id": machine_id,
            "claimed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_heartbeat": time.time(),
            "heartbeat_count": 0,
        }
        storage.write_json(lock_path, lock_data)
        return True

    # 情况2：锁文件存在，检查是否超时
    try:
        lock_data = storage.read_json(lock_path)
        last_hb = lock_data.get("last_heartbeat", 0)
        if (time.time() - last_hb) > DEFAULT_TIMEOUT:
            # 超时，回收锁
            print(f"[TaskQueue] Lock for {task_id} expired (last heartbeat {time.time() - last_hb:.0f}s ago), reclaiming")
            lock_data = {
                "user_id": _get_user_id(),
                "machine_id": machine_id,
                "claimed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "last_heartbeat": time.time(),
                "heartbeat_count": 0,
                "reclaimed_from": lock_data.get("machine_id", "unknown"),
                "reclaimed_from_user": lock_data.get("user_id", "unknown"),
            }
            storage.write_json(lock_path, lock_data)
            return True
        else:
            # 未超时，别人在跑
            return False
    except Exception:
        # 读取失败，尝试覆盖
        lock_data = {
            "user_id": _get_user_id(),
            "machine_id": machine_id,
            "claimed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_heartbeat": time.time(),
            "heartbeat_count": 0,
        }
        storage.write_json(lock_path, lock_data)
        return True


def _release_lock(task_id):
    """释放任务锁"""
    storage = get_storage()
    lock_path = _get_lock_path(task_id)
    try:
        storage.delete(lock_path)
    except Exception:
        pass


def update_heartbeat(task_id):
    """
    更新任务心跳（Worker定期调用）

    Args:
        task_id: 当前正在执行的任务ID
    """
    storage = get_storage()
    lock_path = _get_lock_path(task_id)

    if not storage.exists(lock_path):
        return

    try:
        lock_data = storage.read_json(lock_path)
        lock_data["last_heartbeat"] = time.time()
        lock_data["heartbeat_count"] = lock_data.get("heartbeat_count", 0) + 1
        storage.write_json(lock_path, lock_data)
    except Exception:
        pass


# ============================================================
# 任务完成 & 失败
# ============================================================


def complete_task(task_id, result_summary=None, elapsed_seconds=0):
    """标记任务完成（含可复现性信息和结果校验）"""
    _release_lock(task_id)

    # 结果校验
    validation_warnings = []
    if result_summary:
        is_valid, validation_warnings = validate_result(result_summary)

    done = _load_json(DONE_PATH)
    entry = {
        "id": task_id,
        "user_id": _get_user_id(),
        "machine_id": _get_machine_id(),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed_seconds, 1),
        **({"result": result_summary} if result_summary else {}),
    }

    # 添加可复现性信息
    try:
        entry["reproducibility"] = get_reproducibility_info()
    except Exception:
        pass

    # 添加校验结果
    if validation_warnings:
        entry["validation_warnings"] = validation_warnings
        print(f"[TaskQueue] WARNING: {len(validation_warnings)} validation issues for {task_id}:")
        for w in validation_warnings:
            print(f"  - {w}")

    done.append(entry)
    _save_json(DONE_PATH, done)
    print(f"[TaskQueue] Completed: {task_id} by {_get_user_id()} ({elapsed_seconds:.0f}s)")


def fail_task(task_id, error_msg=""):
    """标记任务失败（放回pending重试）"""
    _release_lock(task_id)

    # 从pending中找到该任务并更新重试计数
    pending = _load_json(PENDING_PATH)
    for t in pending:
        if t.get('id') == task_id:
            t['retry_count'] = t.get('retry_count', 0) + 1
            t['last_error'] = error_msg[:500] if error_msg else ""
            break
    else:
        # 如果不在pending中（可能被其他机器改了），重新添加
        pending.append({
            "id": task_id,
            "retry_count": 1,
            "last_error": error_msg[:500] if error_msg else "",
        })
    _save_json(PENDING_PATH, pending)
    print(f"[TaskQueue] Failed: {task_id} - {error_msg[:100]}")


# ============================================================
# 云端Checkpoint管理
# ============================================================

def get_cloud_checkpoint_dir(task_id):
    """获取任务的云端checkpoint目录路径"""
    return f"{CHECKPOINT_DIR}/{task_id}"


def has_cloud_checkpoint(task_id):
    """检查任务是否有云端checkpoint"""
    storage = get_storage()
    ckpt_dir = get_cloud_checkpoint_dir(task_id)
    files = storage.list_files(ckpt_dir, recursive=True)
    return len(files) > 0


def upload_checkpoint_to_cloud(task_id, local_checkpoint_path):
    """
    上传本地checkpoint到云端

    Args:
        task_id: 任务ID
        local_checkpoint_path: 本地checkpoint文件或目录路径
    """
    storage = get_storage()
    ckpt_dir = get_cloud_checkpoint_dir(task_id)

    if os.path.isfile(local_checkpoint_path):
        filename = os.path.basename(local_checkpoint_path)
        storage.upload(local_checkpoint_path, f"{ckpt_dir}/{filename}")
    elif os.path.isdir(local_checkpoint_path):
        storage.upload_dir(local_checkpoint_path, ckpt_dir)

    print(f"[TaskQueue] Checkpoint uploaded: {task_id} <- {local_checkpoint_path}")


def download_checkpoint_from_cloud(task_id, local_dir):
    """
    从云端下载checkpoint到本地

    Args:
        task_id: 任务ID
        local_dir: 本地目标目录

    Returns:
        bool: 是否成功下载
    """
    storage = get_storage()
    ckpt_dir = get_cloud_checkpoint_dir(task_id)
    files = storage.list_files(ckpt_dir, recursive=True)
    if not files:
        return False

    os.makedirs(local_dir, exist_ok=True)
    for f in files:
        local_path = os.path.join(local_dir, os.path.basename(f))
        storage.download(f, local_path)

    print(f"[TaskQueue] Checkpoint downloaded: {task_id} -> {local_dir} ({len(files)} files)")
    return True


# ============================================================
# 状态查询
# ============================================================

def get_queue_status():
    """获取队列状态统计"""
    pending = _load_json(PENDING_PATH)
    done = _load_json(DONE_PATH)

    # 统计锁文件数量（正在运行的任务）
    storage = get_storage()
    running_count = 0
    running_by_type = {"tabular": 0, "image": 0, "sent": 0}
    try:
        lock_files = storage.list_files(LOCKS_DIR, recursive=False)
        for lf in lock_files:
            if lf.endswith('.json'):
                running_count += 1
                # 从文件名提取任务类型
                task_id = os.path.basename(lf).replace('.json', '')
                if task_id.startswith('tab'):
                    running_by_type['tabular'] += 1
                elif task_id.startswith('img'):
                    running_by_type['image'] += 1
                elif task_id.startswith('sent'):
                    running_by_type['sent'] += 1
    except Exception:
        pass

    def count_by_type(tasks):
        counts = {"tabular": 0, "image": 0, "sent": 0}
        for t in tasks:
            tp = t.get('type', '')
            if tp in counts:
                counts[tp] += 1
        return counts

    return {
        "pending": len(pending),
        "pending_by_type": count_by_type(pending),
        "running": running_count,
        "running_by_type": running_by_type,
        "done": len(done),
        "done_by_type": count_by_type(done),
        "total": len(pending) + running_count + len(done),
    }


def print_queue_status():
    """打印队列状态"""
    status = get_queue_status()
    print("=" * 50)
    print("[TaskQueue Status]")
    print(f"  Total:    {status['total']}")
    print(f"  Pending:  {status['pending']}  (tab:{status['pending_by_type']['tabular']} "
          f"img:{status['pending_by_type']['image']} sent:{status['pending_by_type']['sent']})")
    print(f"  Running:  {status['running']}  (tab:{status['running_by_type']['tabular']} "
          f"img:{status['running_by_type']['image']} sent:{status['running_by_type']['sent']})")
    print(f"  Done:     {status['done']}  (tab:{status['done_by_type']['tabular']} "
          f"img:{status['done_by_type']['image']} sent:{status['done_by_type']['sent']})")
    print("=" * 50)


# ============================================================
# 贡献者注册
# ============================================================

def register_contributor(hardware_profile=None):
    """
    注册/更新贡献者信息到云端

    Args:
        hardware_profile: 硬件画像字典，如果为None则自动获取
    """
    if hardware_profile is None:
        from tool.memory_utils import get_hardware_profile
        hardware_profile = get_hardware_profile()

    storage = get_storage()
    contributors = []
    if storage.exists(CONTRIBUTORS_PATH):
        contributors = storage.read_json(CONTRIBUTORS_PATH)

    user_id = _get_user_id()
    machine_id = _get_machine_id()
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    # 查找是否已注册
    existing = None
    for c in contributors:
        if c.get('machine_id') == machine_id:
            existing = c
            break

    entry = {
        "user_id": user_id,
        "machine_id": machine_id,
        "machine_label": _get_machine_label(),
        "hostname": hardware_profile.get('hostname', ''),
        "os": hardware_profile.get('os', ''),
        "cpu_count": hardware_profile.get('cpu_count', 0),
        "cpu_model": hardware_profile.get('cpu_model', ''),
        "total_memory_gb": hardware_profile.get('total_memory_gb', 0),
        "gpu_info": hardware_profile.get('gpu_info', []),
        "disk_free_gb": hardware_profile.get('disk_free_gb', 0),
        "last_seen": now,
        "total_completed": 0,
        "total_time_seconds": 0,
        # 学术身份
        **_get_user_profile(),
    }

    if existing:
        # 更新已有记录，保留累计统计
        entry['total_completed'] = existing.get('total_completed', 0)
        entry['total_time_seconds'] = existing.get('total_time_seconds', 0)
        entry['first_seen'] = existing.get('first_seen', now)
        contributors = [c for c in contributors if c.get('machine_id') != machine_id]
    else:
        entry['first_seen'] = now

    contributors.append(entry)
    storage.write_json(CONTRIBUTORS_PATH, contributors)


# ============================================================
# P0: 实验可复现性
# ============================================================

def get_reproducibility_info():
    """
    获取实验可复现性信息（git hash + 环境信息）

    Returns:
        dict: 包含git commit、Python版本、PyTorch版本、关键依赖等
    """
    info = {}

    # Git commit hash
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, cwd=project_root,
            shell=True  # Windows需要shell=True来找到git
        )
        if result.returncode == 0 and result.stdout.strip():
            info['git_commit'] = result.stdout.strip()
        else:
            info['git_commit'] = 'unknown'
        # 检查是否有未提交的修改
        status = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, cwd=project_root,
            shell=True
        )
        info['git_dirty'] = bool(status.stdout.strip())
    except Exception:
        info['git_commit'] = 'unknown'
        info['git_dirty'] = True

    # Python版本
    import sys
    info['python_version'] = sys.version.split()[0]

    # PyTorch版本
    try:
        import torch
        info['torch_version'] = torch.__version__
        info['cuda_version'] = torch.version.cuda or 'cpu'
    except ImportError:
        pass

    # 关键依赖版本
    key_packages = ['numpy', 'scikit-learn', 'transformers', 'pandas']
    for pkg in key_packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, '__version__', 'unknown')
            info[f'{pkg}_version'] = ver
        except ImportError:
            pass

    return info


# ============================================================
# P0: 结果自动校验
# ============================================================

def validate_result(result_summary):
    """
    校验实验结果是否合理

    Args:
        result_summary: dict, 包含ACC/DEO/SPD等指标

    Returns:
        tuple: (is_valid: bool, warnings: list)
    """
    warnings = []

    if not result_summary or not isinstance(result_summary, dict):
        return False, ["No result summary provided"]

    # 检查ACC
    acc = result_summary.get('acc') or result_summary.get('accuracy') or result_summary.get('ACC')
    if acc is not None:
        if acc < 0 or acc > 1:
            warnings.append(f"ACC out of range [0,1]: {acc}")
        if acc < 0.3:
            warnings.append(f"ACC suspiciously low: {acc:.4f}")

    # 检查DEO
    deo = result_summary.get('deo') or result_summary.get('DEO')
    if deo is not None:
        if deo < 0 or deo > 1:
            warnings.append(f"DEO out of range [0,1]: {deo}")
        if abs(deo) < 1e-6:
            warnings.append(f"DEO suspiciously close to 0: {deo}")

    # 检查SPD
    spd = result_summary.get('spd') or result_summary.get('SPD')
    if spd is not None:
        if spd < -1 or spd > 1:
            warnings.append(f"SPD out of range [-1,1]: {spd}")

    # 检查NaN
    for key, val in result_summary.items():
        if isinstance(val, float):
            if val != val:  # NaN check
                warnings.append(f"NaN detected in {key}")

    is_valid = len(warnings) == 0
    return is_valid, warnings


# ============================================================
# P0: 数据完整性校验
# ============================================================

def compute_file_md5(filepath):
    """计算文件的MD5哈希"""
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    return md5.hexdigest()


def compute_dir_manifest(dir_path, relative_prefix=""):
    """
    递归计算目录下所有文件的MD5

    Args:
        dir_path: 目录路径
        relative_prefix: 路径前缀（用于云端路径映射）

    Returns:
        dict: {相对路径: md5_hash}
    """
    manifest = {}
    if not os.path.isdir(dir_path):
        return manifest

    for root, dirs, files in os.walk(dir_path):
        for fname in files:
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, dir_path)
            if relative_prefix:
                cloud_path = f"{relative_prefix}/{rel_path}"
            else:
                cloud_path = rel_path
            manifest[cloud_path] = compute_file_md5(full_path)

    return manifest


def upload_data_manifest(dataset_dir, cloud_prefix="datasets"):
    """
    上传数据集的MD5 manifest到云存储

    Args:
        dataset_dir: 本地数据集目录
        cloud_prefix: 云存储中的路径前缀
    """
    manifest = compute_dir_manifest(dataset_dir, cloud_prefix)

    # 合并已有的manifest（支持多次上传不同数据集）
    storage = get_storage()
    existing = {}
    if storage.exists(DATA_MANIFEST_PATH):
        existing = storage.read_json(DATA_MANIFEST_PATH)
    existing.update(manifest)
    storage.write_json(DATA_MANIFEST_PATH, existing)

    print(f"[DataManifest] Uploaded manifest: {len(manifest)} files from {dataset_dir}")
    return existing


def verify_data_manifest(local_dir, cloud_prefix="datasets"):
    """
    校验本地数据集与manifest是否一致

    Args:
        local_dir: 本地数据集目录
        cloud_prefix: 云存储中的路径前缀

    Returns:
        tuple: (all_valid: bool, mismatches: list)
    """
    storage = get_storage()
    if not storage.exists(DATA_MANIFEST_PATH):
        print("[DataManifest] No manifest found, skipping verification")
        return True, []

    manifest = storage.read_json(DATA_MANIFEST_PATH)
    local_manifest = compute_dir_manifest(local_dir, cloud_prefix)

    mismatches = []
    for path, expected_md5 in manifest.items():
        # 只检查属于该cloud_prefix的文件
        if not path.startswith(cloud_prefix):
            continue
        actual_md5 = local_manifest.get(path)
        if actual_md5 is None:
            mismatches.append(f"Missing file: {path}")
        elif actual_md5 != expected_md5:
            mismatches.append(f"MD5 mismatch: {path}")

    all_valid = len(mismatches) == 0
    if all_valid:
        print(f"[DataManifest] All {len(local_manifest)} files verified OK")
    else:
        print(f"[DataManifest] {len(mismatches)} issues found:")
        for m in mismatches:
            print(f"  - {m}")

    return all_valid, mismatches


# ============================================================
# P1: 任务优先级
# ============================================================

# 优先级定义（数值越小优先级越高）
PRIORITY_HIGH = 0
PRIORITY_NORMAL = 1
PRIORITY_LOW = 2


def claim_task(task_type=None):
    """
    从队列中拉取一个任务（使用逐任务锁保证并发安全）

    优先级策略：按priority字段排序，相同优先级保持FIFO顺序

    Args:
        task_type: 只拉取特定类型的任务，None表示不限

    Returns:
        dict or None: 拉取到的任务
    """
    machine_id = _get_machine_id()
    pending = _load_json(PENDING_PATH)
    if not pending:
        return None

    # 过滤任务类型
    if task_type:
        pending = [t for t in pending if t.get('type') == task_type]
    if not pending:
        return None

    # 按优先级排序（priority越小越优先），相同优先级保持原顺序（FIFO）
    pending_sorted = sorted(enumerate(pending), key=lambda x: x[1].get('priority', PRIORITY_NORMAL))

    # 遍历任务，尝试获取锁
    claimed_task = None
    claimed_idx = None

    for orig_idx, task in pending_sorted:
        # 跳过重试次数过多的任务
        if task.get('retry_count', 0) >= MAX_RETRIES:
            continue

        if _try_acquire_lock(task['id'], machine_id):
            claimed_task = task
            claimed_idx = orig_idx
            break

    if claimed_task is None:
        return None

    # 从pending中移除已领取的任务
    remaining = [t for i, t in enumerate(pending) if i != claimed_idx]
    _save_json(PENDING_PATH, remaining)

    priority_str = claimed_task.get('priority', PRIORITY_NORMAL)
    print(f"[TaskQueue] Claimed: {claimed_task['id']} ({claimed_task['type']}, P{priority_str}) "
          f"- {claimed_task['algorithm']}/{claimed_task['dataset']} "
          f"[{claimed_task['split']}/{claimed_task['clients']}/R{claimed_task['repeat']}]")
    return claimed_task
    print(f"[TaskQueue] Contributor registered: {user_id} @ {machine_id}")


def update_contributor_stats(user_id, completed_count=0, time_seconds=0):
    """
    更新贡献者的累计统计

    Args:
        user_id: 用户ID
        completed_count: 本次完成的任务数
        time_seconds: 本次耗时（秒）
    """
    storage = get_storage()
    if not storage.exists(CONTRIBUTORS_PATH):
        return

    contributors = storage.read_json(CONTRIBUTORS_PATH)
    for c in contributors:
        if c.get('user_id') == user_id:
            c['total_completed'] = c.get('total_completed', 0) + completed_count
            c['total_time_seconds'] = c.get('total_time_seconds', 0) + time_seconds
            c['last_seen'] = time.strftime("%Y-%m-%d %H:%M:%S")
            break

    storage.write_json(CONTRIBUTORS_PATH, contributors)
