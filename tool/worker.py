"""
分布式实验 Worker（v2）

改进点：
  1. 心跳线程：后台定期更新锁文件时间戳，防止被超时回收
  2. 云端Checkpoint：训练前检查云端checkpoint并恢复，训练中定期上传
  3. 优雅退出：捕获SIGTERM信号，释放锁文件

使用方式：
  # 启动Worker（自动从队列拉取任务）
  python tool/worker.py

  # 只跑图像任务
  python tool/worker.py --type image

  # P4机器（小显存，自动调小batch_size）
  python tool/worker.py --type image --small_gpu

  # 查看队列状态
  python tool/worker.py --status
"""

import os
import sys
import argparse
import subprocess
import time
import socket
import threading
import shutil
import signal

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tool.config_checker import check_config
from tool.cloud_storage import get_storage
from tool.task_queue import (
    claim_task, complete_task, fail_task, print_queue_status,
    update_heartbeat, HEARTBEAT_INTERVAL,
    has_cloud_checkpoint, upload_checkpoint_to_cloud, download_checkpoint_from_cloud,
    get_cloud_checkpoint_dir,
    register_contributor, update_contributor_stats,
)
from tool.memory_utils import get_system_info, get_hardware_profile, print_hardware_profile


# ============================================================
# 任务类型 → 执行脚本映射
# ============================================================

TASK_SCRIPTS = {
    "tabular": "main_Tabular_CLF.py",
    "image": "main_IMG_CLF.py",
    "sent": "main_SENT_CLF.py",
}

TASK_MODEL_TYPES = {
    "tabular": "ANN",
    "image": "CNNCLASSIFIER",
    "sent": "BERTCLASSIFIER",
}

TASK_TASK_NAMES = {
    "tabular": "Tabular_CLF",
    "image": "IMG_CLF",
    "sent": "SENT_CLF",
}


# ============================================================
# 心跳线程
# ============================================================

class HeartbeatThread(threading.Thread):
    """后台心跳线程，定期更新锁文件时间戳"""

    def __init__(self, task_id, interval=HEARTBEAT_INTERVAL):
        super().__init__(daemon=True)
        self.task_id = task_id
        self.interval = interval
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.wait(self.interval):
            update_heartbeat(self.task_id)

    def stop(self):
        self._stop_event.set()


# ============================================================
# Checkpoint管理
# ============================================================

def _get_local_checkpoint_dir(task):
    """获取任务在本地的checkpoint目录"""
    return os.path.join(PROJECT_ROOT, "result_path", task['dataset'], task['algorithm'],
                        f"{task['split']}_{task['clients']}", "checkpoints")


def _get_local_result_dir(task):
    """获取任务在本地的结果目录"""
    return os.path.join(PROJECT_ROOT, "result_path", task['dataset'], task['algorithm'],
                        f"{task['split']}_{task['clients']}")


def prepare_checkpoint(task):
    """
    准备checkpoint：检查云端是否有checkpoint，有则下载到本地

    Returns:
        bool: 是否从云端恢复了checkpoint
    """
    if has_cloud_checkpoint(task['id']):
        local_ckpt_dir = _get_local_checkpoint_dir(task)
        print(f"[Worker] Found cloud checkpoint for {task['id']}, downloading...")
        success = download_checkpoint_from_cloud(task['id'], local_ckpt_dir)
        if success:
            print(f"[Worker] Cloud checkpoint restored for {task['id']}")
            return True
    return False


def sync_checkpoint_to_cloud(task):
    """将本地checkpoint上传到云端"""
    local_ckpt_dir = _get_local_checkpoint_dir(task)
    if os.path.exists(local_ckpt_dir):
        upload_checkpoint_to_cloud(task['id'], local_ckpt_dir)


def sync_result_to_cloud(task):
    """将实验结果上传到云端"""
    storage = get_storage()
    local_result_dir = _get_local_result_dir(task)
    if not os.path.exists(local_result_dir):
        return

    remote_prefix = f"results/{task['type']}/{task['algorithm']}/{task['dataset']}/{task['split']}_{task['clients']}"
    storage.upload_dir(local_result_dir, remote_prefix)
    print(f"[Worker] Results uploaded: {task['id']} -> {remote_prefix}")


# ============================================================
# 命令构建
# ============================================================

def build_command(task, cuda_id=0, small_gpu=False):
    """根据任务信息构建执行命令"""
    task_type = task['type']
    script = TASK_SCRIPTS[task_type]
    model_type = TASK_MODEL_TYPES[task_type]
    task_name = TASK_TASK_NAMES[task_type]

    num_clients = task['clients'].replace("Clients", "")

    # P4等小显存GPU自动调小batch_size
    batch_size = 32 if small_gpu else 256
    test_batch_size = 32 if small_gpu else 256

    cmd = (
        f"python {script} "
        f"-algorithm {task['algorithm']} "
        f"-dataset {task['dataset']} "
        f"-cuda \"{cuda_id}\" "
        f"-task {task_name} "
        f"-model_type {model_type} "
        f"-learning_rate 3e-4 "
        f"-batch_size {batch_size} "
        f"-test_batch_size {test_batch_size} "
        f"-num_clients {num_clients} "
        f"-split_strategy {task['split']} "
    )

    if task_type == "tabular":
        cmd += " -resume"

    return cmd


# ============================================================
# 任务执行
# ============================================================

def run_single_task(task, cuda_id=0, small_gpu=False):
    """
    执行单个实验任务（含心跳和checkpoint管理）

    Returns:
        tuple: (success: bool, elapsed_seconds: float)
    """
    cmd = build_command(task, cuda_id, small_gpu)
    print(f"\n{'='*60}")
    print(f"[Worker] Running: {task['id']}")
    print(f"[Worker] Command: {cmd}")
    print(f"{'='*60}")

    # 1. 检查云端checkpoint并恢复
    restored = prepare_checkpoint(task)
    if restored:
        print(f"[Worker] Resuming from cloud checkpoint")

    # 2. 启动心跳线程
    hb_thread = HeartbeatThread(task['id'])
    hb_thread.start()

    # 3. 启动checkpoint同步线程（每10分钟上传一次）
    ckpt_sync_interval = 600  # 10分钟
    ckpt_stop = threading.Event()

    def _ckpt_sync_loop():
        while not ckpt_stop.wait(ckpt_sync_interval):
            try:
                sync_checkpoint_to_cloud(task)
            except Exception:
                pass

    ckpt_thread = threading.Thread(target=_ckpt_sync_loop, daemon=True)
    ckpt_thread.start()

    success = False
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=86400,  # 24小时超时（P4最长运行时间）
        )

        if result.returncode == 0:
            print(f"[Worker] Task {task['id']} completed successfully")
            success = True
        else:
            error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
            print(f"[Worker] Task {task['id']} failed (code={result.returncode})")
            print(f"[Worker] Error: {error_msg}")

    except subprocess.TimeoutExpired:
        print(f"[Worker] Task {task['id']} timed out (24h limit)")
    except Exception as e:
        print(f"[Worker] Task {task['id']} exception: {e}")
    finally:
        # 停止心跳和checkpoint同步
        hb_thread.stop()
        ckpt_stop.set()

        # 最后一次上传checkpoint和结果
        try:
            sync_checkpoint_to_cloud(task)
            if success:
                sync_result_to_cloud(task)
        except Exception as e:
            print(f"[Worker] Warning: failed to sync to cloud: {e}")

    elapsed = time.time() - start_time
    return success, elapsed


# ============================================================
# Worker主循环
# ============================================================

_current_task_id = None


def _signal_handler(signum, frame):
    """SIGTERM信号处理：释放锁文件后退出"""
    print(f"\n[Worker] Received signal {signum}, cleaning up...")
    if _current_task_id:
        try:
            fail_task(_current_task_id, "Worker terminated by signal")
        except Exception:
            pass
    sys.exit(0)


def worker_loop(task_type=None, max_tasks=None, cuda_id=0,
                poll_interval=10, small_gpu=False):
    """
    Worker主循环
    """
    global _current_task_id

    machine_id = f"{socket.gethostname()}_pid{os.getpid()}"
    user_id = None
    try:
        from tool.task_queue import _get_user_id
        user_id = _get_user_id()
    except Exception:
        pass

    completed = 0
    failed = 0
    total_time = 0

    # 注册信号处理
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # 打印硬件画像并注册贡献者
    profile = get_hardware_profile()
    print_hardware_profile(profile)
    try:
        register_contributor(profile)
    except Exception as e:
        print(f"[Worker] Warning: failed to register contributor: {e}")

    print(f"\n{'='*60}")
    print(f"[Worker] Started: {user_id or 'unknown'} @ {machine_id}")
    print(f"[Worker] Type filter: {task_type or 'all'}")
    print(f"[Worker] Max tasks: {max_tasks or 'unlimited'}")
    print(f"[Worker] CUDA: {cuda_id}, Small GPU: {small_gpu}")
    print(f"{'='*60}\n")

    while True:
        if max_tasks is not None and completed + failed >= max_tasks:
            print(f"[Worker] Reached max tasks ({max_tasks}), stopping.")
            break

        task = claim_task(task_type=task_type)
        if task is None:
            print(f"[Worker] No tasks, waiting {poll_interval}s...")
            time.sleep(poll_interval)
            continue

        _current_task_id = task['id']
        success, elapsed = run_single_task(task, cuda_id=cuda_id, small_gpu=small_gpu)

        if success:
            complete_task(task['id'], elapsed_seconds=elapsed)
            completed += 1
            total_time += elapsed
            # 更新贡献者统计
            if user_id:
                try:
                    update_contributor_stats(user_id, completed_count=1, time_seconds=elapsed)
                except Exception:
                    pass
        else:
            fail_task(task['id'], "Execution failed")
            failed += 1

        _current_task_id = None
        print(f"[Worker] Progress: {completed} done, {failed} failed, "
              f"total time: {total_time/3600:.1f}h")

    print(f"\n[Worker] Finished. Done: {completed}, Failed: {failed}, "
          f"Total: {total_time/3600:.1f}h")


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Distributed FL Experiment Worker v2")
    parser.add_argument("--type", type=str, choices=["tabular", "image", "sent"],
                        default=None, help="Only run tasks of this type")
    parser.add_argument("--max_tasks", type=int, default=None,
                        help="Maximum number of tasks to run")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--poll_interval", type=int, default=10,
                        help="Polling interval (seconds)")
    parser.add_argument("--small_gpu", action="store_true",
                        help="Use small batch_size (32) for GPUs with <=8GB VRAM")
    parser.add_argument("--status", action="store_true",
                        help="Print queue status and exit")

    args = parser.parse_args()

    if args.status:
        print_queue_status()
        return

    # 启动前检查配置
    if not check_config():
        sys.exit(1)

    worker_loop(
        task_type=args.type,
        max_tasks=args.max_tasks,
        cuda_id=args.cuda,
        poll_interval=args.poll_interval,
        small_gpu=args.small_gpu,
    )


if __name__ == "__main__":
    main()
