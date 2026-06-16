import os
import sys
import subprocess
import time
import multiprocessing
import re

ALGORITHMS = ["LoGoFair", "PraFFL", "FedFACT", "FedAvg"]
TABULAR_DATASETS = [] # ["COMPAS", "DRUG", "DUTCH", "ADULT"]
IMG_DATASETS = ["CelebA", "UTKFace", "LFWA+","FairFace"] 
SENT_DATASETS = []  # ["moji", "bios"]
SPLITS = ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"]
CLIENTS = ["20Clients", "30Clients", "40Clients"]
BATCH_SIZE = 256
IMG_BATCH_SIZE = 32    # 图像实验batch_size（梯度累积等效256）
SENT_BATCH_SIZE = 8   # 文本实验batch_size（梯度累积等效256）

# ============ 资源配置 / Resource Configuration ============
#
# 本模块支持 GPU + CPU 混合并行，充分利用机器所有计算资源。
# This module supports GPU + CPU mixed parallelism to fully utilize all compute resources.
#
# 【配置示例 / Configuration Examples】
#
# 场景1：纯 CPU 机器（如 32 核服务器）
#   GPU_POOL = []              # 无 GPU
#   CPU_SLOTS = 4              # 同时跑 4 个 CPU 实验
#   → 总并行 = 4，OMP 线程 = 32/4 = 8
#
# 场景2：单 GPU + 多核 CPU（如 1 张 GPU + 16 核 CPU）
#   GPU_POOL = ["0"]           # 1 张 GPU
#   CPU_SLOTS = 2              # 额外 2 个 CPU 实验
#   → 总并行 = 1(GPU) + 2(CPU) = 3，GPU 跑图像，CPU 跑表格
#
# 场景3：多 GPU + 多核 CPU（如 4 张 GPU + 64 核 CPU）
#   GPU_POOL = ["0","1","2","3"]  # 4 张 GPU
#   CPU_SLOTS = 4                  # 额外 4 个 CPU 实验
#   → 总并行 = 4(GPU) + 4(CPU) = 8，GPU 跑图像/文本，CPU 跑表格
#
# 场景4：全自动（根据硬件自动计算）
#   GPU_POOL = ["0","1"]       # 2 张 GPU
#   CPU_SLOTS = 0              # 0 = 自动（CPU核数//8，最多4）
#   MAX_PARALLEL = 0           # 0 = 自动（GPU数 + CPU_SLOTS）
#

# GPU 池：可用 GPU ID 列表。不同实验会轮转分配到不同 GPU 上。
# GPU pool: available GPU IDs. Experiments are round-robin assigned across GPUs.
GPU_POOL = []

# CPU 额外并行槽位：在 GPU 任务之外，额外用 CPU 跑多少个实验。
# 设为 0 表示自动计算（CPU 总核数 // 8，最多 4）。
# Extra CPU parallel slots: how many additional experiments run on CPU alongside GPU tasks.
# Set to 0 for auto-calc (total_cpu_cores // 8, max 4).
CPU_SLOTS = 0

# 总最大并行数 = len(GPU_POOL) + CPU_SLOTS。设为 0 则自动计算。
# Total max parallel = len(GPU_POOL) + CPU_SLOTS. Set to 0 for auto-calc.
MAX_PARALLEL = 0

# 每个实验重复几次（不同随机种子），用于统计显著性，结果报告 Mean +/- STD
# Number of times to repeat each experiment with different seeds for statistical significance.
EXP_REPEAT_TIMES = 3

# 几次重复同时并行执行：1=串行（默认），最大不超过 EXP_REPEAT_TIMES
# How many repeat runs execute in parallel via multiprocessing. 1=serial, max=EXP_REPEAT_TIMES.
PARALLEL_REPEATS = 1

REQUIRED_TESTS = 3

PYTHON = sys.executable


def _resolve_resources():
    """解析资源配置，返回 (gpu_count, cpu_slots, max_parallel, threads_per_job)"""
    gpu_count = len(GPU_POOL)
    cpu_cores = multiprocessing.cpu_count()

    # CPU_SLOTS 自动计算
    if CPU_SLOTS <= 0:
        cpu_slots = min(cpu_cores // 8, 4)
    else:
        cpu_slots = CPU_SLOTS

    # MAX_PARALLEL 自动计算
    if MAX_PARALLEL <= 0:
        max_parallel = max(gpu_count, 1) + cpu_slots
    else:
        max_parallel = MAX_PARALLEL

    # OMP 线程数：仅 CPU 任务需要多线程，GPU 任务设为 1（避免和 CUDA 抢资源）
    # GPU 任务主要靠 CUDA 核心，不需要大量 CPU 线程
    threads_gpu = 1
    threads_cpu = max(1, cpu_cores // max(gpu_count + cpu_slots, 1))
    threads_cpu = min(threads_cpu, cpu_cores)

    print(f"  Resource config:")
    print(f"    GPU pool: {GPU_POOL if GPU_POOL else '(none, CPU only)'}")
    print(f"    GPU slots: {gpu_count}")
    print(f"    CPU slots: {cpu_slots} (CPU cores: {cpu_cores})")
    print(f"    Total parallel: {max_parallel}")
    print(f"    OMP threads (GPU tasks): {threads_gpu}")
    print(f"    OMP threads (CPU tasks): {threads_cpu}")

    return gpu_count, cpu_slots, max_parallel, threads_gpu, threads_cpu


def analyze_experiment_log(log_file):
    """分析实验日志，返回已完成的测试次数和是否有最终汇总"""
    if not os.path.exists(log_file):
        return 0, False
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        test_count = len(re.findall(r'Trained Global Model Testing', content))
        has_summary = 'Mean' in content and 'STD' in content
        
        return test_count, has_summary
    except:
        return 0, False


def has_experiment_progress(log_file):
    """检查实验是否有训练进度（已开始但未完成）"""
    if not os.path.exists(log_file):
        return False
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        return len(re.findall(r'Communication Round: \d+', content)) > 0
    except:
        return False


def get_experiment_id(split, clients):
    """根据split和clients计算实验编号（1-12）"""
    split_idx = SPLITS.index(split)
    client_idx = CLIENTS.index(clients)
    return split_idx * 3 + client_idx + 1


def get_experiment_config(exp_id):
    """根据实验编号获取对应的split和clients配置"""
    idx = exp_id - 1
    split_idx = idx // 3
    client_idx = idx % 3
    return SPLITS[split_idx], CLIENTS[client_idx]


def is_experiment_complete(log_file):
    """检查单个实验是否真正完成（3次测试+汇总）"""
    test_count, has_summary = analyze_experiment_log(log_file)
    return test_count >= REQUIRED_TESTS and has_summary


def is_task_done(algorithm, dataset, hypothesis):
    """检查整个算法/数据集组合是否全部完成"""
    log_dir = os.path.join("log_path", dataset)
    if not os.path.exists(log_dir):
        return False
    for split in SPLITS:
        for clients in CLIENTS:
            client_dir = os.path.join(log_dir, split, algorithm, hypothesis, clients)
            if not os.path.exists(client_dir):
                return False
            exp_id = get_experiment_id(split, clients)
            log_file = os.path.join(client_dir, f"{exp_id}.txt")
            if not is_experiment_complete(log_file):
                return False
    return True


def get_next_experiment_id(algorithm, dataset, hypothesis):
    """获取下一个未完成实验的编号（从1开始）"""
    log_dir = os.path.join("log_path", dataset)
    for exp_id in range(1, 13):
        split, clients = get_experiment_config(exp_id)
        client_dir = os.path.join(log_dir, split, algorithm, hypothesis, clients)
        if os.path.exists(client_dir):
            log_file = os.path.join(client_dir, f"{exp_id}.txt")
            if not is_experiment_complete(log_file):
                return exp_id
        else:
            return exp_id
    return None


def get_experiment_status(algorithm, dataset, hypothesis):
    """获取所有实验的状态统计"""
    log_dir = os.path.join("log_path", dataset)
    status = []
    for exp_id in range(1, 13):
        split, clients = get_experiment_config(exp_id)
        client_dir = os.path.join(log_dir, split, algorithm, hypothesis, clients)
        if os.path.exists(client_dir):
            log_file = os.path.join(client_dir, f"{exp_id}.txt")
            test_count, has_summary = analyze_experiment_log(log_file)
            status.append({
                'id': exp_id,
                'split': split,
                'clients': clients,
                'test_count': test_count,
                'has_summary': has_summary,
                'complete': test_count >= REQUIRED_TESTS and has_summary,
                'needs_summary': test_count >= REQUIRED_TESTS and not has_summary,
                'needs_more_tests': test_count > 0 and test_count < REQUIRED_TESTS,
                'needs_start': test_count == 0
            })
        else:
            status.append({
                'id': exp_id,
                'split': split,
                'clients': clients,
                'test_count': 0,
                'has_summary': False,
                'complete': False,
                'needs_summary': False,
                'needs_more_tests': False,
                'needs_start': True
            })
    return status


def build_jobs():
    jobs = []
    gpu_counter = 0   # GPU 任务计数器，用于 GPU 轮转分配
    cpu_counter = 0   # CPU 任务计数器
    has_gpu = len(GPU_POOL) > 0
    
    def _assign_device(job_type):
        """根据任务类型和 GPU 可用性，分配计算设备（GPU 或 CPU）"""
        nonlocal gpu_counter, cpu_counter
        if has_gpu and job_type in ["image", "sent"]:
            # 图像/文本任务优先分配 GPU（计算量大）
            gpu_id = GPU_POOL[gpu_counter % len(GPU_POOL)]
            gpu_counter += 1
            return gpu_id, "gpu"
        elif has_gpu and job_type == "tabular":
            # 表格任务：如果 GPU 还有空位也可以用，否则用 CPU
            # 简单策略：表格任务全部走 CPU，把 GPU 留给图像/文本
            return "", "cpu"
        else:
            # 无 GPU，全部走 CPU
            return "", "cpu"
    
    def _make_cmd(main_script, algo, ds, batch_size, hypothesis, cuda_arg, extra_args=""):
        base = f'{PYTHON} {main_script} -algorithm {algo} -dataset {ds}'
        base += f' -batch_size {batch_size} -test_batch_size {batch_size}'
        base += f' -cuda {cuda_arg} -learning_rate 3e-4'
        base += f' -exp_repeat_times {EXP_REPEAT_TIMES} -parallel_repeats {PARALLEL_REPEATS}'
        base += extra_args
        return base
    
    # 构建所有表格实验任务
    tabular_jobs = []
    for algo in ALGORITHMS:
        for ds in TABULAR_DATASETS:
            if is_task_done(algo, ds, "ANN"):
                print(f"  [SKIP] {algo} / {ds} (all 12 experiments complete)")
                continue
            
            status = get_experiment_status(algo, ds, "ANN")
            incomplete_count = sum(1 for s in status if not s['complete'])
            needs_summary_count = sum(1 for s in status if s['needs_summary'])
            
            if needs_summary_count > 0:
                print(f"  [PENDING] {algo} / {ds} - {needs_summary_count} experiments need summary calculation")
            
            next_exp_id = get_next_experiment_id(algo, ds, "ANN")
            if next_exp_id is not None:
                first_incomplete = next_exp_id
                print(f"  [RESUME] {algo} / {ds} (starting from experiment {first_incomplete}/12, {incomplete_count} incomplete)")
            else:
                print(f"  [START] {algo} / {ds} (starting from experiment 1/12)")
            
            gpu_id, device = _assign_device("tabular")
            cuda_arg = gpu_id if gpu_id != "" else '""'
            cmd = _make_cmd("main_Tabular_CLF.py", algo, ds, BATCH_SIZE, "ANN", cuda_arg,
                          f' -task Tabular_CLF -model_type ANN -resume')
            tabular_jobs.append({"cmd": cmd, "name": f"[Tabular] {algo} / {ds}", "type": "tabular", 
                               "algo": algo, "dataset": ds, "hypothesis": "ANN", "gpu": gpu_id, "device": device})
            cpu_counter += 1
    
    # 构建所有图像实验任务
    image_jobs = []
    for algo in ALGORITHMS:
        for ds in IMG_DATASETS:
            if is_task_done(algo, ds, "BERTCLASSIFIER"):
                print(f"  [SKIP] {algo} / {ds} (all 12 experiments complete)")
                continue
            
            next_exp_id = get_next_experiment_id(algo, ds, "BERTCLASSIFIER")
            if next_exp_id is not None:
                print(f"  [RESUME] {algo} / {ds} (starting from experiment {next_exp_id}/12)")
            else:
                print(f"  [START] {algo} / {ds} (starting from experiment 1/12)")
            
            gpu_id, device = _assign_device("image")
            cuda_arg = gpu_id if gpu_id != "" else '""'
            cmd = _make_cmd("main_IMG_CLF.py", algo, ds, IMG_BATCH_SIZE, "BERTCLASSIFIER", cuda_arg,
                          f' -task IMG_CLF -resume')
            image_jobs.append({"cmd": cmd, "name": f"[Image] {algo} / {ds}", "type": "image",
                              "algo": algo, "dataset": ds, "hypothesis": "BERTCLASSIFIER", "gpu": gpu_id, "device": device})
    
    # 构建所有文本实验任务
    sent_jobs = []
    for algo in ALGORITHMS:
        for ds in SENT_DATASETS:
            if is_task_done(algo, ds, "BERTCLASSIFIER"):
                print(f"  [SKIP] {algo} / {ds} (all 12 experiments complete)")
                continue
            
            next_exp_id = get_next_experiment_id(algo, ds, "BERTCLASSIFIER")
            if next_exp_id is not None:
                print(f"  [RESUME] {algo} / {ds} (starting from experiment {next_exp_id}/12)")
            else:
                print(f"  [START] {algo} / {ds} (starting from experiment 1/12)")
            
            gpu_id, device = _assign_device("sent")
            cuda_arg = gpu_id if gpu_id != "" else '""'
            cmd = _make_cmd("main_SENT_CLF.py", algo, ds, SENT_BATCH_SIZE, "BERTCLASSIFIER", cuda_arg,
                          f' -task SENT_CLF -resume')
            sent_jobs.append({"cmd": cmd, "name": f"[Sent] {algo} / {ds}", "type": "sent",
                              "algo": algo, "dataset": ds, "hypothesis": "BERTCLASSIFIER", "gpu": gpu_id, "device": device})
    
    # 按顺序添加任务：表格 -> 图像 -> 文本
    for job in tabular_jobs:
        jobs.append(job)
    for job in image_jobs:
        jobs.append(job)
    for job in sent_jobs:
        jobs.append(job)
    
    return jobs


def run_jobs(jobs):
    global MAX_PARALLEL
    
    # 解析资源配置
    gpu_count, cpu_slots, max_parallel, threads_gpu, threads_cpu = _resolve_resources()
    MAX_PARALLEL = max_parallel
    
    running = []
    done = 0
    failed = 0
    total = len(jobs)
    idx = 0

    print(f"\nTotal jobs: {total}")
    print(f"Max parallel: {MAX_PARALLEL} ({gpu_count} GPU + {cpu_slots} CPU)")
    print()

    while idx < total or running:
        tab_running = sum(1 for (_, t, _, _, _) in running if t == "tabular")
        img_running = sum(1 for (_, t, _, _, _) in running if t == "image")
        sent_running = sum(1 for (_, t, _, _, _) in running if t == "sent")

        # 计算剩余任务数
        remaining_tabular = sum(1 for j in jobs[idx:] if j["type"] == "tabular")
        remaining_image = sum(1 for j in jobs[idx:] if j["type"] == "image")
        remaining_sent = sum(1 for j in jobs[idx:] if j["type"] == "sent")

        # 优化：当表格实验只剩2个时，允许启动图像/文本实验
        allow_early_start = remaining_tabular <= 2 and (remaining_image > 0 or remaining_sent > 0)

        while idx < total:
            job = jobs[idx]
            
            # 检查是否达到最大并行度
            total_running = tab_running + img_running + sent_running
            if total_running >= MAX_PARALLEL:
                break
            
            # 如果是表格实验，直接启动
            if job["type"] == "tabular":
                pass  # 表格实验优先
            # 如果是图像/文本实验，检查是否允许提前启动
            elif job["type"] in ["image", "sent"]:
                if not allow_early_start and remaining_tabular > 2:
                    # 表格实验还很多，不启动图像/文本实验
                    break
            
            # 为子进程设置正确的 OMP 线程数
            env = os.environ.copy()
            device = job.get("device", "cpu")
            env["OMP_NUM_THREADS"] = str(threads_gpu if device == "gpu" else threads_cpu)
            env["MKL_NUM_THREADS"] = env["OMP_NUM_THREADS"]
            
            p = subprocess.Popen(job["cmd"], shell=True, env=env)
            pid = p.pid
            gpu_info = job.get("gpu", "")
            device_tag = f"GPU:{gpu_info}" if gpu_info else "CPU"
            print(f"  Starting: {job['name']} ({idx+1}/{total}) [PID: {pid}, {device_tag}, OMP:{env['OMP_NUM_THREADS']}, Tab: {tab_running}, Img: {img_running}, Sent: {sent_running}]")
            
            with open("run_pids.log", "a") as f:
                f.write(f"{time.strftime('%Y/%m/%d %H:%M:%S')} - PID: {pid} - {job['name']}\n")
            
            running.append((p, job["type"], job["name"], pid, job))
            if job["type"] == "tabular":
                tab_running += 1
            elif job["type"] == "image":
                img_running += 1
            else:  # sent
                sent_running += 1
            idx += 1

        if running:
            time.sleep(10)

        still_running = []
        for p, t, name, pid, job in running:
            if p.poll() is not None:
                if p.returncode == 0:
                    done += 1
                    print(f"  [DONE] {name} (PID: {pid}) ({done}/{total})")
                    with open("run_progress.log", "a") as f:
                        f.write(f"[DONE] {time.strftime('%Y/%m/%d %H:%M:%S')} - {name} (PID: {pid})\n")
                else:
                    failed += 1
                    print(f"  [ERROR] {name} (PID: {pid}) ({failed} failed)")
                    with open("run_errors.log", "a") as f:
                        f.write(f"[ERROR] {time.strftime('%Y/%m/%d %H:%M:%S')} - {name} (PID: {pid})\n")
            else:
                still_running.append((p, t, name, pid, job))
        running = still_running

    print(f"\n{'='*60}")
    print(f"  All done! Success: {done}, Failed: {failed}")
    print(f"  End Time: {time.strftime('%Y/%m/%d %H:%M:%S')}")
    print(f"{'='*60}")

    try:
        from tool.notification import notify_batch_done
        notify_batch_done(f"All experiments completed. Success: {done}, Failed: {failed}")
    except Exception:
        pass


if __name__ == "__main__":
    print("=" * 60)
    print("  Batch Experiment Runner")
    print(f"  Start Time: {time.strftime('%Y/%m/%d %H:%M:%S')}")
    print("=" * 60)
    
    print("\nChecking experiment status...")
    print(f"  Each experiment requires {REQUIRED_TESTS} runs for mean/std calculation")
    print()
    jobs = build_jobs()
    run_jobs(jobs)