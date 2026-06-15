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
IMG_BATCH_SIZE = 64    # 图像实验batch_size（梯度累积等效256）
SENT_BATCH_SIZE = 32   # 文本实验batch_size（梯度累积等效256）

# ============ 资源配置 / Resource Configuration ============
# GPU 池：可用 GPU ID 列表。空列表 "" 表示仅用 CPU。
# GPU pool: list of available GPU IDs. Empty string "" means CPU-only.
# 示例 / Examples:
#   GPU_POOL = []           → 全部用 CPU / All CPU
#   GPU_POOL = ["0"]        → 单卡 / Single GPU
#   GPU_POOL = ["0", "1"]   → 双卡，不同实验分配不同卡 / Dual GPU, different exp on different GPU
#   GPU_POOL = ["0", "1", "2", "3"] → 四卡 / Quad GPU
GPU_POOL = []

# 最大并行实验数。设为 0 则自动根据 GPU 数和 CPU 核心数计算。
# Max parallel experiments. Set to 0 to auto-calculate based on GPU count and CPU cores.
# 自动计算逻辑：max(len(GPU_POOL), 1) + cpu_extra（CPU核数//8，最多4）
# Auto-calc: max(len(GPU_POOL), 1) + cpu_extra (cpu_cores//8, max 4)
MAX_PARALLEL = 0

# 每个实验重复几次（不同随机种子），用于统计显著性，结果报告 Mean +/- STD
# Number of times to repeat each experiment with different seeds for statistical significance.
EXP_REPEAT_TIMES = 3

# 几次重复同时并行执行：1=串行（默认），最大不超过 EXP_REPEAT_TIMES
# How many repeat runs execute in parallel via multiprocessing. 1=serial, max=EXP_REPEAT_TIMES.
PARALLEL_REPEATS = 1

REQUIRED_TESTS = 3

PYTHON = sys.executable


def _auto_max_parallel():
    """根据 GPU 数量和 CPU 核心数自动计算最大并行数"""
    gpu_count = max(len(GPU_POOL), 1)
    cpu_cores = multiprocessing.cpu_count()
    # 每个 CPU 实验大约需要 2-4 核，留一些余量给系统
    cpu_extra = min(cpu_cores // 8, 4)
    auto = gpu_count + cpu_extra
    print(f"  Auto MAX_PARALLEL: {auto} (GPU: {gpu_count}, CPU cores: {cpu_cores}, CPU extra slots: {cpu_extra})")
    return auto


def _get_threads_per_job():
    """根据并行数和 CPU 核心数，计算每个实验进程分配的 OMP 线程数"""
    cpu_cores = multiprocessing.cpu_count()
    effective_parallel = MAX_PARALLEL if MAX_PARALLEL > 0 else _auto_max_parallel()
    # 确保每个进程至少 1 线程，最多不超过总核心数
    threads = max(1, cpu_cores // effective_parallel)
    threads = min(threads, cpu_cores)
    return threads


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
    job_counter = 0  # 全局任务计数器，用于 GPU 轮转分配
    
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
            
            # GPU 分配：从 GPU_POOL 中轮转
            gpu_id = GPU_POOL[job_counter % len(GPU_POOL)] if GPU_POOL else ""
            cuda_arg = gpu_id if gpu_id != "" else '""'
            
            cmd = f'{PYTHON} main_Tabular_CLF.py -algorithm {algo} -dataset {ds} -batch_size {BATCH_SIZE} -test_batch_size {BATCH_SIZE} -cuda {cuda_arg} -task Tabular_CLF -learning_rate 3e-4 -model_type ANN -resume -exp_repeat_times {EXP_REPEAT_TIMES} -parallel_repeats {PARALLEL_REPEATS}'
            tabular_jobs.append({"cmd": cmd, "name": f"[Tabular] {algo} / {ds}", "type": "tabular", 
                               "algo": algo, "dataset": ds, "hypothesis": "ANN", "gpu": gpu_id})
            job_counter += 1
    
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
            
            gpu_id = GPU_POOL[job_counter % len(GPU_POOL)] if GPU_POOL else ""
            cuda_arg = gpu_id if gpu_id != "" else '""'
            
            cmd = f'{PYTHON} main_IMG_CLF.py -algorithm {algo} -dataset {ds} -batch_size {IMG_BATCH_SIZE} -test_batch_size {IMG_BATCH_SIZE} -cuda {cuda_arg} -task IMG_CLF -learning_rate 3e-4 -resume -exp_repeat_times {EXP_REPEAT_TIMES} -parallel_repeats {PARALLEL_REPEATS}'
            image_jobs.append({"cmd": cmd, "name": f"[Image] {algo} / {ds}", "type": "image",
                              "algo": algo, "dataset": ds, "hypothesis": "BERTCLASSIFIER", "gpu": gpu_id})
            job_counter += 1
    
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
            
            gpu_id = GPU_POOL[job_counter % len(GPU_POOL)] if GPU_POOL else ""
            cuda_arg = gpu_id if gpu_id != "" else '""'
            
            cmd = f'{PYTHON} main_SENT_CLF.py -algorithm {algo} -dataset {ds} -batch_size {SENT_BATCH_SIZE} -test_batch_size {SENT_BATCH_SIZE} -cuda {cuda_arg} -task SENT_CLF -learning_rate 3e-4 -resume -exp_repeat_times {EXP_REPEAT_TIMES} -parallel_repeats {PARALLEL_REPEATS}'
            sent_jobs.append({"cmd": cmd, "name": f"[Sent] {algo} / {ds}", "type": "sent",
                              "algo": algo, "dataset": ds, "hypothesis": "BERTCLASSIFIER", "gpu": gpu_id})
            job_counter += 1
    
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
    
    # 自动计算 MAX_PARALLEL
    if MAX_PARALLEL <= 0:
        MAX_PARALLEL = _auto_max_parallel()
    
    # 动态计算每个进程的 OMP 线程数
    threads_per_job = _get_threads_per_job()
    os.environ["OMP_NUM_THREADS"] = str(threads_per_job)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_job)
    
    running = []
    done = 0
    failed = 0
    total = len(jobs)
    idx = 0

    print(f"\nTotal jobs: {total}")
    print(f"Max parallel: {MAX_PARALLEL}")
    print(f"OMP threads per job: {threads_per_job}")
    print(f"GPU pool: {GPU_POOL if GPU_POOL else 'CPU only'}")
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
            
            p = subprocess.Popen(job["cmd"], shell=True)
            pid = p.pid
            gpu_info = job.get("gpu", "")
            gpu_tag = f" GPU:{gpu_info}" if gpu_info else " CPU"
            print(f"  Starting: {job['name']} ({idx+1}/{total}) [PID: {pid},{gpu_tag}, Tab: {tab_running}, Img: {img_running}, Sent: {sent_running}]")
            
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