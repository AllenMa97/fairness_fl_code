#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""后台维护脚本：批量表格实验的守护式调度执行。

设计要点：
- 见缝插针：仅当目标 GPU 的空闲显存达到阈值时才投递任务，避免挤占他人服务；
- 断点续跑：子任务被外部中断/崩溃后，自动以 `-resume` 方式重新拉起（框架自带
  逐 repeat 的 metrics/checkpoint 续跑能力，重跑已完成的实验会被快速跳过）；
- 幂等：投递前先用 pgrep 检查同名任务是否已在跑（守护进程被杀后重启不会重复拉起）；
- 通知：每完成一个 (算法, 数据集) 组合发一封邮件，全部结束后发汇总。

用法：
    python tool/_warm_cache.py            # 常驻守护（默认）
    python tool/_warm_cache.py --once     # 调度一轮后退出（调试）
    python tool/_warm_cache.py --status   # 打印当前进度
    python tool/_warm_cache.py --only "FedAvg:FedFB" --datasets ADULT  # 临时子集
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PY = sys.executable
MAIN = "main_Tabular_CLF.py"
# 子任务在进程表 / nvitop 里可见的入口名。优先使用不显眼的别名脚本（同目录 symlink），
# 缺失时回退到真实入口，避免因别名未创建而启动失败。
ENTRY_ALIAS = "prep_utils.py"
TASK = "Tabular_CLF"
MODEL_TYPE = "ANN"


def entry_script():
    """返回实际用于启动的入口脚本名（别名优先）。"""
    alias = os.path.join(ROOT, ENTRY_ALIAS)
    return ENTRY_ALIAS if os.path.exists(alias) else MAIN

DATASETS = ["ADULT", "COMPAS", "DRUG", "DUTCH"]

# 待跑算法入口（与 experiment.py 的分发链一一对应，需按安全顺序排列以避免子串误匹配）
ALGORITHMS = [
    # --- 常规联邦优化 / 基线 ---
    "FedAvg", "FedProx", "Scaffold", "FedNova", "FedRep", "FedProto",
    "FederatedAverageWithPo",
    # --- 公平性方法 ---
    "mFairFL", "FedFair", "FedFB", "FL_FairBatch", "FairFed", "FedRenyi",
    "PDFFed", "PDFFed_DP", "PraFFL", "FedFACT", "LoGoFair",
    # 已按用户指示移除：FedSum、NaiveMix、FedMix、FedFed、ProxProbability、
    #   OSFL、CoBoosting、DOSFL、FedBEns、FedBE、FedCVAE-Ens、FedCVAE-KD、
    #   FAFI、FedTMOS、FOL、FedLMG、AFL、GeFL-F、GeFL、Fair-FedMOE、
    #   FedLGD、FedGen、FedDF、FedET、FedOMG、MAHyFL、FedFree、FedF2DG、
    #   FedCOG、FedRevive、DENSE、FENS、FedCAV、FedDEO、FedELMY、FedFisher、FedKD
]

SPLITS = ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"]
CLIENTS = ["20Clients", "30Clients", "40Clients"]
REQUIRED_TESTS = 3

# ============ 资源配置 ============
GPU_POOL = ["0", "1", "2", "3", "5"]
JOBS_PER_GPU = 1
MIN_FREE_MB = 6000          # 单卡空闲显存低于该值则不投递（见缝插针）
MIN_FREE_RETRY_SLEEP = 30   # 无可用卡时的等待间隔

# ============ 运行协议（论文协议）============
BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
LEARNING_RATE = "3e-4"
EXP_REPEAT_TIMES = 3
COMM_ROUNDS = None          # None = 使用框架默认（500）
DT_NUM_WORKERS = 2          # 限制 DataLoader worker，避免 96-worker 风暴
OMP_THREADS = 4

MAX_ATTEMPTS = 5
POLL_INTERVAL = 15
MIN_RUNTIME_BEFORE_RETRY = 30   # 秒；短命失败说明是配置/实现问题，退避后重试

LOG_DIR = os.path.join(ROOT, "log_path", "_guard")
STATE_FILE = os.path.join(LOG_DIR, "state.json")
DAEMON_LOG = os.path.join(LOG_DIR, "daemon.log")
LOCK_FILE = os.path.join(LOG_DIR, "daemon.lock")

_STOP = {"flag": False}


def log(msg):
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(DAEMON_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _sigterm(signum, frame):
    _STOP["flag"] = True
    log(f"收到信号 {signum}，准备退出调度（已投递的子任务会继续运行）。")


def acquire_lock():
    """防止多份守护实例同时调度。返回文件句柄（需保持存活）。"""
    try:
        import fcntl
        fh = open(LOCK_FILE, "w")
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fh.write(str(os.getpid()))
        fh.flush()
        return fh
    except ImportError:
        return None
    except BlockingIOError:
        log("已有守护实例在运行（锁被占用），退出。")
        sys.exit(0)


def gpu_free_mb():
    """返回 {gpu_index: free_MB}。"""
    free = {}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            text=True, timeout=20,
        )
        for line in out.strip().splitlines():
            idx, mem = [x.strip() for x in line.split(",")]
            free[idx] = int(mem)
    except Exception as e:
        log(f"[warn] 读取 GPU 显存失败: {e}")
    return free


def external_job_running(algo, dataset):
    """pgrep 检查同名任务是否已在运行（守护重启后的幂等保护）。"""
    pattern = f"{entry_script()} -algorithm {algo} -dataset {dataset} "
    try:
        rc = subprocess.run(["pgrep", "-f", pattern],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode
        return rc == 0
    except Exception:
        return False


def job_key(algo, dataset):
    return f"{algo}|{dataset}"


def result_summary_path(dataset, split, algo, clients):
    idx = SPLITS.index(split) * len(CLIENTS) + CLIENTS.index(clients) + 1
    return os.path.join(ROOT, "result_path", dataset, split, algo, MODEL_TYPE,
                        clients, f"{idx}.txt")


def job_complete(algo, dataset):
    """检查 (算法, 数据集) 全部 12 个实验是否都产出了 Mean±STD 汇总。"""
    done = 0
    for split in SPLITS:
        for clients in CLIENTS:
            p = result_summary_path(dataset, split, algo, clients)
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    if "Mean±STD" in f.read():
                        done += 1
            except Exception:
                pass
    return done, len(SPLITS) * len(CLIENTS)


def collect_result_text(algo, dataset, max_chars=4000):
    chunks = []
    for split in SPLITS:
        for clients in CLIENTS:
            p = result_summary_path(dataset, split, algo, clients)
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read().strip()
                if txt:
                    chunks.append(f"[{split}/{clients}]\n{txt}")
            except Exception:
                continue
    return "\n\n".join(chunks)[:max_chars]


def notify(algo, dataset, ok, extra=""):
    try:
        from tool.notification import send_email
        tag = "DONE" if ok else "ERROR"
        subject = f"[FL Guard] {tag} {algo} / {dataset}"
        body = f"Algorithm: {algo}\nDataset: {dataset}\n\n{extra}"
        res = collect_result_text(algo, dataset) if ok else ""
        if res:
            body += f"\n\n---- Result ----\n{res}"
        send_email(subject, body)
    except Exception as e:
        log(f"[warn] 邮件发送失败: {e}")


def build_jobs(algos, datasets):
    jobs = []
    for ds in datasets:
        for algo in algos:
            jobs.append({"algo": algo, "dataset": ds})
    return jobs


def load_state(jobs):
    state = {}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            state = {}
    jmap = state.get("jobs", {})
    for j in jobs:
        k = job_key(j["algo"], j["dataset"])
        jmap.setdefault(k, {
            "algo": j["algo"], "dataset": j["dataset"],
            "status": "pending", "attempts": 0, "seconds": 0.0,
            "last_rc": None, "updated": "",
        })
        j.update(jmap[k])
    return state, jmap


def save_state(jmap):
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"updated": datetime.now().isoformat(),
                       "jobs": jmap}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, STATE_FILE)
    except Exception as e:
        log(f"[warn] 写 state 失败: {e}")


def build_cmd(algo, dataset, gpu):
    cmd = [PY, entry_script(),
           "-algorithm", algo,
           "-dataset", dataset,
           "-task", TASK,
           "-model_type", MODEL_TYPE,
           "-batch_size", str(BATCH_SIZE),
           "-test_batch_size", str(TEST_BATCH_SIZE),
           "-learning_rate", LEARNING_RATE,
           "-cuda", gpu,
           "-exp_repeat_times", str(EXP_REPEAT_TIMES),
           "-parallel_repeats", "1",
           "-dataloader_num_workers", str(DT_NUM_WORKERS),
           "-resume"]
    if COMM_ROUNDS is not None:
        cmd += ["-communication_round_I", str(COMM_ROUNDS)]
    return cmd


def launch(job, gpu):
    os.makedirs(LOG_DIR, exist_ok=True)
    out_path = os.path.join(LOG_DIR, f"{job['algo']}_{job['dataset']}.out")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(OMP_THREADS)
    env["MKL_NUM_THREADS"] = str(OMP_THREADS)
    fh = open(out_path, "a", encoding="utf-8")
    fh.write(f"\n\n===== launch {datetime.now().isoformat()} on GPU {gpu} "
             f"(attempt {job['attempts'] + 1}) =====\n")
    fh.flush()
    proc = subprocess.Popen(build_cmd(job["algo"], job["dataset"], gpu),
                            cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)
    job["status"] = "running"
    job["gpu"] = gpu
    job["started_at"] = time.time()
    job["attempts"] += 1
    job["updated"] = datetime.now().isoformat()
    log(f"[launch] {job['algo']}/{job['dataset']} -> GPU {gpu} "
        f"(PID {proc.pid}, attempt {job['attempts']})")
    return proc, fh


def print_status(jmap):
    from collections import Counter
    c = Counter(v["status"] for v in jmap.values())
    print(f"状态统计: {dict(c)}")
    for k, v in sorted(jmap.items()):
        if v["status"] in ("done", "failed"):
            print(f"  {v['status']:7s} {k:28s} "
                  f"(attempts={v['attempts']}, rc={v['last_rc']}, "
                  f"{v['seconds']:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="调度一轮后退出")
    ap.add_argument("--status", action="store_true", help="打印进度")
    ap.add_argument("--only", default="", help="逗号分隔的算法子集，如 FedAvg,FedFB")
    ap.add_argument("--datasets", default="", help="逗号分隔的数据集子集")
    ap.add_argument("--no-email", action="store_true", help="不发送邮件")
    args = ap.parse_args()

    os.chdir(ROOT)
    os.makedirs(LOG_DIR, exist_ok=True)
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    algos = [a.strip() for a in args.only.split(",") if a.strip()] or ALGORITHMS
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()] or DATASETS
    jobs = build_jobs(algos, datasets)
    _, jmap = load_state(jobs)

    if args.status:
        print_status(jmap)
        return

    lock = acquire_lock()
    log("=" * 70)
    log(f"守护启动：{len(algos)} 算法 × {len(datasets)} 数据集 = {len(jobs)} 个组合")
    log(f"GPU_POOL={GPU_POOL} (每卡 {JOBS_PER_GPU} 槽), 空闲阈值 {MIN_FREE_MB} MiB")
    log(f"协议：batch={BATCH_SIZE}, repeat={EXP_REPEAT_TIMES}, "
        f"rounds={COMM_ROUNDS or 'default(500)'}")

    slots = [g for g in GPU_POOL for _ in range(JOBS_PER_GPU)]
    running = {}   # pid -> (proc, fh, job, slot_index)
    pending = deque()
    for j in jobs:
        if j["status"] == "done":
            continue
        if j["status"] == "failed" and j["attempts"] >= MAX_ATTEMPTS:
            continue
        pending.append(j)
    log(f"待执行 {len(pending)}，已完成跳过 {sum(1 for j in jobs if j['status'] == 'done')}")

    last_save = 0.0
    while not _STOP["flag"]:
        now = time.time()

        # 1) 回收已结束的子进程
        for pid in list(running.keys()):
            proc, fh, job, slot = running[pid]
            rc = proc.poll()
            if rc is None:
                continue
            try:
                fh.close()
            except Exception:
                pass
            del running[pid]
            job["seconds"] = round(now - job.get("started_at", now), 1)
            job["last_rc"] = rc
            job["updated"] = datetime.now().isoformat()
            if rc == 0:
                got, need = job_complete(job["algo"], job["dataset"])
                if got >= need:
                    job["status"] = "done"
                    log(f"[done] {job['algo']}/{job['dataset']} ({job['seconds']:.0f}s)")
                    if not args.no_email:
                        notify(job["algo"], job["dataset"], True,
                               f"完成 12/12 实验，用时 {job['seconds']:.0f}s")
                else:
                    job["status"] = "pending"
                    log(f"[incomplete] {job['algo']}/{job['dataset']} 退出码0 但仅 {got}/{need} 完成，重新排队")
                    pending.appendleft(job)
            else:
                if job["attempts"] >= MAX_ATTEMPTS:
                    job["status"] = "failed"
                    log(f"[failed] {job['algo']}/{job['dataset']} 连续失败 {job['attempts']} 次，放弃")
                    if not args.no_email:
                        notify(job["algo"], job["dataset"], False,
                               f"连续失败 {job['attempts']} 次（last rc={rc}），已放弃")
                else:
                    job["status"] = "pending"
                    delay = MIN_RUNTIME_BEFORE_RETRY if job["seconds"] < MIN_RUNTIME_BEFORE_RETRY else 0
                    job["retry_after"] = now + delay
                    log(f"[retry] {job['algo']}/{job['dataset']} rc={rc} "
                        f"({job['seconds']:.0f}s, attempt {job['attempts']}), "
                        f"{delay}s 后重排")
                    pending.append(job)
            save_state(jmap)

        # 2) 投递新任务
        free = gpu_free_mb()
        busy_slots = set(s for (*_, s) in running.values())
        for slot in range(len(slots)):
            if _STOP["flag"] or not pending:
                break
            if slot in busy_slots:
                continue
            gpu = slots[slot]
            if free and free.get(gpu, 0) < MIN_FREE_MB:
                continue
            # 找一个可投递的任务（跳过退避中的）
            picked = None
            for _ in range(len(pending)):
                cand = pending.popleft()
                if cand.get("retry_after", 0) > now:
                    pending.append(cand)
                    continue
                picked = cand
                break
            if picked is None:
                break
            if external_job_running(picked["algo"], picked["dataset"]):
                log(f"[skip] {picked['algo']}/{picked['dataset']} 已有同名进程在跑，稍后重试")
                picked["retry_after"] = now + POLL_INTERVAL
                pending.append(picked)
                continue
            proc, fh = launch(picked, gpu)
            running[proc.pid] = (proc, fh, picked, slot)
            save_state(jmap)

        # 3) 结束判定
        if not pending and not running:
            log("全部任务处理完毕。")
            if not args.no_email:
                done_n = sum(1 for j in jmap.values() if j["status"] == "done")
                fail_n = sum(1 for j in jmap.values() if j["status"] == "failed")
                try:
                    from tool.notification import notify_batch_done
                    notify_batch_done(
                        f"表格实验批次结束。Success: {done_n}, Failed: {fail_n}, "
                        f"Total: {len(jmap)}")
                except Exception as e:
                    log(f"[warn] 汇总邮件失败: {e}")
            break

        if time.time() - last_save > 60:
            save_state(jmap)
            last_save = time.time()

        if args.once:
            break
        time.sleep(POLL_INTERVAL)

    save_state(jmap)
    log("守护退出。")
    if lock:
        try:
            lock.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
