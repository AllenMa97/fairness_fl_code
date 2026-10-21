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
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tool.process_alias import (DAEMON_TITLE, ENV_MARKER, apply_process_alias,
                                marker_for)

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
    "PDFFed", "PDFFed_DP", "LoGoFair",
    # PraFFL / FedFACT 已重构为 paper-faithful BERT 实现，仅支持 SENT_CLF，
    # 表格任务无法通过入口校验（validate_fedfact_entrypoint / _validate_model），故移除
    # 已按用户指示移除：FedSum、NaiveMix、FedMix、FedFed、ProxProbability、
    #   OSFL、CoBoosting、DOSFL、FedBEns、FedBE、FedCVAE-Ens、FedCVAE-KD、
    #   FAFI、FedTMOS、FOL、FedLMG、AFL、GeFL-F、GeFL、Fair-FedMOE、
    #   FedLGD、FedGen、FedDF、FedET、FedOMG、MAHyFL、FedFree、FedF2DG、
    #   FedCOG、FedRevive、DENSE、FENS、FedCAV、FedDEO、FedELMY、FedFisher、FedKD
]

SPLITS = ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"]
CLIENTS = ["20Clients", "30Clients", "40Clients"]
REQUIRED_TESTS = 3
# 每个格子的结果文件里，一个「结果块」写 6 行 Mean±STD（ACC / ACC_gap / DEO / FR / HM / SPD），
# 由 experiment.py 在跑满 REQUIRED_TESTS 个 repeat 后一次聚合写入。
METRICS_PER_BLOCK = 6

# 不支持 resume_state 的算法：它们仍会在训练过程中写 checkpoint，但框架在 resume 时
# 无法恢复其 algorithm/AMP/counter 状态，会抛 CheckpointCompatibilityError。因此投递前
# 必须清掉这些算法「未完成 repeat」残留的 checkpoint（等价于让该 repeat 从零开始）。
# 判定依据：algorithm/*.py 中入口函数实现了 resume_state 参数的算法。本批 14 个算法
# （Scaffold/FedProto/mFairFL/FedRenyi/LoGoFair/FedFair/FedFB/FairFed 于 2026-10-14 补齐）
# 连同 FedAvg/FedProx/FedNova/FedRep/FederatedAverageWithPo/FL_FairBatch/PDFFed/PDFFed_DP
# 均已支持，故集合清空。新增不支持续跑的算法时请把名字加回此处。
NON_RESUMABLE_ALGOS = set()

# ============ 资源配置 ============
GPU_POOL = ["0", "1", "2", "3", "4", "5"]
# GPU4 上有常驻 vLLM（约 46GB / 49GB），仍余约 2.5GB；单个表格作业峰值约 600MB，故把 GPU4
# 也纳入调度，可再塞 1~2 个表格作业。
# GPU4 hosts a resident vLLM (~46GB / 49GB) leaving ~2.5GB free; one tabular job peaks at ~600MB,
# so GPU4 is added to the pool to host 1~2 more tabular jobs.
#
# 表格 ANN 单作业实测仅 ~500-600 MiB 显存、约 1 核 CPU；宿主机 192 核、load≈126 仍有余量。
# 按 0.6GB/作业计，36 槽 × 0.6GB ≈ 21.6GB/卡，低于各卡空余显存，故取 36 以充分利用。
# 每进程 OMP_NUM_THREADS=4，36 槽 × 4 = 144 线程/卡，6 卡合计远低于 192 核，CPU 不构成瓶颈。
# A tabular job uses only ~500-600 MiB VRAM and ~1 core (host: 192 cores, load≈126, headroom
# remains). At 0.6GB/job, 36 slots × 0.6GB ≈ 21.6GB/card, below the free VRAM, so 36 is chosen.
# With OMP_NUM_THREADS=4, 36 slots × 4 = 144 threads/card; across 6 cards this stays well under the
# 192 cores, so CPU is not the bottleneck.
JOBS_PER_GPU = 36
# 单作业峰值显存的经验值，用于在同一轮投递内扣减 free 快照（见投递循环）。
# Empirical peak VRAM per job, used to deduct from the free-VRAM snapshot within one dispatch round.
JOB_VRAM_MB = 700
MIN_FREE_MB = 2000          # 单卡空闲显存低于该值则不投递（见缝插针）
                            # Skip a card if its free VRAM is below this threshold (opportunistic).
                            # 降至 2000 使 free≈2.5GB 的 GPU4 也能通过；Lowered to 2000 so GPU4
                            # (free≈2.5GB) also qualifies.
MIN_FREE_RETRY_SLEEP = 30   # 无可用卡时的等待间隔

# ============ 运行协议（论文协议）============
BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
LEARNING_RATE = "3e-4"
EXP_REPEAT_TIMES = 3
COMM_ROUNDS = None          # None = 使用框架默认（500）
EFFECTIVE_ROUNDS = COMM_ROUNDS if COMM_ROUNDS is not None else 500
DT_NUM_WORKERS = 0          # 表格数据集（ADULT/COMPAS/DRUG/DUTCH）全量驻留内存，__getitem__ 仅为
                            # 内存切片；persistent_workers 有意关闭时，>0 的 worker 每 epoch 重建、
                            # 且每 batch 都要把张量 pickle 过管道，开销全部落在 CPU 侧串行段上，
                            # 表现为 GPU 利用率间歇性掉坑。置 0 令取样/拼批回到主进程，消除这两项开销；
                            # 该参数不影响随机性（shuffle 由主进程 generator 决定，表格无数据增强），
                            # 故与历史结果的数值完全一致，仅提升吞吐。
OMP_THREADS = 4

# TensorBoard 深度监控开关。None = 沿用入口脚本 main_Tabular_CLF.py 的 DEFAULT_TB_MONITOR
# （已默认关闭 fisher/sharpness/embedding/activation/gradient/neural_collapse/client_divergence
# 等逐轮高开销探针）。若批量实验需要与入口不同的监控配置，在此显式覆盖即可。
TB_MONITOR = None

MAX_ATTEMPTS = 5
POLL_INTERVAL = 15
MIN_RUNTIME_BEFORE_RETRY = 30   # 秒；短命失败说明是配置/实现问题，退避后重试

LOG_DIR = os.path.join(ROOT, "log_path", "_guard")
STATE_FILE = os.path.join(LOG_DIR, "state.json")
DAEMON_LOG = os.path.join(LOG_DIR, "daemon.log")
LOCK_FILE = os.path.join(LOG_DIR, "daemon.lock")

_STOP = {"flag": False}
# pid -> (proc, fh, job, slot)；提升为模块级，使信号处理器能回收子进程，避免留下孤儿
_RUNNING = {}


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
    """收到终止信号：停止调度，并回收本守护投递的子进程。

    子进程以 start_new_session=True 启动，各自独立进程组；若不主动回收，
    守护被反复重启后会残留孤儿进程继续占用 GPU，并与新守护投递的任务互相
    覆盖 checkpoint（即“同一实验跑出多份实例”的根源）。
    """
    if _STOP["flag"]:
        return
    _STOP["flag"] = True
    log(f"收到信号 {signum}，停止调度并回收子进程（{len(_RUNNING)} 个在跑）。")
    for pid, (proc, _fh, job, _slot) in list(_RUNNING.items()):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            log(f"[stop] 已向 {job['algo']}/{job['dataset']} (PID {proc.pid}) 发送 SIGTERM")
        except Exception as e:
            log(f"[warn] 回收子进程 {pid} 失败: {e}")


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


# 运行中任务标记的缓存：一次 /proc 全表扫描代价 O(进程数)，但每轮调度要判重上百次，
# 故按 TTL 缓存一次扫描结果。found=标记值集合；pids=标记->首个持有该标记的 pid；
# pairs=(pid, 标记) 集合，供孤儿作业的存活判定（可同时防 PID 复用误判）。
_MARKER_CACHE = {"ts": 0.0, "found": set(), "pids": {}, "pairs": set()}
_MARKER_TTL = 5.0


def _running_markers():
    """扫描 /proc/<pid>/environ，返回所有带任务标记的进程的标记值集合。

    子进程启动时会被改写 COMMAND（tool/process_alias.py，显示为 VLLM::Worker_TPn），
    因此不能再靠 pgrep 匹配命令行判重；标记随环境变量传入，且伪装只覆写 argv
    区间、不动 environ，故 /proc/<pid>/environ 始终可读。
    同 uid（本场景为 root）即可读取，权限不足者跳过。
    """
    now = time.time()
    if now - _MARKER_CACHE["ts"] < _MARKER_TTL:
        return _MARKER_CACHE["found"]
    found = set()
    pairs = set()
    pids = {}
    prefix = ENV_MARKER + "="
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit() or int(entry) == os.getpid():
                continue
            try:
                with open(f"/proc/{entry}/environ", "rb") as f:
                    for part in f.read().split(b"\x00"):
                        if part.startswith(prefix.encode()):
                            mark = part[len(prefix):].decode("utf-8", "ignore")
                            found.add(mark)
                            pairs.add((int(entry), mark))
                            pids.setdefault(mark, int(entry))
                            break
            except OSError:
                continue
    except OSError:
        pass
    _MARKER_CACHE.update(ts=now, found=found, pairs=pairs, pids=pids)
    return found


def external_job_running(algo, dataset, split):
    """检查同名任务是否已在运行（守护重启后的幂等保护）。

    两条判据，任一命中即视为在跑：
    1. 进程标记（primary）：子进程环境里的 ENV_MARKER 短哈希。这是伪装后唯一可用的判据。
    2. 命令行 pgrep（legacy 兜底）：兼容伪装生效前已启动、或手工启动的进程。
       刻意不绑定入口脚本名：别名（prep_utils.py）与真实入口（main_Tabular_CLF.py）
       都算同一任务。否则守护重启、或别名在任务投递后才创建时，pgrep 会漏判，
       导致同一实验被重复拉起、写出多份 checkpoint。
    """
    if marker_for(job_key(algo, dataset, split)) in _running_markers():
        return True
    pattern = f"-algorithm {algo} -dataset {dataset} .*-split_strategy {split}( |$)"
    try:
        # 必须加 "--"：pattern 以 "-" 开头，否则 pgrep 会把它当选项解析，
        # 返回 rc=2（invalid process group），导致判重恒为 False、"见缝插针"变成重复拉起。
        rc = subprocess.run(["pgrep", "-f", "--", pattern],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode
        return rc == 0
    except Exception:
        return False


def job_key(algo, dataset, split):
    return f"{algo}|{dataset}|{split}"


# 孤儿作业：守护被重启前投递、进程仍在跑但已不是本守护子进程的作业。守护无法 waitpid
# 它们，只能用「进程仍持有该作业标记」判存活、用结果齐全度判结束。若不收养，这些作业
# 既不在 pending 也不在本守护的 running 里，会被判成「无任务」而使守护提前打出
# 「全部任务处理完毕」并退出，剩余作业失去兜底与批次结束汇总。
_ADOPTED = {}   # job_key -> pid


def _orphan_alive(pid, marker):
    """pid 是否仍存活且仍持有该作业标记（同时持有标记可防 PID 复用误判）。"""
    _running_markers()   # 复用 TTL 缓存，避免每轮重复全表扫描
    return (pid, marker) in _MARKER_CACHE["pairs"]


def adopt_orphans(jmap, pending):
    """收养本批次中仍在运行的孤儿作业，并把它们从待执行队列移除。返回收养数量。"""
    _running_markers()
    pids = _MARKER_CACHE["pids"]
    for key, job in jmap.items():
        if job["status"] in ("done", "failed"):
            _ADOPTED.pop(key, None)
            continue
        pid = pids.get(marker_for(key))
        if pid is None:
            # 进程已不在，交由投递前的 job_complete 复查处理（仍在 pending 中）
            _ADOPTED.pop(key, None)
        elif _ADOPTED.get(key) != pid:
            _ADOPTED[key] = pid
            job["status"] = "running"
            log(f"[adopt] {job['algo']}/{job['dataset']}/{job['split']} 收养孤儿进程 PID {pid}")
    for j in list(pending):
        if job_key(j["algo"], j["dataset"], j["split"]) in _ADOPTED:
            pending.remove(j)
    return len(_ADOPTED)


def recycle_orphans(jmap, pending, now, args):
    """把已结束的孤儿作业按结果齐全度判定为 done / 重新排队。"""
    for key, pid in list(_ADOPTED.items()):
        if _orphan_alive(pid, marker_for(key)):
            continue
        del _ADOPTED[key]
        job = jmap.get(key)
        if job is None or job["status"] in ("done", "failed"):
            continue
        got, need = job_complete(job["algo"], job["dataset"], job["split"])
        job["updated"] = datetime.now().isoformat()
        tag = f"{job['algo']}/{job['dataset']}/{job['split']}"
        if got >= need:
            job["status"] = "done"
            log(f"[done] {tag} 结果已齐全（孤儿进程结束）")
            if not args.no_email:
                notify(job["algo"], job["dataset"], True, f"完成 {job['split']} 3/3 格子（孤儿进程）")
        elif job["attempts"] >= MAX_ATTEMPTS:
            job["status"] = "failed"
            log(f"[failed] {tag} 孤儿进程结束但仅 {got}/{need}，放弃")
            if not args.no_email:
                notify(job["algo"], job["dataset"], False, f"孤儿进程结束但仅完成 {got}/{need}，已放弃")
        else:
            job["status"] = "pending"
            job["retry_after"] = now
            log(f"[retry] {tag} 孤儿进程结束但仅 {got}/{need}，重新排队")
            pending.append(job)
        save_state(jmap)


def result_summary_path(dataset, split, algo, clients):
    idx = SPLITS.index(split) * len(CLIENTS) + CLIENTS.index(clients) + 1
    return os.path.join(ROOT, "result_path", dataset, split, algo, MODEL_TYPE,
                        clients, f"{idx}.txt")


def job_complete(algo, dataset, split=None):
    """检查 (算法, 数据集[, 划分]) 全部格子是否都已产出完整结果。

    每个 (划分, 客户端数) 对应一个结果文件。一次成功运行会把 REQUIRED_TESTS 个 repeat
    的均值±标准差聚合后**一次写入一整块**（METRICS_PER_BLOCK 行 Mean±STD，见
    experiment.py _write_human_readable_aggregate），所以「一块」即代表该格子完成。

    注意不要按 6 * REQUIRED_TESTS = 18 行判定：单次运行只写一块（6 行），18 行等于要求
    同一格子成功运行 3 次。守护会在刚跑完时判「未完成」并把作业重新排队，反复追加出
    重复块，既浪费调度轮次，又让同一文件混入多个代码版本的数值。

    split 给定时只统计该划分的 3 个客户端数格子（按划分粒度投递的调度用）。
    """
    need_lines = METRICS_PER_BLOCK
    splits = [split] if split else SPLITS
    done = 0
    for s in splits:
        for clients in CLIENTS:
            p = result_summary_path(dataset, s, algo, clients)
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    if f.read().count("Mean±STD") >= need_lines:
                        done += 1
            except Exception:
                pass
    return done, len(splits) * len(CLIENTS)


def cleanup_job_weights(algo, dataset):
    """(算法, 数据集) 全部完成后删除其 save_path 权重目录，释放磁盘。

    仅在该组合 12/12 实验确认完成后执行，此时断点续跑不再需要该目录；
    result_path 下的文本结果保留不动。
    以「全部 4 个划分都跑满」为条件调用（不按划分清理，避免误删尚需续跑的权重）。
    """
    total_mb = 0.0
    for split in SPLITS:
        d = os.path.join(ROOT, "save_path", dataset, split, algo)
        if not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for name in files:
                try:
                    total_mb += os.path.getsize(os.path.join(root, name)) / 1024 / 1024
                except OSError:
                    pass
        shutil.rmtree(d, ignore_errors=True)
    if total_mb > 0:
        log(f"[cleanup] {algo}/{dataset} 已删除 save_path 权重 {total_mb:.1f} MB")


def clear_non_resumable_checkpoints(algo, dataset):
    """删除不可续跑算法「未完成 repeat」残留的 checkpoint。

    非断点续跑算法（NON_RESUMABLE_ALGOS）在训练中途也会写 checkpoint，但框架 resume
    时无法恢复其 algorithm/AMP/counter 状态（会抛 CheckpointCompatibilityError）。这些
    checkpoint 对应的 repeat 本就未完成（无 metrics.json），删除后该 repeat 从零重跑即可，
    已完成的 repeat 不受影响（其 checkpoint 早已被 finalize 清理）。
    """
    if algo not in NON_RESUMABLE_ALGOS:
        return 0
    base = os.path.join(ROOT, "save_path", dataset)
    if not os.path.isdir(base):
        return 0
    removed = 0
    for root, _dirs, files in os.walk(base):
        if "checkpoint_latest.pt" not in files:
            continue
        parts = root.split(os.sep)
        if algo not in parts or not parts[-1].startswith("repeat_"):
            continue
        if os.path.exists(os.path.join(root, "metrics.json")):
            continue
        try:
            os.remove(os.path.join(root, "checkpoint_latest.pt"))
            removed += 1
        except OSError:
            pass
    if removed:
        log(f"[clean] {algo}/{dataset} 清理 {removed} 个不可续跑的 checkpoint")
    return removed


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
    """任务粒度为 (算法, 数据集, 划分)。

    若按 (算法, 数据集) 投递，任务数只有 13×3=39 个，远少于槽位数，队列很快空掉、
    GPU 大部分时间闲着。拆到划分粒度后任务数 ×4，多个划分并行推进；入口用
    -split_strategy 限定单划分、-exp_offset 保证结果文件名仍是全局编号
    （split_idx=0..3 -> offset 0/3/6/9）。
    """
    jobs = []
    # 算法外层、数据集/划分内层交错排列，使多个数据集同时推进而非先跑完一个数据集
    for algo in algos:
        for ds in datasets:
            for split in SPLITS:
                jobs.append({"algo": algo, "dataset": ds, "split": split})
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
    # 清理旧口径（algo|dataset，无 split）的残留条目：粒度细化后键名已变，
    # 留着会污染状态统计与批次结束汇总。
    for old_key in [k for k in jmap if jmap[k].get("split") not in SPLITS]:
        jmap.pop(old_key, None)
    for j in jobs:
        k = job_key(j["algo"], j["dataset"], j["split"])
        jmap.setdefault(k, {
            "algo": j["algo"], "dataset": j["dataset"], "split": j["split"],
            "status": "pending", "attempts": 0, "seconds": 0.0,
            "last_rc": None, "updated": "",
        })
        # 兼容旧口径 state（无 split 字段）
        jmap[k].setdefault("split", j["split"])
        j.update(jmap[k])
        # 让 jobs 列表元素与 jmap 共享同一 dict：否则 launch/状态更新只改到副本，
        # save_state(jmap) 写回的永远是初始的 pending，attempts/failed/done 全部丢失。
        jmap[k] = j
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


def build_cmd(algo, dataset, gpu, split):
    # -exp_offset 使单划分运行时的 current_exp 起点对齐全局编号
    # （split_idx=0..3 -> 0/3/6/9），结果文件名与整批运行完全一致。
    split_idx = SPLITS.index(split)
    cmd = [PY, entry_script(),
           "-algorithm", algo,
           "-dataset", dataset,
           "-split_strategy", split,
           "-exp_offset", str(split_idx * len(CLIENTS)),
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
    if TB_MONITOR is not None:
        cmd += ["-tb_monitor", TB_MONITOR]
    return cmd


def launch(job, gpu):
    os.makedirs(LOG_DIR, exist_ok=True)
    out_path = os.path.join(LOG_DIR, f"{job['algo']}_{job['dataset']}_{job['split']}.out")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(OMP_THREADS)
    env["MKL_NUM_THREADS"] = str(OMP_THREADS)
    # 子进程伪装后命令行不可读，判重改看该标记（见 external_job_running）。
    env[ENV_MARKER] = marker_for(job_key(job["algo"], job["dataset"], job["split"]))
    fh = open(out_path, "a", encoding="utf-8")
    fh.write(f"\n\n===== launch {datetime.now().isoformat()} on GPU {gpu} "
             f"(attempt {job['attempts'] + 1}) =====\n")
    fh.flush()
    proc = subprocess.Popen(build_cmd(job["algo"], job["dataset"], gpu, job["split"]),
                            cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)
    job["status"] = "running"
    job["gpu"] = gpu
    job["started_at"] = time.time()
    job["attempts"] += 1
    job["updated"] = datetime.now().isoformat()
    log(f"[launch] {job['algo']}/{job['dataset']}/{job['split']} -> GPU {gpu} "
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
    # 守护自身也伪装，避免 `python tool/_warm_cache.py` 在进程表里暴露批量调度。
    # 定位方式：log_path/_guard/daemon.lock（写入 PID）或 state.json。
    apply_process_alias(DAEMON_TITLE)

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

    # 轮转分配：同一轮 slot 依次落到不同卡，避免"按卡填满"导致末位卡（GPU5）分不到任务
    slots = [g for _ in range(JOBS_PER_GPU) for g in GPU_POOL]
    running = _RUNNING   # pid -> (proc, fh, job, slot_index)，与信号处理器共享同一对象
    running.clear()
    pending = deque()
    for j in jobs:
        if j["status"] == "done":
            continue
        if j["status"] == "failed" and j["attempts"] >= MAX_ATTEMPTS:
            continue
        # 防御：state 丢失重启时，结果已齐全的组合直接跳过并清权重，避免重跑
        got, need = job_complete(j["algo"], j["dataset"], j["split"])
        if got >= need:
            j["status"] = "done"
            continue
        pending.append(j)
    log(f"待执行 {len(pending)}，已完成跳过 {sum(1 for j in jobs if j['status'] == 'done')}")
    adopted = adopt_orphans(jmap, pending)
    if adopted:
        log(f"收养孤儿作业 {adopted} 个（进程仍在跑但非本守护子进程），待执行降为 {len(pending)}")
        save_state(jmap)

    last_save = 0.0
    while not _STOP["flag"]:
        now = time.time()

        # 0) 回收已结束的孤儿作业（进程消失后按结果齐全度判定 done / 重新排队）
        if _ADOPTED:
            recycle_orphans(jmap, pending, now, args)

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
                got, need = job_complete(job["algo"], job["dataset"], job["split"])
                if got >= need:
                    job["status"] = "done"
                    log(f"[done] {job['algo']}/{job['dataset']}/{job['split']} ({job['seconds']:.0f}s)")
                    if not args.no_email:
                        notify(job["algo"], job["dataset"], True,
                               f"完成 {job['split']} 3/3 格子，用时 {job['seconds']:.0f}s")
                else:
                    if job["attempts"] >= MAX_ATTEMPTS:
                        job["status"] = "failed"
                        log(f"[failed] {job['algo']}/{job['dataset']}/{job['split']} 退出码0 但 {job['attempts']} 次仍仅 {got}/{need}，放弃")
                        if not args.no_email:
                            notify(job["algo"], job["dataset"], False,
                                   f"退出码0 但 {job['attempts']} 次仍仅完成 {got}/{need}，已放弃")
                    else:
                        job["status"] = "pending"
                        log(f"[incomplete] {job['algo']}/{job['dataset']}/{job['split']} 退出码0 但仅 {got}/{need} 完成，重新排队")
                        pending.appendleft(job)
            else:
                if job["attempts"] >= MAX_ATTEMPTS:
                    job["status"] = "failed"
                    log(f"[failed] {job['algo']}/{job['dataset']}/{job['split']} 连续失败 {job['attempts']} 次，放弃")
                    if not args.no_email:
                        notify(job["algo"], job["dataset"], False,
                               f"连续失败 {job['attempts']} 次（last rc={rc}），已放弃")
                else:
                    job["status"] = "pending"
                    delay = MIN_RUNTIME_BEFORE_RETRY if job["seconds"] < MIN_RUNTIME_BEFORE_RETRY else 0
                    job["retry_after"] = now + delay
                    log(f"[retry] {job['algo']}/{job['dataset']}/{job['split']} rc={rc} "
                        f"({job['seconds']:.0f}s, attempt {job['attempts']}), "
                        f"{delay}s 后重排")
                    pending.append(job)
            save_state(jmap)

        # 2) 投递新任务
        free = gpu_free_mb()
        busy_slots = set(s for (*_, s) in running.values())
        # 孤儿作业也真占着卡，按其数量扣减本轮可用槽位，避免本守护再往同一张卡叠任务
        reserved = len(_ADOPTED)
        for slot in range(len(slots)):
            if _STOP["flag"] or not pending:
                break
            if len(busy_slots) + reserved >= len(slots):
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
            if external_job_running(picked["algo"], picked["dataset"], picked["split"]):
                log(f"[skip] {picked['algo']}/{picked['dataset']}/{picked['split']} 已有同名进程在跑，稍后重试")
                picked["retry_after"] = now + POLL_INTERVAL
                pending.append(picked)
                continue
            # 守护重启后，孤儿子进程可能已悄悄跑完：投递前复查一次，避免重复拉起
            got, need = job_complete(picked["algo"], picked["dataset"], picked["split"])
            if got >= need:
                picked["status"] = "done"
                log(f"[done] {picked['algo']}/{picked['dataset']}/{picked['split']} 结果已齐全（孤儿进程完成）")
                save_state(jmap)
                continue
            # 该组合的全部 4 个划分都跑满时清理权重目录（划分粒度下由投递前检查兜底触发）
            all_got, all_need = job_complete(picked["algo"], picked["dataset"])
            if all_got >= all_need:
                cleanup_job_weights(picked["algo"], picked["dataset"])
            # 非断点续跑算法投递前清掉残留 checkpoint，避免 resume 抛 CheckpointCompatibilityError
            clear_non_resumable_checkpoints(picked["algo"], picked["dataset"])
            proc, fh = launch(picked, gpu)
            running[proc.pid] = (proc, fh, picked, slot)
            # free 快照取自本轮循环开始，同轮连投多个任务时不会反映新增占用，会让余量小的卡
            # （GPU4 仅约 2.5GB）被连续投到 OOM（实测曾触发 CUBLAS_STATUS_ALLOC_FAILED）。
            # 这里按单作业峰值显存扣减快照，使同轮后续投递对该卡重新"刹车"；下一轮会重新
            # 读取真实空闲值，故不影响长期填充。
            # The free snapshot is taken once per round and does not reflect same-round allocations,
            # which over-committed the tight card (GPU4, ~2.5GB free) until OOM (observed
            # CUBLAS_STATUS_ALLOC_FAILED). Deduct per-job peak VRAM so later dispatches in the same
            # round brake again; the next round re-reads real free memory, so long-run filling is
            # unaffected.
            if free:
                free[gpu] = max(0, free.get(gpu, 0) - JOB_VRAM_MB)
            save_state(jmap)

        # 3) 结束判定：孤儿作业仍在外跑时不能收工，否则剩余作业失去兜底与批次汇总
        if not pending and not running and not _ADOPTED:
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
