"""Phase A 队列执行器。

用法：
    python patch_experiment/run_queue.py <queue.jsonl> <log_path>

职责：
- 逐格串行执行（并发由"启动多少个执行器"控制，便于把 RAM 压在阈值内）
- 每格注入 PDFFED_PROBE=1，采集 Δ_rep 与逐评测轨迹
- 日志里打印 start/done + 返回码，便于远程巡检

注意：本脚本只做调度，不改变任何训练超参；超参由队列文件里的 argv 决定。
"""

import json
import os
import re
import subprocess
import sys
import time

PYTHON = sys.executable
ENV = dict(os.environ)
ENV["PDFFED_PROBE"] = "1"
ENV.setdefault("OMP_NUM_THREADS", "8")
ENV.setdefault("MKL_NUM_THREADS", "8")


def log(path, msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def done_cells(log_path):
    """从已有日志里读出已成功跑完的格编号（done #N rc=0），用于重启后续跑。

    重启（换卡/换机器）时不必重跑已完成的格：日志是按顺序追加的，
    编号与队列文件的行号一一对应，只要队列文件不被重排就能直接对上。
    """
    done = set()
    if not log_path or not os.path.exists(log_path):
        return done
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = re.search(r"done\s+#(\d+) rc=0", line)
                if m:
                    done.add(int(m.group(1)))
    except OSError:
        pass
    return done


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    queue, log_path = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
    if not os.path.isabs(queue):
        queue = os.path.abspath(queue)

    cells = []
    with open(queue, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cells.append(json.loads(line))

    log(log_path, "queue=%s cells=%d" % (os.path.basename(queue), len(cells)))
    skip = done_cells(log_path)
    if skip:
        log(log_path, "resume: skip done cells %s" % sorted(skip))
    for i, cell in enumerate(cells, 1):
        if i in skip:
            continue
        if cell.get("skip"):
            # 该格已由另一条队列（或另一台机器）认领/跑完：保留行号占位，
            # 避免删除行导致日志里 done #N 与行号错位，只跳过执行。
            log(log_path, "skip  #%d %s/%s/%s" % (
                i, cell["dataset"], cell["split"], cell["algo"]))
            continue
        argv = [PYTHON, cell.get("train", "main_Tabular_CLF.py")] + cell["argv"]
        log(log_path, "start #%d %s/%s/%s%s" % (
            i, cell["dataset"], cell["split"], cell["algo"],
            ("_" + cell["tag"]) if cell.get("tag") else ""))
        t0 = time.time()
        # 每格的训练输出单独落盘，失败时能直接看到 traceback（否则只有返回码）
        out_path = ("%s_cell%02d.out" % (os.path.splitext(log_path)[0], i)
                    if log_path else None)
        if out_path:
            with open(out_path, "w", encoding="utf-8", errors="replace") as out:
                rc = subprocess.call(argv, env=ENV, cwd=os.getcwd(),
                                     stdout=out, stderr=subprocess.STDOUT)
        else:
            rc = subprocess.call(argv, env=ENV, cwd=os.getcwd())
        log(log_path, "done  #%d rc=%d %.1fmin" % (i, rc, (time.time() - t0) / 60.0))
    log(log_path, "queue finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
