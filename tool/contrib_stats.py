"""
贡献统计工具

查看项目贡献情况：谁跑了多少实验、用了什么硬件、贡献了多少算力。

使用方式：
  python tool/contrib_stats.py              # 查看完整统计
  python tool/contrib_stats.py --leaderboard  # 只看排行榜
  python tool/contrib_stats.py --hardware    # 只看硬件信息
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tool.cloud_storage import get_storage
from tool.task_queue import get_queue_status, print_queue_status, CONTRIBUTORS_PATH, DONE_PATH


def load_contributors():
    """加载贡献者列表"""
    storage = get_storage()
    if storage.exists(CONTRIBUTORS_PATH):
        return storage.read_json(CONTRIBUTORS_PATH)
    return []


def load_done_tasks():
    """加载已完成的任务列表"""
    storage = get_storage()
    if storage.exists(DONE_PATH):
        return storage.read_json(DONE_PATH)
    return []


def print_leaderboard():
    """打印贡献排行榜"""
    contributors = load_contributors()
    done_tasks = load_done_tasks()

    if not contributors and not done_tasks:
        print("[Stats] No data yet. Start running workers first!")
        return

    # 按用户聚合统计
    user_stats = {}
    for c in contributors:
        uid = c.get('user_id', 'unknown')
        if uid not in user_stats:
            user_stats[uid] = {
                'completed': c.get('total_completed', 0),
                'time_hours': round(c.get('total_time_seconds', 0) / 3600, 1),
                'machines': [],
                'gpus': [],
                'email': c.get('email', ''),
                'affiliation': c.get('affiliation', ''),
                'openreview': c.get('openreview', ''),
                'google_scholar': c.get('google_scholar', ''),
                'orcid': c.get('orcid', ''),
                'github': c.get('github', ''),
            }
        machine_label = c.get('machine_label', '') or c.get('hostname', '')
        if machine_label and machine_label not in user_stats[uid]['machines']:
            user_stats[uid]['machines'].append(machine_label)
        for gpu in c.get('gpu_info', []):
            gpu_name = f"{gpu['name']} ({gpu['vram_gb']}GB)"
            if gpu_name not in user_stats[uid]['gpus']:
                user_stats[uid]['gpus'].append(gpu_name)

    # 从done任务补充统计
    for t in done_tasks:
        uid = t.get('user_id', 'unknown')
        if uid not in user_stats:
            user_stats[uid] = {'completed': 0, 'time_hours': 0, 'machines': [], 'gpus': [],
                               'email': '', 'affiliation': '', 'openreview': '', 'google_scholar': '', 'orcid': '', 'github': ''}
        user_stats[uid]['completed'] += 1
        user_stats[uid]['time_hours'] += t.get('elapsed_seconds', 0) / 3600

    sorted_users = sorted(user_stats.items(), key=lambda x: x[1]['completed'], reverse=True)

    print(f"\n{'='*70}")
    print(f"[Contribution Leaderboard]")
    print(f"{'='*70}")

    for uid, stats in sorted_users:
        print(f"\n  {uid}")
        if stats['affiliation']:
            print(f"    Affiliation:  {stats['affiliation']}")
        if stats['email']:
            print(f"    Email:        {stats['email']}")
        if stats['orcid']:
            print(f"    ORCID:        {stats['orcid']}")
        if stats['openreview']:
            print(f"    OpenReview:   {stats['openreview']}")
        if stats['google_scholar']:
            print(f"    Scholar:      {stats['google_scholar']}")
        if stats['github']:
            print(f"    GitHub:       {stats['github']}")
        print(f"    Tasks:        {stats['completed']}")
        print(f"    Compute time: {stats['time_hours']:.1f} hours")
        machines = ', '.join(stats['machines'][:3]) or '-'
        gpus = ', '.join(stats['gpus'][:2]) or 'CPU only'
        print(f"    Machines:     {machines}")
        print(f"    GPUs:         {gpus}")

    total_tasks = sum(s['completed'] for s in user_stats.values())
    total_hours = sum(s['time_hours'] for s in user_stats.values())
    print(f"\n  {'─'*40}")
    print(f"  TOTAL: {total_tasks} tasks, {total_hours:.1f} hours, {len(sorted_users)} contributors")
    print(f"{'='*70}")


def print_hardware_info():
    """打印所有贡献者的硬件信息"""
    contributors = load_contributors()

    if not contributors:
        print("[Stats] No contributors registered yet.")
        return

    print(f"\n{'='*60}")
    print(f"[Hardware Inventory]")
    print(f"{'='*60}")

    for c in contributors:
        uid = c.get('user_id', 'unknown')
        label = c.get('machine_label', '') or c.get('hostname', '')
        print(f"\n  {uid} @ {label}")
        print(f"    OS:       {c.get('os', '?')} ({c.get('cpu_count', '?')} cores)")
        print(f"    CPU:      {c.get('cpu_model', '?')}")
        print(f"    Memory:   {c.get('total_memory_gb', '?')} GB")
        print(f"    Disk:     {c.get('disk_free_gb', '?')} GB free")
        gpus = c.get('gpu_info', [])
        if gpus:
            for gpu in gpus:
                print(f"    GPU:      {gpu['name']} ({gpu['vram_gb']} GB)")
        else:
            print(f"    GPU:      None (CPU only)")
        print(f"    First seen: {c.get('first_seen', '?')}")
        print(f"    Last seen:  {c.get('last_seen', '?')}")

    print(f"\n{'='*60}")


def print_full_stats():
    """打印完整统计"""
    print_queue_status()
    print()
    print_leaderboard()
    print()
    print_hardware_info()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Contribution Statistics")
    parser.add_argument("--leaderboard", action="store_true", help="Show leaderboard only")
    parser.add_argument("--hardware", action="store_true", help="Show hardware info only")
    args = parser.parse_args()

    if args.leaderboard:
        print_leaderboard()
    elif args.hardware:
        print_hardware_info()
    else:
        print_full_stats()
