#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
磁盘清理工具 - 定期清理运行时产生的模型权重和日志文件

用法:
    python -m tool.cleanup --scan              # 扫描磁盘占用，不删除
    python -m tool.cleanup --clean-save        # 清理 save_path 中的中间模型
    python -m tool.cleanup --clean-save --all  # 清理 save_path 中的所有模型（包括最终模型）
    python -m tool.cleanup --clean-log         # 清理 log_path 中的旧日志
    python -m tool.cleanup --clean-log --days 7  # 清理7天前的日志
    python -m tool.cleanup --clean-all         # 清理所有运行时产物
    python -m tool.cleanup --dry-run           # 预览模式，只显示将要删除的文件
"""

import os
import sys
import argparse
import shutil
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAVE_PATH = os.path.join(PROJECT_ROOT, "save_path")
LOG_PATH = os.path.join(PROJECT_ROOT, "log_path")
RESULT_PATH = os.path.join(PROJECT_ROOT, "result_path")


def get_dir_size(path):
    total = 0
    count = 0
    if not os.path.exists(path):
        return total, count
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
                count += 1
            except OSError:
                pass
    return total, count


def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.1f} MB"
    else:
        return f"{size_bytes / 1024 / 1024 / 1024:.2f} GB"


def scan():
    print("=" * 60)
    print("  磁盘占用扫描")
    print("=" * 60)

    for name, path in [("save_path (模型权重)", SAVE_PATH),
                       ("log_path (日志)", LOG_PATH),
                       ("result_path (结果)", RESULT_PATH)]:
        if os.path.exists(path):
            size, count = get_dir_size(path)
            print(f"\n  {name}: {format_size(size)} ({count} 文件)")
            if count > 0 and size > 0:
                subdirs = []
                for d in os.listdir(path):
                    dp = os.path.join(path, d)
                    if os.path.isdir(dp):
                        s, c = get_dir_size(dp)
                        subdirs.append((d, s, c))
                subdirs.sort(key=lambda x: -x[1])
                for d, s, c in subdirs[:10]:
                    print(f"    {d:40s} {format_size(s):>10s} ({c} 文件)")
                if len(subdirs) > 10:
                    print(f"    ... 还有 {len(subdirs) - 10} 个子目录")
        else:
            print(f"\n  {name}: 不存在")

    print("\n" + "=" * 60)


def clean_save_path(remove_all=False, dry_run=False):
    if not os.path.exists(SAVE_PATH):
        print("[跳过] save_path 不存在")
        return

    print(f"\n{'[预览]' if dry_run else '[清理]'} save_path 中的模型文件")
    freed = 0
    removed_count = 0

    for root, dirs, files in os.walk(SAVE_PATH, topdown=False):
        for f in files:
            fp = os.path.join(root, f)
            should_remove = False

            if remove_all:
                should_remove = True
            else:
                if f.startswith("client_") or f.endswith(".pkl"):
                    should_remove = True
                elif f.startswith("step_") and f.endswith(".pt"):
                    parent_files = os.listdir(root)
                    step_files = [x for x in parent_files if x.startswith("step_") and x.endswith(".pt")]
                    if len(step_files) > 1:
                        step_files.sort(key=lambda x: int(x.split("_")[1]))
                        if f != step_files[-1]:
                            should_remove = True

            if should_remove:
                try:
                    size = os.path.getsize(fp)
                    if dry_run:
                        print(f"  [将删除] {os.path.relpath(fp, PROJECT_ROOT)} ({format_size(size)})")
                    else:
                        os.remove(fp)
                        print(f"  [已删除] {os.path.relpath(fp, PROJECT_ROOT)} ({format_size(size)})")
                    freed += size
                    removed_count += 1
                except OSError as e:
                    print(f"  [错误] 无法删除 {fp}: {e}")

        for d in dirs:
            if d.startswith("client_"):
                dp = os.path.join(root, d)
                if remove_all or True:
                    size, count = get_dir_size(dp)
                    if dry_run:
                        print(f"  [将删除] {os.path.relpath(dp, PROJECT_ROOT)}/ ({format_size(size)}, {count} 文件)")
                    else:
                        try:
                            shutil.rmtree(dp)
                            print(f"  [已删除] {os.path.relpath(dp, PROJECT_ROOT)}/ ({format_size(size)}, {count} 文件)")
                            freed += size
                            removed_count += count
                        except OSError as e:
                            print(f"  [错误] 无法删除 {dp}: {e}")

    action = "将释放" if dry_run else "已释放"
    print(f"\n  {action}: {format_size(freed)} ({removed_count} 文件)")


def clean_log_path(days=0, dry_run=False):
    if not os.path.exists(LOG_PATH):
        print("[跳过] log_path 不存在")
        return

    print(f"\n{'[预览]' if dry_run else '[清理]'} log_path 中的日志文件")
    if days > 0:
        print(f"  (保留最近 {days} 天的日志)")
    else:
        print(f"  (清理所有日志)")

    freed = 0
    removed_count = 0
    cutoff = time.time() - days * 86400

    for root, dirs, files in os.walk(LOG_PATH):
        for f in files:
            fp = os.path.join(root, f)
            try:
                mtime = os.path.getmtime(fp)
                if days == 0 or mtime < cutoff:
                    size = os.path.getsize(fp)
                    if dry_run:
                        print(f"  [将删除] {os.path.relpath(fp, PROJECT_ROOT)} ({format_size(size)})")
                    else:
                        os.remove(fp)
                        print(f"  [已删除] {os.path.relpath(fp, PROJECT_ROOT)} ({format_size(size)})")
                    freed += size
                    removed_count += 1
            except OSError as e:
                print(f"  [错误] 无法删除 {fp}: {e}")

    action = "将释放" if dry_run else "已释放"
    print(f"\n  {action}: {format_size(freed)} ({removed_count} 文件)")


def clean_empty_dirs(path):
    if not os.path.exists(path):
        return
    for root, dirs, files in os.walk(path, topdown=False):
        for d in dirs:
            dp = os.path.join(root, d)
            try:
                if not os.listdir(dp):
                    os.rmdir(dp)
                    print(f"  [清理空目录] {os.path.relpath(dp, PROJECT_ROOT)}")
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description="磁盘清理工具")
    parser.add_argument("--scan", action="store_true", help="扫描磁盘占用")
    parser.add_argument("--clean-save", action="store_true", help="清理 save_path 中的中间模型")
    parser.add_argument("--clean-log", action="store_true", help="清理 log_path 中的日志")
    parser.add_argument("--clean-all", action="store_true", help="清理所有运行时产物")
    parser.add_argument("--all", action="store_true", help="（配合 --clean-save）删除所有模型，包括最终模型")
    parser.add_argument("--days", type=int, default=0, help="（配合 --clean-log）保留最近N天的日志")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不实际删除")
    args = parser.parse_args()

    if not any([args.scan, args.clean_save, args.clean_log, args.clean_all]):
        args.scan = True

    if args.scan:
        scan()

    if args.clean_save or args.clean_all:
        clean_save_path(remove_all=args.all, dry_run=args.dry_run)
        if not args.dry_run:
            clean_empty_dirs(SAVE_PATH)

    if args.clean_log or args.clean_all:
        clean_log_path(days=args.days, dry_run=args.dry_run)
        if not args.dry_run:
            clean_empty_dirs(LOG_PATH)


if __name__ == "__main__":
    main()
