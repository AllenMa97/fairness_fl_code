"""
配置检查模块

Worker启动前检查配置完整性，缺失则报错并提示。
独立模块，不依赖torch等重型库。
"""

import os
import sys


def check_config():
    """
    启动前检查配置完整性，缺失则报错并提示。

    Returns:
        bool: 配置是否完整
    """
    errors = []
    warnings = []

    # 检查用户身份
    user_name = os.environ.get('FL_USER_NAME', '')
    try:
        from tool.user_config import USER_NAME
        if not user_name:
            user_name = USER_NAME
    except ImportError:
        pass
    if not user_name:
        errors.append(
            "USER_NAME 未设置！请在 tool/user_config.py 中填写你的姓名，"
            "或设置环境变量 FL_USER_NAME。\n"
            "  例如: export FL_USER_NAME=张三"
        )

    # 检查存储后端配置
    backend = os.environ.get('STORAGE_BACKEND', '')
    try:
        from tool.storage_config import STORAGE_BACKEND
        if not backend:
            backend = STORAGE_BACKEND
    except ImportError:
        pass
    backend = backend or 'local'

    if backend == 'github':
        # 检查GitHub Token
        token = os.environ.get('GITHUB_TOKEN', '')
        try:
            from tool.storage_config import GITHUB_TOKEN
            if not token:
                token = GITHUB_TOKEN
        except ImportError:
            pass
        if not token:
            errors.append(
                "GITHUB_TOKEN 未设置！GitHub后端需要Token才能写入队列和结果。\n"
                "  创建方式: GitHub -> Settings -> Developer settings -> Personal access tokens\n"
                "           -> Generate new token (classic) -> 勾选 'repo' 权限\n"
                "  设置方式: export GITHUB_TOKEN=ghp_xxxxxxxxxxxx"
            )

    elif backend == 'cos':
        token = os.environ.get('COS_SECRET_ID', '')
        try:
            from tool.storage_config import COS_SECRET_ID
            if not token:
                token = COS_SECRET_ID
        except ImportError:
            pass
        if not token:
            errors.append("COS_SECRET_ID 未设置！请设置环境变量 COS_SECRET_ID。")

    elif backend == 'oss':
        token = os.environ.get('OSS_ACCESS_KEY_ID', '')
        try:
            from tool.storage_config import OSS_ACCESS_KEY_ID
            if not token:
                token = OSS_ACCESS_KEY_ID
        except ImportError:
            pass
        if not token:
            errors.append("OSS_ACCESS_KEY_ID 未设置！请设置环境变量 OSS_ACCESS_KEY_ID。")

    elif backend == 's3':
        token = os.environ.get('S3_ACCESS_KEY', '')
        try:
            from tool.storage_config import S3_ACCESS_KEY
            if not token:
                token = S3_ACCESS_KEY
        except ImportError:
            pass
        if not token:
            errors.append("S3_ACCESS_KEY 未设置！请设置环境变量 S3_ACCESS_KEY。")

    # 打印结果
    if errors:
        print(f"\n{'='*60}")
        print("[CONFIG ERROR] Worker 启动失败！以下配置必须修复：")
        print(f"{'='*60}")
        for i, err in enumerate(errors, 1):
            print(f"\n  {i}. {err}")
        print(f"\n{'='*60}")
        return False

    if warnings:
        print(f"\n[CONFIG WARNING]")
        for w in warnings:
            print(f"  - {w}")

    return True
