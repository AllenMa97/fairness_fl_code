"""
统一云存储抽象层

支持多种后端，通过配置切换：
  - local: 本地文件系统
  - cos:   腾讯云 COS
  - oss:   阿里云 OSS
  - s3:    AWS S3 / 兼容S3的服务（如MinIO）

配置方式：
  1. 在 tool/storage_config.py 中设置 STORAGE_BACKEND 和对应凭证
  2. 或通过环境变量设置：
     export STORAGE_BACKEND=cos
     export COS_SECRET_ID=xxx
     export COS_SECRET_KEY=xxx
     export COS_BUCKET=xxx
     export COS_REGION=xxx
"""

import os
import json
import shutil
import time
from abc import ABC, abstractmethod

# ============================================================
# 配置（优先从 storage_config.py 读取，环境变量作为回退）
# ============================================================

def _load_storage_config():
    """加载存储配置"""
    # 先尝试从配置文件读取
    file_config = {}
    try:
        from tool.storage_config import (
            STORAGE_BACKEND, STORAGE_LOCAL_ROOT,
            COS_SECRET_ID, COS_SECRET_KEY, COS_BUCKET, COS_REGION,
            OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET, OSS_BUCKET, OSS_ENDPOINT,
            S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET, S3_ENDPOINT, S3_REGION,
            GITHUB_TOKEN, GITHUB_REPO, GITHUB_BRANCH,
        )
        file_config = {
            'backend': STORAGE_BACKEND,
            'local_root': STORAGE_LOCAL_ROOT,
            'cos_secret_id': COS_SECRET_ID,
            'cos_secret_key': COS_SECRET_KEY,
            'cos_bucket': COS_BUCKET,
            'cos_region': COS_REGION,
            'oss_access_key_id': OSS_ACCESS_KEY_ID,
            'oss_access_key_secret': OSS_ACCESS_KEY_SECRET,
            'oss_bucket': OSS_BUCKET,
            'oss_endpoint': OSS_ENDPOINT,
            's3_access_key': S3_ACCESS_KEY,
            's3_secret_key': S3_SECRET_KEY,
            's3_bucket': S3_BUCKET,
            's3_endpoint': S3_ENDPOINT,
            's3_region': S3_REGION,
            'github_token': GITHUB_TOKEN,
            'github_repo': GITHUB_REPO,
            'github_branch': GITHUB_BRANCH,
        }
    except ImportError:
        pass

    # 环境变量回退（仅当配置文件中值为空时才使用环境变量）
    def _get(key, env_key, default=''):
        val = file_config.get(key, '')
        if val:
            return val
        return os.environ.get(env_key, default)

    return {
        'backend': _get('backend', 'STORAGE_BACKEND', 'local'),
        'local_root': _get('local_root', 'STORAGE_LOCAL_ROOT', ''),
        'cos_secret_id': _get('cos_secret_id', 'COS_SECRET_ID'),
        'cos_secret_key': _get('cos_secret_key', 'COS_SECRET_KEY'),
        'cos_bucket': _get('cos_bucket', 'COS_BUCKET'),
        'cos_region': _get('cos_region', 'COS_REGION'),
        'oss_access_key_id': _get('oss_access_key_id', 'OSS_ACCESS_KEY_ID'),
        'oss_access_key_secret': _get('oss_access_key_secret', 'OSS_ACCESS_KEY_SECRET'),
        'oss_bucket': _get('oss_bucket', 'OSS_BUCKET'),
        'oss_endpoint': _get('oss_endpoint', 'OSS_ENDPOINT'),
        's3_access_key': _get('s3_access_key', 'S3_ACCESS_KEY'),
        's3_secret_key': _get('s3_secret_key', 'S3_SECRET_KEY'),
        's3_bucket': _get('s3_bucket', 'S3_BUCKET'),
        's3_endpoint': _get('s3_endpoint', 'S3_ENDPOINT'),
        's3_region': _get('s3_region', 'S3_REGION'),
        'github_token': _get('github_token', 'GITHUB_TOKEN'),
        'github_repo': _get('github_repo', 'GITHUB_REPO'),
        'github_branch': _get('github_branch', 'GITHUB_BRANCH'),
    }

    return config


# ============================================================
# 抽象基类
# ============================================================

class CloudStorageBase(ABC):
    """云存储抽象基类"""

    @abstractmethod
    def upload(self, local_path, remote_path):
        """上传文件"""
        ...

    @abstractmethod
    def download(self, remote_path, local_path):
        """下载文件"""
        ...

    @abstractmethod
    def exists(self, remote_path) -> bool:
        """检查文件是否存在"""
        ...

    @abstractmethod
    def list_files(self, prefix, recursive=False):
        """列出文件"""
        ...

    @abstractmethod
    def delete(self, remote_path):
        """删除文件"""
        ...

    @abstractmethod
    def read_json(self, remote_path) -> dict:
        """读取JSON文件"""
        ...

    @abstractmethod
    def write_json(self, remote_path, data):
        """写入JSON文件"""
        ...

    def get_file_size(self, remote_path):
        """获取文件大小（字节），不支持的后端返回-1"""
        return -1

    def download_range(self, remote_path, start, end):
        """下载文件的指定字节范围，不支持的后端抛出NotImplementedError"""
        raise NotImplementedError(f"download_range not supported by {self.__class__.__name__}")

    def upload_dir(self, local_dir, remote_prefix):
        """上传整个目录"""
        for root, dirs, files in os.walk(local_dir):
            for f in files:
                local_path = os.path.join(root, f)
                rel_path = os.path.relpath(local_path, local_dir).replace('\\', '/')
                remote_path = f"{remote_prefix}/{rel_path}"
                self.upload(local_path, remote_path)

    def download_dir(self, remote_prefix, local_dir):
        """下载整个目录（仅下载文件，不创建子目录结构）"""
        os.makedirs(local_dir, exist_ok=True)
        files = self.list_files(remote_prefix, recursive=True)
        for f in files:
            local_path = os.path.join(local_dir, os.path.basename(f))
            self.download(f, local_path)


# ============================================================
# 本地文件系统后端
# ============================================================

class LocalStorage(CloudStorageBase):
    """本地文件系统存储"""

    def __init__(self, root_dir=''):
        self.root = root_dir

    def _full_path(self, path):
        return os.path.join(self.root, path.lstrip('/'))

    def upload(self, local_path, remote_path):
        full = self._full_path(remote_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        shutil.copy2(local_path, full)

    def download(self, remote_path, local_path):
        full = self._full_path(remote_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copy2(full, local_path)

    def exists(self, remote_path) -> bool:
        return os.path.exists(self._full_path(remote_path))

    def list_files(self, prefix, recursive=False):
        full = self._full_path(prefix)
        if not os.path.exists(full):
            return []
        if os.path.isfile(full):
            return [prefix]
        result = []
        if recursive:
            for root, dirs, files in os.walk(full):
                for f in files:
                    rel = os.path.relpath(os.path.join(root, f), self.root).replace('\\', '/')
                    result.append(rel)
        else:
            for f in os.listdir(full):
                result.append(f"{prefix}/{f}")
        return result

    def delete(self, remote_path):
        full = self._full_path(remote_path)
        if os.path.exists(full):
            os.remove(full)

    def read_json(self, remote_path) -> dict:
        full = self._full_path(remote_path)
        with open(full, 'r', encoding='utf-8') as f:
            return json.load(f)

    def write_json(self, remote_path, data):
        full = self._full_path(remote_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_file_size(self, remote_path):
        full = self._full_path(remote_path)
        if os.path.exists(full):
            return os.path.getsize(full)
        return -1

    def download_range(self, remote_path, start, end):
        full = self._full_path(remote_path)
        with open(full, 'rb') as f:
            f.seek(start)
            return f.read(end - start)


# ============================================================
# 腾讯云 COS 后端
# ============================================================

class CosStorage(CloudStorageBase):
    """腾讯云 COS 存储"""

    def __init__(self, secret_id, secret_key, bucket, region):
        from qcloud_cos import CosConfig, CosS3Client
        config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key)
        self.client = CosS3Client(config)
        self.bucket = bucket
        self.region = region

    def upload(self, local_path, remote_path):
        self.client.upload_file(
            Bucket=self.bucket,
            Key=remote_path,
            LocalFilePath=local_path,
        )

    def download(self, remote_path, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.client.download_file(
            Bucket=self.bucket,
            Key=remote_path,
            DestFilePath=local_path,
        )

    def exists(self, remote_path) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=remote_path)
            return True
        except Exception:
            return False

    def list_files(self, prefix, recursive=False):
        result = []
        marker = ''
        while True:
            resp = self.client.list_objects(
                Bucket=self.bucket,
                Prefix=prefix,
                Marker=marker,
                MaxKeys=1000,
            )
            if 'Contents' in resp:
                for obj in resp['Contents']:
                    key = obj['Key']
                    if not recursive and '/' in key[len(prefix):]:
                        continue
                    result.append(key)
            if resp.get('IsTruncated') == 'false':
                break
            marker = resp.get('NextMarker', '')
            if not marker:
                break
        return result

    def delete(self, remote_path):
        self.client.delete_object(Bucket=self.bucket, Key=remote_path)

    def read_json(self, remote_path) -> dict:
        resp = self.client.get_object(Bucket=self.bucket, Key=remote_path)
        return json.loads(resp['Body'].get_raw_stream().read().decode('utf-8'))

    def write_json(self, remote_path, data):
        import io
        content = json.dumps(data, indent=2, ensure_ascii=False).encode('utf-8')
        self.client.put_object(
            Bucket=self.bucket,
            Key=remote_path,
            Body=io.BytesIO(content),
        )

    def get_file_size(self, remote_path):
        try:
            resp = self.client.head_object(Bucket=self.bucket, Key=remote_path)
            return int(resp.get('Content-Length', -1))
        except Exception:
            return -1

    def download_range(self, remote_path, start, end):
        resp = self.client.get_object(
            Bucket=self.bucket,
            Key=remote_path,
            Range=f"bytes={start}-{end-1}",
        )
        return resp['Body'].get_raw_stream().read()


# ============================================================
# 阿里云 OSS 后端
# ============================================================

class OssStorage(CloudStorageBase):
    """阿里云 OSS 存储"""

    def __init__(self, access_key_id, access_key_secret, bucket, endpoint):
        import oss2
        auth = oss2.Auth(access_key_id, access_key_secret)
        self.bucket = oss2.Bucket(auth, endpoint, bucket)

    def upload(self, local_path, remote_path):
        self.bucket.put_object_from_file(remote_path, local_path)

    def download(self, remote_path, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.bucket.get_object_to_file(remote_path, local_path)

    def exists(self, remote_path) -> bool:
        return self.bucket.object_exists(remote_path)

    def list_files(self, prefix, recursive=False):
        result = []
        for obj in oss2.ObjectIterator(self.bucket, prefix=prefix):
            key = obj.key
            if not recursive and '/' in key[len(prefix):]:
                continue
            result.append(key)
        return result

    def delete(self, remote_path):
        self.bucket.delete_object(remote_path)

    def read_json(self, remote_path) -> dict:
        result = self.bucket.get_object(remote_path)
        return json.loads(result.read().decode('utf-8'))

    def write_json(self, remote_path, data):
        content = json.dumps(data, indent=2, ensure_ascii=False).encode('utf-8')
        self.bucket.put_object(remote_path, content)

    def get_file_size(self, remote_path):
        try:
            return self.bucket.head_object(remote_path).content_length
        except Exception:
            return -1

    def download_range(self, remote_path, start, end):
        result = self.bucket.get_object(remote_path, headers={'Range': f'bytes={start}-{end-1}'})
        return result.read()


# ============================================================
# S3 兼容后端（AWS S3 / MinIO 等）
# ============================================================

class S3Storage(CloudStorageBase):
    """S3 兼容存储（AWS S3 / MinIO）"""

    def __init__(self, access_key, secret_key, bucket, endpoint=None, region='us-east-1'):
        import boto3
        from botocore.config import Config as BotoConfig
        kwargs = {
            'aws_access_key_id': access_key,
            'aws_secret_access_key': secret_key,
            'region_name': region,
        }
        if endpoint:
            kwargs['endpoint_url'] = endpoint
            kwargs['config'] = BotoConfig(s3={'addressing_style': 'path'})
        self.s3 = boto3.client('s3', **kwargs)
        self.bucket = bucket

    def upload(self, local_path, remote_path):
        self.s3.upload_file(local_path, self.bucket, remote_path)

    def download(self, remote_path, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.s3.download_file(self.bucket, remote_path, local_path)

    def exists(self, remote_path) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=remote_path)
            return True
        except Exception:
            return False

    def list_files(self, prefix, recursive=False):
        result = []
        paginator = self.s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if not recursive and '/' in key[len(prefix):]:
                    continue
                result.append(key)
        return result

    def delete(self, remote_path):
        self.s3.delete_object(Bucket=self.bucket, Key=remote_path)

    def read_json(self, remote_path) -> dict:
        resp = self.s3.get_object(Bucket=self.bucket, Key=remote_path)
        return json.loads(resp['Body'].read().decode('utf-8'))

    def write_json(self, remote_path, data):
        content = json.dumps(data, indent=2, ensure_ascii=False).encode('utf-8')
        self.s3.put_object(Bucket=self.bucket, Key=remote_path, Body=content)

    def get_file_size(self, remote_path):
        try:
            resp = self.s3.head_object(Bucket=self.bucket, Key=remote_path)
            return resp.get('ContentLength', -1)
        except Exception:
            return -1

    def download_range(self, remote_path, start, end):
        resp = self.s3.get_object(
            Bucket=self.bucket,
            Key=remote_path,
            Range=f"bytes={start}-{end-1}",
        )
        return resp['Body'].read()


# ============================================================
# GitHub 后端（使用 GitHub Contents API）
# ============================================================

class GitHubStorage(CloudStorageBase):
    """
    GitHub 仓库作为存储后端（通过 Contents API）

    适用场景：
      - 免费、无需额外云服务
      - 数据集放 Git Release，JSON队列文件放仓库
      - 多机器通过 GitHub 协调任务

    限制：
      - 单文件 < 100MB（Contents API 限制）
      - API rate limit: 5000次/小时（认证后）
      - 适合小文件（JSON、日志、小checkpoint）

    配置：
      GITHUB_TOKEN: GitHub Personal Access Token（需要 repo 权限）
      GITHUB_REPO: 仓库名（格式: "owner/repo"）
      GITHUB_BRANCH: 分支名（默认: "main"）
    """

    def __init__(self, token, repo, branch='main'):
        from github import Github
        self.gh = Github(token)
        self.repo = self.gh.get_repo(repo)
        self.branch = branch
        self._cache = {}  # 简单的内存缓存，减少API调用

    def upload(self, local_path, remote_path):
        """上传文件到 GitHub 仓库"""
        with open(local_path, 'rb') as f:
            content = f.read()

        # GitHub Contents API 需要 base64 编码
        import base64
        b64_content = base64.b64encode(content).decode('utf-8')

        try:
            # 尝试更新已有文件
            file = self.repo.get_contents(remote_path, ref=self.branch)
            self.repo.update_file(
                remote_path,
                f"update {remote_path}",
                b64_content,
                file.sha,
                branch=self.branch,
            )
        except Exception:
            # 文件不存在，创建新文件
            self.repo.create_file(
                remote_path,
                f"add {remote_path}",
                b64_content,
                branch=self.branch,
            )

        # 更新缓存
        self._cache[remote_path] = content
        print(f"[GitHub] Uploaded: {remote_path} ({len(content)} bytes)")

    def download(self, remote_path, local_path):
        """从 GitHub 仓库下载文件"""
        # 先查缓存
        if remote_path in self._cache:
            content = self._cache[remote_path]
        else:
            file = self.repo.get_contents(remote_path, ref=self.branch)
            content = file.decoded_content
            self._cache[remote_path] = content

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(content)

    def exists(self, remote_path) -> bool:
        """检查文件是否存在"""
        if remote_path in self._cache:
            return True
        try:
            self.repo.get_contents(remote_path, ref=self.branch)
            return True
        except Exception:
            return False

    def list_files(self, prefix, recursive=False):
        """列出文件"""
        try:
            contents = self.repo.get_contents(prefix, ref=self.branch)
        except Exception:
            return []

        result = []
        # get_contents 返回的是列表或单个Content对象
        if not isinstance(contents, list):
            contents = [contents]

        for item in contents:
            if item.type == 'file':
                result.append(item.path)
            elif item.type == 'dir' and recursive:
                # 递归获取子目录
                sub_files = self.list_files(item.path, recursive=True)
                result.extend(sub_files)

        return result

    def delete(self, remote_path):
        """删除文件"""
        try:
            file = self.repo.get_contents(remote_path, ref=self.branch)
            self.repo.delete_file(
                remote_path,
                f"delete {remote_path}",
                file.sha,
                branch=self.branch,
            )
            self._cache.pop(remote_path, None)
        except Exception:
            pass

    def read_json(self, remote_path) -> dict:
        """读取JSON文件"""
        if remote_path in self._cache:
            content = self._cache[remote_path]
        else:
            file = self.repo.get_contents(remote_path, ref=self.branch)
            content = file.decoded_content
            self._cache[remote_path] = content

        if isinstance(content, bytes):
            return json.loads(content.decode('utf-8'))
        return json.loads(content)

    def write_json(self, remote_path, data):
        """写入JSON文件"""
        content = json.dumps(data, indent=2, ensure_ascii=False).encode('utf-8')
        import base64
        b64_content = base64.b64encode(content).decode('utf-8')

        try:
            file = self.repo.get_contents(remote_path, ref=self.branch)
            self.repo.update_file(
                remote_path,
                f"update {remote_path}",
                b64_content,
                file.sha,
                branch=self.branch,
            )
        except Exception:
            self.repo.create_file(
                remote_path,
                f"add {remote_path}",
                b64_content,
                branch=self.branch,
            )

        self._cache[remote_path] = content

    def get_file_size(self, remote_path):
        """获取文件大小"""
        try:
            file = self.repo.get_contents(remote_path, ref=self.branch)
            return file.size
        except Exception:
            return -1

    def download_range(self, remote_path, start, end):
        """下载文件指定范围（GitHub API不原生支持，回退到全量下载）"""
        self.download(remote_path, f"__github_range_tmp_{os.path.basename(remote_path)}")
        tmp_path = f"__github_range_tmp_{os.path.basename(remote_path)}"
        with open(tmp_path, 'rb') as f:
            f.seek(start)
            chunk = f.read(end - start)
        os.remove(tmp_path)
        return chunk


# ============================================================
# 混合存储：本地优先 + 云端回退
# ============================================================

class HybridStorage(CloudStorageBase):
    """
    混合存储：先查本地，本地没有就从云端下载并缓存到本地。

    适用场景：
      - 数据集：本地有就直接用，没有就从云端下载
      - 任务队列：始终走云端（保证多机器一致性）
      - Checkpoint：上传到云端（防机器清空），本地也留一份

    Args:
        cloud: 云存储后端实例
        local_root: 本地缓存根目录
        always_cloud_prefixes: 始终走云端的路径前缀列表（如队列文件）
    """

    def __init__(self, cloud, local_root='./local_cache', always_cloud_prefixes=None):
        self.cloud = cloud
        self.local = LocalStorage(root_dir=local_root)
        self.always_cloud = set(always_cloud_prefixes or ['queue/'])

    def _should_use_cloud(self, path):
        """判断是否应该直接走云端（不走本地缓存）"""
        for prefix in self.always_cloud:
            if path.startswith(prefix):
                return True
        return False

    def upload(self, local_path, remote_path):
        """上传到云端，同时更新本地缓存"""
        self.cloud.upload(local_path, remote_path)
        if not self._should_use_cloud(remote_path):
            self.local.upload(local_path, remote_path)

    def download(self, remote_path, local_path):
        """下载：先查本地缓存，没有就从云端下载并缓存"""
        if not self._should_use_cloud(remote_path):
            local_cached = self.local._full_path(remote_path)
            if os.path.exists(local_cached):
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                shutil.copy2(local_cached, local_path)
                return
        # 从云端下载
        self.cloud.download(remote_path, local_path)
        # 缓存到本地
        if not self._should_use_cloud(remote_path):
            self.local.upload(local_path, remote_path)

    def download_resumable(self, remote_path, local_path, chunk_size=8*1024*1024):
        """
        断点续传下载（仅云端后端支持）

        如果本地已有部分文件，从断点处继续下载，避免重复传输。

        Args:
            remote_path: 云端路径
            local_path: 本地目标路径
            chunk_size: 分块大小（默认8MB）
        """
        # 检查本地已有大小
        existing_size = 0
        if os.path.exists(local_path):
            existing_size = os.path.getsize(local_path)

        # 获取云端文件大小
        try:
            remote_size = self.cloud.get_file_size(remote_path)
        except (AttributeError, Exception):
            # 后端不支持get_file_size，回退到普通下载
            self.download(remote_path, local_path)
            return

        if existing_size >= remote_size:
            # 文件已完整下载
            return

        if existing_size > 0:
            print(f"[CloudStorage] Resuming download: {remote_path} "
                  f"({existing_size}/{remote_size} bytes, {existing_size*100//remote_size}%)")

        # 分块下载
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        mode = 'ab' if existing_size > 0 else 'wb'
        offset = existing_size

        while offset < remote_size:
            end = min(offset + chunk_size, remote_size)
            chunk = self.cloud.download_range(remote_path, offset, end)
            with open(local_path, mode) as f:
                f.write(chunk)
            offset = end
            mode = 'ab'  # 后续都是追加

        # 缓存到本地
        if not self._should_use_cloud(remote_path):
            self.local.upload(local_path, remote_path)

    def exists(self, remote_path) -> bool:
        """检查存在：队列文件只查云端，其他先查本地"""
        if self._should_use_cloud(remote_path):
            return self.cloud.exists(remote_path)
        return self.local.exists(remote_path) or self.cloud.exists(remote_path)

    def list_files(self, prefix, recursive=False):
        """列出文件：队列文件只查云端，其他查本地"""
        if self._should_use_cloud(prefix):
            return self.cloud.list_files(prefix, recursive)
        local_files = self.local.list_files(prefix, recursive)
        if local_files:
            return local_files
        return self.cloud.list_files(prefix, recursive)

    def delete(self, remote_path):
        """删除：云端和本地都删"""
        self.cloud.delete(remote_path)
        if not self._should_use_cloud(remote_path):
            self.local.delete(remote_path)

    def read_json(self, remote_path) -> dict:
        """读取JSON：队列文件只从云端读，其他先查本地"""
        if self._should_use_cloud(remote_path):
            return self.cloud.read_json(remote_path)
        if self.local.exists(remote_path):
            return self.local.read_json(remote_path)
        return self.cloud.read_json(remote_path)

    def write_json(self, remote_path, data):
        """写入JSON：始终写云端，队列文件不同步本地"""
        self.cloud.write_json(remote_path, data)
        if not self._should_use_cloud(remote_path):
            self.local.write_json(remote_path, data)

    def download_to_local_cache(self, remote_path):
        """强制从云端下载到本地缓存（用于预热数据集）"""
        local_cached = self.local._full_path(remote_path)
        os.makedirs(os.path.dirname(local_cached), exist_ok=True)
        self.cloud.download(remote_path, local_cached)
        return local_cached


# ============================================================
# 工厂函数 & 全局实例
# ============================================================

_storage_instance = None


def get_storage() -> CloudStorageBase:
    """获取全局存储实例（单例）"""
    global _storage_instance
    if _storage_instance is not None:
        return _storage_instance

    config = _load_storage_config()
    backend = config['backend']

    if backend == 'local':
        _storage_instance = LocalStorage(root_dir=config.get('local_root', ''))
        print(f"[CloudStorage] Using local storage: {config.get('local_root', './')}")

    elif backend == 'cos':
        cloud = CosStorage(
            secret_id=config['cos_secret_id'],
            secret_key=config['cos_secret_key'],
            bucket=config['cos_bucket'],
            region=config['cos_region'],
        )
        print(f"[CloudStorage] Using Tencent COS: bucket={config['cos_bucket']}, region={config['cos_region']}")
        # 自动包装为混合存储
        local_root = config.get('local_root', './local_cache')
        _storage_instance = HybridStorage(cloud, local_root=local_root)
        print(f"[CloudStorage] Hybrid mode: local cache at {local_root}")

    elif backend == 'oss':
        cloud = OssStorage(
            access_key_id=config['oss_access_key_id'],
            access_key_secret=config['oss_access_key_secret'],
            bucket=config['oss_bucket'],
            endpoint=config['oss_endpoint'],
        )
        print(f"[CloudStorage] Using Aliyun OSS: bucket={config['oss_bucket']}, endpoint={config['oss_endpoint']}")
        local_root = config.get('local_root', './local_cache')
        _storage_instance = HybridStorage(cloud, local_root=local_root)
        print(f"[CloudStorage] Hybrid mode: local cache at {local_root}")

    elif backend == 's3':
        cloud = S3Storage(
            access_key=config['s3_access_key'],
            secret_key=config['s3_secret_key'],
            bucket=config['s3_bucket'],
            endpoint=config.get('s3_endpoint', ''),
            region=config.get('s3_region', 'us-east-1'),
        )
        print(f"[CloudStorage] Using S3: bucket={config['s3_bucket']}")
        local_root = config.get('local_root', './local_cache')
        _storage_instance = HybridStorage(cloud, local_root=local_root)
        print(f"[CloudStorage] Hybrid mode: local cache at {local_root}")

    elif backend == 'github':
        cloud = GitHubStorage(
            token=config['github_token'],
            repo=config['github_repo'],
            branch=config.get('github_branch', 'main'),
        )
        print(f"[CloudStorage] Using GitHub: repo={config['github_repo']}, branch={config.get('github_branch', 'main')}")
        local_root = config.get('local_root', './local_cache')
        _storage_instance = HybridStorage(cloud, local_root=local_root)
        print(f"[CloudStorage] Hybrid mode: local cache at {local_root}")

    else:
        raise ValueError(f"[CloudStorage] Unknown backend: {backend}. Supported: local, cos, oss, s3, github")

    return _storage_instance


def reset_storage():
    """重置全局存储实例（用于切换后端）"""
    global _storage_instance
    _storage_instance = None
