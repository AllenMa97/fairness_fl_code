"""
云存储配置文件

使用方式：
  1. 复制此文件为 storage_config.py（如果不存在）
  2. 填写您使用的云存储凭证
  3. 设置 STORAGE_BACKEND 为对应的后端名称

支持的后端：local / cos / oss / s3 / github

注意：此文件包含敏感信息，请勿提交到版本控制！
      建议在 .gitignore 中添加 tool/storage_config.py
"""

# ============================================================
# 存储后端选择：local / cos / oss / s3 / github
# ============================================================
STORAGE_BACKEND = "local"

# ============================================================
# 本地存储配置（STORAGE_BACKEND="local" 时使用）
# ============================================================
STORAGE_LOCAL_ROOT = ""  # 留空表示当前目录，或设为 "/mnt/nas" 等挂载路径

# ============================================================
# 腾讯云 COS 配置（STORAGE_BACKEND="cos" 时使用）
# 安装SDK: pip install cos-python-sdk-v5
# ============================================================
COS_SECRET_ID = ""
COS_SECRET_KEY = ""
COS_BUCKET = ""     # 例如: "fl-experiments-1234567890"
COS_REGION = ""      # 例如: "ap-guangzhou"

# ============================================================
# 阿里云 OSS 配置（STORAGE_BACKEND="oss" 时使用）
# 安装SDK: pip install oss2
# ============================================================
OSS_ACCESS_KEY_ID = ""
OSS_ACCESS_KEY_SECRET = ""
OSS_BUCKET = ""     # 例如: "fl-experiments"
OSS_ENDPOINT = ""   # 例如: "https://oss-cn-guangzhou.aliyuncs.com"

# ============================================================
# S3 兼容配置（STORAGE_BACKEND="s3" 时使用）
# 支持 AWS S3 / MinIO / Cloudflare R2 等
# 安装SDK: pip install boto3
# ============================================================
S3_ACCESS_KEY = ""
S3_SECRET_KEY = ""
S3_BUCKET = ""      # 例如: "fl-experiments"
S3_ENDPOINT = ""    # 留空表示AWS S3，MinIO填: "http://minio-server:9000"
S3_REGION = "us-east-1"

# ============================================================
# GitHub 配置（STORAGE_BACKEND="github" 时使用）
# 安装SDK: pip install PyGithub
#
# 使用方式：
#   1. 创建一个 GitHub Personal Access Token (需要 repo 权限)
#      Settings -> Developer settings -> Personal access tokens -> Generate new token
#   2. 创建一个专用仓库（建议私有）用于存储队列和结果
#   3. 数据集放 Git Release（不走仓库）
# ============================================================
GITHUB_TOKEN = ""     # GitHub Personal Access Token
GITHUB_REPO = ""      # 仓库名，格式: "owner/repo"，例如: "alice/fairness-fl-infra"
GITHUB_BRANCH = "main"  # 分支名
