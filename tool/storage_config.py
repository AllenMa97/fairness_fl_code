"""
云存储配置文件

此文件可以安全提交到版本控制（Token等敏感信息留空，通过环境变量注入）。

使用方式：
  1. 设置 STORAGE_BACKEND 为对应的后端名称
  2. 非敏感配置（如仓库名、bucket名）直接填写在此文件中
  3. 敏感信息（Token、Secret Key）通过环境变量设置：
     export GITHUB_TOKEN=ghp_xxxxx
     export COS_SECRET_ID=xxxxx
     ...

支持的后端：local / cos / oss / s3 / github
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
# 敏感信息通过环境变量设置: COS_SECRET_ID, COS_SECRET_KEY
# ============================================================
COS_SECRET_ID = ""      # ← 请设置环境变量 COS_SECRET_ID
COS_SECRET_KEY = ""     # ← 请设置环境变量 COS_SECRET_KEY
COS_BUCKET = ""         # 例如: "fl-experiments-1234567890"
COS_REGION = ""         # 例如: "ap-guangzhou"

# ============================================================
# 阿里云 OSS 配置（STORAGE_BACKEND="oss" 时使用）
# 安装SDK: pip install oss2
# 敏感信息通过环境变量设置: OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET
# ============================================================
OSS_ACCESS_KEY_ID = ""          # ← 请设置环境变量 OSS_ACCESS_KEY_ID
OSS_ACCESS_KEY_SECRET = ""     # ← 请设置环境变量 OSS_ACCESS_KEY_SECRET
OSS_BUCKET = ""                # 例如: "fl-experiments"
OSS_ENDPOINT = ""              # 例如: "https://oss-cn-guangzhou.aliyuncs.com"

# ============================================================
# S3 兼容配置（STORAGE_BACKEND="s3" 时使用）
# 支持 AWS S3 / MinIO / Cloudflare R2 等
# 安装SDK: pip install boto3
# 敏感信息通过环境变量设置: S3_ACCESS_KEY, S3_SECRET_KEY
# ============================================================
S3_ACCESS_KEY = ""     # ← 请设置环境变量 S3_ACCESS_KEY
S3_SECRET_KEY = ""     # ← 请设置环境变量 S3_SECRET_KEY
S3_BUCKET = ""          # 例如: "fl-experiments"
S3_ENDPOINT = ""        # 留空表示AWS S3，MinIO填: "http://minio-server:9000"
S3_REGION = "us-east-1"

# ============================================================
# GitHub 配置（STORAGE_BACKEND="github" 时使用）
# 安装SDK: pip install PyGithub
#
# ★ 必须设置环境变量 GITHUB_TOKEN（不要把Token写在这里！）
#   Linux/Mac:  export GITHUB_TOKEN=ghp_xxxxxxxxxxxx
#   Windows:    $env:GITHUB_TOKEN = "ghp_xxxxxxxxxxxx"
#
# Token创建方式：
#   GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
#   → Generate new token → 勾选 "repo" 权限
# ============================================================
GITHUB_TOKEN = ""      # ← 请设置环境变量 GITHUB_TOKEN（不要在此填写！）
GITHUB_REPO = "AllenMa97/fairness_fl_code"  # 仓库名（固定）
GITHUB_BRANCH = "main" # 分支名
