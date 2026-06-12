"""
用户配置文件

每个参与贡献的协作者在此文件中设置自己的信息。
此文件不应提交到版本控制（加入 .gitignore）。

也可以通过环境变量设置：
  export FL_USER_NAME=YourName
  export FL_USER_EMAIL=your@university.edu
  export FL_OPENREVIEW=YourOpenReviewID
  export FL_GOOGLE_SCHOLAR=https://scholar.google.com/citations?user=XXXXX
  export FL_MACHINE_LABEL=lab-server
"""

# ============================================================
# 基本信息（必填）
# ============================================================
USER_NAME = ""       # 你的姓名/昵称，例如: "张三"

# ============================================================
# 学术身份（强烈建议填写，用于论文致谢和贡献归属）
# ============================================================
USER_EMAIL = ""      # 学术邮箱，例如: "zhangsan@pku.edu.cn"
USER_AFFILIATION = "" # 所属机构，例如: "Peking University"
USER_OPENREVIEW = ""  # OpenReview 用户名，例如: "San_Zhang"
USER_GOOGLE_SCHOLAR = ""  # Google Scholar 主页URL
USER_ORCID = ""       # ORCID ID（16位数字），例如: "0000-0002-1825-0097"
USER_GITHUB = ""      # GitHub 用户名（可选）

# ============================================================
# 机器标签（可选，帮助区分不同机器）
# ============================================================
# 例如: "lab-server", "home-pc", "cloud-p4-01"
MACHINE_LABEL = ""
