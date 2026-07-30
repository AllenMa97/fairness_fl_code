"""
PDFFed 折纸概览图生成脚本（精修版）
设计哲学：Folded Unity
精修要点：
  1. 原型箭真正贯穿折叠体三层（从左下穿入 → 右上穿出）
  2. 三层叠合透视更锐利，边缘阴影线增强折纸感
  3. 折痕线用渐变 alpha，优雅汇聚
  4. 色块更饱满，文字呼吸空间更充裕
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle, PathPatch
from matplotlib.path import Path
from matplotlib.font_manager import FontProperties
import numpy as np

try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

# 色彩系统
COLOR_BLUE   = '#2C5F8D'
COLOR_RED    = '#B5524A'
COLOR_GREEN  = '#5A7D52'
COLOR_INK    = '#1A1A1A'
COLOR_PAPER  = '#F5F1E8'
COLOR_CREASE = '#8A8580'
COLOR_ARROW  = '#2A2A2A'
COLOR_SHADOW = '#D8D2C4'

fig, ax = plt.subplots(figsize=(14, 7.2), dpi=220)
fig.patch.set_facecolor(COLOR_PAPER)
ax.set_facecolor(COLOR_PAPER)
ax.set_xlim(0, 14)
ax.set_ylim(0, 7.2)
ax.set_aspect('equal')
ax.axis('off')

# ============================================================
# 左侧：平铺正方形折纸，3 个水平色块
# ============================================================
square_x, square_y = 0.7, 1.6
square_size = 3.9
band_h = square_size / 3

bands = [
    (COLOR_BLUE,  '挑战 1：隐私泄露风险',           square_y + 2 * band_h),
    (COLOR_RED,   '挑战 2：公平性-精度协同退化',    square_y + 1 * band_h),
    (COLOR_GREEN, '挑战 3：统计异构下全局公平性',   square_y + 0 * band_h),
]

for color, label, y_bottom in bands:
    rect = Rectangle((square_x, y_bottom), square_size, band_h,
                     facecolor=color, edgecolor=COLOR_INK, linewidth=1.0, alpha=0.92, zorder=2)
    ax.add_patch(rect)
    ax.text(square_x + square_size / 2, y_bottom + band_h / 2, label,
            ha='center', va='center', fontsize=10, color='white',
            fontweight='bold', family='sans-serif', zorder=3)

# 外框
ax.add_patch(Rectangle((square_x, square_y), square_size, square_size,
                       facecolor='none', edgecolor=COLOR_INK, linewidth=1.8, zorder=3))

# 左侧标注
ax.text(square_x + square_size / 2, square_y + square_size + 0.4,
        '三个独立挑战（分散）', ha='center', va='bottom',
        fontsize=11, color=COLOR_INK, family='sans-serif',
        style='italic')

# ============================================================
# 中间：折叠过渡（渐变折痕线 + 弧形折叠箭头）
# ============================================================
mid_x_start = square_x + square_size + 0.25
mid_x_end   = 8.55
mid_center_y = square_y + square_size / 2

# 渐变折痕线：每条线从左到右 alpha 递减，表现汇聚
n_seg = 30
for i, (color, _, y_band_center) in enumerate([
    (COLOR_BLUE,  None, square_y + 2 * band_h + band_h / 2),
    (COLOR_RED,   None, square_y + 1 * band_h + band_h / 2),
    (COLOR_GREEN, None, square_y + 0 * band_h + band_h / 2),
]):
    y_left = y_band_center
    y_right = mid_center_y
    for j in range(n_seg):
        t1 = j / n_seg
        t2 = (j + 1) / n_seg
        x1 = mid_x_start + t1 * (mid_x_end - mid_x_start)
        x2 = mid_x_start + t2 * (mid_x_end - mid_x_start)
        y1 = y_left + t1 * (y_right - y_left)
        y2 = y_left + t2 * (y_right - y_left)
        alpha = 0.65 * (1 - t1 * 0.85)
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.6, alpha=alpha, zorder=2)

# 折叠动作弧形箭头
fold_arrow = FancyArrowPatch(
    (mid_x_start + 0.15, mid_center_y + 1.9),
    (mid_x_end - 0.15, mid_center_y + 0.35),
    connectionstyle="arc3,rad=-0.35",
    arrowstyle='->,head_width=5,head_length=7',
    color=COLOR_CREASE, linewidth=1.4, alpha=0.85, zorder=3
)
ax.add_patch(fold_arrow)
ax.text((mid_x_start + mid_x_end) / 2, mid_center_y + 2.35,
        'PDFFed 折叠（统一框架）', ha='center', va='bottom',
        fontsize=10, color=COLOR_CREASE, family='sans-serif',
        style='italic')

# ============================================================
# 右侧：折叠后的立体结构（3层透视叠合）+ 原型箭真正贯穿
# ============================================================
fold_base_x = 9.1
fold_base_y = 2.15
fold_w = 3.0
fold_h = 2.6
skew = 0.32

# 三层叠合（底→顶：绿、红、蓝），逐层向右上错位
layers = [
    (COLOR_GREEN, fold_base_x - skew * 1.4, fold_base_y - skew * 1.4),
    (COLOR_RED,   fold_base_x - skew * 0.7, fold_base_y - skew * 0.7),
    (COLOR_BLUE,  fold_base_x,              fold_base_y),
]

layer_polys = []
for color, x, y in layers:
    poly = Polygon([
        (x, y),
        (x + fold_w, y),
        (x + fold_w + skew, y + fold_h),
        (x + skew, y + fold_h)
    ], facecolor=color, edgecolor=COLOR_INK, linewidth=1.0, alpha=0.80, zorder=4)
    ax.add_patch(poly)
    layer_polys.append((x, y))

# 顶层外轮廓强调
top_x, top_y = fold_base_x, fold_base_y
outline = Polygon([
    (top_x, top_y),
    (top_x + fold_w, top_y),
    (top_x + fold_w + skew, top_y + fold_h),
    (top_x + skew, top_y + fold_h)
], facecolor='none', edgecolor=COLOR_INK, linewidth=1.6, zorder=5)
ax.add_patch(outline)

# 折纸边缘阴影线（增强折纸质感）—— 在顶层右侧和顶边加细阴影
shadow_top = Polygon([
    (top_x + skew, top_y + fold_h),
    (top_x + fold_w + skew, top_y + fold_h),
    (top_x + fold_w + skew - 0.06, top_y + fold_h - 0.06),
    (top_x + skew - 0.06, top_y + fold_h - 0.06)
], facecolor=COLOR_SHADOW, edgecolor='none', alpha=0.5, zorder=6)
ax.add_patch(shadow_top)

# ============================================================
# 原型箭：真正贯穿折叠体（从左下穿入 → 穿过三层 → 右上射出）
# ============================================================
# 穿入点：折叠体底层左下区域；穿出点：顶层右上区域
arrow_start_x = fold_base_x - skew * 1.4 + 0.3
arrow_start_y = fold_base_y - skew * 1.4 + 0.3
arrow_end_x   = fold_base_x + fold_w + skew + 0.5
arrow_end_y   = fold_base_y + fold_h + 0.5

# 箭身（贯穿，zorder 高于色块）
proto_arrow = FancyArrowPatch(
    (arrow_start_x, arrow_start_y),
    (arrow_end_x, arrow_end_y),
    arrowstyle='-|>,head_width=13,head_length=16',
    color=COLOR_ARROW, linewidth=3.0, zorder=10,
    mutation_scale=1.0
)
ax.add_patch(proto_arrow)

# 穿入点与穿出点的小标记（表现"穿透"）
ax.plot(arrow_start_x, arrow_start_y, 'o', markersize=5,
        color=COLOR_ARROW, zorder=11)
ax.plot(arrow_end_x - 0.02, arrow_end_y - 0.02, 'o', markersize=4,
        color=COLOR_ARROW, zorder=11, alpha=0.6)

# 箭标注
ax.text(arrow_end_x + 0.15, arrow_end_y + 0.3,
        '原型 (Prototype)', ha='left', va='bottom',
        fontsize=11, color=COLOR_INK, family='sans-serif',
        fontweight='bold')

# 右侧标注
ax.text(fold_base_x + fold_w / 2 + skew / 2, fold_base_y + fold_h + 0.55,
        '统一框架（原型贯穿）', ha='center', va='bottom',
        fontsize=11, color=COLOR_INK, family='sans-serif',
        style='italic')

# ============================================================
# 图题与图释
# ============================================================
fig.text(0.5, 0.95, 'PDFFed：原型驱动的统一框架',
         ha='center', va='top', fontsize=15.5, color=COLOR_INK,
         fontweight='bold', family='sans-serif')
fig.text(0.5, 0.045,
         '折叠前：3 个挑战分散独立   →   折叠后：PDFFed 统一到一个原型驱动框架，原型作为核心机制贯穿所有挑战',
         ha='center', va='bottom', fontsize=9.5, color=COLOR_CREASE,
         family='sans-serif', style='italic')

plt.tight_layout(rect=[0, 0.07, 1, 0.92])
output_path = r'd:\最新PDFFed\fairness_fl_code\docs\pdffed_origami_overview.png'
plt.savefig(output_path, dpi=220, facecolor=COLOR_PAPER, bbox_inches='tight', pad_inches=0.35)
plt.close()
print(f'精修版图已保存：{output_path}')
