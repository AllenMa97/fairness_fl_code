# -*- coding: utf-8 -*-
"""Render PDFFed concept diagram to PNG (2800x1980) with matplotlib."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.patheffects import withStroke
import matplotlib.font_manager as fm

for f in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc"]:
    fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False


def C(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


fig, ax = plt.subplots(figsize=(14, 9.9), dpi=200)
ax.set_xlim(0, 1400)
ax.set_ylim(990, 0)  # top-left origin
ax.axis("off")
ax.set_position([0, 0, 1, 1])


def box(x, y, w, h, bg, ec, lw=2.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=16",
                                fc=C(bg), ec=C(ec), lw=lw, zorder=2))


def txt(cx, cy, s, size, color, weight="normal", italic=False, halo=True):
    t = ax.text(cx, cy, s, ha="center", va="center", fontsize=size,
                color=C(color), fontweight=weight,
                fontstyle="italic" if italic else "normal", zorder=4)
    if halo:
        t.set_path_effects([withStroke(linewidth=4, foreground="white")])
    return t


def box_text(x, y, w, h, items, line_h=32):
    cx = x + w / 2
    total = (len(items) - 1) * line_h
    y_top = y + (h - total) / 2
    for i, (s, size, color, weight) in enumerate(items):
        txt(cx, y_top + i * line_h, s, size, color, weight)


def arrow(p1, p2, color, lw=3, ls="-"):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", color=C(color), lw=lw,
                                linestyle=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=18), zorder=3)


def seg(p1, p2, color, lw=2.2, ls="--"):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=C(color), lw=lw,
            linestyle=ls, zorder=3, solid_capstyle="round")


# ---- title / subtitle ----
txt(700, 44, "PDFFed：一个载体 · 两条路径 · 一个闭环", 26, "#1F3864", weight="bold")
txt(700, 106, "类-组原型驱动的群组公平—精度协同优化", 14, "#7F7F7F", italic=True)

# ---- center ----
box(503, 348, 396, 180, "#DAE8FC", "#6C8EBF", lw=2.6)
box_text(503, 348, 396, 180, [
    ("类-组原型（CG Proto）", 19, "#1F3864", "bold"),
    ("统一知识载体", 14, "#2F4F7F", "normal"),
    ("同时编码 类别 × 群组 分布信息", 12.5, "#4A6A9A", "normal"),
], line_h=34)

# ---- FR ----
box(72, 258, 348, 228, "#F8CECC", "#B85450", lw=2.6)
box_text(72, 258, 348, 228, [
    ("公平性路径（FR）", 16.5, "#843030", "bold"),
    ("L_PA：压缩同类异群组", 13, "#843030", "normal"),
    ("原型距离 Δ_rep → 收窄 EO 上界", 13, "#843030", "normal"),
    ("梯度作用域：φ（引理2）", 13, "#B85450", "bold"),
], line_h=33)

# ---- ACC ----
box(980, 258, 348, 228, "#D5E8D4", "#82B366", lw=2.6)
box_text(980, 258, 348, 228, [
    ("精度路径（ACC）", 16.5, "#376027", "bold"),
    ("L_l2l：原型分类一致性", 13, "#376027", "normal"),
    ("→ 传导至样本级精度", 13, "#376027", "normal"),
    ("梯度作用域：w（引理4）", 13, "#5E9949", "bold"),
], line_h=33)

# ---- loop label ----
txt(700, 596, "一个闭环：原型贯穿 训练 — 通信 — 聚合", 14.5, "#7F6000",
    italic=True, weight="bold")

# ---- loop nodes ----
box(144, 636, 276, 96, "#FFF2CC", "#D6B656", lw=2.4)
box_text(144, 636, 276, 96, [
    ("客户端局部训练", 14.5, "#7F6000", "bold"),
    ("L_PA + L_l2l 同时执行", 12.5, "#8A6D1F", "normal"),
], line_h=30)

box(563, 660, 276, 96, "#FFF2CC", "#D6B656", lw=2.4)
box_text(563, 660, 276, 96, [
    ("通信传输", 14.5, "#7F6000", "bold"),
    ("仅传 模型参数 + 原型", 12.5, "#8A6D1F", "normal"),
    ("（不传群组统计量）", 12.5, "#8A6D1F", "normal"),
], line_h=26)

box(984, 636, 276, 96, "#FFF2CC", "#D6B656", lw=2.4)
box_text(984, 636, 276, 96, [
    ("服务器聚合", 14.5, "#7F6000", "bold"),
    ("全局原型 / 分类头校正", 12.5, "#8A6D1F", "normal"),
], line_h=30)

# ---- slogan ----
box(341, 828, 720, 120, "#E1D5E7", "#9673A6", lw=2.6)
box_text(341, 828, 720, 120, [
    ("协同闭环成立", 18, "#4C3A63", "bold"),
    ("φ 管公平（L_PA）、w 管精度（L_l2l）", 14, "#4C3A63", "normal"),
    ("参数作用域不相交 → 同时执行、互不冲突", 14, "#4C3A63", "normal"),
], line_h=33)

# ---- edges ----
arrow((503, 438), (420, 372), "#B85450")
txt(461, 424, "公平性信号", 12.5, "#B85450", weight="bold")

arrow((899, 438), (980, 372), "#82B366")
txt(940, 424, "精度信号", 12.5, "#82B366", weight="bold")

arrow((420, 684), (563, 708), "#D6B656")
arrow((839, 708), (984, 684), "#D6B656")

# return path (agg -> train)
seg((1122, 732), (1122, 796), "#D6B656", lw=2.6, ls="-")
seg((1122, 796), (282, 796), "#D6B656", lw=2.6, ls="-")
arrow((282, 796), (282, 744), "#D6B656")

# dashed connectors center -> loop nodes
seg((620, 528), (282, 636), "#6C8EBF", lw=2.0)
seg((701, 528), (701, 660), "#6C8EBF", lw=2.0)
seg((782, 528), (1122, 636), "#6C8EBF", lw=2.0)

out = r"d:\最新PDFFed\fairness_fl_code\PDFFed概念图.png"
fig.savefig(out, dpi=200, facecolor="white")
print("saved:", out)

# ---- ASCII self-check of layout ----
from PIL import Image
im = Image.open(out).convert("L").resize((110, 66))
px = im.load()
chars = " .:-=+*#%@"
for j in range(im.height):
    row = ""
    for i in range(im.width):
        v = px[i, j]
        row += chars[min(len(chars) - 1, (255 - v) * (len(chars) - 1) // 255)]
    print(row)
