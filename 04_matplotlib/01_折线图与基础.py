# -*- coding: utf-8 -*-
"""
模块4 · 第1课：折线图与基础
==========================
Matplotlib 是 Python 的"画布"。ML 中用于观察数据趋势、画出损失曲线等。
约定俗成：import matplotlib.pyplot as plt

【中文字体】Matplotlib 默认不支持中文，需配置字体。Windows 用 SimHei（黑体）。
【运行模式】本课脚本把图保存成 PNG，便于无界面环境运行；交互时改用 plt.show()。
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ============================================================
# 0. 全局配置：中文字体 + 保存目录
# ============================================================
# Windows 中文字体配置（让标题、坐标轴中文不乱码）
plt.rcParams["font.sans-serif"] = ["SimHei"]        # 黑体
plt.rcParams["axes.unicode_minus"] = False          # 负号正常显示

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "_images")
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 一、最简单的折线图（5 行代码出图）
# ============================================================
# 画图三件套：plt.plot(数据) → 加标题/标签 → plt.show() / savefig

x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 5, 3]

plt.figure(figsize=(6, 4))     # 画布大小（宽6英寸×高4英寸）
plt.plot(x, y)                 # 画折线
plt.title("最简单的折线图")
plt.savefig(os.path.join(OUT_DIR, "01_最简单折线.png"), dpi=100, bbox_inches="tight")
plt.close()                    # 关闭当前图（释放内存，循环画图必备）
print("已保存: 01_最简单折线.png")


# ============================================================
# 二、完善一张"合格"的图（标题/坐标轴/图例/网格）
# ============================================================
# 一张能给别人看的图，至少要有：标题、x轴标签、y轴标签、图例

months = np.arange(1, 13)
sales_2024 = np.array([12, 15, 18, 25, 30, 42, 45, 43, 38, 28, 20, 35])
sales_2025 = np.array([15, 18, 22, 28, 35, 50, 55, 52, 45, 35, 25, 40])

plt.figure(figsize=(8, 4))
plt.plot(months, sales_2024, color="blue", linestyle="-",  marker="o", label="2024年")
plt.plot(months, sales_2025, color="red",  linestyle="--", marker="s", label="2025年")

plt.title("全年销售额对比", fontsize=14)
plt.xlabel("月份")
plt.ylabel("销售额（万元）")
plt.xticks(months)                              # x 轴刻度用 1~12
plt.legend(loc="upper left")                    # 图例位置
plt.grid(True, alpha=0.3)                       # 网格，alpha 透明度
plt.savefig(os.path.join(OUT_DIR, "02_完善折线.png"), dpi=100, bbox_inches="tight")
plt.close()
print("已保存: 02_完善折线.png")


# ============================================================
# 三、plot 常用参数速查
# ============================================================
# color:    "red"/"blue"/"#FF5733" 十六进制
# linestyle: "-"实线  "--"虚线  "-."点划  ":"点线  "None"/""无线
# marker:   "o"圆  "s"方  "^"三角  "*"星  "D"菱形  "+"加号
# linewidth(lw): 线宽
# markersize(ms): 标记大小

x = np.linspace(0, 2 * np.pi, 50)
plt.figure(figsize=(7, 4))
plt.plot(x, np.sin(x), color="blue",  ls="-",  marker="", lw=2, label="sin")
plt.plot(x, np.cos(x), color="green", ls="--", marker="", lw=2, label="cos")
plt.plot(x, np.sin(x) + np.cos(x),    ls=":",  marker="^", ms=4,
         markevery=5, label="sin+cos")          # markevery 每隔5个点画标记
plt.title("plot 参数演示")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUT_DIR, "03_plot参数.png"), dpi=100, bbox_inches="tight")
plt.close()
print("已保存: 03_plot参数.png")


# ============================================================
# 四、【ML 场景】绘制训练过程的损失曲线
# ============================================================
# 这是最常见的 ML 可视化需求：观察 loss 是否下降、是否收敛

np.random.seed(0)
epochs = np.arange(1, 51)
train_loss = 2.0 * np.exp(-epochs / 10) + np.random.randn(50) * 0.05   # 训练损失下降
val_loss   = 2.0 * np.exp(-epochs / 12) + np.random.randn(50) * 0.08   # 验证损失

plt.figure(figsize=(8, 4))
plt.plot(epochs, train_loss, label="训练损失", color="blue")
plt.plot(epochs, val_loss,   label="验证损失", color="orange")
plt.axhline(y=0.1, color="red", linestyle="--", alpha=0.5, label="目标值")  # 水平参考线
plt.title("模型训练损失曲线（ML 最常见可视化）")
plt.xlabel("训练轮次 Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUT_DIR, "04_损失曲线.png"), dpi=100, bbox_inches="tight")
plt.close()
print("已保存: 04_损失曲线.png")


# ============================================================
# 五、figure 与坐标系的概念（理解层次）
# ============================================================
# plt 是脚本式画图（简单）；面向对象方式更灵活（高级）
#   Figure  : 整张画布
#   Axes    : 一个坐标系（一张 Figure 可有多个 Axes）

fig, ax = plt.subplots(figsize=(6, 4))     # fig 是画布，ax 是坐标系
ax.plot(months, sales_2024, label="2024")
ax.set_title("面向对象画法")
ax.set_xlabel("月份"); ax.set_ylabel("销售额")
ax.legend(); ax.grid(alpha=0.3)
plt.savefig(os.path.join(OUT_DIR, "05_面向对象画法.png"), dpi=100, bbox_inches="tight")
plt.close()
print("已保存: 05_面向对象画法.png")


# ============================================================
# 小结：本课你应掌握的"ML 高频"操作
# ============================================================
# 1. 中文字体：plt.rcParams["font.sans-serif"]=["SimHei"]
# 2. 三件套：plot → title/legend → savefig/show
# 3. plot 参数：color/linestyle/marker/linewidth
# 4. axhline 画参考线（标注目标值/阈值）
# 5. plt vs 面向对象（fig, ax）—— 复杂图用面向对象
# 6. plt.close() 释放内存（循环画图必备）
#
# 练习（可选）：生成一个 sin 和 cos 的图，加上标题、图例、网格，保存为 png。
