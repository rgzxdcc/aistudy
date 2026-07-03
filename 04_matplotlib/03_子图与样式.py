# -*- coding: utf-8 -*-
"""
模块4 · 第3课：子图、样式与注释
===============================
本课讲"怎么把多张图组织在一起"和"怎么让图更美观可读"。
核心：subplot 子图、样式主题、图例、注释。
"""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "_images")
os.makedirs(OUT_DIR, exist_ok=True)

x = np.linspace(0, 2 * np.pi, 100)


# ============================================================
# 一、subplot —— 一张画布放多张图（脚本式）
# ============================================================
# plt.subplot(行数, 列数, 第几个)
# 例如 subplot(2, 2, 1) 表示 2行2列 网格的第 1 个位置

plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)            # 左上
plt.plot(x, np.sin(x), "b-")
plt.title("sin(x)"); plt.grid(alpha=0.3)

plt.subplot(2, 2, 2)            # 右上
plt.plot(x, np.cos(x), "r-")
plt.title("cos(x)"); plt.grid(alpha=0.3)

plt.subplot(2, 2, 3)            # 左下
plt.plot(x, np.tan(x), "g-")
plt.ylim(-5, 5)                 # 限制 y 范围（tan 有渐近线，不限制会爆炸）
plt.title("tan(x)"); plt.grid(alpha=0.3)

plt.subplot(2, 2, 4)            # 右下
plt.plot(x, -np.sin(x), "m-")
plt.title("-sin(x)"); plt.grid(alpha=0.3)

plt.suptitle("四合一子图（subplot）", fontsize=14)   # 总标题
plt.tight_layout()              # 自动调整间距（强烈推荐！）
plt.savefig(os.path.join(OUT_DIR, "13_四合一子图.png"), dpi=100, bbox_inches="tight")
plt.close()
print("已保存: 13_四合一子图.png")


# ============================================================
# 二、subplots —— 面向对象式（推荐！更灵活）
# ============================================================
# fig, axes = plt.subplots(行, 列) 一次拿到画布和"坐标系数组"
# axes 可以像二维数组那样索引：axes[0,1]

fig, axes = plt.subplots(2, 3, figsize=(12, 6))

functions = [
    (np.sin, "sin"),
    (np.cos, "cos"),
    (lambda t: np.sin(2*t), "sin(2x)"),
    (lambda t: np.sin(t) * np.exp(-t/5), "阻尼振荡"),
    (lambda t: t % 2, "锯齿"),
    (lambda t: np.abs(np.sin(t)), "|sin|"),
]

for i in range(2):
    for j in range(3):
        func, name = functions[i * 3 + j]
        axes[i, j].plot(x, func(x))
        axes[i, j].set_title(name)
        axes[i, j].grid(alpha=0.3)

plt.suptitle("subplots 网格（2×3）", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "14_subplots网格.png"), dpi=100, bbox_inches="tight")
plt.close()
print("已保存: 14_subplots网格.png")


# ============================================================
# 三、样式主题 style —— 一键美化
# ============================================================
# 内置主题：ggplot / seaborn-v0_8 / bmh / dark_background / classic 等
# 用法：with plt.style.context("名字"):  （推荐用 with 临时切换）

themes = ["default", "ggplot", "dark_background", "seaborn-v0_8-bright"]
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
for ax, theme in zip(axes.flat, themes):
    with plt.style.context(theme):       # 临时应用主题
        ax.plot(x, np.sin(x), lw=2)
        ax.plot(x, np.cos(x), lw=2)
        ax.set_title(f"主题: {theme}")
        ax.grid(alpha=0.3)
plt.suptitle("四种主题对比", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "15_主题对比.png"), dpi=100, bbox_inches="tight")
plt.close()
print("已保存: 15_主题对比.png")

# 列出所有可用主题
# print(plt.style.available)


# ============================================================
# 四、图例、注释、参考线
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4))

# 多条线 + 图例
ax.plot(x, np.sin(x), label="sin(x)", lw=2)
ax.plot(x, np.cos(x), label="cos(x)", lw=2)
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)  # framealpha 图例透明度

# 参考线：水平 + 垂直
ax.axhline(y=0,  color="gray", ls="--", alpha=0.5)        # y=0 水平线
ax.axvline(x=np.pi, color="red", ls=":", alpha=0.5, label="x=π")  # x=π 垂直线

# 文字注释 annotate
ax.annotate("最大值点",                                          # 注释文字
            xy=(np.pi/2, 1),                                     # 箭头指向的点
            xytext=(2, 1.5),                                     # 文字位置
            arrowprops=dict(arrowstyle="->", color="red"),       # 箭头样式
            fontsize=11, color="red")

ax.set_title("图例、参考线、注释")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.grid(alpha=0.3)
plt.savefig(os.path.join(OUT_DIR, "16_图例注释.png"), dpi=100, bbox_inches="tight")
plt.close()
print("已保存: 16_图例注释.png")


# ============================================================
# 五、【ML 场景】子图实战：同时画 4 张诊断图
# ============================================================
# 模型评估时常用一张画布展示：训练曲线/预测散点/残差/特征重要性

np.random.seed(0)
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# 子图1：损失曲线
epochs = np.arange(1, 51)
train_loss = 2.0 * np.exp(-epochs / 8)
axes[0, 0].plot(epochs, train_loss, "b-", lw=2)
axes[0, 0].set_title("训练损失"); axes[0, 0].grid(alpha=0.3)

# 子图2：预测 vs 真实（理想情况应贴近对角线）
y_true = np.linspace(0, 10, 50)
y_pred = y_true + np.random.randn(50) * 0.8
axes[0, 1].scatter(y_true, y_pred, alpha=0.6)
axes[0, 1].plot([0, 10], [0, 10], "r--", alpha=0.5)            # 对角线参考
axes[0, 1].set_title("预测 vs 真实"); axes[0, 1].grid(alpha=0.3)

# 子图3：残差分布（应为正态、均值0）
residuals = y_pred - y_true
axes[1, 0].hist(residuals, bins=15, color="steelblue", edgecolor="white")
axes[1, 0].set_title("残差分布"); axes[1, 0].grid(alpha=0.3)

# 子图4：特征重要性（条形图）
features = ["面积", "房龄", "楼层", "地段", "朝向"]
importance = [0.45, 0.20, 0.12, 0.18, 0.05]
axes[1, 1].barh(features, importance, color="coral")
axes[1, 1].set_title("特征重要性")

plt.suptitle("ML 模型诊断仪表盘", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "17_ML诊断仪表盘.png"), dpi=100, bbox_inches="tight")
plt.close()
print("已保存: 17_ML诊断仪表盘.png")


# ============================================================
# 小结：本课你应掌握的"ML 高频"操作
# ============================================================
# 1. subplot(行,列,第几) 脚本式；subplots(行,列) 面向对象式（推荐）
# 2. plt.style.context("ggplot") 切换主题；tight_layout 自动间距
# 3. axhline / axvline 参考线；annotate 箭头注释
# 4. legend 的 loc / framealpha 参数
# 5. ML 诊断仪表盘：一张画布组合多张图（损失/散点/残差/重要性）
