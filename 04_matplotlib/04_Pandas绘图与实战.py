# -*- coding: utf-8 -*-
"""
模块4 · 第4课：Pandas 绘图与综合实战（可视化收官）
==================================================
Pandas 把 matplotlib 封装得超简单：df.plot() 一行出图。
本课演示如何用最少的代码完成数据可视化，并做一个综合分析实战。
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "_images")
DATA_DIR = os.path.join(HERE, "..", "06_datasets")
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 一、Pandas 一行绘图 df.plot()
# ============================================================
# df.plot(kind=...) —— 一行搞定折线/柱状/散点/直方/箱线/饼图
# kind 可选: line / bar / barh / hist / box / area / scatter / pie

# 1.1 折线图（默认）
months = pd.date_range("2024-01-01", periods=12, freq="ME")
sales = pd.DataFrame({
    "2024年": [12, 15, 18, 25, 30, 42, 45, 43, 38, 28, 20, 35],
    "2023年": [10, 13, 16, 22, 27, 38, 40, 38, 33, 25, 17, 30],
}, index=range(1, 13))

ax = sales.plot(kind="line", figsize=(8, 4), marker="o", title="月度销售对比")
ax.set_xlabel("月份"); ax.set_ylabel("销售额（万元）")
ax.grid(alpha=0.3)
plt.savefig(os.path.join(OUT_DIR, "18_pandas折线.png"), dpi=100, bbox_inches="tight")
plt.close()
print("已保存: 18_pandas折线.png")

# 1.2 柱状图
ax = sales.plot(kind="bar", figsize=(9, 4), title="两年月度销售对比")
ax.set_xlabel("月份"); ax.set_ylabel("销售额")
plt.savefig(os.path.join(OUT_DIR, "19_pandas柱状.png"), dpi=100, bbox_inches="tight")
plt.close()
print("已保存: 19_pandas柱状.png")

# 1.3 直方图
np.random.seed(42)
df_tmp = pd.DataFrame({"分数": np.random.normal(75, 15, 200)})
ax = df_tmp.plot(kind="hist", bins=20, figsize=(7, 4), edgecolor="white",
                 title="分数分布", legend=False)
ax.set_xlabel("分数")
plt.savefig(os.path.join(OUT_DIR, "20_pandas直方.png"), dpi=100, bbox_inches="tight")
plt.close()
print("已保存: 20_pandas直方.png")

# 1.4 箱线图（按类别分组）
df_box = pd.DataFrame({
    "班级": ["A"] * 50 + ["B"] * 50 + ["C"] * 50,
    "分数": np.concatenate([
        np.random.normal(70, 10, 50),
        np.random.normal(75, 12, 50),
        np.random.normal(65, 8, 50),
    ]),
})
df_box.pivot(columns="班级", values="分数").plot(
    kind="box", figsize=(7, 4), title="各班分数箱线图")
plt.ylabel("分数")
plt.savefig(os.path.join(OUT_DIR, "21_pandas箱线.png"), dpi=100, bbox_inches="tight")
plt.close()
print("已保存: 21_pandas箱线.png")


# ============================================================
# 二、综合实战：房价数据分析（端到端可视化）
# ============================================================
# 读取前面 Pandas 课生成的房价数据
csv = os.path.join(DATA_DIR, "houses.csv")
if os.path.exists(csv):
    df = pd.read_csv(csv)
    print(f"\n=== 房价分析：{len(df)} 条数据 ===")
    print(df.head())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 2.1 直方图：价格分布
    axes[0, 0].hist(df["价格"], bins=25, color="steelblue", edgecolor="white")
    axes[0, 0].set_title("房价分布"); axes[0, 0].set_xlabel("价格")
    axes[0, 0].grid(alpha=0.3)

    # 2.2 散点图：面积 vs 价格（最重要的关系图）
    sc = axes[0, 1].scatter(df["面积"], df["价格"],
                             c=df["房龄"], cmap="viridis", alpha=0.6)
    axes[0, 1].set_title("面积 vs 价格（颜色=房龄）")
    axes[0, 1].set_xlabel("面积"); axes[0, 1].set_ylabel("价格")
    fig.colorbar(sc, ax=axes[0, 1], label="房龄")
    axes[0, 1].grid(alpha=0.3)

    # 2.3 箱线图：不同房间数的价格差异
    df.boxplot(column="价格", by="房间数", ax=axes[1, 0])
    axes[1, 0].set_title("不同房间数的价格分布")
    axes[1, 0].set_xlabel("房间数"); axes[1, 0].set_ylabel("价格")

    # 2.4 相关性柱状图：各特征与价格的相关系数
    corr = df.corr()["价格"].drop("价格").sort_values()
    axes[1, 1].barh(corr.index, corr.values, color="coral")
    axes[1, 1].set_title("特征与价格的相关系数")
    axes[1, 1].axvline(x=0, color="gray", ls="--", alpha=0.5)
    axes[1, 1].grid(alpha=0.3, axis="x")

    plt.suptitle("房价数据综合分析", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "22_房价综合分析.png"), dpi=100, bbox_inches="tight")
    plt.close()
    print("已保存: 22_房价综合分析.png")
else:
    print("未找到 houses.csv，请先运行 03_pandas/02_数据读写与查看.py")


# ============================================================
# 三、饼图（类别占比）—— 补充图型
# ============================================================
classes = ["A班", "B班", "C班", "D班"]
counts = [25, 30, 20, 15]
plt.figure(figsize=(6, 6))
plt.pie(counts, labels=classes, autopct="%1.1f%%", startangle=90,
        colors=["steelblue", "coral", "limegreen", "violet"])
plt.title("班级人数占比")
plt.savefig(os.path.join(OUT_DIR, "23_饼图.png"), dpi=100, bbox_inches="tight")
plt.close()
print("已保存: 23_饼图.png")


# ============================================================
# 小结：本课你应掌握的"ML 高频"操作
# ============================================================
# 1. df.plot(kind=...) 一行出图：line/bar/hist/box/scatter/pie
# 2. df.boxplot(column=, by=) 按类别分组看分布
# 3. scatter 的 c/cmap 用颜色编码第三个变量（多维可视化）
# 4. corr() 算相关系数 + 柱状图 —— 选特征的核心方法
# 5. 综合仪表盘：subplots + 多种图型组合
#
# === 模块 4 · Matplotlib 数据可视化 结业 ===
# 你已掌握 ML 必备的可视化能力：折线/柱状/散点/直方/箱线/饼图 + 子图组合。
# 下一阶段进入模块 5：机器学习实战（Scikit-learn），这才是真正"训练模型"！
