# -*- coding: utf-8 -*-
"""
模块4 · 第2课：柱状图、散点图、直方图
====================================
本课介绍 ML 最常用的三种"非折线"图：
- 柱状图 bar    ：比较不同类别的"数量/均值"
- 散点图 scatter：观察两个变量的"关系"，特征工程的利器
- 直方图 hist   ：看一个变量的"分布"，检查正态/偏态
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "_images")
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 一、柱状图 bar —— 比较类别
# ============================================================
cities = ["北京", "上海", "广州", "深圳", "杭州"]
gdp = [40269, 43215, 28232, 30632, 18753]

plt.figure(figsize=(7, 4))
plt.bar(cities, gdp, color="steelblue", edgecolor="black")
plt.title("2022年主要城市 GDP（亿元）")
plt.ylabel("GDP")
# 在柱顶标注数值
for i, v in enumerate(gdp):
    plt.text(i, v + 500, str(v), ha="center", fontsize=9)
plt.savefig(os.path.join(OUT_DIR, "06_柱状图.png"), dpi=100, bbox_inches="tight")
plt.close()
print("已保存: 06_柱状图.png")

# 横向柱状图 barh（类别名长时更清晰）
plt.figure(figsize=(7, 4))
plt.barh(cities, gdp, color="coral")
plt.title("横向柱状图")
plt.savefig(os.path.join(OUT_DIR, "07_横向柱状图.png"), dpi=100, bbox_inches="tight")
plt.close()
print("已保存: 07_横向柱状图.png")

# 分组柱状图（两组数据对比）
index = np.arange(len(cities))
width = 0.35
gdp_2022 = gdp
gdp_2023 = [g * 1.05 + np.random.randint(-500, 500) for g in gdp]
plt.figure(figsize=(8, 4))
plt.bar(index - width/2, gdp_2022, width, label="2022", color="steelblue")
plt.bar(index + width/2, gdp_2023, width, label="2023", color="orange")
plt.xticks(index, cities)
plt.title("两年 GDP 对比（分组柱状图）")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "08_分组柱状图.png"), dpi=100, bbox_inches="tight")
plt.close()
print("已保存: 08_分组柱状图.png")


# ============================================================
# 二、散点图 scatter —— 观察关系（特征工程利器！）
# ============================================================
np.random.seed(42)
n = 100
area = np.random.randint(50, 200, n)
price = area * 5 + np.random.randn(n) * 80 + 100     # 正相关 + 噪声

plt.figure(figsize=(7, 5))
plt.scatter(area, price, c="steelblue", alpha=0.6, edgecolors="white", s=40)
plt.title("房屋面积 vs 价格（正相关）")
plt.xlabel("面积（㎡）")
plt.ylabel("价格（万元）")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUT_DIR, "09_散点图.png"), dpi=100, bbox_inches="tight")
plt.close()
print("已保存: 09_散点图.png")

# 【ML 场景】用颜色区分类别（分类问题可视化）
# 假设有两类点
np.random.seed(0)
class0_x = np.random.randn(50) + 2;   class0_y = np.random.randn(50) + 2
class1_x = np.random.randn(50) + 5;   class1_y = np.random.randn(50) + 5
plt.figure(figsize=(6, 5))
plt.scatter(class0_x, class0_y, c="blue", label="类别 0", alpha=0.6)
plt.scatter(class1_x, class1_y, c="red",  label="类别 1", alpha=0.6)
plt.title("两个类别的分布（分类可视化）")
plt.xlabel("特征1"); plt.ylabel("特征2")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUT_DIR, "10_分类散点.png"), dpi=100, bbox_inches="tight")
plt.close()
print("已保存: 10_分类散点.png")


# ============================================================
# 三、直方图 hist —— 看分布（检查正态/偏态/异常值）
# ============================================================
np.random.seed(42)
normal_data = np.random.randn(1000)             # 标准正态分布
skewed_data = np.random.exponential(2, 1000)    # 指数分布（右偏）

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)                             # 1行2列的第1个（下节课详讲）
plt.hist(normal_data, bins=30, color="steelblue", edgecolor="white")
plt.title("正态分布（钟形对称）")
plt.xlabel("值"); plt.ylabel("频数")

plt.subplot(1, 2, 2)
plt.hist(skewed_data, bins=30, color="coral", edgecolor="white")
plt.title("右偏分布（长尾向右）")
plt.xlabel("值")
plt.tight_layout()                               # 自动调整间距，避免重叠
plt.savefig(os.path.join(OUT_DIR, "11_直方图.png"), dpi=100, bbox_inches="tight")
plt.close()
print("已保存: 11_直方图.png")

# bins 参数：控制分多少个"桶"
# bins 大 → 细节多但噪点多；bins 小 → 平滑但丢细节
# 【ML 场景】直方图常用于：
#   检查特征是否正态（很多模型假设正态）
#   发现异常值（直方图最左/最右的孤立柱）


# ============================================================
# 四、箱线图 boxplot —— 五数概括 + 异常值检测
# ============================================================
# 一图看清：最小值、下四分位、中位数、上四分位、最大值、异常值
np.random.seed(0)
group_A = np.random.normal(70, 10, 100)         # 均值70标准差10
group_B = np.random.normal(75, 15, 100)
group_C = np.concatenate([np.random.normal(60, 8, 80), [20, 105, 110]])  # 加异常值

plt.figure(figsize=(7, 5))
# 注意：matplotlib 3.9+ 用 tick_labels 替代旧版 labels 参数
plt.boxplot([group_A, group_B, group_C], tick_labels=["A班", "B班", "C班"])
plt.title("三班成绩箱线图（圈=异常值）")
plt.ylabel("分数")
plt.grid(alpha=0.3, axis="y")
plt.savefig(os.path.join(OUT_DIR, "12_箱线图.png"), dpi=100, bbox_inches="tight")
plt.close()
print("已保存: 12_箱线图.png")


# ============================================================
# 小结：本课你应掌握的"ML 高频"图表
# ============================================================
# 1. bar / barh：类别比较；分组柱状图对比多组
# 2. scatter：看两个变量关系（特征工程）；用颜色区分类别
# 3. hist：看分布（正态/偏态/异常值）
# 4. boxplot：五数概括 + 异常值圈出
# 5. plt.text 在柱顶标值；alpha 控制透明度
#
# 练习（可选）：用 students.csv 的数据
#   (1) 画各城市人数的柱状图
#   (2) 画分数的直方图，bins=5
#   提示：先 df = pd.read_csv(...)，再用 value_counts() / plt.hist(df["分数"])
