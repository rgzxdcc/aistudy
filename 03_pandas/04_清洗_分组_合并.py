# -*- coding: utf-8 -*-
"""
模块3 · 第4课：清洗、分组、合并（Pandas 实战收官）
==================================================
本课是数据分析最核心的能力：把"脏数据"变干净，再分组统计、合并多张表。
学完即具备处理真实业务数据的能力，可直接进入 ML 实战模块。
"""
import os
import numpy as np
import pandas as pd


# ============================================================
# 一、数据清洗：缺失值处理
# ============================================================
# 真实数据几乎都有缺失（NaN），ML 模型无法直接处理，必须先清洗。

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "..", "06_datasets", "students.csv")
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

print("=== 原始数据 ===")
print(df)
print("\n缺失值统计:\n", df.isnull().sum())

# 1.1 检测缺失值
print("\nisnull 前 3 行:\n", df.head(3).isnull())    # True 表示缺失

# 1.2 丢弃缺失：dropna
#   how="any"  只要有一个缺失就删行（默认）
#   how="all"  全部缺失才删
#   subset=["列"]  只看特定列
df_drop = df.dropna(subset=["分数", "年龄"])   # 只在分数/年龄缺失时删行
print(f"\ndropna 后: {len(df)} → {len(df_drop)} 行")
print(df_drop[["学号", "姓名", "分数"]])

# 1.3 填充缺失：fillna（通常比删除更好，保留数据）
df_fill = df.copy()
df_fill["年龄"] = df_fill["年龄"].fillna(df_fill["年龄"].mean())   # 用均值填
df_fill["分数"] = df_fill["分数"].fillna(df_fill["分数"].median()) # 用中位数填
print("\n填充后缺失检查:", df_fill[["年龄", "分数"]].isnull().sum().to_dict())

# 【ML 场景】Scikit-learn 的 SimpleImputer 就是在做这件事


# ============================================================
# 二、数据清洗：重复值处理
# ============================================================
print("\n=== 重复值 ===")
print("重复行数:", df.duplicated().sum())        # 标记重复行
print("\n重复的行:\n", df[df.duplicated(keep=False)])

# 删除重复行（keep="first" 保留第一次出现的）
df_clean = df_fill.drop_duplicates()
print(f"\n去重后: {len(df_fill)} → {len(df_clean)} 行")


# ============================================================
# 三、分组聚合 groupby（数据分析的灵魂！）
# ============================================================
# SQL: SELECT 城市, AVG(分数) FROM df GROUP BY 城市
# Pandas: df.groupby("城市")["分数"].mean()

print("\n=== 分组聚合 ===")
# 单列分组 + 单列聚合
city_mean = df_clean.groupby("城市")["分数"].mean()
print("各城市平均分:\n", city_mean.sort_values(ascending=False))

# 多种聚合一起算：agg
stats = df_clean.groupby("城市")["分数"].agg(["count", "mean", "min", "max"])
print("\n各城市分数统计:\n", stats)

# 按多列分组
gender_city = df_clean.groupby(["性别", "城市"])["分数"].mean()
print("\n性别×城市的平均分:\n", gender_city)

# 【ML 场景】groupby 常用于：
#   检查不同类别的样本是否均衡
#   生成"群体均值"作为新特征（特征工程）


# ============================================================
# 四、合并多张表 merge / concat
# ============================================================
# merge  ：按某列"对齐"两张表（类似 SQL JOIN）
# concat ：直接拼接（行变多或列变多）

# 4.1 准备第二张表（学生附加信息）
extra = pd.DataFrame({
    "学号":   [1001, 1002, 1003, 1004],
    "班级":   ["A班", "A班", "B班", "B班"],
    "电话":   ["138xxx", "139xxx", "137xxx", "136xxx"],
})

# merge：按"学号"对齐（how="left" 保留左表所有行）
merged = df_clean.merge(extra, on="学号", how="left")
print("\n=== merge 按学号对齐 ===")
print(merged[["学号", "姓名", "班级", "分数"]])

# how 参数：
#   how="inner" 交集（两表都有的学号）—— 默认
#   how="left"  以左表为主
#   how="right" 以右表为主
#   how="outer" 并集（全部保留，缺失补 NaN）

# 4.2 concat 纵向拼接（加样本）
more = pd.DataFrame({
    "学号":   [1006, 1007],
    "姓名":   ["新1", "新2"],
    "性别":   ["男", "女"],
    "年龄":   [20, 21],
    "分数":   [88, 76],
    "城市":   ["成都", "重庆"],
})
combined = pd.concat([df_clean, more], ignore_index=True)
print(f"\nconcat 加样本后: {len(df_clean)} → {len(combined)} 行")


# ============================================================
# 五、透视表 pivot_table（Excel 神器在 Python 版）
# ============================================================
# 把"长表"转成"宽表"，方便交叉观察

print("\n=== 透视表 ===")
pivot = pd.pivot_table(
    df_clean,
    values="分数",          # 要聚合的值
    index="城市",           # 行
    columns="性别",         # 列
    aggfunc="mean",         # 聚合方式
)
print("城市×性别 的平均分透视表:\n", pivot)


# ============================================================
# 六、ML 实战：完整的数据预处理流程
# ============================================================
# 模拟：从原始数据到可直接喂模型的 X, y

print("\n=== ML 数据预处理流水线 ===")
print(f"原始数据: {len(df)} 行, {len(df.columns)} 列")

# Step 1: 去重
d = df.drop_duplicates()
print(f"Step1 去重: {len(d)} 行")

# Step 2: 处理缺失（删除关键列缺失的行）
d = d.dropna(subset=["分数"])
print(f"Step2 去缺失: {len(d)} 行")

# Step 3: 类别编码（文字 → 数字）
d = d.copy()
d["性别编码"] = d["性别"].map({"男": 0, "女": 1})
d["城市编码"] = d["城市"].astype("category").cat.codes   # 自动编码

# Step 4: 切分 X, y
feature_cols = ["年龄", "性别编码", "城市编码"]
X = d[feature_cols]
y = d["分数"]
print(f"\n最终 X 形状: {X.shape}, y 形状: {y.shape}")
print("X 前 3 行:\n", X.head(3))
print("\ny 前 3 行:", y.head(3).tolist())
print("\n特征矩阵已准备就绪，可交给 Scikit-learn 训练模型！")


# ============================================================
# 小结：本课你应掌握的"ML 高频"操作
# ============================================================
# 1. isnull().sum() 查缺失；dropna 删；fillna 用均值/中位数填
# 2. duplicated() / drop_duplicates() 处理重复
# 3. groupby + agg 分组聚合（数据分析灵魂）
# 4. merge 按键对齐（JOIN）；concat 拼接
# 5. pivot_table 透视表
# 6. 完整预处理流水线：去重→去缺失→编码→切分 X/y
#
# === 模块 3 · Pandas 数据处理 结业 ===
# 你已具备真实数据的"读入-清洗-分析-导出"全流程能力。
# 下一阶段：模块 4 Matplotlib 可视化（让数据"看得见"）。
