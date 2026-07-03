# -*- coding: utf-8 -*-
"""
模块3 · 第2课：数据读写与查看
============================
真实 ML 项目的第一步：把外部数据（CSV/Excel）读进来 → 摸底 → 处理后写回。
本课聚焦 IO（输入输出）和初步查看技巧。
"""
import os
import numpy as np
import pandas as pd


# ============================================================
# 一、准备一份示例 CSV 数据（模拟真实场景）
# ============================================================
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "06_datasets")
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, "students.csv")

# 先生成一个带"瑕疵"的 CSV（缺失值、重复行）—— 后面清洗课会用到
raw = pd.DataFrame({
    "学号":   [1001, 1002, 1003, 1004, 1005, 1002],
    "姓名":   ["小明", "小红", "小刚", "小华", "小强", "小红"],
    "性别":   ["男", "女", "男", "女", "男", "女"],
    "年龄":   [20, 21, 19, 22, None, 21],
    "分数":   [85, 92, 78, None, 88, 92],
    "城市":   ["北京", "上海", "广州", "深圳", "杭州", "上海"],
})
raw.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
# encoding="utf-8-sig"：带 BOM 头，Excel 打开中文不乱码
# index=False：不把行索引(0,1,2,...)写进文件
print(f"示例数据已写入: {os.path.basename(CSV_PATH)}")


# ============================================================
# 二、读取 CSV（ML 项目最常见入口）
# ============================================================
# pd.read_csv() 是用得最多的函数，参数众多

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
print("\n=== 读取结果 ===")
print(df)

# 常用参数（按需使用）：
#   encoding="utf-8" / "gbk"   处理中文编码
#   index_col=0                 把第 0 列作为行索引
#   usecols=["姓名", "分数"]    只读需要的列（大数据集省内存）
#   nrows=1000                   只读前 N 行（先探查数据用）
#   na_values=["NA", "?"]        把这些符号当作缺失值
#   sep="," / "\t"               分隔符（TSV 用 \t）

# 只读指定列
df_sub = pd.read_csv(CSV_PATH, usecols=["姓名", "分数"])
print("\n只读两列:\n", df_sub)


# ============================================================
# 三、数据摸底的标准三件套
# ============================================================
# 拿到数据必做：head / info / describe

print("\n--- head(3) 前几行 ---")
print(df.head(3))

print("\n--- info() 结构信息（行数/列/类型/非空数） ---")
df.info()

print("\n--- describe() 数值列统计 ---")
print(df.describe())

# 非数值列的统计（include="object" 看字符串列；pandas 3.0 建议用 "str"）
print("\n--- describe(include='object') 文本列统计 ---")
print(df.describe(include="str"))   # 兼容旧写法；新版本可用 include="str"
# unique: 唯一值数；top: 出现最多的值；freq: 出现次数

# 每列缺失值数量（极其重要！）
print("\n--- 缺失值统计 ---")
print(df.isnull().sum())           # 每列有多少个 NaN


# ============================================================
# 四、查看分布与唯一值（理解数据特征）
# ============================================================
print("\n--- 性别分布 ---")
print(df["性别"].value_counts())    # 每个值出现次数

print("\n--- 唯一城市 ---")
print(df["城市"].unique())          # 唯一值列表
print("城市数量:", df["城市"].nunique())  # 唯一值个数


# ============================================================
# 五、写回文件（保存处理结果）
# ============================================================
OUT_PATH = os.path.join(DATA_DIR, "students_cleaned.csv")

# 假设我们做个简单处理：按分数降序后保存
result = df.sort_values("分数", ascending=False)
result.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print(f"\n结果已保存: {os.path.basename(OUT_PATH)}")

# 其他写法：
#   df.to_excel("xxx.xlsx", index=False)        Excel（需装 openpyxl）
#   df.to_json("xxx.json", orient="records", force_ascii=False)
#   df.to_csv(..., sep="\t")                    TSV


# ============================================================
# 六、实战：用真实经典数据集（鸢尾花 iris）
# ============================================================
# 演示从 URL 读 CSV（在线数据）——但通常我们离线用
# 这里用 NumPy 模拟生成一份"房价"数据演示
np.random.seed(42)
houses = pd.DataFrame({
    "面积": np.random.randint(50, 200, 100),
    "房间数": np.random.randint(1, 6, 100),
    "楼层": np.random.randint(1, 30, 100),
    "房龄": np.random.randint(0, 30, 100),
})
# 用公式生成价格（加噪声），让数据有规律
houses["价格"] = (
    houses["面积"] * 5
    + houses["房间数"] * 20
    - houses["房龄"] * 3
    + np.random.randn(100) * 30
).round(1)

HOUSE_PATH = os.path.join(DATA_DIR, "houses.csv")
houses.to_csv(HOUSE_PATH, index=False, encoding="utf-8-sig")

# 重新读入并摸底
h = pd.read_csv(HOUSE_PATH)
print(f"\n=== 房价数据集（{len(h)} 行）摸底 ===")
print(h.head())
print("\n相关系数矩阵（看特征与目标的相关性）:")
print(h.corr()["价格"].sort_values(ascending=False))
# corr() 接近 1 强正相关，-1 强负相关，0 无线性相关


# ============================================================
# 小结：本课你应掌握的"ML 工程"能力
# ============================================================
# 1. read_csv 常用参数（encoding/usecols/nrows/index_col）
# 2. 摸底三件套：head / info / describe
# 3. isnull().sum() 查缺失值；value_counts 看分布；unique 看唯一值
# 4. to_csv 保存结果（encoding="utf-8-sig" 让 Excel 不乱码）
# 5. corr() 相关系数 —— 选特征的第一步
