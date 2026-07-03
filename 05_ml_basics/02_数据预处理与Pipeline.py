# -*- coding: utf-8 -*-
"""
模块5 · 第2课：数据预处理与 Pipeline
====================================
"垃圾进，垃圾出"——模型效果好不好，数据预处理占 60% 的工作量。
本课聚焦 sklearn 的预处理工具：标准化、编码、缺失值填充、Pipeline。
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


# ============================================================
# 一、为什么需要预处理？
# ============================================================
# (1) 量纲不一致：身高 170、体重 65、收入 50000。KNN 算距离时收入会主导
#     → 需要"标准化"把各特征拉到同一量级
# (2) 模型只认数字：性别"男/女"、城市"北京"模型看不懂
#     → 需要"编码"把文字转数字
# (3) 缺失值：模型无法处理 NaN
#     → 需要"填充"


# ============================================================
# 二、标准化 StandardScaler（最常用的预处理）
# ============================================================
# 公式：x' = (x - μ) / σ   转换后均值0、标准差1
# 适合：大多数 ML 算法（KNN、SVM、逻辑回归、神经网络）

np.random.seed(42)
X = np.array([[170, 65, 50000],
              [160, 55, 30000],
              [180, 80, 80000]], dtype=float)

print("=== 标准化 ===")
print("原始（量纲差异巨大）:\n", X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)        # fit 计算均值方差，transform 转换
print("\n标准化后:\n", np.round(X_scaled, 3))
print("均值(每列≈0):", np.round(X_scaled.mean(axis=0), 10))
print("标准差(每列≈1):", np.round(X_scaled.std(axis=0), 10))

# 【关键】fit 只能在训练集做！测试集用 transform
# 防止"数据泄漏"：模型不该知道测试集的统计信息
X_train, X_test = train_test_split(X, test_size=0.33, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # 训练集 fit + transform
X_test_s = scaler.transform(X_test)         # 测试集只 transform！用训练集的均值方差
print("\n测试集用训练集参数转换:\n", np.round(X_test_s, 3))


# ============================================================
# 三、归一化 MinMaxScaler（缩放到 [0,1]）
# ============================================================
# 公式：x' = (x - min) / (max - min)
# 适合：图像像素（0~255→0~1）、不假设正态分布的算法
mm = MinMaxScaler()
X_mm = mm.fit_transform(X)
print("\n归一化到 [0,1]:\n", np.round(X_mm, 3))


# ============================================================
# 四、类别编码
# ============================================================
# 4.1 LabelEncoder：类别 → 整数（用于"标签"y）
le = LabelEncoder()
y_text = ["猫", "狗", "猫", "鸟", "狗"]
y_num = le.fit_transform(y_text)
print("\n=== 编码 ===")
print("LabelEncoder:", y_text, "->", y_num)
print("反查:", le.inverse_transform([0, 1, 2]))   # 数字还原成文字

# 4.2 OrdinalEncoder：特征矩阵 X 的类别 → 整数（多列）
oe = OneHotEncoder(sparse_output=False)    # sparse_output=False 返回普通数组
cities = np.array([["北京"], ["上海"], ["北京"], ["广州"]])
onehot = oe.fit_transform(cities)
print("\nOneHot 编码:\n", onehot)
print("类别:", oe.categories_)
# OneHot 适合"无大小关系"的类别（如城市、颜色），避免模型误以为 0<1<2


# ============================================================
# 五、缺失值填充 SimpleImputer
# ============================================================
X_missing = np.array([[1, 2, np.nan],
                      [4, np.nan, 6],
                      [7, 8, 9]], dtype=float)
print("\n=== 缺失值填充 ===")
print("含缺失:\n", X_missing)

# 用列均值填充（默认）
imp_mean = SimpleImputer(strategy="mean")     # mean/median/most_frequent/constant
X_filled = imp_mean.fit_transform(X_missing)
print("均值填充:\n", X_filled)


# ============================================================
# 六、Pipeline 管道 —— 把多步预处理 + 模型串成一条
# ============================================================
# 痛点：标准化→编码->模型 每步都要分别 fit/transform，测试集要重复一遍
# 解决：Pipeline 把多步封装成一个对象，用一套 fit/predict

from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# 创建管道：标准化 + KNN
pipe = Pipeline([
    ("scaler", StandardScaler()),              # 第 1 步：标准化
    ("knn", KNeighborsClassifier(n_neighbors=5)) # 第 2 步：KNN
])

# 用起来就像普通模型——一个 fit 搞定所有预处理 + 训练
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
acc = pipe.score(X_test, y_test)
print(f"\n=== Pipeline ===\n测试集准确率: {acc:.4f}")

# 好处：
# 1. 代码简洁，避免漏掉测试集的某步预处理
# 2. 防数据泄漏：fit 只在训练集，transform 自动应用到测试集
# 3. 可直接用于交叉验证、网格搜索（后续会学）


# ============================================================
# 七、预处理前后效果对比（用 KNN 演示）
# ============================================================
# 制造量纲差异巨大的数据：第 0 列 0~1，第 1 列 0~1000
np.random.seed(42)
n = 100
X1 = np.random.rand(n)                    # 0~1
X2 = np.random.rand(n) * 1000             # 0~1000
y_bin = (X1 + X2/1000 > 1).astype(int)    # 人造标签
X_raw = np.column_stack([X1, X2])

X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y_bin, test_size=0.3, random_state=42)

# 不标准化
m1 = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)
print(f"\n不标准化准确率: {m1.score(X_te, y_te):.4f}")

# 标准化（用 Pipeline）
m2 = Pipeline([("s", StandardScaler()), ("k", KNeighborsClassifier(5))]).fit(X_tr, y_tr)
print(f"标准化后准确率:  {m2.score(X_te, y_te):.4f}")
print("（差距来自第2列数值过大压制了第1列的信息）")


# ============================================================
# 小结：本课你应掌握的"ML 工程"能力
# ============================================================
# 1. StandardScaler 标准化（均值0标准差1），fit 只在训练集！
# 2. MinMaxScaler 归一化到 [0,1]
# 3. LabelEncoder 编码标签；OneHotEncoder 编码无序类别特征
# 4. SimpleImputer 填充缺失值（mean/median/constant）
# 5. Pipeline 串联预处理+模型，防泄漏、代码简洁
#
# 下一课：进入分类算法实战（逻辑回归/KNN/决策树/随机森林）。
