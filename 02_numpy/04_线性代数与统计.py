# -*- coding: utf-8 -*-
"""
模块2 · 第4课：线性代数与统计（NumPy 实战收官）
==============================================
本课把线性代数和统计的常用操作收口，并用 NumPy 手动实现两个 ML 基础算法：
标准化（Standardization）和 KNN 单步预测。学完即具备进入 Scikit-learn 的数学基础。
"""
import numpy as np


# ============================================================
# 一、矩阵乘法（ML 里出现频率最高！）
# ============================================================
# 三种写法等价：np.matmul(A, B)  /  A @ B  /  np.dot(A, B)
# 注意：A * B 是逐元素乘（不是矩阵乘法！），别混淆

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print("逐元素乘 A*B:\n", A * B)         # [[5 12],[21 32]]
print("矩阵乘 A@B:\n", A @ B)           # [[19 22],[43 50]]

# 矩阵 × 向量（线性变换）
W = np.array([[1, 2, 3],
              [4, 5, 6]])               # shape (2,3)  权重矩阵
x = np.array([1, 0, -1])                # shape (3,)   输入特征
y = W @ x                               # shape (2,)   输出
print("\n线性变换 W@x =", y)

# 【ML 场景】神经网络的前向传播：y = W @ x + b
# 线性回归：y = X @ w + b   （X 样本矩阵，w 权重向量）
# 这是几乎所有 ML 算法的核心运算


# ============================================================
# 二、常用线性代数操作
# ============================================================

# 转置
M = np.array([[1, 2, 3],
              [4, 5, 6]])               # (2,3)
print("\n转置 M.T 形状:", M.T.shape)    # (3,2)

# 单位矩阵、对角矩阵
I = np.eye(3)
D = np.diag([1, 2, 3])
print("对角阵:\n", D)

# 行列式与逆（方阵）
S = np.array([[4, 7],
              [2, 6]])
print("行列式:", np.linalg.det(S))           # 10.0
print("逆矩阵:\n", np.linalg.inv(S))
print("验证 S @ inv(S) = I:\n", S @ np.linalg.inv(S))

# 解线性方程组 Ax = b
A = np.array([[3, 1],
              [1, 2]])
b = np.array([9, 8])
x = np.linalg.solve(A, b)                   # 解出 x
print("\n方程组解:", x)                       # [2. 3.]
print("验证 A@x:", A @ x)                     # [9. 8.]


# ============================================================
# 三、统计函数（理解数据分布的利器）
# ============================================================
np.random.seed(42)
data = np.random.randn(1000, 3)             # 1000 样本 × 3 特征（标准正态）

print("\n均值(每列):", data.mean(axis=0))    # 每个特征的均值 ≈ 0
print("标准差(每列):", data.std(axis=0))     # ≈ 1
print("最大值(每列):", data.max(axis=0))
print("最小值(每列):", data.min(axis=0))
print("中位数(每列):", np.median(data, axis=0))

# 分位数（了解数据分布，常用于找异常值）
print("95 分位(每列):", np.percentile(data, 95, axis=0))

# 协方差矩阵（特征之间的关系）
cov = np.cov(data, rowvar=False)            # rowvar=False: 每列是一个特征
print("协方差矩阵形状:", cov.shape)          # (3,3)
# 相关系数矩阵
corr = np.corrcoef(data, rowvar=False)
print("相关系数矩阵:\n", np.round(corr, 3))


# ============================================================
# 四、拼接与堆叠（合并数据集）
# ============================================================
a = np.array([[1, 2],
              [3, 4]])
b = np.array([[5, 6],
              [7, 8]])

# vstack 竖直堆叠（行变多，相当于"加样本"）
v = np.vstack([a, b])
print("\nvstack（加样本）:\n", v)            # shape (4,2)

# hstack 水平堆叠（列变多，相当于"加特征"）
h = np.hstack([a, b])
print("hstack（加特征）:\n", h)              # shape (2,4)

# concatenate 更通用，用 axis 指定方向
c = np.concatenate([a, b], axis=0)          # 等价 vstack
print("concatenate axis=0:\n", c)

# 【ML 场景】
#   加样本：np.vstack([X_train, X_new])      数据集越来越大
#   加特征：np.hstack([X, X_extra])          把新特征拼到原特征后


# ============================================================
# 五、ML 实战 1：数据标准化（Z-score Normalization）
# ============================================================
# 公式：x' = (x - μ) / σ    μ=均值，σ=标准差
# 标准化后数据均值 0、标准差 1，是很多 ML 算法（SVM、KNN、神经网络）的预处理标配。

np.random.seed(0)
X = np.random.randint(0, 100, (5, 3)).astype(np.float64)
print("\n=== 标准化实战 ===")
print("原始数据:\n", X)

mu = X.mean(axis=0)                         # 每个特征的均值，shape (3,)
sigma = X.std(axis=0)                       # 每个特征的标准差
X_norm = (X - mu) / sigma                   # 广播！每行都减去同一个 mu
print("标准化后:\n", np.round(X_norm, 3))
print("验证均值≈0:", np.round(X_norm.mean(axis=0), 10))
print("验证标准差≈1:", np.round(X_norm.std(axis=0), 10))


# ============================================================
# 六、ML 实战 2：手写 KNN 单步预测（欧氏距离）
# ============================================================
# KNN 思想：找离新样本最近的 k 个训练样本，投票决定类别。
# 这里演示"计算新样本到所有训练样本的距离"这一核心步骤。

X_train = np.array([[1.0, 2.0],
                    [1.5, 1.8],
                    [5.0, 8.0],
                    [6.0, 8.0],
                    [1.0, 0.6],
                    [9.0, 11.0]])
y_train = np.array(['A', 'A', 'B', 'B', 'A', 'B'])   # 标签

x_new = np.array([2.0, 2.0])                # 待预测的新样本

# 方法 1：用循环（易懂但慢）
dist_loop = np.array([np.sqrt(np.sum((x_new - x) ** 2)) for x in X_train])

# 方法 2：向量化（推荐，快得多）
# 利用广播：(6,2) - (2,) → (6,2)，每行减同一个 x_new
diff = X_train - x_new                      # shape (6,2)
dist_vec = np.sqrt((diff ** 2).sum(axis=1)) # shape (6,) 每个样本的距离

print("\n=== KNN 距离计算 ===")
print("循环法:", np.round(dist_loop, 3))
print("向量化:", np.round(dist_vec, 3))

# 找最近的 3 个，投票决定类别
k = 3
nearest_idx = np.argsort(dist_vec)[:k]      # 距离从小到大排序，取前 k 个索引
nearest_labels = y_train[nearest_idx]
print(f"新样本 {x_new} 最近 {k} 个邻居:", nearest_labels)

# 投票（统计每个类别出现次数）
from collections import Counter
vote = Counter(nearest_labels)
pred = vote.most_common(1)[0][0]
print(f"预测类别: {pred}")


# ============================================================
# 小结：本课你应掌握的"ML 高频"操作
# ============================================================
# 1. 矩阵乘 @ / matmul / dot；区分 @ 和 *
# 2. 转置 .T、逆 linalg.inv、解方程 linalg.solve
# 3. 统计：mean/std/median/percentile，配合 axis
# 4. vstack/hstack 拼接数据（加样本 vs 加特征）
# 5. 实战：标准化 (X-μ)/σ、KNN 欧氏距离的向量化实现
#
# === 模块 2 · NumPy 数值计算 结业 ===
# 你已掌握 ML 所需的全部 NumPy 基础。
# 下一阶段进入模块 3：Pandas 数据处理——更贴近真实数据清洗。
