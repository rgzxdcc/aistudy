# -*- coding: utf-8 -*-
"""
模块2 · 第1课：ndarray 基础
==========================
NumPy 是 ML 的底层计算引擎：所有数据（图像、文本特征、表格）最终都会变成
"多维数组"喂给模型。本课聚焦：如何创建数组、数组的关键属性、数据类型。

约定俗成：import numpy as np   ← 全世界 ML 代码都这么写，请记住。
"""
import numpy as np


# ============================================================
# 一、从列表创建 ndarray —— 最直观的入门方式
# ============================================================
# np.array() 把 Python 列表转成 NumPy 数组（ndarray = N-dimensional array）

arr1 = np.array([1, 2, 3, 4, 5])          # 一维：从列表创建
print("一维数组:", arr1)
print("类型:", type(arr1))                 # <class 'numpy.ndarray'>

# 二维：从"列表的列表"创建（相当于矩阵）
arr2 = np.array([[1, 2, 3],
                 [4, 5, 6]])
print("二维数组:\n", arr2)

# 【ML 场景】一条样本 = 一维数组；一个数据集 = 二维数组（样本×特征）
sample = np.array([1.7, 65.0, 30])         # 一个人的 [身高, 体重, 年龄]
dataset = np.array([[1.7, 65, 30],         # 多个人的数据
                    [1.6, 55, 25],
                    [1.8, 80, 35]])
print("数据集形状:", dataset.shape)         # (3, 3) = 3样本 × 3特征


# ============================================================
# 二、ndarray 的关键属性（必须牢记！）
# ============================================================
a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8]])

print("shape  形状:", a.shape)     # (2, 4)  ← 最常用，表示 2行×4列
print("ndim   维度:", a.ndim)      # 2       ← 几维数组
print("size   元素总数:", a.size)  # 8       ← shape 各维度乘积
print("dtype  数据类型:", a.dtype) # int64
print("itemsize 每元素字节数:", a.itemsize)  # 8（int64=8字节）

# 形状(shape) 是 ML 中最频繁打交道的概念：
#   (样本数,)         一维标签，如 1000 个类别
#   (样本数, 特征数)   二维特征矩阵 X
#   (样本数, 高, 宽)   灰度图像
#   (样本数, 高, 宽, 3) 彩色图像（3 通道 RGB）


# ============================================================
# 三、快速创建数组的内置函数（ML 中极常用）
# ============================================================

# 3.1 全 0 / 全 1 数组（初始化权重、占位）
zeros = np.zeros(5)                      # 一维全 0
zeros2d = np.zeros((2, 3))               # 二维全 0，参数是"形状元组"
ones2d = np.ones((3, 3))                 # 二维全 1
print("全0:\n", zeros2d)
print("全1:\n", ones2d)

# 3.2 等差数列
r1 = np.arange(0, 10, 2)                 # [0,2,4,6,8]  起/止/步长（不含止）
r2 = np.linspace(0, 1, 5)                # [0,0.25,0.5,0.75,1]  起/止/个数（含止）
print("arange:", r1)
print("linspace:", r2)
# 区别：arange 指定"步长"，linspace 指定"数量"

# 3.3 单位矩阵（线性代数基础）
I = np.eye(3)                            # 3×3 单位阵，对角线为1
print("单位阵:\n", I)

# 3.4 随机数（ML 最常用，后面会单独讲）
np.random.seed(42)                       # 设随机种子，保证结果可复现
rand_arr = np.random.rand(2, 3)          # [0,1) 均匀分布，形状 (2,3)
randn_arr = np.random.randn(2, 3)        # 标准正态分布（均值0方差1）
rand_int = np.random.randint(0, 10, 5)   # [0,10) 的随机整数 5 个
print("均匀分布随机:\n", rand_arr)
print("正态分布随机:\n", randn_arr)
print("随机整数:", rand_int)


# ============================================================
# 四、数据类型 dtype —— ML 性能与精度的关键
# ============================================================
# NumPy 数组所有元素必须是同一类型（这点和 Python 列表不同！）
# 这让 NumPy 比列表快几十倍，因为内存连续、可向量化。

# 常见 dtype：
#   整数  int8 / int16 / int32 / int64（默认）
#   无符号 uint8（图像像素 0~255 专用！）
#   浮点  float16 / float32 / float64（默认）
#   布尔  bool

img_pixels = np.array([0, 128, 255], dtype=np.uint8)     # 图像像素
weights = np.array([0.1, -0.2, 0.05], dtype=np.float32)  # 神经网络权重
print("uint8 像素:", img_pixels, img_pixels.dtype)
print("float32 权重:", weights, weights.dtype)

# 类型转换：astype()
float_arr = np.array([1.7, 2.3, 3.9])
int_arr = float_arr.astype(np.int64)      # 转整数（直接截断，不四舍五入！）
print("转整数:", int_arr)                 # [1 2 3]

# 【ML 场景】图像数据通常是 uint8，喂给模型前常转成 float32 并归一化到 [0,1]
pixels = np.array([0, 128, 255], dtype=np.uint8)
normalized = pixels.astype(np.float32) / 255.0
print("归一化像素:", normalized)          # [0.   0.502 1.  ]


# ============================================================
# 五、形状变换 reshape（ML 数据预处理核心操作）
# ============================================================
# reshape 不改变数据，只改变"看待数据的形状"
original = np.arange(12)                  # [0,1,...,11] 共12个元素
print("原始:", original, "shape:", original.shape)

# 改成 3行4列
matrix = original.reshape(3, 4)
print("reshape(3,4):\n", matrix)

# 用 -1 让 NumPy 自动推断该维度
auto = original.reshape(2, -1)            # 2行，列数自动算 = 6
print("reshape(2,-1):\n", auto)

# 【ML 场景】把 60000 张 28×28 图像展平成 60000×784 的矩阵喂给全连接层
# images.shape = (60000, 28, 28)
# flat = images.reshape(60000, -1)   # 自动变成 (60000, 784)

# 一维化：flatten() 或 reshape(-1)
flat = matrix.reshape(-1)                 # 重新变回一维
print("展平:", flat)

# 转置：T 属性（行变列、列变行）
print("转置:\n", matrix.T)                # 4行3列


# ============================================================
# 小结：本课你应掌握的"ML 高频"操作
# ============================================================
# 1. np.array() / np.zeros / np.ones / np.arange / np.linspace
# 2. 关键属性：shape（最重要！） / ndim / size / dtype
# 3. 随机数：np.random.seed 保证可复现；rand / randn / randint
# 4. dtype 概念：uint8 存图像、float32 存权重；astype 转换
# 5. reshape 改形状、-1 自动推断、flatten 展平、T 转置
#
# 练习（可选）：
#   1) 创建一个 5×5 的全 1 数组，把中间 3×3 改成 0
#   2) 用 arange 生成 0~99，reshape 成 10×10，再展平验证 size 不变
