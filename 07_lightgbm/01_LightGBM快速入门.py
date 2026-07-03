# -*- coding: utf-8 -*-
"""
模块7 第1课：LightGBM 快速入门
================================
本课目标：
  1. 理解 LightGBM 是什么、为什么它又快又准
  2. 掌握「原生 API」(lgb.Dataset + lgb.train) 的最小工作流
  3. 了解 LightGBM 的参数体系与模型保存/加载

LightGBM = Light Gradient Boosting Machine
它是微软开源的「梯度提升决策树(GBDT)」实现，和 XGBoost 同类，
但在大规模数据上更快、内存更省，是 Kaggle 表格数据的常胜选手。

【为什么快】（先混个眼熟，细节后面课程会用到）
  - 直方图算法：把连续特征离散化成桶，分裂只需遍历桶，而非每个值
  - Leaf-wise 生长：每次选当前收益最大的叶子继续分裂（vs 别人的层增长）
  - 原生支持类别特征：不用手动 one-hot，效率更高
"""

import os
import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("第1课：LightGBM 快速入门")
print("=" * 60)

# ------------------------------------------------------------
# 1. 准备数据（沿用前面的乳腺癌数据集）
# ------------------------------------------------------------
print("\n【1】加载数据")
data = load_breast_cancer()
X, y = data.data, data.target          # (569, 30) 特征矩阵, (569,) 标签
print(f"  样本数={X.shape[0]}  特征数={X.shape[1]}")
print(f"  类别: {data.target_names.tolist()}")   # ['malignant' 恶性, 'benign' 良性]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------
# 2. 构造 LightGBM 专属数据结构：lgb.Dataset
# ------------------------------------------------------------
# 【核心概念】lgb.Dataset 是 LightGBM 的高效数据容器
#   - 内部用直方图格式存储，训练时更快、更省内存
#   - 原生 API 必须用它；sklearn 接口可以直接传 ndarray
print("\n【2】构造 lgb.Dataset")
train_data = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
valid_data = lgb.Dataset(X_te, label=y_te, reference=train_data, free_raw_data=False)
print(f"  train_data: {len(X_tr)} 行  valid_data: {len(X_te)} 行")

# ------------------------------------------------------------
# 3. 设置参数（参数体系）
# ------------------------------------------------------------
# 【ML 场景】LightGBM 参数分三大类，理解分类就不会乱调：
#   (1) 核心任务参数(objective/metric):  解决什么问题
#   (2) 学习控制参数(num_leaves/learning_rate/max_depth): 树的形状与学习速度
#   (3) IO/效率参数(num_threads等): 性能相关
params = {
    # —— 任务参数 ——
    "objective": "binary",         # 二分类 (回归用 regression, 多分类用 multiclass)
    "metric": "binary_logloss",    # 评估指标: 二分类对数损失
    # —— 学习控制参数(后面调参课会重点讲) ——
    "num_leaves": 31,              # 一棵树的最大叶子数(LightGBM 最关键参数!)
    "learning_rate": 0.1,          # 学习率(每棵树贡献的权重)
    "feature_fraction": 0.9,       # 每棵树随机选 90% 的特征(防过拟合)
    "verbose": -1,                 # 关闭训练过程中的日志刷屏
}
print("\n【3】参数:", params)

# ------------------------------------------------------------
# 4. 训练：lgb.train
# ------------------------------------------------------------
# 【核心概念】num_boost_round = 树的棵数(迭代次数)
#   GBDT 是「串行」加树: 每棵树修正前面所有树的残差
#   early_stopping: 在验证集上连续 N 轮没提升就提前停止，防过拟合
print("\n【4】训练模型")
model = lgb.train(
    params,
    train_data,
    num_boost_round=200,                    # 最多训练 200 棵树
    valid_sets=[train_data, valid_data],    # 观察训练集 + 验证集表现
    valid_names=["train", "valid"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=20),   # 连续 20 轮无提升则停止
        lgb.log_evaluation(period=50),             # 每 50 轮打印一次
    ],
)
print(f"  最佳迭代轮数 best_iteration = {model.best_iteration}")

# ------------------------------------------------------------
# 5. 预测与评估
# ------------------------------------------------------------
print("\n【5】预测与评估")
# 【注意】原生 API 的 predict 返回的是「概率」(objective=binary 时)
y_prob = model.predict(X_te, num_iteration=model.best_iteration)
y_pred = (y_prob > 0.5).astype(int)         # 概率 > 0.5 判为正类
acc = accuracy_score(y_te, y_pred)
print(f"  测试集准确率 = {acc:.4f}")

# ------------------------------------------------------------
# 6. 模型保存与加载
# ------------------------------------------------------------
# 【ML 场景】训练好的模型要持久化，才能上线服务/给别人用
print("\n【6】模型保存与加载")
model_path = "model_breast_cancer.txt"
model.save_model(model_path)
print(f"  已保存到 {model_path}  (文件大小: {os.path.getsize(model_path)} 字节)")

# 重新加载并验证一致性
model2 = lgb.Booster(model_file=model_path)
acc2 = accuracy_score(y_te, (model2.predict(X_te) > 0.5).astype(int))
print(f"  重新加载后准确率 = {acc2:.4f}  (应与上面一致)")
print(f"  一致性检查: {'通过' if acc == acc2 else '失败'}")

# 清理临时模型文件
os.remove(model_path)

# ------------------------------------------------------------
# 7. 关键参数速查表（打印出来当备忘）
# ------------------------------------------------------------
print("\n【7】关键参数速查")
print("-" * 60)
print(f"{'参数':25s} {'作用':20s} {'调参方向'}")
print("-" * 60)
cheatsheet = [
    ("num_leaves",            "树的复杂度",        "越大越易过拟合, 一般 < 2^max_depth"),
    ("learning_rate",         "每棵树的步长",      "0.01~0.3, 小则需更多树"),
    ("num_boost_round",       "树的数量",          "配合 early_stopping 自动定"),
    ("max_depth",             "树的最大深度",      "限制深度防过拟合"),
    ("min_child_samples",     "叶子最小样本数",    "增大可防过拟合"),
    ("feature_fraction",      "特征采样比例",      "0.5~1.0, 防过拟合"),
    ("bagging_fraction",      "样本采样比例",      "需配合 bagging_freq"),
    ("reg_alpha / reg_lambda", "L1/L2 正则化",     "增大抑制过拟合"),
]
for name, effect, tip in cheatsheet:
    print(f"  {name:23s} {effect:20s} {tip}")
print("-" * 60)

# 清理临时文件
if os.path.exists("model_breast_cancer.txt"):
    os.remove("model_breast_cancer.txt")

print("\n" + "=" * 60)
print("第1课小结")
print("=" * 60)
print("""
  [OK] LightGBM 是高效的梯度提升树实现，表格数据之王
  [OK] 原生 API 三步走: lgb.Dataset -> 设 params -> lgb.train
  [OK] 参数分三类: 任务参数 / 学习控制 / IO效率
  [OK] early_stopping 让模型自动决定树的数量，是防过拟合标配
  [OK] 模型用 save_model/load_model 保存加载(文本格式, 体积小)

  下一课: 用 sklearn 风格接口做分类与回归实战。
""")


# ============================================================
# 练习（可选）
# ============================================================
# 1. 把 learning_rate 改成 0.01，观察 best_iteration 和准确率的变化
#    (学习率变小 → 需要更多树 → 但可能更准)
# 2. 把 num_leaves 从 31 改成 127，观察训练集和验证集表现
#    (叶子变多 → 模型更复杂 → 容易过拟合训练集)
# 3. 尝试 objective='regression'，把标签改成连续值(如特征均值)，跑回归
