# -*- coding: utf-8 -*-
"""
模块7 第3课：特征重要性与参数调优
==================================
本课目标：
  1. 三种特征重要性的区别与选择
  2. 用 GridSearchCV / RandomizedSearchCV 系统调参
  3. 交叉验证评估模型稳定性

【为什么需要调参】
  LightGBM 默认参数已经不错，但要拿到最优结果必须调参。
  调参核心是平衡「欠拟合 - 过拟合」: 模型太简单学不到，太复杂会死记硬背。
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     RandomizedSearchCV, cross_val_score)
from sklearn.metrics import accuracy_score

# 静音 sklearn 1.x 的「feature names 不一致」良性警告
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("_images", exist_ok=True)

print("=" * 60)
print("第3课：特征重要性与参数调优")
print("=" * 60)

# 加载数据
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  训练集 {X_tr.shape}  测试集 {X_te.shape}  特征数 {X.shape[1]}")


# ============================================================
# 一、三种特征重要性对比
# ============================================================
print("\n" + "#" * 60)
print("# 一、三种特征重要性")
print("#" * 60)

# 训练一个基线模型
base_model = lgb.LGBMClassifier(
    n_estimators=100, num_leaves=31, learning_rate=0.1,
    random_state=42, verbose=-1
)
base_model.fit(X_tr, y_tr)

# 【核心概念】LightGBM 的三种重要性(importance_type):
#   1) split:   该特征被用作分裂节点的次数(默认) -- 频次视角
#   2) gain:    该特征带来的总收益(分裂后损失下降量) -- 质量视角, 更可靠
#   两种视角可能差异很大: 一个高频但低效的特征, split 高但 gain 低
print("\n  对比三种 importance_type:")
importance_dict = {}
for imp_type in ["split", "gain"]:
    imp = base_model.booster_.feature_importance(importance_type=imp_type)
    importance_dict[imp_type] = imp
    top5 = np.argsort(imp)[::-1][:5]
    print(f"\n  [{imp_type}] 前5重要特征:")
    for idx in top5:
        print(f"    {feature_names[idx]:30s} = {imp[idx]:.1f}")

# 可视化对比
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, imp_type in zip(axes, ["split", "gain"]):
    imp = importance_dict[imp_type]
    order = np.argsort(imp)[::-1][:15]   # 只画前15个，否则太挤
    ax.barh(range(len(order)), imp[order][::-1], color="steelblue")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order][::-1], fontsize=8)
    ax.set_xlabel(f"importance ({imp_type})")
    label = "分裂次数" if imp_type == "split" else "总收益"
    ax.set_title(f"Top15 特征重要性 [{imp_type}]\n({imp_type}={label})")
plt.tight_layout()
plt.savefig("_images/lgb_03_importance_compare.png", dpi=100)
print("\n  [图] 已保存: _images/lgb_03_importance_compare.png")
plt.close()


# ============================================================
# 二、系统调参：GridSearchCV（网格搜索）
# ============================================================
print("\n" + "#" * 60)
print("# 二、GridSearchCV 网格搜索")
print("#" * 60)

# 【ML 场景】网格搜索 = 穷举所有参数组合
#   优点: 不会遗漏; 缺点: 组合多时非常慢
#   技巧: 先粗调(大范围) → 再细调(小范围)

# 粗调: 只调最关键的 2 个参数
print("\n  [粗调] 搜索 num_leaves 和 learning_rate...")
t0 = time.time()
param_grid = {
    "num_leaves": [15, 31, 63],          # 树的复杂度
    "learning_rate": [0.01, 0.05, 0.1],  # 学习率
}
grid_search = GridSearchCV(
    estimator=lgb.LGBMClassifier(
        n_estimators=100, random_state=42, verbose=-1
    ),
    param_grid=param_grid,
    cv=5,                 # 5折交叉验证
    scoring="accuracy",
    n_jobs=-1,            # 用满所有 CPU 核心
)
grid_search.fit(X_tr, y_tr)
t1 = time.time()
print(f"  耗时 {t1-t0:.1f}s")
print(f"  最佳参数: {grid_search.best_params_}")
print(f"  最佳CV准确率: {grid_search.best_score_:.4f}")

# 用最佳模型在测试集评估
best_clf = grid_search.best_estimator_
test_acc = accuracy_score(y_te, best_clf.predict(X_te))
print(f"  对应测试集准确率: {test_acc:.4f}")


# ============================================================
# 三、RandomizedSearchCV（随机搜索）—— 更高效的调参
# ============================================================
print("\n" + "#" * 60)
print("# 三、RandomizedSearchCV 随机搜索")
print("#" * 60)

# 【ML 场景】随机搜索 = 在参数空间里随机采样 N 组
#   优点: 比网格快, 且对"只有少数参数重要"的情况效果更好
#   论文实证: 同样时间下, RandomizedSearch 通常不比 Grid 差
from scipy.stats import randint, uniform      # 用于定义参数分布

print("\n  [随机搜索] 采样 20 组参数...")
t0 = time.time()
param_dist = {
    "num_leaves":       randint(10, 100),
    "learning_rate":    uniform(0.01, 0.3),     # 均匀分布 U(0.01, 0.31)
    "max_depth":        randint(-1, 15),        # -1 表示不限制
    "min_child_samples": randint(5, 50),
    "feature_fraction": uniform(0.5, 0.5),
}
random_search = RandomizedSearchCV(
    estimator=lgb.LGBMClassifier(
        n_estimators=100, random_state=42, verbose=-1
    ),
    param_distributions=param_dist,
    n_iter=20,          # 只采样 20 组(组合爆炸时远少于网格)
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    random_state=42,
)
random_search.fit(X_tr, y_tr)
t1 = time.time()
print(f"  耗时 {t1-t0:.1f}s")
print(f"  最佳参数: {random_search.best_params_}")
print(f"  最佳CV准确率: {random_search.best_score_:.4f}")


# ============================================================
# 四、交叉验证评估稳定性
# ============================================================
print("\n" + "#" * 60)
print("# 四、交叉验证评估模型稳定性")
print("#" * 60)

# 【ML 场景】只看一次 train/test 切分会有运气成分
#   交叉验证(CV) 把数据切 N 份, 轮流做训练/测试, 得到 N 个分数
#   看均值和标准差: 均值高+标准差小 = 模型稳定可靠
print("\n  用最佳参数做 10 折交叉验证:")
final_model = lgb.LGBMClassifier(
    n_estimators=100, random_state=42, verbose=-1,
    **random_search.best_params_
)
scores = cross_val_score(final_model, X, y, cv=10, scoring="accuracy")
print(f"  各折准确率: {[f'{s:.3f}' for s in scores]}")
print(f"  均值 = {scores.mean():.4f}  标准差 = {scores.std():.4f}")
print("  (标准差小 → 模型对数据切分不敏感 → 稳定)")

# 与默认参数对比
default_scores = cross_val_score(
    lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    X, y, cv=10, scoring="accuracy"
)
print(f"\n  默认参数: 均值 = {default_scores.mean():.4f}  标准差 = {default_scores.std():.4f}")
print(f"  调参后:   均值 = {scores.mean():.4f}  标准差 = {scores.std():.4f}")
print(f"  提升: {scores.mean() - default_scores.mean():+.4f}")


# ============================================================
# 五、调参策略经验总结
# ============================================================
print("\n" + "#" * 60)
print("# 五、LightGBM 调参策略（经验）")
print("#" * 60)
print("""
  推荐调参顺序(影响从大到小):
  ┌──────────────────────────────────────────────────────┐
  │ 1. num_leaves + max_depth      ← 树的复杂度, 最关键   │
  │ 2. min_child_samples           ← 防叶子过拟合         │
  │ 3. feature_fraction + bagging  ← 随机性, 防过拟合     │
  │ 4. reg_alpha + reg_lambda      ← L1/L2 正则化         │
  │ 5. learning_rate + n_estimators← 最后微调(小步长+多树)│
  └──────────────────────────────────────────────────────┘

  实战流程:
    (a) 先用默认参数 + early_stopping 跑出 baseline
    (b) 用 GridSearchCV/RandomizedSearchCV 调 1~3
    (c) 固定树相关参数, 调 4 正则化
    (d) 最后把 learning_rate 减小, n_estimators 增大, 跑最终模型
""")


print("=" * 60)
print("第3课小结")
print("=" * 60)
print("""
  [OK] 三种重要性: split(频次) / gain(质量) -- 推荐 gain
  [OK] GridSearchCV: 穷举, 慢但全; RandomizedSearchCV: 采样, 快且够用
  [OK] cross_val_score: 用 CV 均值+标准差评估模型稳定性
  [OK] 调参顺序: 树复杂度 > 随机性 > 正则化 > 学习率(最后微调)

  下一课: Kaggle 风格完整项目实战(收官)。
""")


# ============================================================
# 练习（可选）
# ============================================================
# 1. 把 GridSearchCV 的 param_grid 扩展到 4 个参数, 观察耗时增长(组合爆炸)
# 2. 尝试只调 learning_rate 一个参数, 画出 learning_rate vs CV准确率 曲线
# 3. 对比: 调参前后, gain 类前 5 重要特征是否发生变化
