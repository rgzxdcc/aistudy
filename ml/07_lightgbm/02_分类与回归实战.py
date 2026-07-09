# -*- coding: utf-8 -*-
"""
模块7 第2课：分类与回归实战 (sklearn 接口)
==========================================
本课目标：
  1. 用 sklearn 风格 API (LGBMClassifier / LGBMRegressor) 做二分类、回归
  2. 掌握 early_stopping 在 sklearn 接口中的用法
  3. 与随机森林对比，理解 GBDT 家族的优势

【两种 API 怎么选】
  - 原生 API (lgb.train): 更灵活，适合高级用法、自定义损失、大规模数据
  - sklearn API (LGBM*): 与 Pipeline/GridSearchCV 无缝衔接，学习曲线低
  本课全部用 sklearn 接口(日常 80% 场景用它就够了)
"""

import os
import warnings
import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_squared_error, r2_score)

# 静音 sklearn 1.x 的「feature names 不一致」良性警告
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("第2课：分类与回归实战 (sklearn 接口)")
print("=" * 60)


# ============================================================
# 一、二分类实战：乳腺癌诊断
# ============================================================
print("\n" + "#" * 60)
print("# 一、二分类实战：乳腺癌诊断")
print("#" * 60)

# 1. 数据准备
data = load_breast_cancer()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  训练集样本数 = {X_tr.shape[0]}")
print(f"  测试集样本数 = {X_te.shape[0]}")

# 2. 创建分类器并训练
# 【ML 场景】LGBMClassifier 的参数名与原生 API 略有不同:
#   num_leaves / learning_rate  →  保持原名
#   num_boost_round(原生)       →  n_estimators(sklearn 风格)
#   feature_fraction            →  保持原名 (也可写 colsample_bytree)
print("\n【训练 LGBMClassifier】")
clf = lgb.LGBMClassifier(
    n_estimators=200,        # 最多 200 棵树
    num_leaves=31,
    learning_rate=0.1,
    feature_fraction=0.9,
    random_state=42,
    verbose=-1,
)

# 【关键】sklearn 接口的 early_stopping 用 callbacks 实现
#   fit 时传 eval_set + callbacks=[lgb.early_stopping(N)]
clf.fit(
    X_tr, y_tr,
    eval_set=[(X_te, y_te)],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)],
)
print(f"  best_iteration_ = {clf.best_iteration_}")

# 3. 评估
y_pred = clf.predict(X_te)
y_prob = clf.predict_proba(X_te)[:, 1]   # 正类概率
acc = accuracy_score(y_te, y_pred)
print(f"\n  准确率 = {acc:.4f}")
print("  分类报告:")
print(classification_report(y_te, y_pred, target_names=data.target_names))


# ============================================================
# 二、与随机森林对比
# ============================================================
print("#" * 60)
print("# 二、与随机森林对比 (同一个数据集)")
print("#" * 60)

# 【ML 场景】随机森林 vs LightGBM 的核心差异:
#   - 随机森林: Bagging, 树独立训练, 投票, 不易过拟合但偏差大
#   - LightGBM: Boosting, 树串行纠错, 精度高但需调参防过拟合
#   经验: 中小数据两者接近; 大数据 LightGBM 明显更优
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_tr, y_tr)
rf_acc = accuracy_score(y_te, rf.predict(X_te))

print(f"\n  随机森林准确率 = {rf_acc:.4f}")
print(f"  LightGBM 准确率 = {acc:.4f}")
print(f"  差距 = {abs(acc - rf_acc):+.4f}")
print("  (这个数据集较小，两者通常很接近；数据越大 LightGBM 优势越明显)")


# ============================================================
# 三、回归实战：合成数据
# ============================================================
print("\n" + "#" * 60)
print("# 三、回归实战")
print("#" * 60)

# 1. 生成回归数据
# 【ML 场景】make_regression 专门生成回归测试数据
#   n_informative=10: 真正有用的特征数(其余是噪声)
X, y = make_regression(
    n_samples=1000, n_features=20, n_informative=10,
    noise=20.0, random_state=42
)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n  训练集: {X_tr.shape}  测试集: {X_te.shape}")

# 2. 训练回归器
# 【注意】回归用 LGBMRegressor，objective 默认 regression(L2 均方误差)
reg = lgb.LGBMRegressor(
    n_estimators=300,
    num_leaves=31,
    learning_rate=0.05,
    feature_fraction=0.9,
    random_state=42,
    verbose=-1,
)
reg.fit(
    X_tr, y_tr,
    eval_set=[(X_te, y_te)],
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
)

# 3. 评估
# 【ML 场景】回归常用指标:
#   - MSE / RMSE: 预测值与真实值的均方误差(越小越好)
#   - R^2 (R方): 解释方差比例(越接近1越好, >0.8 算不错)
y_pred = reg.predict(X_te)
mse = mean_squared_error(y_te, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_te, y_pred)
print(f"\n  MSE  = {mse:.2f}")
print(f"  RMSE = {rmse:.2f}")
print(f"  R^2  = {r2:.4f}   (>0.8 视为良好)")
print(f"  best_iteration_ = {reg.best_iteration_}")


# ============================================================
# 四、类别特征与样本权重（实用技巧）
# ============================================================
print("\n" + "#" * 60)
print("# 四、实用技巧：样本不均衡处理")
print("#" * 60)

# 【ML 场景】类别不均衡(如欺诈检测, 正样本仅 5%)时:
#   - 可以用 sample_weight 给少数类更大权重
#   - LightGBM 还支持 is_unbalance=True 自动平衡
#   演示: 重新加载乳腺癌数据，手动制造不均衡
data_imb = load_breast_cancer()
Xi, yi = data_imb.data, data_imb.target.astype(int)
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(
    Xi, yi, test_size=0.2, random_state=42, stratify=yi
)
mask_pos = (yi_tr == 1)
# 模拟: 保留全部正类, 但只取 30% 负类 → 制造不均衡
idx_keep = np.where(mask_pos)[0].tolist()
idx_keep += np.where(~mask_pos)[0][: int((~mask_pos).sum() * 0.3)].tolist()
idx_keep = np.array(sorted(idx_keep))
y_imb = yi_tr[idx_keep].astype(int)
X_imb = Xi_tr[idx_keep]
counts = np.bincount(y_imb)
print(f"  不均衡后各类样本数: {counts.tolist()}")

# 给少数类(样本少的那类)更大权重
weight_ratio = counts.max() / counts.min()
sample_weight = np.where(y_imb == int(np.argmin(counts)), weight_ratio, 1.0)
print(f"  少数类权重放大 {weight_ratio:.1f} 倍")

clf_bal = lgb.LGBMClassifier(n_estimators=50, num_leaves=15,
                             random_state=42, verbose=-1)
clf_bal.fit(X_imb, y_imb, sample_weight=sample_weight)
print(f"  加权训练完成 (这只是演示样本权重的用法)")


print("\n" + "=" * 60)
print("第2课小结")
print("=" * 60)
print("""
  [OK] sklearn 接口: LGBMClassifier / LGBMRegressor, 用法和其他 sklearn 模型一致
  [OK] early_stopping 用法: fit(eval_set=..., callbacks=[lgb.early_stopping(N)])
  [OK] 分类用 predict/proba; 回归看 MSE/RMSE/R^2
  [OK] GBDT(Boosting) vs RF(Bagging): 大数据选 GBDT, 求稳选 RF
  [OK] 样本不均衡: 用 sample_weight 给少数类加权

  下一课: 特征重要性分析与参数调优。
""")


# ============================================================
# 练习（可选）
# ============================================================
# 1. 把 n_estimators 调大到 1000，learning_rate 调小到 0.01，
#    观察 best_iteration_ 和准确率的变化(小步长 + 多棵树通常更稳)
# 2. 对比: 同样参数下 LGBM 和 RandomForest 的训练耗时(可用 %%time 或 time)
# 3. 尝试让 num_leaves=127, max_depth=-1, 看看是否过拟合(训练集准 vs 测试集准)
