# -*- coding: utf-8 -*-
"""
模块5 · 第3课：分类算法与评估
============================
本课实战四大经典分类算法，并学会用指标客观评估"分类得好不好"。
- 逻辑回归 LogisticRegression（线性分类基线）
- KNN KNeighborsClassifier（基于距离）
- 决策树 DecisionTreeClassifier（基于规则）
- 随机森林 RandomForestClassifier（集成学习，常用强基线）
"""
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, precision_score, recall_score,
                             f1_score)


# ============================================================
# 一、四大分类算法简介
# ============================================================
#
# 逻辑回归：用直线/超平面把类别分开，输出概率（名字有"回归"实是分类！）
#   优点：快、可解释、输出概率
#   适合：线性可分问题、做基线
#
# KNN (K-Nearest Neighbors)：新样本找最近的 k 个邻居，投票决定类别
#   优点：简单、无需训练
#   缺点：需标准化、预测慢（每次都要算距离）
#
# 决策树：学一串 if-else 规则（如"花瓣长度<2.5→山鸢尾"）
#   优点：可解释性极强、不需标准化
#   缺点：容易过拟合
#
# 随机森林：训练很多棵决策树，投票决定（集成学习）
#   优点：准确、抗过拟合、常用作"强基线"
#   缺点：可解释性弱于单棵树


# ============================================================
# 二、准备数据（二分类：乳腺癌数据集，良/恶性）
# ============================================================
# 鸢尾花是 3 分类，这里换成更适合评估指标的"二分类"数据集
data = load_breast_cancer()
X, y = data.data, data.target
print("=== 乳腺癌数据集（良/恶性）===")
print(f"特征数: {X.shape[1]}, 样本数: {X.shape[0]}")
print(f"类别: {data.target_names}（0=恶性, 1=良性）")
print(f"类别分布: 恶性 {np.sum(y==0)} 例, 良性 {np.sum(y==1)} 例")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)


# ============================================================
# 三、训练四大模型，对比准确率
# ============================================================
# 用 Pipeline 把标准化 + 模型打包（KNN/逻辑回归需要标准化）

models = {
    "逻辑回归": Pipeline([("s", StandardScaler()),
                          ("m", LogisticRegression(max_iter=1000, random_state=42))]),
    "KNN(k=5)": Pipeline([("s", StandardScaler()),
                          ("m", KNeighborsClassifier(n_neighbors=5))]),
    "决策树":   DecisionTreeClassifier(max_depth=5, random_state=42),  # 树不需要标准化
    "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
}

print("\n=== 四模型准确率对比 ===")
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    results[name] = model
    print(f"  {name:6s}: {acc:.4f}")


# ============================================================
# 四、评估指标（核心！准确率不够用）
# ============================================================
# 准确率的陷阱：类别不均衡时（如 99% 是良性），全猜良性也有 99% 准确率！
# 需要更细的指标，对"二分类"尤其重要：
#
#   混淆矩阵 Confusion Matrix:
#                  预测正  预测负
#   实际正  →  TP(真阳)   FN(假阴)
#   实际负  →  FP(假阳)   TN(真阴)
#
#   精确率 Precision = TP/(TP+FP)：预测为正的，有多少真的是正（少误报）
#   召回率 Recall    = TP/(TP+FN)：真正的正例，有多少被找出来了（少漏报）
#   F1              = 精确率和召回率的调和平均
#
#   医疗场景：召回率更重要（宁可误诊也别漏诊，漏掉一个癌症病人代价大）
#   垃圾邮件：精确率更重要（宁可放过也别误杀正常邮件）

best = results["随机森林"]
y_pred = best.predict(X_test)

print("\n=== 随机森林详细评估 ===")
print(f"准确率 Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\n混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
# [[真恶性数  误判良性]
#  [误判恶性  真良性数]]

# 把"恶性(0)"当正例来看（pos_label=0），因为医疗上关心"查癌"
print(f"\n以'恶性'为正例：")
print(f"  精确率 Precision: {precision_score(y_test, y_pred, pos_label=0):.4f}")
print(f"  召回率 Recall:    {recall_score(y_test, y_pred, pos_label=0):.4f}")
print(f"  F1:               {f1_score(y_test, y_pred, pos_label=0):.4f}")

print("\n分类报告 classification_report（一次看全所有指标）:")
print(classification_report(y_test, y_pred, target_names=data.target_names))


# ============================================================
# 五、predict_proba：输出概率（不只是类别）
# ============================================================
# 很多场景需要"概率"而非硬标签（如风控评分、推荐排序）
proba = best.predict_proba(X_test[:3])
print("=== 预测概率 ===")
print("前 3 个样本的预测概率（每行和为1）:")
for i, p in enumerate(proba):
    print(f"  样本{i}: 恶性={p[0]:.3f}, 良性={p[1]:.3f} → 预测 {data.target_names[best.predict(X_test[:3])[i]]}")


# ============================================================
# 六、决策树的可解释性（看模型学到的规则）
# ============================================================
tree = results["决策树"]
print("\n=== 决策树重要性 ===")
# feature_importances_：每个特征的重要程度（和为1）
importances = tree.feature_importances_
top5 = np.argsort(importances)[::-1][:5]
print("Top 5 重要特征:")
for idx in top5:
    print(f"  {data.feature_names[idx]:25s} 重要度={importances[idx]:.4f}")

# 随机森林也能看重要性，且通常更稳定
rf_importances = results["随机森林"].feature_importances_
print("\n随机森林 Top 5 重要特征:")
for idx in np.argsort(rf_importances)[::-1][:5]:
    print(f"  {data.feature_names[idx]:25s} 重要度={rf_importances[idx]:.4f}")


# ============================================================
# 七、超参数 k 的影响（KNN 的 k 选多少合适？）
# ============================================================
print("\n=== KNN 的 k 值实验 ===")
ks = list(range(1, 22, 2))
for k in ks:
    m = Pipeline([("s", StandardScaler()),
                  ("k", KNeighborsClassifier(n_neighbors=k))]).fit(X_train, y_train)
    train_acc = m.score(X_train, y_train)
    test_acc = m.score(X_test, y_test)
    print(f"  k={k:2d}: 训练={train_acc:.4f}  测试={test_acc:.4f}")
# 观察：k 太小（如1）容易过拟合；k 太大欠拟合；需选平衡点（后续用网格搜索自动找）


# ============================================================
# 小结：本课你应掌握的"ML 实战"能力
# ============================================================
# 1. 四大分类器的特点与适用场景
# 2. 混淆矩阵 / 精确率 / 召回率 / F1，知道何时该侧重哪个
# 3. predict_proba 输出概率；feature_importances_ 看特征重要度
# 4. 用 Pipeline 串联标准化 + 模型
# 5. 超参数（如 k）影响效果，需实验找最佳
#
# 下一课：回归算法（预测连续数值）。
