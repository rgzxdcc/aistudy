# -*- coding: utf-8 -*-
"""
模块5 · 第5课：无监督学习 + 完整项目收官
========================================
本课两部分：
(1) 无监督学习：K-Means 聚类、PCA 降维
(2) 端到端实战：鸢尾花分类项目（贯穿前 4 课所有技能）
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 第一部分：无监督学习
# ============================================================

# ---------- 一、K-Means 聚类 ----------
# 目标：把相似的样本自动分到一组，不需要标签
# 算法：随机选 K 个中心点 → 把每个样本归到最近的中心 → 重新算中心 → 重复
#
# 关键超参数：K（簇数）需要事先指定
# 应用：客户分群、异常检测、图像压缩

np.random.seed(42)
# 生成 3 簇数据
from sklearn.datasets import make_blobs
X_blob, y_true = make_blobs(n_samples=300, centers=3, cluster_std=1.0,
                             random_state=42)

print("=== K-Means 聚类 ===")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_blob)   # 聚类，返回每个样本的簇标签
print(f"聚类中心:\n{np.round(kmeans.cluster_centers_, 2)}")
print(f"惯性 inertia_（越小越紧凑）: {kmeans.inertia_:.2f}")

# 用肘部法则选 K：画 K vs inertia，找"拐点"
inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_blob)
    inertias.append(km.inertia_)

# ---------- 二、PCA 降维 ----------
# 目标：把高维特征压缩到低维，同时保留尽可能多的信息
# 应用：可视化高维数据、加速训练、降噪
#
# 例：4 维鸢尾花特征 → 2 维便于画图

iris = load_iris()
X = iris.data
y = iris.target

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
print(f"\n=== PCA 降维 ===")
print(f"原始: {X.shape} → 降维后: {X_2d.shape}")
print(f"两个主成分解释的方差比例: {pca.explained_variance_ratio_}")
print(f"累计保留信息: {sum(pca.explained_variance_ratio_):.2%}")
# 如果 2 维能保留 95%+ 信息，说明 4 维里有很多冗余


# ---------- 三、画图：肘部法则 + PCA 可视化 ----------
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "_images")
os.makedirs(OUT_DIR, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 子图1：K-Means 聚类结果（聚类 vs 真实）
axes[0].scatter(X_blob[:, 0], X_blob[:, 1], c=labels, cmap="viridis", s=15, alpha=0.6)
axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c="red", marker="X", s=200, label="聚类中心")
axes[0].set_title("K-Means 聚类"); axes[0].legend()

# 子图2：肘部法则
axes[1].plot(list(K_range), inertias, "o-")
axes[1].axvline(x=3, color="red", ls="--", alpha=0.5, label="肘部 K=3")
axes[1].set_xlabel("K（簇数）"); axes[1].set_ylabel("inertia")
axes[1].set_title("肘部法则选 K"); axes[1].legend(); axes[1].grid(alpha=0.3)

# 子图3：PCA 降维后的鸢尾花（按真实类别上色）
for i, name in enumerate(iris.target_names):
    axes[2].scatter(X_2d[y == i, 0], X_2d[y == i, 1], label=name, alpha=0.7)
axes[2].set_xlabel(f"主成分1 ({pca.explained_variance_ratio_[0]:.1%})")
axes[2].set_ylabel(f"主成分2 ({pca.explained_variance_ratio_[1]:.1%})")
axes[2].set_title("PCA: 鸢尾花 4维→2维"); axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "无监督学习.png"), dpi=100, bbox_inches="tight")
plt.close()
print(f"已保存: {OUT_DIR}/无监督学习.png")


# ============================================================
# 第二部分：端到端实战项目（模块 5 收官）
# ============================================================
# 任务：用鸢尾花数据集，完整走一遍 ML 项目流程
# 这是真实项目要做的事，把前 4 课所有技能串起来。

print("\n" + "=" * 60)
print("           鸢尾花分类项目（端到端实战）")
print("=" * 60)

# ---------- Step 1: 加载数据 & 探索 ----------
print("\n【Step 1】数据加载与探索")
iris = load_iris(as_frame=True)
df = iris.frame
print(f"形状: {df.shape}")
print(f"特征: {list(iris.feature_names)}")
print(f"类别: {list(iris.target_names)}")
print("\n类别分布:")
print(df["target"].value_counts())
print("\n统计摘要:")
print(df.describe().round(2))

# ---------- Step 2: 划分数据 ----------
print("\n【Step 2】划分训练/测试集（7:3，分层抽样）")
X = df.drop(columns=["target"]).values
y = df["target"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
print(f"训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")

# ---------- Step 3: 模型对比（用交叉验证）----------
print("\n【Step 3】四模型对比（5 折交叉验证）")
# cross_val_score：把训练集再分 5 份，轮流当验证集，平均得到更稳定的评估
candidates = {
    "逻辑回归": Pipeline([("s", StandardScaler()),
                           ("m", LogisticRegression(max_iter=200))]),
    "KNN":      Pipeline([("s", StandardScaler()),
                           ("m", KNeighborsClassifier(n_neighbors=5))]),
    "决策树":   DecisionTreeClassifier(max_depth=5, random_state=42),
    "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
}

best_name, best_score, best_model = None, 0, None
for name, model in candidates.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    mean_score = scores.mean()
    print(f"  {name:6s}: {mean_score:.4f} (±{scores.std():.4f})")
    if mean_score > best_score:
        best_name, best_score, best_model = name, mean_score, model

print(f"\n  → 最佳模型: {best_name}（CV={best_score:.4f}）")

# ---------- Step 4: 训练最佳模型 ----------
print("\n【Step 4】训练最佳模型")
best_model.fit(X_train, y_train)

# ---------- Step 5: 测试集评估 ----------
print("\n【Step 5】测试集最终评估")
y_pred = best_model.predict(X_test)
print(f"测试集准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ---------- Step 6: 预测新样本 ----------
print("\n【Step 6】预测新样本")
new_samples = np.array([
    [5.1, 3.5, 1.4, 0.2],   # 像山鸢尾
    [6.7, 3.1, 4.7, 1.5],   # 像变色鸢尾
    [7.2, 3.6, 6.1, 2.5],   # 像维吉尼亚鸢尾
])
preds = best_model.predict(new_samples)
for i, p in enumerate(preds):
    print(f"  样本{new_samples[i]} → {iris.target_names[p]}")

# ---------- Step 7: 项目总结 ----------
print("\n【项目总结】")
print(f"  数据: 150 样本 × 4 特征，3 类均衡")
print(f"  最佳模型: {best_name}")
print(f"  交叉验证: {best_score:.2%}")
print(f"  测试集: {accuracy_score(y_test, y_pred):.2%}")
print("  结论: 鸢尾花是简单数据集，多数模型都能达到 95%+ 准确率")


# ============================================================
# 小结：本课 + 整个模块 5 的核心能力
# ============================================================
# 1. K-Means 聚类：需指定 K，用肘部法则选；inertia 越小越紧凑
# 2. PCA 降维：高维→低维，explained_variance_ratio_ 看保留多少信息
# 3. 端到端项目 7 步流程：加载→探索→划分→对比→训练→评估→预测
# 4. cross_val_score 交叉验证：更稳定的模型评估
#
# === 恭喜！整个 ML 学习路径（Python→NumPy→Pandas→Matplotlib→Scikit-learn）已结业 ===
# 你现在具备了完整的"数据加载→清洗→可视化→建模→评估"能力。
# 进阶方向：深度学习（PyTorch/TensorFlow）、Kaggle 实战、特征工程深入。
