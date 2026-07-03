# -*- coding: utf-8 -*-
"""
模块5 · 第4课：回归算法与评估
============================
分类预测"类别"，回归预测"数值"。本课实战回归任务：
- 线性回归 LinearRegression（最经典的回归）
- 随机森林回归 RandomForestRegressor（非线性强基线）
并学习回归的评估指标（MSE / RMSE / R^2）。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无界面环境用 Agg 后端
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 一、回归任务简介
# ============================================================
# 回归：标签 y 是连续数值
#   例：房价预测、销量预测、温度预测
#
# 评估指标（与分类完全不同！）：
#   MSE  均方误差 = 平均((y真 - y预)^2)   对大误差敏感
#   RMSE = √MSE                        与 y 同量纲，更直观
#   MAE  平均绝对误差 = 平均(|y真 - y预|)
#   R^2  决定系数 ∈ (-∞, 1]
#        1 = 完美预测；0 = 等同于直接猜均值；负数 = 还不如猜均值
#        R^2 是回归最常用的指标，越接近 1 越好


# ============================================================
# 二、准备数据：使用前面的房价数据集
# ============================================================
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "06_datasets", "houses.csv")

if not os.path.exists(CSV):
    # 数据不存在则生成（与 Pandas 课一致）
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "面积": np.random.randint(50, 200, n),
        "房间数": np.random.randint(1, 6, n),
        "楼层": np.random.randint(1, 30, n),
        "房龄": np.random.randint(0, 30, n),
    })
    df["价格"] = (
        df["面积"] * 5 + df["房间数"] * 20 - df["房龄"] * 3
        + np.random.randn(n) * 30
    ).round(1)
else:
    df = pd.read_csv(CSV)

print("=== 房价数据 ===")
print(df.head())
print(f"共 {len(df)} 条数据")

# 切分 X, y
feature_cols = ["面积", "房间数", "楼层", "房龄"]
X = df[feature_cols].values
y = df["价格"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# ============================================================
# 三、线性回归：最经典的回归算法
# ============================================================
# 学一个线性方程：y = w1*x1 + w2*x2 + ... + b
# 学习目标：找一组 w 让预测值与真实值的误差最小（最小二乘法）

lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\n=== 线性回归 ===")
print(f"R^2:  {r2_score(y_test, y_pred_lr):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_lr):.2f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred_lr) ** 0.5:.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_lr):.2f}")

# 查看学到的权重（每个特征对价格的影响）
model_lr = lr.named_steps["model"]
print("\n学到的权重（标准化后的系数，可比较特征重要性）:")
for name, w in zip(feature_cols, model_lr.coef_):
    print(f"  {name:6s}: {w:+.2f}")
print(f"  截距 b: {model_lr.intercept_:+.2f}")


# ============================================================
# 四、随机森林回归：非线性强基线
# ============================================================
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n=== 随机森林回归 ===")
print(f"R^2:  {r2_score(y_test, y_pred_rf):.4f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred_rf) ** 0.5:.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.2f}")

# 特征重要性
print("\n特征重要性:")
for name, imp in sorted(zip(feature_cols, rf.feature_importances_),
                        key=lambda x: -x[1]):
    print(f"  {name:6s}: {imp:.4f}")


# ============================================================
# 五、可视化：预测值 vs 真实值
# ============================================================
OUT_DIR = os.path.join(HERE, "_images")
os.makedirs(OUT_DIR, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, name, y_pred in [(axes[0], "线性回归", y_pred_lr),
                          (axes[1], "随机森林", y_pred_rf)]:
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolors="white")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", alpha=0.5)         # 对角线（理想情况点全在线上）
    ax.set_xlabel("真实价格"); ax.set_ylabel("预测价格")
    r2 = r2_score(y_test, y_pred)
    ax.set_title(f"{name}（R^2={r2:.3f}）")
    ax.grid(alpha=0.3)

plt.suptitle("回归模型：预测 vs 真实", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "回归对比.png"), dpi=100, bbox_inches="tight")
plt.close()
print(f"\n已保存图: {OUT_DIR}/回归对比.png")


# ============================================================
# 六、理解 R^2 的含义（重要直觉）
# ============================================================
# R^2 = 1 - SS_res/SS_tot
#   SS_res = 残差平方和 Σ(y真 - y预)^2    模型的误差
#   SS_tot = 总平方和 Σ(y真 - y均值)^2    "全猜均值"的误差
#
# R^2 = 1   → 模型完美，误差为 0
# R^2 = 0   → 模型等同于"直接猜平均值"，没学到任何规律
# R^2 < 0  → 模型比"猜均值"还差（很少见，说明严重过拟合或选错模型）
#
# 直觉：R^2 表示"模型解释了多少数据波动"

print("\n=== R^2 直觉验证 ===")
# 用均值预测的"基线"
y_mean_pred = np.full_like(y_test, y_train.mean())
r2_baseline = r2_score(y_test, y_mean_pred)
print(f"用均值预测的 R^2: {r2_baseline:.4f}  (≈0，说明没学到规律)")
print(f"线性回归 R^2:     {r2_score(y_test, y_pred_lr):.4f}  (远高于0，学到了)")


# ============================================================
# 七、预测新样本
# ============================================================
new_house = np.array([[100, 3, 15, 10]])   # 100㎡、3室、15层、10年
price_lr = lr.predict(new_house)[0]
price_rf = rf.predict(new_house)[0]
print(f"\n新房子预测价格: 线性回归={price_lr:.1f}万, 随机森林={price_rf:.1f}万")


# ============================================================
# 小结：本课你应掌握的"ML 实战"能力
# ============================================================
# 1. 回归与分类的区别（标签连续 vs 离散）
# 2. 评估指标：MSE/RMSE/MAE/R^2，重点是 R^2
# 3. LinearRegression 的系数可解释性强；随机森林能抓非线性
# 4. "预测 vs 真实"散点图是回归的标配可视化
# 5. R^2 的直觉：解释了多少数据波动，0 = 没学到
#
# 下一课：无监督学习（聚类/降维）+ 鸢尾花完整项目收官。
