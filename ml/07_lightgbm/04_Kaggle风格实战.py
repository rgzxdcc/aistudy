# -*- coding: utf-8 -*-
"""
模块7 第4课：Kaggle 风格端到端实战（收官）
==========================================
本课目标：走完一个完整的表格数据竞赛流程

  1. 数据生成与加载 (模拟「客户流失预测」竞赛数据)
  2. EDA 探索性分析
  3. 特征工程 (衍生特征、编码)
  4. 交叉验证 + LightGBM 训练
  5. 特征重要性分析与模型可解释性
  6. 生成 submission.csv (Kaggle 标准提交格式)
  7. 模型持久化

【为什么这个流程重要】
  Kaggle / 真实工作中, 模型代码只占 20%, 80% 时间花在
  理解数据 + 特征工程 + 验证策略上。本课完整呈现这套流程。
"""

import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("_images", exist_ok=True)
os.makedirs("_output", exist_ok=True)

print("=" * 60)
print("第4课：Kaggle 风格端到端实战 —— 客户流失预测")
print("=" * 60)


# ============================================================
# 第1步：数据准备（模拟竞赛数据集）
# ============================================================
print("\n[步骤 1/7] 数据准备")
print("-" * 50)

# 【ML 场景】真实 Kaggle 比赛会给你 train.csv + test.csv
#   train 有标签, test 没标签, 你要预测 test 的标签并提交
#   这里我们用合成数据模拟这个场景
rng = np.random.RandomState(42)
N_TRAIN, N_TEST = 4000, 1000

def make_churn_data(n, rng):
    """模拟电信客户流失数据"""
    tenure = rng.randint(1, 72, n)                        # 在网月数
    monthly_charges = rng.uniform(18, 120, n).round(2)    # 月费
    total_charges = (tenure * monthly_charges * rng.uniform(0.8, 1.2, n)).round(2)
    age = rng.randint(18, 75, n)                          # 年龄
    support_calls = rng.poisson(2, n)                     # 客服来电次数
    contract = rng.choice(["月付", "年付", "两年"], n, p=[0.5, 0.3, 0.2])
    internet = rng.choice(["光纤", "DSL", "无"], n, p=[0.4, 0.4, 0.2])

    # 流失逻辑: 月费高+来电多+月付 → 更易流失 (非线性关系)
    logit = (-3.0
             + 0.02 * monthly_charges
             + 0.4 * support_calls
             - 0.05 * tenure
             + (contract == "月付") * 1.5
             + (internet == "光纤") * 0.5)
    prob = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < prob).astype(int)
    return tenure, monthly_charges, total_charges, age, support_calls, contract, internet, churn

raw = make_churn_data(N_TRAIN + N_TEST, rng)
cols = ["tenure", "monthly_charges", "total_charges", "age",
        "support_calls", "contract", "internet", "churn"]
df_all = pd.DataFrame(zip(*raw), columns=cols)

train_df = df_all.iloc[:N_TRAIN].copy()
test_df = df_all.iloc[N_TRAIN:].copy()
# 竞赛中 test 没有 churn 列
test_labels = test_df.pop("churn")     # 暂存「标准答案」用于最后自评

print(f"  训练集: {train_df.shape}  测试集: {test_df.shape}")
print(f"  流失率(训练集): {train_df['churn'].mean():.1%}")
print(f"  特征: {list(train_df.columns[:-1])}")


# ============================================================
# 第2步：EDA 探索性分析
# ============================================================
print("\n[步骤 2/7] EDA 探索性分析")
print("-" * 50)

print("\n  训练集概览:")
print(train_df.describe(include="all").round(2).to_string())

# 缺失值检查
print(f"\n  缺失值: {train_df.isnull().sum().sum()} 个 (合成数据无缺失)")

# 流失 vs 合同类型 交叉表
print("\n  各合同类型流失率:")
ct = pd.crosstab(train_df["contract"], train_df["churn"], normalize="index")
print(ct.round(3).to_string())

# 可视化 EDA
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
# (a) 月费分布按流失分组
ax = axes[0, 0]
for label, color, name in [(0, "steelblue", "留存"), (1, "coral", "流失")]:
    ax.hist(train_df[train_df.churn == label]["monthly_charges"],
            bins=30, alpha=0.6, color=color, label=name)
ax.set_xlabel("月费"); ax.set_ylabel("人数"); ax.legend(); ax.set_title("月费分布 vs 流失")

# (b) 在网月数 vs 流失
ax = axes[0, 1]
ax.hist(train_df[train_df.churn == 0]["tenure"], bins=30, alpha=0.6,
        color="steelblue", label="留存")
ax.hist(train_df[train_df.churn == 1]["tenure"], bins=30, alpha=0.6,
        color="coral", label="流失")
ax.set_xlabel("在网月数"); ax.legend(); ax.set_title("在网时长 vs 流失")

# (c) 客服来电次数 vs 流失率
ax = axes[1, 0]
grouped = train_df.groupby("support_calls")["churn"].mean()
ax.bar(grouped.index, grouped.values, color="coral")
ax.set_xlabel("客服来电次数"); ax.set_ylabel("流失率"); ax.set_title("来电次数 vs 流失率")

# (d) 合同类型 vs 流失率
ax = axes[1, 1]
ct2 = train_df.groupby("contract")["churn"].mean()
ax.bar(ct2.index, ct2.values, color=["tomato", "orange", "seagreen"])
ax.set_ylabel("流失率"); ax.set_title("合同类型 vs 流失率")
plt.tight_layout()
plt.savefig("_images/lgb_04_eda.png", dpi=100)
print("\n  [图] 已保存: _images/lgb_04_eda.png")
plt.close()


# ============================================================
# 第3步：特征工程
# ============================================================
print("\n[步骤 3/7] 特征工程")
print("-" * 50)

def feature_engineering(df):
    """统一处理训练集和测试集的特征工程"""
    df = df.copy()
    # (a) 类别特征编码
    df["contract"] = df["contract"].map({"月付": 0, "年付": 1, "两年": 2})
    df["internet"] = df["internet"].map({"无": 0, "DSL": 1, "光纤": 2})

    # (b) 衍生特征 —— 业务理解驱动的特征创造
    # 【ML 场景】特征工程是 Kaggle 涨分关键, 好特征 > 复杂模型
    df["avg_charge_per_month"] = df["total_charges"] / (df["tenure"] + 1)  # 月均消费
    df["charge_ratio"] = df["monthly_charges"] / (df["total_charges"] + 1)  # 月费占比
    df["is_monthly"] = (df["contract"] == 0).astype(int)                  # 是否月付
    df["high_charge"] = (df["monthly_charges"] > 70).astype(int)          # 是否高消费
    df["risk_score"] = df["support_calls"] * 0.3 + df["is_monthly"] * 0.4 # 风险评分
    return df

train_fe = feature_engineering(train_df)
test_fe = feature_engineering(test_df)
print(f"  原始特征数: {train_df.shape[1]-1}  衍生后: {train_fe.shape[1]-1}")
print(f"  新增特征: avg_charge_per_month, charge_ratio, is_monthly, high_charge, risk_score")

feature_cols = [c for c in train_fe.columns if c != "churn"]
X = train_fe[feature_cols].values
y = train_fe["churn"].values
X_test_final = test_fe[feature_cols].values
print(f"  最终特征矩阵: X{X.shape}  X_test{X_test_final.shape}")


# ============================================================
# 第4步：交叉验证 + 训练
# ============================================================
print("\n[步骤 4/7] 5折交叉验证训练")
print("-" * 50)

# 【Kaggle 核心技巧】StratifiedKFold 保证每折的正负样本比例一致
#   对不均衡数据尤其重要
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 【Kaggle 核心技巧】OOF (Out-Of-Fold) 预测
#   每折用 4/5 训练, 预测剩下 1/5, 拼起来得到完整训练集的「诚实预测」
#   OOF 分数比单次 split 更可靠, 也是集成学习的基础
oof_pred = np.zeros(len(X))
test_pred = np.zeros(len(X_test_final))
cv_aucs = []

lgb_params = dict(
    n_estimators=300,
    num_leaves=31,
    learning_rate=0.05,
    max_depth=6,
    min_child_samples=30,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    # OOF 预测
    oof_pred[val_idx] = model.predict_proba(X_val)[:, 1]
    # 累加 test 预测(最后取平均 = 5个模型的集成)
    test_pred += model.predict_proba(X_test_final)[:, 1] / skf.n_splits

    fold_auc = roc_auc_score(y_val, oof_pred[val_idx])
    cv_aucs.append(fold_auc)
    print(f"  Fold {fold}: AUC = {fold_auc:.4f}  (best_iter={model.best_iteration_})")

test_pred /= 1  # 已经在上面除了 n_splits
oof_auc = roc_auc_score(y, oof_pred)
print(f"\n  OOF AUC = {oof_auc:.4f}  (5折平均: {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f})")


# ============================================================
# 第5步：特征重要性分析
# ============================================================
print("\n[步骤 5/7] 特征重要性分析")
print("-" * 50)

# 用最后一折的模型看重要性(gain)
importance = model.booster_.feature_importance(importance_type="gain")
imp_df = pd.DataFrame({"feature": feature_cols, "importance": importance})
imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
print("\n  特征重要性 (gain):")
for _, row in imp_df.head(10).iterrows():
    print(f"    {row['feature']:25s} {row['importance']:>10.1f}")

# 可视化
fig, ax = plt.subplots(figsize=(8, 6))
order = imp_df.index[:12]
ax.barh(range(len(order)), imp_df.loc[order, "importance"][::-1], color="steelblue")
ax.set_yticks(range(len(order)))
ax.set_yticklabels(imp_df.loc[order, "feature"][::-1])
ax.set_xlabel("importance (gain)")
ax.set_title("Top12 特征重要性")
plt.tight_layout()
plt.savefig("_images/lgb_04_importance.png", dpi=100)
print("\n  [图] 已保存: _images/lgb_04_importance.png")
plt.close()


# ============================================================
# 第6步：评估与提交
# ============================================================
print("\n[步骤 6/7] 评估与生成提交文件")
print("-" * 50)

# 用 OOF 预测评估
oof_class = (oof_pred > 0.5).astype(int)
print("\n  OOF 整体性能:")
print(f"    准确率 = {accuracy_score(y, oof_class):.4f}")
print(f"    AUC    = {oof_auc:.4f}")
print("  混淆矩阵:")
print(pd.DataFrame(
    confusion_matrix(y, oof_class),
    index=["实际:留存", "实际:流失"],
    columns=["预测:留存", "预测:流失"]
).to_string())

# 用「标准答案」自评 test 集
test_class = (test_pred > 0.5).astype(int)
test_auc = roc_auc_score(test_labels, test_pred)
test_acc = accuracy_score(test_labels, test_class)
print(f"\n  测试集(自评): 准确率 = {test_acc:.4f}  AUC = {test_auc:.4f}")

# 生成 Kaggle 标准提交文件
submission = pd.DataFrame({
    "id": range(1, len(test_pred) + 1),
    "churn": test_pred     # 提交概率(Kaggle 用 AUC 评分时需要概率)
})
submission.to_csv("_output/submission.csv", index=False)
print(f"\n  [文件] 已生成: _output/submission.csv ({len(submission)} 行)")
print("  前5行预览:")
print(submission.head().to_string(index=False))


# ============================================================
# 第7步：模型持久化
# ============================================================
print("\n[步骤 7/7] 模型保存")
print("-" * 50)

# 训练一个最终的全量模型用于部署
final_model = lgb.LGBMClassifier(**lgb_params)
final_model.fit(X, y)
final_model.booster_.save_model("_output/final_model.txt")
print(f"  [文件] 已保存: _output/final_model.txt")
print(f"  (全量数据训练, 可直接用于线上预测)")


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("第4课小结 + LightGBM 模块全部结业")
print("=" * 60)
print("""
  本课完整走了一遍 Kaggle 表格竞赛标准流程:
  [OK] 数据准备 -> EDA -> 特征工程 -> CV训练 -> 重要性 -> 提交 -> 持久化

  关键技术点回顾:
  [OK] OOF (Out-Of-Fold): 交叉验证的诚实预测, 集成学习的基础
  [OK] StratifiedKFold: 保持每折正负比例一致
  [OK] 特征工程: 衍生业务特征是涨分关键(好特征 > 复杂模型)
  [OK] 5模型平均: test_pred 取 5 折平均 = 简单的模型集成

  =========== LightGBM 模块(模块7)全部结业 ===========
    第1课: 原生API入门, Dataset/params/train, early_stopping
    第2课: sklearn接口, 分类/回归, RF对比, 样本权重
    第3课: 特征重要性, GridSearch/RandomSearch, CV稳定性
    第4课: Kaggle风格端到端实战(本项目)

  下一步方向:
    -> 模型集成: XGBoost + LightGBM + CatBlend 加权融合
    -> CatBoost: 另一个 GBDT 利器, 类别特征更强
    -> Optuna: 比 GridSearch 更高效的自动调参框架
    -> 真实竞赛: 注册 Kaggle, 找一个 Getting Started 赛事练手
""")


# ============================================================
# 练习（可选）
# ============================================================
# 1. 在特征工程里加入「年龄段」分箱特征(18-30, 30-50, 50+), 观察 AUC 变化
# 2. 调整 lgb_params, 把 learning_rate 改 0.01, n_estimators 改 2000, 跑出最优 OOF AUC
# 3. 训练一个 XGBoost 模型, 把它的 OOF 预测和 LightGBM 做 0.5/0.5 加权融合, 看 AUC 能否提升
