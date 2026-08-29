# -*- coding: utf-8 -*-
"""
模块8 第3课：Transformer 与注意力机制
================================
本课目标：
  1. 直观理解 Self-Attention 在做什么
  2. 手写 Q/K/V 计算 + 缩放点积注意力
  3. 理解 Multi-Head Attention 与位置编码
  4. 拼出一个最小 Transformer 块

这一课是 LLM 的灵魂。任何 GPT/Qwen/DeepSeek 的核心都是这一行公式：
    Attention(Q, K, V) = softmax( Q·K^T / √d_k ) · V

依赖: numpy (在 dl/ 模块已用过，不依赖 PyTorch/TensorFlow)
"""

import os
import sys
import numpy as np

# Windows 控制台默认 GBK 编码，强制改成 UTF-8 以支持特殊符号
sys.stdout.reconfigure(encoding="utf-8")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
np.random.seed(42)

# 数值稳定版 softmax
def softmax(x, axis=-1):
    """数值稳定的 softmax: 减去最大值防溢出"""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

print("=" * 60)
print("第3课：Transformer 与注意力机制")
print("=" * 60)

# ============================================================
# 1. 为什么需要 Attention？—— 直观动机
# ============================================================
# 【核心概念】"注意力"是给序列中每个位置"打分"，决定关注谁
#
#   例: "The cat sat on the mat because it was tired"
#       这里的 "it" 指代谁？人类一眼就知道是 "cat"。
#       注意力机制让模型能学到 "it" 应该把注意力集中在 "cat" 上。
#
#   对比 RNN: RNN 靠"逐步遗忘"传信息，远处信息会衰减
#   Attention: 任意两个位置直接计算相似度，距离成本 O(1)
print("\n【1】为什么需要 Attention？")
print('  例: "猫坐在垫子上因为 它 累了"  →  "它"指代"猫"?')
print("  注意力让模型能跨距离直接关联两个词 (O(1) 直接交互)")

# ============================================================
# 2. Q / K / V —— 注意力的三剑客
# ============================================================
# 【核心概念】Q/K/V 来自数据库检索的类比：
#   Q (Query, 查询): 当前词"想找什么"
#   K (Key, 键):    每个词"能提供什么标签"
#   V (Value, 值):  每个词"实际携带的信息"
#
#   计算流程:
#     1) Q · K^T : 当前词与所有词的相似度 → 注意力分数
#     2) softmax : 归一化成概率 (和为 1)
#     3) 乘 V    : 按概率加权求和 → 输出
#
#   Q/K/V 都是输入 X 经过不同线性变换得到的 (可学习参数 W_Q, W_K, W_V)
print("\n【2】Q / K / V 是什么？(数据库类比)")
print("  Q (Query): 我在找什么 → '它'在找主语")
print("  K (Key):   我能提供什么 → '猫'提供 [动物, 主语] 标签")
print("  V (Value): 我的实际信息 → '猫'的语义向量")
print("  → 相似度高的(K匹配Q)就拿走更多的V")

# ============================================================
# 3. 手写 Scaled Dot-Product Attention
# ============================================================
# 【核心公式】Attention(Q, K, V) = softmax( Q·K^T / √d_k ) · V
print("\n【3】手写 Scaled Dot-Product Attention")

def attention(Q, K, V, mask=None):
    """缩放点积注意力
    Q: (seq_len, d_k)  查询
    K: (seq_len, d_k)  键
    V: (seq_len, d_v)  值
    return: (seq_len, d_v) 注意力输出
    """
    d_k = Q.shape[-1]
    # ① 计算注意力分数: Q · K^T / √d_k
    scores = Q @ K.T / np.sqrt(d_k)
    # ② 应用 mask (可选): 把不该看的位置分数设成 -∞
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    # ③ softmax 归一化 → 注意力权重 (每行和为 1)
    weights = softmax(scores, axis=-1)
    # ④ 加权求和 V
    output = weights @ V
    return output, weights

# 构造一个 4 词序列，每词 8 维向量 (用随机数模拟嵌入)
seq_len, d_model = 4, 8
X = np.random.randn(seq_len, d_model)
print(f"  输入 X shape: {X.shape}  (4 个词, 每词 8 维向量)")

# 用线性变换得到 Q/K/V (这里直接用随机权重模拟)
W_Q = np.random.randn(d_model, d_model)
W_K = np.random.randn(d_model, d_model)
W_V = np.random.randn(d_model, d_model)
Q, K, V = X @ W_Q, X @ W_K, X @ W_V

out, weights = attention(Q, K, V)
print(f"  输出 shape: {out.shape}")
print(f"  注意力权重 (每行和应为 1):")
np.set_printoptions(precision=3, suppress=True)
print(f"    {weights}")
print(f"  每行和: {weights.sum(axis=1)}")

# ============================================================
# 4. 为什么除以 √d_k ？ (缩放的数学意义)
# ============================================================
# 【核心概念】当 d_k 较大时，Q·K 的点积数值会很大 → softmax 进入饱和区
#   饱和后梯度几乎为 0，模型无法学习。除以 √d_k 把方差拉回 1 附近。
print("\n【4】为什么要除以 √d_k？")
print("  当 d_k 大时, Q·K 的方差 ≈ d_k → 数值大 → softmax 梯度消失")
print("  除以 √d_k → 方差拉回 1 → 训练稳定")
print(f"  例: d_k=8, √d_k ≈ {np.sqrt(8):.3f}")

# ============================================================
# 5. Multi-Head Attention (多头注意力)
# ============================================================
# 【核心概念】一个注意力头只能学一种"关注模式"
#   多头 = 把 d_model 拆成 h 份，每份独立做 attention → 学到多种关注模式
#   最后把所有头的结果拼起来再用线性变换合并。
#
#   例: d_model=512, h=8 → 每个头 d_k=64 维
#   不同头可能分别学到: 语法依赖 / 实体指代 / 语义相似 ...
print("\n【5】Multi-Head Attention (多头注意力)")

def multi_head_attention(X, num_heads, d_k, d_v):
    """简化版多头注意力 (没有投影回原维度的最后一步)"""
    seq_len, d_model = X.shape
    assert d_model == num_heads * d_k
    heads_outputs = []
    for h in range(num_heads):
        # 每个头用自己的 Q/K/V 投影权重
        W_Q = np.random.randn(d_model, d_k)
        W_K = np.random.randn(d_model, d_k)
        W_V = np.random.randn(d_model, d_v)
        Q_h = X @ W_Q        # (seq_len, d_k)
        K_h = X @ W_K
        V_h = X @ W_V
        out_h, _ = attention(Q_h, K_h, V_h)
        heads_outputs.append(out_h)
    # 拼接所有头: (seq_len, num_heads * d_v)
    return np.concatenate(heads_outputs, axis=-1)

num_heads = 4
d_k = d_v = d_model // num_heads    # 每头 2 维
mha_out = multi_head_attention(X, num_heads, d_k, d_v)
print(f"  num_heads={num_heads}, 每头 d_k=d_v={d_k}")
print(f"  输入 shape:  {X.shape}")
print(f"  每头输出:    (4, {d_v})")
print(f"  拼接后 shape: {mha_out.shape}  (= num_heads × d_v)")

# ============================================================
# 6. 因果掩码 Causal Mask —— GPT 的核心机制
# ============================================================
# 【核心概念】GPT 是 Decoder-only，只能"看前面"不能"看后面" (避免作弊)
#   实现方式: 用一个下三角矩阵把上三角(未来位置)设为 -∞
#
#   mask 矩阵长这样 (4×4):
#     [1, 0, 0, 0]    位置0 只能看 0
#     [1, 1, 0, 0]    位置1 能看 0,1
#     [1, 1, 1, 0]    位置2 能看 0,1,2
#     [1, 1, 1, 1]    位置3 能看 0,1,2,3
print('\n【6】因果掩码 (GPT 不能"偷看"未来)')
seq_len_demo = 4
causal_mask = np.tril(np.ones((seq_len_demo, seq_len_demo)))
print(f"  下三角 mask:\n{causal_mask.astype(int)}")

Q2 = np.random.randn(4, 4)
K2 = np.random.randn(4, 4)
V2 = np.random.randn(4, 4)
out2, weights2 = attention(Q2, K2, V2, mask=causal_mask)
print(f"  带因果 mask 的注意力权重:")
print(f"    {weights2}")
print("  → 上三角全 0, 模型只能关注自己和之前的词")

# ============================================================
# 7. 位置编码 Positional Encoding
# ============================================================
# 【核心概念】Attention 本身是"位置无关"的 (打乱顺序结果不变)
#   必须显式注入位置信息。常见方法:
#   ① 正弦/余弦编码 (原版 Transformer): 用不同频率的 sin/cos
#   ② 可学习位置编码 (BERT/GPT-2): 直接学一组位置向量
#   ③ RoPE 旋转位置编码 (LLaMA/Qwen): 把位置"旋转"进向量 (当前主流)
print("\n【7】位置编码 (给序列注入顺序信息)")

def sinusoidal_pos_encoding(seq_len, d_model):
    """原版 Transformer 的正弦位置编码"""
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)   # 偶数维用 sin
    pe[:, 1::2] = np.cos(position * div_term)   # 奇数维用 cos
    return pe

pe = sinusoidal_pos_encoding(seq_len=4, d_model=8)
print(f"  正弦位置编码 shape: {pe.shape}")
print(f"  前 2 个位置的前 4 维:")
print(f"    位置0: {pe[0, :4]}")
print(f"    位置1: {pe[1, :4]}")
print("  → 不同位置有唯一编码，模型能区分顺序")

# ============================================================
# 8. 残差连接 + LayerNorm (Transformer 的两个稳定剂)
# ============================================================
# 【核心概念】① 残差连接: x_out = x + Sublayer(x)
#             作用: 缓解梯度消失，让深层网络可训练
#            ② LayerNorm: 对每个样本的所有维度做归一化 (均值0方差1)
#             作用: 稳定训练，加速收敛
print("\n【8】残差连接 & LayerNorm")

def layer_norm(x, eps=1e-6):
    """对最后一维做 LayerNorm"""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

x = np.random.randn(2, 4) * 3 + 5     # 均值约5，方差约9
x_norm = layer_norm(x)
print(f"  LayerNorm 前: 均值={x.mean(axis=-1).round(2)}, 方差={x.var(axis=-1).round(2)}")
print(f"  LayerNorm 后: 均值={x_norm.mean(axis=-1).round(3)}, 方差={x_norm.var(axis=-1).round(3)}")
print("  → 归一化到均值0、方差1，让训练更稳定")

# ============================================================
# 9. 组装一个完整的 Transformer Block (GPT Decoder 风格)
# ============================================================
# 【架构图】 GPT 的一个 Block:
#   x → LayerNorm → MultiHeadAttention(因果mask) → + 残差 →
#       → LayerNorm → FeedForward(MLP) → + 残差 → 输出
print("\n【9】组装一个最小 Transformer Block")

def feed_forward(x, d_ff=16):
    """前馈网络: 两层线性 + 激活"""
    W1 = np.random.randn(x.shape[-1], d_ff) * 0.1
    W2 = np.random.randn(d_ff, x.shape[-1]) * 0.1
    return np.maximum(0, x @ W1) @ W2    # ReLU 激活

def transformer_block(x, num_heads=4):
    """一个 GPT 风格的 Transformer 块 (Pre-LN 结构)"""
    d_model = x.shape[-1]
    d_k = d_v = d_model // num_heads

    # 子层 1: LayerNorm + Multi-Head Attention + 残差
    normed = layer_norm(x)
    attn_out = multi_head_attention(normed, num_heads, d_k, d_v)
    # 为了维度对齐，截断/补齐到 d_model
    if attn_out.shape[-1] != d_model:
        attn_out = attn_out[..., :d_model]
    x = x + attn_out                       # 残差连接

    # 子层 2: LayerNorm + FeedForward + 残差
    normed = layer_norm(x)
    ff_out = feed_forward(normed)
    x = x + ff_out                         # 残差连接
    return x

X_in = np.random.randn(seq_len, d_model)
X_out = transformer_block(X_in, num_heads=4)
print(f"  输入 shape:  {X_in.shape}")
print(f"  输出 shape:  {X_out.shape}  (与输入同形，可堆叠多层)")
print("  GPT-3 就是把这种 Block 堆了 96 层！")

# ============================================================
# 10. GPT 完整结构回顾
# ============================================================
print("\n【10】GPT 完整结构 (从输入到输出)")
print("-" * 60)
gpt_structure = [
    ("1. Token Embedding",   "每个 token id → 一个 d 维向量"),
    ("2. + Position Encoding", "加上位置编码，让模型知道顺序"),
    ("3. ×N Transformer Block", "堆叠 N 层 (GPT-2 small=12, GPT-3=96)"),
    ("   ├─ LayerNorm",      ""),
    ("   ├─ Multi-Head Attention (因果mask)", ""),
    ("   └─ FeedForward + 残差", ""),
    ("4. Final LayerNorm",   ""),
    ("5. Linear (输出投影)",  "把 d 维向量映射回词表大小 V"),
    ("6. Softmax",           "得到下一个 token 的概率分布"),
]
for layer, desc in gpt_structure:
    print(f"  {layer:32s} {desc}")
print("-" * 60)

print("\n" + "=" * 60)
print("第3课小结")
print("=" * 60)
print("""
  [OK] Attention: 让序列中任意位置直接交互 (O(1) 距离成本)
  [OK] 核心公式: softmax(Q·K^T / √d_k) · V
  [OK] Multi-Head: 多个头并行学多种关注模式
  [OK] 因果 mask: GPT 不能看未来 (下三角矩阵)
  [OK] 位置编码: 注入顺序信息 (sin / 可学习 / RoPE)
  [OK] 残差 + LayerNorm: 让深层网络可训练
  [OK] GPT = Embedding + N×TransformerBlock + Linear

  下一课: 用 HuggingFace 加载真实大模型 (理论落地为代码)。
""")

# ============================================================
# 练习 (可选)
# ============================================================
# 1. 修改 attention() 的 mask，实现 BERT 的双向注意力 (mask 全 1)
# 2. 思考: 为什么 GPT 用 Pre-LN (先 LayerNorm 再 Attention) 而不是 Post-LN？
#    (提示: Pre-LN 训练更稳定，深层不发散)
# 3. 进阶: 用 PyTorch 把这套代码重写一遍，对比 nn.MultiheadAttention 的输出
