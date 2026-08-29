# -*- coding: utf-8 -*-
"""
模块8 第1课：LLM 通识与发展史
================================
本课目标：
  1. 建立对「语言模型 (LM)」的直观理解：它到底在干什么？
  2. 把握技术演进主线：n-gram → Word2Vec → RNN → Transformer → LLM
  3. 理解三大预训练范式：BERT / GPT / T5
  4. 认识关键概念：涌现能力、Scaling Law、上下文学习

语言模型 (Language Model, LM) 本质就一句话：
    给定一段文本的历史，预测下一个 token(词/字) 的概率分布。
        P(next_token | token_1, token_2, ..., token_k)
现代 LLM(GPT 系列)只是把这件事做到极致：数据多、模型大、训练久。

【一句话总结发展史】
    统计计数(n-gram)  →  神经网络拟合(Word2Vec/RNN)  →  并行注意力(Transformer)  →  海量预训练(LLM)
"""

import os
import sys
import random
from collections import Counter, defaultdict

# Windows 控制台默认 GBK 编码，强制改成 UTF-8 以支持特殊符号
sys.stdout.reconfigure(encoding="utf-8")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
random.seed(42)

print("=" * 60)
print("第1课：LLM 通识与发展史")
print("=" * 60)

# ============================================================
# 1. 语言模型最朴素的形态：n-gram (统计语言模型)
# ============================================================
# 【核心概念】n-gram: 用「最近 n-1 个词」预测下一个词
#   思想非常朴素：在大量文本里数一数 "我爱" 后面跟什么最多
#   缺点：n 大时组合爆炸，且无法捕捉长距离依赖
print("\n【1】n-gram 语言模型演示")

# 准备一个微型"训练语料" (3 句话)
corpus = [
    "我 爱 自然 语言 处理",
    "我 爱 机器 学习",
    "自然 语言 处理 很 有趣",
    "我 喜欢 机器 学习 和 深度 学习",
]

# 统计 bigram(2-gram): 给定前一个词，下一个词的频次
bigram_count = defaultdict(Counter)   # bigram_count[w] = {next_w: 次数}
for sentence in corpus:
    words = sentence.split()
    for i in range(len(words) - 1):
        bigram_count[words[i]][words[i + 1]] += 1

# 计算条件概率 P(next | current) = count(current, next) / count(current)
def predict_next(word: str) -> dict:
    """给定前一个词，返回下一个词的概率分布"""
    counter = bigram_count[word]
    total = sum(counter.values())
    return {w: c / total for w, c in counter.items()}

print("  语料:", corpus)
print("  P(下一个词 | '我') =", predict_next("我"))
print("  P(下一个词 | '爱') =", predict_next("爱"))
# 输出会显示: '我' 后面 50% 跟 '爱', 50% 跟 '喜欢'
# 这就是语言模型最原始的形态：用统计计数估计概率

# 生成一段文本（贪心：每次选概率最高的下一个词）
def generate_bigram(start: str, n_words: int = 4) -> str:
    words = [start]
    for _ in range(n_words):
        cur = words[-1]
        if cur not in bigram_count or not bigram_count[cur]:
            break
        next_word = max(bigram_count[cur], key=bigram_count[cur].get)
        words.append(next_word)
    return " ".join(words)

print("  从 '我' 开始生成:", generate_bigram("我"))
# n-gram 只能生成这种"死板"的句子，因为它只看前 1 个词

# ============================================================
# 2. 词的语义：从离散符号到稠密向量 (Word2Vec)
# ============================================================
# 【核心概念】Word2Vec (2013, Google)
#   传统方法中 "猫" 和 "狗" 是完全无关的两个符号。
#   Word2Vec 训练一个神经网络，把每个词映射到一个稠密向量(如 300 维)，
#   使得 "意思相近的词" 在向量空间中距离也近。
#
#   著名等式: king - man + woman ≈ queen   (词向量上的代数关系)
#
#   这是神经网络时代语言模型的开端——模型开始"理解"语义。
print("\n【2】词向量思想 (Word2Vec 概念演示)")
print("  传统: 词 = 离散符号  →  '猫'和'狗'毫无关联")
print("  现代: 词 = 稠密向量  →  '猫'和'狗'向量很接近")
print("  经典代数: vec('king') - vec('man') + vec('woman') ≈ vec('queen')")

# 用随机向量模拟"相似词的向量也接近"这一性质
def cosine_sim(v1, v2):
    """余弦相似度: 衡量两个向量的方向相似程度 (1=完全相同)"""
    import math
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    return dot / (n1 * n2 + 1e-9)

# 模拟: 把同义词的向量设成接近的随机值
word_vec = {
    "猫":   [0.9, 0.1, 0.05, 0.0],
    "狗":   [0.85, 0.15, 0.0, 0.05],   # 与"猫"很近
    "汽车": [0.05, 0.1, 0.9, 0.95],   # 与"猫/狗"很远
}
print(f"  sim(猫, 狗)   = {cosine_sim(word_vec['猫'], word_vec['狗']):.3f}  (相近)")
print(f"  sim(猫, 汽车) = {cosine_sim(word_vec['猫'], word_vec['汽车']):.3f}  (无关)")

# ============================================================
# 3. 序列建模：RNN 与 Transformer 的分水岭
# ============================================================
# 【核心概念】为什么需要 Transformer (2017, 《Attention is All You Need》)
#
#   RNN (循环神经网络): 一个词一个词串行处理，有"记忆"但有两个致命伤：
#     ① 无法并行 → 训练慢
#     ② 长序列遗忘 → 前面信息会丢失
#
#   Transformer 的破局点: 用 Self-Attention 让序列中任意两个位置
#   「直接」交互，距离成本 O(1)，并且可以高度并行 → 训练大模型成为可能。
#
#   GPT 全称 Generative Pre-trained Transformer，名字本身就说明了这点。
print("\n【3】RNN vs Transformer (理论对比)")
print("-" * 60)
print(f"  {'特性':15s} {'RNN':25s} {'Transformer'}")
print("-" * 60)
for feature, rnn, tf in [
    ("处理方式", "串行(按时间步)", "并行(一次看全句)"),
    ("长距离依赖", "差(易遗忘)", "好(O(1)直接交互)"),
    ("训练速度", "慢", "快(适合大规模训练)"),
    ("可扩展性", "差", "极强 → 催生 LLM"),
]:
    print(f"  {feature:15s} {rnn:25s} {tf}")
print("-" * 60)

# ============================================================
# 4. 三大预训练范式：BERT / GPT / T5
# ============================================================
# 【核心概念】预训练 = 在海量无标注文本上自监督学习 → 得到通用模型
#   再通过「微调 / 提示」适配具体任务，避免每个任务都从零训练。
#
#   按结构可分为三类，理解它们的差异就理解了现代 NLP 全貌：
print("\n【4】三大预训练范式")
paradigms = [
    {
        "name": "BERT (Encoder-only)",
        "代表": "BERT / RoBERTa / ALBERT",
        "训练目标": "完形填空 (Masked LM): 随机遮挡15%的词让模型猜",
        "擅长": "文本理解 (分类、NER、问答)",
        "生成能力": "弱 (不擅长续写)",
    },
    {
        "name": "GPT (Decoder-only)",
        "代表": "GPT-2/3/4, LLaMA, Qwen, ChatGLM, DeepSeek",
        "训练目标": "预测下一个 token (Causal LM)",
        "擅长": "文本生成 (对话、写作、代码)",
        "生成能力": "强 (ChatGPT 等都是这一类)",
    },
    {
        "name": "T5 (Encoder-Decoder)",
        "代表": "T5, BART, mT5",
        "训练目标": "Seq2Seq: 把输入文本转成输出文本",
        "擅长": "翻译、摘要等转换任务",
        "生成能力": "中等",
    },
]
for p in paradigms:
    print(f"\n  · {p['name']}")
    print(f"      代表模型: {p['代表']}")
    print(f"      训练目标: {p['训练目标']}")
    print(f"      擅长: {p['擅长']}")

print("\n  【关键】当前主流对话大模型 (ChatGPT/Qwen/DeepSeek) 都是 Decoder-only!")
print("        原因: 生成任务 + 大规模并行训练 + 易于扩展 → Scaling Law 加持")

# ============================================================
# 5. LLM 的三个标志性能力
# ============================================================
# 【核心概念】现代 LLM 最让人震撼的三个"涌现现象"
print("\n【5】LLM 的标志性能力")

abilities = [
    ("涌现能力 Emergent Abilities",
     "模型规模到一定程度后突然出现的能力 (如: 多步推理、指令遵循)。\n"
     "      小模型完全不会，大模型突然就会了——这就是'涌现'。"),
    ("上下文学习 In-Context Learning (ICL)",
     "不给示例(zero-shot)或给几个示例(few-shot)就能完成新任务，\n"
     "      而且不需要更新模型参数——这在预训练时代是不可想象的。"),
    ("思维链 Chain-of-Thought (CoT)",
     "让模型'一步步思考'再给答案，准确率大幅提升 (尤其数学/推理)。\n"
     "      例如提示: '请一步一步推理后给出答案'。"),
]
for name, desc in abilities:
    print(f"\n  · {name}")
    print(f"      {desc}")

# ============================================================
# 6. Scaling Law (缩放定律) —— LLM 时代的"摩尔定律"
# ============================================================
# 【核心概念】OpenAI 在 2020 年发现的经验规律：
#   模型性能 ≈ f(参数量 N, 数据量 D, 计算量 C)
#   三者按比例放大，损失函数会以幂律方式下降 (可预测)。
#
#   含义: 想要更强的模型 → 加参数 + 加数据 + 加算力，方向明确。
#   这就是为什么 GPT-3 (175B)、GPT-4 (推测万亿级) 越做越大。
print("\n【6】Scaling Law (缩放定律)")
print("  公式简化: Loss ≈ A / N^α + B / D^β + C₀")
print("  含义: 参数 N ↑、数据 D ↑ → 损失 ↓ (性能提升可预测)")
print("  → 这是大模型'大力出奇迹'的理论依据")

# 演示一个简化的 Scaling Law 曲线
print("\n  模拟: 参数量 vs 性能(预测准确率)")
for params_b in [0.1, 1, 10, 100, 1000]:
    # 简化的幂律: acc = a - b * N^(-c)，仅作示意
    acc = 0.95 - 0.5 * (params_b ** -0.25)
    print(f"    {params_b:>7.1f}B 参数  →  准确率 {acc*100:>5.1f}%")

# ============================================================
# 7. 中文开源大模型生态 (2024-2025)
# ============================================================
print("\n【7】主流中文开源大模型 (随时在更新)")
models_cn = [
    ("Qwen (通义千问)",   "阿里",   "开源生态最完整, Qwen2.5/Qwen3 系列, 中英文能力强"),
    ("DeepSeek",          "深度求索", "MoE 架构, 推理能力突出, 训练成本低"),
    ("ChatGLM",           "智谱",   "GLM 系列结构, 工具调用强"),
    ("Baichuan",          "百川",   "中文语料占比高, 适用中文场景"),
    ("Yi",                "零一万物", "01.ai 出品, 综合性能强"),
    ("Llama 3 (中文版)",  "Meta",   "国际主流, 社区中文微调版多"),
]
for name, org, feature in models_cn:
    print(f"  · {name:22s} [{org}]  {feature}")

# ============================================================
# 8. 学习路径建议
# ============================================================
print("\n【8】后续学习路径")
print("-" * 60)
roadmap = [
    ("第2课", "Tokenizer 分词",        "理解文本如何变成模型输入"),
    ("第3课", "Transformer 注意力",    "理解模型内部的核心计算"),
    ("第4课", "预训练模型实战",        "用 HuggingFace 调用真实大模型"),
    ("第5课", "Prompt 提示工程",       "不训练参数也能完成任务"),
    ("第6课", "LoRA 微调",             "低成本定制专属模型"),
    ("第7课", "RAG 检索增强",          "让模型基于私有知识回答"),
]
for lesson, topic, goal in roadmap:
    print(f"  {lesson} · {topic:18s} → {goal}")
print("-" * 60)

print("\n" + "=" * 60)
print("第1课小结")
print("=" * 60)
print("""
  [OK] 语言模型本质: 预测下一个 token 的概率分布
  [OK] 发展主线: n-gram(计数) → Word2Vec(向量) → Transformer(注意力)
  [OK] 三大范式: BERT(理解) / GPT(生成, 主流) / T5(转换)
  [OK] 三个关键能力: 涌现 / ICL(上下文学习) / CoT(思维链)
  [OK] Scaling Law: 参数 + 数据 → 性能，"大力出奇迹"
  [OK] 中文开源生态: Qwen / DeepSeek / ChatGLM / Baichuan ...

  下一课: Tokenizer 分词原理 —— 模型看到的"文本"长什么样。
""")

# ============================================================
# 练习 (可选)
# ============================================================
# 1. 把语料改成中文(无空格)，思考 n-gram 该如何处理？(提示: 需要"分词")
# 2. 自己想 5 个任务，分别属于 BERT / GPT / T5 哪类范式更合适？
#    例: 机器翻译 → T5; 情感分析 → BERT 或 GPT; 写诗 → GPT
# 3. 思考题: 既然 Scaling Law 说"加参数就行"，为什么还要做 LoRA、RAG？
#    (提示: 加参数成本极高，且私有知识无法靠预训练覆盖)
