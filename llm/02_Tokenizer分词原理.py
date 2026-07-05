# -*- coding: utf-8 -*-
"""
模块8 第2课：Tokenizer 分词原理
================================
本课目标：
  1. 理解文本到模型输入的转换全过程
  2. 掌握三种分词粒度及各自优缺点
  3. 从零手写 BPE 算法，看清"子词"是怎么来的
  4. 学会使用 HuggingFace tokenizers 库

模型从不直接"读"文字，它只认数字 (token id)：
    原始文本 → [Tokenizer] → token 序列 → [词表查表] → id 序列 → [模型]
                              ↑ 本课重点
"""

import os
import sys
from collections import Counter, defaultdict

# Windows 控制台默认 GBK 编码，强制改成 UTF-8 以支持特殊符号
sys.stdout.reconfigure(encoding="utf-8")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("第2课：Tokenizer 分词原理")
print("=" * 60)

# ============================================================
# 1. 为什么不能直接用"字符"或"整词"？
# ============================================================
# 【核心概念】三种分词粒度对比
print("\n【1】分词粒度对比")
print("-" * 70)
print(f"  {'粒度':12s} {'示例 (\"unhappiness\")':25s} {'优点':18s} {'缺点'}")
print("-" * 70)
granularity = [
    ("字符 char", "['u','n','h','a','p','p','i','n','e','s','s']",
     "词表极小(<300)", "序列太长，语义弱"),
    ("整词 word", "['unhappiness']",
     "语义完整", "词表爆炸(百万级)，OOV 严重"),
    ("子词 subword", "['un', 'happ', 'iness']",
     "平衡长度与词表", "需要训练算法"),
]
for g, ex, pro, con in granularity:
    print(f"  {g:12s} {ex:25s} {pro:18s} {con}")
print("-" * 70)
print("  【结论】现代 LLM 普遍采用「子词 subword」粒度，平衡长度与语义")

# 【关键术语】OOV (Out-Of-Vocabulary)
#   整词分词时，遇到训练时没见过的词(新词、拼写错误、生僻字)就抓瞎，
#   只能映射为 <UNK> (unknown) 符号。子词方法几乎不存在 OOV 问题。

# ============================================================
# 2. 主流子词算法：BPE / WordPiece / Unigram
# ============================================================
# 【核心概念】三大子词算法对比
print("\n【2】三大子词算法概览")
algorithms = [
    ("BPE (Byte-Pair Encoding)",
     "GPT/LLaMA/Qwen 使用",
     "从字符开始，反复合并出现频率最高的相邻 token 对"),
    ("WordPiece",
     "BERT 使用",
     '类似 BPE，但合并标准是"互信息最大化"而非频率'),
    ("Unigram LM",
     "T5/mBART 使用",
     "先有大词表，再逐步删除低收益 token，留概率高的"),
]
for name, used, idea in algorithms:
    print(f"\n  · {name}")
    print(f"      使用者: {used}")
    print(f"      核心思想: {idea}")
print("\n  【重点】BPE 是当前主流 (GPT 系列都用它)，下面我们手写一个")

# ============================================================
# 3. 手写 BPE 算法 —— 从字符到子词
# ============================================================
# 【核心概念】BPE (Byte Pair Encoding) 算法
#   输入: 一堆单词及其出现频次
#   输出: 一个子词词表 + 分词规则
#
#   算法流程:
#     1. 把每个单词拆成字符序列，末尾加 </w> 标记词尾
#     2. 统计所有相邻字符对的频次
#     3. 选频次最高的字符对合并成一个新 token，加入词表
#     4. 重复 2-3 直到达到目标词表大小
print("\n【3】手写 BPE 算法 (从字符开始合并)")
print("-" * 60)

def get_pair_stats(vocab):
    """统计所有相邻 token 对的出现频次"""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def merge_pair(pair, vocab):
    """把指定 token 对合并到词表中"""
    new_vocab = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word, freq in vocab.items():
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = freq
    return new_vocab

# 训练语料: 单词 -> 频次 (注意每个字符用空格分隔, </w> 表示词尾)
vocab = {
    "l o w </w>": 5,
    "l o w e r </w>": 2,
    "n e w e s t </w>": 6,
    "w i d e s t </w>": 3,
}
print("  初始词表 (字符级别):")
for w, f in vocab.items():
    print(f"    {w}  ×{f}")

# 执行 5 次合并，观察子词如何形成
merges = []
num_merges = 5
print(f"\n  开始执行 {num_merges} 次 BPE 合并:")
for i in range(num_merges):
    pairs = get_pair_stats(vocab)
    if not pairs:
        break
    best = pairs.most_common(1)[0]      # 频次最高的 token 对
    print(f"    第{i+1}轮: 合并 {best[0]}  (频次={best[1]})")
    vocab = merge_pair(best[0], vocab)
    merges.append(best[0])

print("\n  合并后的词表:")
for w, f in vocab.items():
    print(f"    {w}  ×{f}")
print("  → 注意 'es'、'est'、'er' 等子词已经形成！这就是 BPE 的产物")

# 用学到的合并规则对新词分词
def encode_bpe(word, merges):
    """对未知单词应用 BPE 合并规则"""
    symbols = list(word) + ["</w>"]
    for pair in merges:
        i = 0
        while i < len(symbols) - 1:
            if (symbols[i], symbols[i + 1]) == pair:
                symbols[i:i + 2] = ["".join(pair)]
            else:
                i += 1
    return symbols

test_words = ["lowest", "newer", "wider"]
print("\n  应用 BPE 规则对新词分词:")
for w in test_words:
    print(f"    {w:8s} → {encode_bpe(w, merges)}")
# 'lowest' 会被分成 ['low', 'est', '</w>']，因为它见过 'low' 和 'est'
# 这就是 BPE 的妙处: 既能处理已知词，也能用子词组合处理新词！

# ============================================================
# 4. 特殊 Token (Special Tokens)
# ============================================================
# 【核心概念】每个模型都有一组特殊 token，不参与普通分词，但有专门功能
print("\n【4】特殊 Token 速查")
special_tokens = [
    ("[PAD]",    "填充符",   "把短句子补齐到等长，便于批量处理"),
    ("[UNK]",    "未知符",   "词表中没有的 token 用它代替 (现代 LLM 几乎不用)"),
    ("[BOS]",    "序列起始", "Beginning Of Sequence，标记句子开始"),
    ("[EOS]",    "序列终止", "End Of Sequence，模型生成到这里就停"),
    ("[CLS]",    "分类符",   "BERT 用，该位置的向量用于分类任务"),
    ("[SEP]",    "分隔符",   "BERT 用，分隔两个句子"),
    ("<|im_start|>", "ChatML起始", "Qwen/ChatGLM 用，标记对话角色开始"),
]
for tok, name, use in special_tokens:
    print(f"  {tok:18s}  {name:8s}  {use}")

# ============================================================
# 5. Padding 与 Attention Mask
# ============================================================
# 【核心概念】批处理时，不同句子长度不一，必须 padding 到等长。
#   attention_mask 告诉模型哪些位置是真实的，哪些是 padding (应忽略)
print("\n【5】Padding 与 Attention Mask")
sentences = ["你好", "今天天气真不错", "我"]
# 假设用字符级分词，最大长度 = 6
max_len = 6
print(f"  原始句子: {sentences}")
print(f"  最大长度: {max_len}")
print()
for s in sentences:
    ids = list(s)                            # 模拟分词
    mask = [1] * len(ids)                    # 真实 token mask=1
    pad_len = max_len - len(ids)
    ids = ids + ["[PAD]"] * pad_len          # 补齐
    mask = mask + [0] * pad_len              # padding mask=0
    print(f"    {''.join(s):10s}  input_ids  = {ids}")
    print(f"    {'':12s}attention_mask = {mask}")
print("\n  → attention_mask 让模型在 attention 计算时忽略 [PAD] 位置")

# ============================================================
# 6. 实战：使用 HuggingFace tokenizers
# ============================================================
# 【核心概念】生产环境不会自己写 BPE，而是用现成的库
#   transformers.AutoTokenizer 会根据模型名自动加载对应分词器
print("\n【6】HuggingFace Tokenizer 调用模板 (代码示例)")
print("-" * 60)
print("""
  from transformers import AutoTokenizer

  # 加载分词器 (会自动下载对应模型的词表)
  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

  text = "大语言模型真有趣"
  # 编码: 文本 → token id
  ids = tokenizer(text)["input_ids"]
  # 解码: token id → 文本
  decoded = tokenizer.decode(ids)

  # 查看具体分了哪些 token
  tokens = tokenizer.tokenize(text)
  print(f"原文: {text}")
  print(f"分词: {tokens}")
  print(f"ID:   {ids}")
  print(f"还原: {decoded}")
""")
print("-" * 60)
print("  【运行提示】需要 pip install transformers，并联网下载模型词表")

# ============================================================
# 7. Token 计费与长度限制 (实用知识)
# ============================================================
print("\n【7】实用知识：Token 计费与上下文长度")
facts = [
    "1 token ≈ 4 个英文字符 ≈ 0.75 个英文单词",
    "1 个中文字 ≈ 1~2 个 token (取决于分词器)",
    "API 按 token 数计费 (输入 + 输出都算钱)",
    '上下文长度 = 模型一次能"看到"的最大 token 数 (如 GPT-4 是 128K)',
    "中文相对英文 token 利用率低 → 同样内容中文更贵",
]
for i, f in enumerate(facts, 1):
    print(f"  {i}. {f}")

# 一个简单的 token 数估算函数
def estimate_tokens(text: str) -> int:
    """粗略估算 token 数 (中英文混合场景)"""
    chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    english_chars = len(text) - chinese
    # 中文约 1 字 = 1.5 token; 英文约 4 字符 = 1 token
    return int(chinese * 1.5 + english_chars / 4)

sample = "今天学习了LLM的Tokenizer，感觉很interesting！"
print(f"\n  示例: '{sample}'")
print(f"  估算 token 数 ≈ {estimate_tokens(sample)}")

print("\n" + "=" * 60)
print("第2课小结")
print("=" * 60)
print("""
  [OK] 模型只认数字: 文本 → token → id → 模型
  [OK] 三种粒度: 字符(小但语义弱) / 整词(大且OOV) / 子词(平衡)
  [OK] BPE 算法: 反复合并最高频相邻 token 对
  [OK] 特殊 token: [PAD] [UNK] [BOS] [EOS] 等控制符
  [OK] attention_mask: 标记真实位置，忽略 padding
  [OK] 实战: AutoTokenizer.from_pretrained(模型名)

  下一课: Transformer 与注意力机制 —— LLM 的"心脏"。
""")

# ============================================================
# 练习 (可选)
# ============================================================
# 1. 把语料改成中文，思考 BPE 该如何处理 (提示: 字符就是单字)
# 2. 安装 transformers，加载一个真实 tokenizer，对比 "Hello" 和 "你好" 各占几个 token
# 3. 思考题: 为什么 GPT-4 上下文是 128K，但 API 调用建议别超过 4K？
#    (提示: 1.成本 2.注意力可能稀释 3.延迟)
