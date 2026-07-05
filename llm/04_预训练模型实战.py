# -*- coding: utf-8 -*-
"""
模块8 第4课：预训练模型实战 (HuggingFace Transformers)
================================
本课目标：
  1. 掌握 HuggingFace 三大核心 API：pipeline / AutoTokenizer / AutoModel
  2. 理解文本生成的自回归解码过程
  3. 学会调节采样参数 (temperature / top_k / top_p)
  4. 跑通中文情感分析与文本生成

【运行说明】
  本课代码需要 transformers 库 + 网络下载模型。
  首次运行会从 HuggingFace 下载模型 (几百 MB)，请耐心等待。
  无 GPU 也能跑 (只是慢)，建议用 Qwen2-0.5B 这种小模型演示。

  安装: pip install transformers accelerate sentencepiece
"""

import os
import sys

# Windows 控制台默认 GBK 编码，强制改成 UTF-8 以支持特殊符号
sys.stdout.reconfigure(encoding="utf-8")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("第4课：预训练模型实战 (HuggingFace Transformers)")
print("=" * 60)

# ============================================================
# 0. 依赖检查与导入策略
# ============================================================
# 【设计哲学】本课采用"渐进式"演示:
#   - 没有 transformers 库时 → 打印代码模板供阅读
#   - 有库但下载失败时 → 优雅降级，不报错
# 这样无论什么环境都能跑通，教学友好
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    HAS_HF = True
except ImportError:
    HAS_HF = False
    print("  [提示] 未安装 transformers，本课以代码模板形式展示")
    print("        安装命令: pip install transformers accelerate sentencepiece")

print(f"\n  transformers 可用: {HAS_HF}")

# ============================================================
# 1. HuggingFace 生态全景
# ============================================================
print("\n【1】HuggingFace 生态全景")
ecosystem = [
    ("Transformers",  "模型加载/推理/训练的核心库"),
    ("Hub",           "20万+ 模型托管平台 (类似 GitHub for AI)"),
    ("Datasets",      "海量数据集一键加载"),
    ("Tokenizers",    "高性能分词库 (Rust 实现)"),
    ("PEFT",          "参数高效微调 (LoRA 等)"),
    ("Accelerate",    "分布式训练/推理加速"),
    ("Spaces",        "在线 Demo 托管 (类似 Hugging Face 版 Gradio)"),
]
for name, desc in ecosystem:
    print(f"  · {name:14s}  {desc}")

print("\n  【模型命名规则】组织名/模型名-参数规模")
naming_examples = [
    ("Qwen/Qwen2-0.5B",       "通义千问 0.5B (5亿参数，CPU可跑)"),
    ("Qwen/Qwen2-7B-Instruct", "通义千问 7B 指令微调版"),
    ("meta-llama/Llama-3-8B", "Meta Llama3 8B"),
    ("THUDM/chatglm3-6b",     "智谱 ChatGLM3 6B"),
    ("bert-base-chinese",     "BERT 中文版 (编码器，非生成)"),
]
for name, desc in naming_examples:
    print(f"    {name:30s}  {desc}")

# ============================================================
# 2. 最简方式：pipeline 一行搞定
# ============================================================
# 【核心概念】pipeline 是最高层封装：内部自动完成
#   分词 → 模型推理 → 后处理，一行代码出结果
print("\n【2】pipeline: 最高层封装 (一行代码)")
print("-" * 60)
print("""
  # 情感分析 (默认下载 distilbert-base-uncased-finetuned-sst-2)
  from transformers import pipeline

  classifier = pipeline("sentiment-analysis")
  result = classifier("I love learning about LLMs!")
  print(result)
  # [{'label': 'POSITIVE', 'score': 0.9998}]

  # 支持的任务类型:
  #   "text-classification"   文本分类 (情感/主题)
  #   "token-classification"  序列标注 (NER/词性)
  #   "question-answering"    阅读理解
  #   "summarization"         文本摘要
  #   "translation"           机器翻译
  #   "text-generation"       文本生成 (GPT 类)
  #   "fill-mask"             完形填空 (BERT 类)
""")
print("-" * 60)

# 真实运行 pipeline (如果环境允许)
if HAS_HF:
    try:
        print("\n  [实跑] 加载情感分析 pipeline...")
        classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        result = classifier("I love learning about Large Language Models!")
        print(f"  结果: {result}")
    except Exception as e:
        print(f"  [跳过实跑] 模型下载失败或无网络: {type(e).__name__}")
        print("  (这很正常，阅读代码模板即可理解用法)")

# ============================================================
# 3. 进阶方式：AutoTokenizer + AutoModel 分步控制
# ============================================================
# 【核心概念】Auto 系列自动根据模型名选择对应类
#   - AutoTokenizer: 文本 ↔ token id 互转
#   - AutoModel: 加载模型权重做推理
#   - AutoModelForCausalLM: 专门用于 GPT 风格的生成任务
print("\n【3】分步控制: AutoTokenizer + AutoModel")
print("-" * 60)
print("""
  from transformers import AutoTokenizer, AutoModelForCausalLM
  import torch

  model_name = "Qwen/Qwen2-0.5B"   # 0.5B 参数 CPU 也能跑
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      torch_dtype="auto",          # 自动选 dtype (fp16/bf16)
      device_map="auto",           # 自动分配到 GPU/CPU
  )

  # 编码 → 推理 → 解码
  inputs = tokenizer("大语言模型是", return_tensors="pt")
  outputs = model.generate(**inputs, max_new_tokens=30)
  print(tokenizer.decode(outputs[0], skip_special_tokens=True))
""")
print("-" * 60)

# ============================================================
# 4. 文本生成原理：自回归解码
# ============================================================
# 【核心概念】GPT 生成 = 一个 token 一个 token 地"吐"出来
#   每一步:
#     1) 把当前已生成的 token 序列喂给模型
#     2) 模型输出下一个 token 的概率分布 (logits)
#     3) 根据采样策略选一个 token
#     4) 加到序列末尾，重复 1-3 直到 EOS 或达最大长度
print("\n【4】自回归解码 (GPT 的生成过程)")
print("  循环: 输入序列 → 模型 → 预测下一个 token → 拼接 → 再输入...")
print("  示意:")
steps = [
    ("输入",  "大语言"),
    ("生成",  "大语言 模型"),
    ("生成",  "大语言 模型 是"),
    ("生成",  "大语言 模型 是 一种"),
    ("生成",  "大语言 模型 是 一种 人工智能"),
    ("生成",  "大语言 模型 是 一种 人工智能 技术"),
    ("...",   "直到遇到 <EOS> 或达 max_length"),
]
for step, text in steps:
    print(f"    {step:6s}  →  {text}")

# ============================================================
# 5. 采样策略：让生成更多样/更可控
# ============================================================
# 【核心概念】模型输出的不是单一 token，而是概率分布。
#   怎么从分布中"挑" token？有几种策略:
print("\n【5】采样策略详解")
sampling_strategies = [
    ("Greedy (贪心)",      "max_new_tokens",  "每步选概率最高的 → 确定性，但容易重复无聊"),
    ("Beam Search",        "num_beams=4",     "保留 top-k 整体路径，质量高但慢"),
    ("Temperature",        "temperature=0.7", "温度高→分布更平(更多样)，低→更尖锐(保守)"),
    ("Top-k Sampling",     "top_k=50",        "只在概率最高的 k 个里选，过滤长尾噪声"),
    ("Top-p Sampling",     "top_p=0.9",       "累积概率达 p 的候选集合中选 (核采样 Nucleus)"),
]
print(f"  {'策略':22s} {'参数':18s} 说明")
print("  " + "-" * 60)
for s, p, desc in sampling_strategies:
    print(f"  {s:22s} {p:18s} {desc}")

# 直观演示 temperature 对概率分布的影响
print("\n  Temperature 对概率分布的影响 (直观示例):")
import math
def softmax_with_temp(logits, T):
    z = [x / T for x in logits]
    e = [math.exp(i - max(z)) for i in z]
    s = sum(e)
    return [i / s for i in e]

# 假设模型输出 3 个候选 token 的 logits
demo_logits = [2.0, 1.0, 0.1]
print(f"    原始 logits: {demo_logits}")
for T in [0.5, 1.0, 2.0]:
    probs = softmax_with_temp(demo_logits, T)
    print(f"    T={T}: 概率 {[round(p, 3) for p in probs]}")
print("  → T 越小: 越倾向选最高概率的 → 保守、确定")
print("  → T 越大: 分布越平均 → 多样、有创意 (但易跑偏)")

# 推荐组合
print("\n  【常见推荐组合】")
recommendations = [
    "事实问答 / 代码生成:  temperature=0.1, top_p=0.9 (求准)",
    "对话 / 创意写作:      temperature=0.7~0.9, top_p=0.9 (求多样)",
    "Brainstorm 头脑风暴: temperature=1.0, top_k=50, top_p=0.95",
]
for r in recommendations:
    print(f"    · {r}")

# ============================================================
# 6. 完整生成代码示例
# ============================================================
print("\n【6】完整生成代码示例 (Chat 模板)")
print("-" * 60)
print("""
  from transformers import AutoTokenizer, AutoModelForCausalLM
  import torch

  # 1. 加载模型 (Qwen2-0.5B-Instruct 是对话微调版)
  model_name = "Qwen/Qwen2-0.5B-Instruct"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(
      model_name, torch_dtype="auto", device_map="auto"
  )

  # 2. 构造对话消息 (OpenAI ChatML 风格)
  messages = [
      {"role": "system", "content": "你是一个耐心的AI老师"},
      {"role": "user",   "content": "用一句话解释什么是Transformer"},
  ]

  # 3. 应用 chat 模板 (不同模型格式不同，自动处理)
  text = tokenizer.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=True
  )
  inputs = tokenizer([text], return_tensors="pt").to(model.device)

  # 4. 生成
  outputs = model.generate(
      **inputs,
      max_new_tokens=100,
      temperature=0.7,
      top_p=0.9,
      do_sample=True,             # 启用采样 (否则贪心)
      pad_token_id=tokenizer.eos_token_id,
  )
  # 5. 解码 (去掉输入部分，只看新生成)
  response = tokenizer.batch_decode(
      outputs[:, inputs["input_ids"].shape[1]:],
      skip_special_tokens=True,
  )[0]
  print(response)
""")
print("-" * 60)

# ============================================================
# 7. 本地运行常见问题
# ============================================================
print("\n【7】常见问题与解决")
issues = [
    ("下载模型慢/失败",       "设置 HF_ENDPOINT=https://hf-mirror.com 用国内镜像"),
    ("显存不足 OOM",          "用更小模型 / device_map='auto' / load_in_4bit=True"),
    ("中文乱码",              "确保加载的是中文模型，tokenizer 自带 vocab"),
    ("生成重复",              "调高 temperature 或加 repetition_penalty=1.1"),
    ("不支持 chat_template",  "老模型没有对话模板，需要手动拼 prompt"),
    ("CPU 太慢",              "换 Qwen2-0.5B 这种小模型，或用 Colab T4 免费 GPU"),
]
for issue, solution in issues:
    print(f"  · {issue:24s} → {solution}")

# ============================================================
# 8. 试运行真实模型 (可选)
# ============================================================
if HAS_HF:
    print("\n【8】尝试加载真实模型 (Qwen2-0.5B)")
    print("  正在加载... (首次约 1GB 下载)")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        model_name = "Qwen/Qwen2-0.5B-Instruct"
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        messages = [{"role": "user", "content": "你好，请用一句话自我介绍"}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok([text], return_tensors="pt").to(mdl.device)
        out = mdl.generate(**inputs, max_new_tokens=50, pad_token_id=tok.eos_token_id)
        resp = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        print(f"  模型回复: {resp}")
    except Exception as e:
        print(f"  [跳过] {type(e).__name__}: {str(e)[:80]}")
        print("  → 这是网络/环境问题，不影响理解本课内容")

print("\n" + "=" * 60)
print("第4课小结")
print("=" * 60)
print("""
  [OK] HuggingFace 三件套: pipeline / AutoTokenizer / AutoModel
  [OK] 命名规则: 组织/模型-规模-Instruct
  [OK] 文本生成 = 自回归解码 (一次吐一个 token)
  [OK] 采样策略: temperature / top_k / top_p 控制多样性与质量
  [OK] chat_template: 不同模型对话格式不同，用 apply_chat_template 自动处理
  [OK] 国内镜像: HF_ENDPOINT=https://hf-mirror.com

  下一课: Prompt 提示工程 —— 不训练参数也能玩转大模型。
""")

# ============================================================
# 练习 (可选)
# ============================================================
# 1. 用 pipeline 跑一个中文情感分析 (model="uer/roberta-base-finetuned-jd-binary-chinese")
# 2. 加载 Qwen2-0.5B-Instruct，分别用 temperature=0.1 和 1.0 各生成 3 次，观察差异
# 3. 思考: 为什么 do_sample=False 时 temperature 不起作用？
#    (提示: 贪心解码不看概率分布，永远选 argmax)
