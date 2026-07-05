# -*- coding: utf-8 -*-
"""
模块8 第7课：RAG 检索增强生成
================================
本课目标：
  1. 理解为什么需要 RAG (幻觉 + 时效 + 私有知识)
  2. 掌握 RAG 的完整链路: 切分 → 嵌入 → 检索 → 生成
  3. 用 numpy 手写一个可运行的最小 RAG 系统
  4. 了解生产级 RAG 框架 (LangChain / LlamaIndex)

RAG (Retrieval-Augmented Generation) 的本质：
    先从知识库"检索"相关片段，再把片段塞进 prompt 让模型"生成"答案。
    = 搜索引擎 + 大模型

【一句话】"给大模型配一个外接大脑"
"""

import os
import sys
import numpy as np

# Windows 控制台默认 GBK 编码，强制改成 UTF-8 以支持特殊符号
sys.stdout.reconfigure(encoding="utf-8")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
np.random.seed(42)

print("=" * 60)
print("第7课：RAG 检索增强生成")
print("=" * 60)

# ============================================================
# 1. 为什么需要 RAG？
# ============================================================
# 【核心概念】LLM 的三大痛点，RAG 都能解决：
print("\n【1】LLM 的痛点 vs RAG 的解药")
print("-" * 65)
pain_points = [
    ("幻觉 Hallucination", "模型会一本正经地胡说八道",
     "RAG 提供真实来源 → 模型基于事实回答"),
    ("知识时效",          "训练截止后的事模型不知道",
     "RAG 实时检索最新文档 → 时效自由"),
    ("私有知识",          "企业内部文档模型没见过",
     "RAG 接入企业知识库 → 私域问答"),
    ("不可溯源",          "不知道答案从哪来",
     "RAG 可返回引用来源 → 可信"),
]
for pain, desc, cure in pain_points:
    print(f"  · {pain:18s} {desc:25s}")
    print(f"    {'→ RAG:':18s} {cure}")
print("-" * 65)

# ============================================================
# 2. RAG vs 微调：什么时候用哪个？
# ============================================================
print("\n【2】RAG vs 微调 选型指南")
print("-" * 60)
comparison = [
    ("适用场景",  "事实型/知识型问答",      "风格/格式/能力调整"),
    ("数据需求",  "文档库 (无需标注)",      "高质量指令对 (需标注)"),
    ("更新成本",  "极低 (重建索引即可)",    "高 (需重新训练)"),
    ("可解释性",  "高 (能给出处)",          "低 (黑盒)"),
    ("解决幻觉",  "✓ 直接缓解",             "✗ 只能间接"),
    ("改变风格",  "✗ 不能",                 "✓ 直接改变"),
]
print(f"  {'维度':14s} {'RAG':28s} {'微调'}")
print("  " + "-" * 60)
for dim, rag, ft in comparison:
    print(f"  {dim:14s} {rag:28s} {ft}")
print("-" * 60)
print("  【经验】大多数企业场景，RAG 优先；微调补充。两者可叠加。")

# ============================================================
# 3. RAG 完整流程 (5 个核心步骤)
# ============================================================
print("\n【3】RAG 完整流程")
print("-" * 60)
steps = [
    ("1. 文档加载",   "Load",     "读取 PDF/Word/HTML/Markdown"),
    ("2. 文本切分",   "Split",    "切成 chunk (200~500 字)"),
    ("3. 向量嵌入",   "Embed",    "每个 chunk → 一个向量"),
    ("4. 相似检索",   "Retrieve", "用问题向量找最相关的 chunk"),
    ("5. 增强生成",   "Generate", "把检索结果塞进 prompt，让 LLM 回答"),
]
for step, en, desc in steps:
    print(f"  {step} ({en:9s})  {desc}")
print("-" * 60)
print("\n  数据流:")
print("  ┌────────────┐    ┌──────────┐    ┌──────────────┐")
print("  │ 用户问题   │ →  │ 向量化   │ →  │ 向量数据库   │")
print("  └────────────┘    └──────────┘    └──────┬───────┘")
print("                                            │ top-k")
print("                                            ▼")
print("  ┌────────────┐    ┌──────────┐    ┌──────────────┐")
print("  │ LLM 生成   │ ←  │ 拼接     │ ←  │ 相关 chunks  │")
print("  └────────────┘    └──────────┘    └──────────────┘")

# ============================================================
# 4. 文本切分策略 (Chunking)
# ============================================================
# 【核心概念】切分质量直接决定 RAG 效果
#   - 太长: 一个 chunk 包含多个主题 → 检索精度低
#   - 太短: 语义不完整 → 模型难以理解
#   - 推荐: 200~500 字符，重叠 50~100 字符
print("\n【4】文本切分策略")
print("-" * 60)
print("""
  固定长度切分 (最简单):
    chunk_size = 300 字符
    overlap    = 50  字符   ← 防止切断语义

  递归切分 (LangChain 默认, 推荐):
    优先按 \\n\\n → \\n → 句号 → 空格 → 字符 递归切分
    尽量保持语义完整

  语义切分 (进阶):
    用 embedding 检测语义断点，按主题切分
""")

# 实现一个简单的固定长度切分函数
def split_text(text: str, chunk_size: int = 50, overlap: int = 10):
    """固定长度 + 重叠的文本切分"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap     # 步长 = chunk_size - overlap
    return chunks

demo_text = "大语言模型是人工智能的重要分支。Transformer是其核心架构。RAG技术让模型具备实时知识检索能力。"
chunks = split_text(demo_text, chunk_size=30, overlap=10)
print(f"  切分演示 (chunk_size=30, overlap=10):")
print(f"  原文 ({len(demo_text)} 字): {demo_text}")
for i, c in enumerate(chunks):
    print(f"  chunk[{i}]: {c}")

# ============================================================
# 5. 向量嵌入 (Embedding) 原理
# ============================================================
# 【核心概念】Embedding 模型把文本映射到稠密向量 (如 768/1024/1536 维)
#   关键特性: 语义相近的文本 → 向量距离也近
#   常见模型:
#     - OpenAI text-embedding-3-small (1536维)
#     - BGE-M3 (中文优秀, 1024维)
#     - Qwen3-Embedding-0.6B
print("\n【5】向量嵌入 (Embedding)")
print("-" * 60)
print("""
  # 用 sentence-transformers (本地，免费)
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
  vectors = model.encode(['你好', 'Hello', '天气真好'])
  # vectors.shape = (3, 512)

  # 用 OpenAI 兼容 API (云上，付费但质量高)
  from openai import OpenAI
  client = OpenAI(api_key="sk-xxx", base_url="...")
  resp = client.embeddings.create(
      model="text-embedding-v3",
      input="你好世界"
  )
  vec = resp.data[0].embedding   # 1024 维向量
""")

# ============================================================
# 6. 相似度检索 (核心算法)
# ============================================================
# 【核心概念】向量相似度的三种度量:
#   ① 余弦相似度 (最常用): 看方向，不看长度
#   ② 点积 (内积): 同时考虑方向和长度
#   ③ 欧氏距离 (L2): 距离越小越相似
print("\n【6】相似度检索算法")

def cosine_similarity(v1, v2):
    """余弦相似度: cos(θ) = (A·B) / (|A|·|B|)"""
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot / (norm + 1e-9)

# 用随机向量模拟: 让"问题"和"相关文档"向量接近
print("\n  余弦相似度演示:")
v_question =  np.array([0.8, 0.6, 0.1])           # "如何训练模型"
v_relevant = np.array([0.75, 0.55, 0.15])         # "模型训练方法" (相关)
v_unrelated = np.array([0.1, 0.2, 0.95])          # "今天的午餐" (无关)
print(f"    sim(问题, 相关文档)   = {cosine_similarity(v_question, v_relevant):.3f}")
print(f"    sim(问题, 无关文档)   = {cosine_similarity(v_question, v_unrelated):.3f}")
print("  → 相关文档相似度更高，会被检索出来")

# ============================================================
# 7. 手写一个完整的最小 RAG 系统 (numpy 实现)
# ============================================================
# 【动手实践】用纯 numpy 实现可运行的 RAG (无需任何外部依赖)
#   模拟 embedding: 用"词袋 + 哈希"代替真实模型
print("\n【7】最小 RAG 系统 (numpy 手写，可立即运行!)")
print("-" * 60)

class MiniEmbedder:
    """用词袋 + 哈希模拟 embedding (演示用，生产请用 BGE/OpenAI)"""
    def __init__(self, dim=64):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        """把文本编码成 dim 维向量 (词袋哈希)"""
        vec = np.zeros(self.dim)
        for word in text:
            # 用字符的 hash 映射到向量维度
            idx = hash(word) % self.dim
            vec[idx] += 1
        # L2 归一化 (便于余弦相似度)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

class MiniVectorDB:
    """超简单的向量数据库 (本质就是列表 + 余弦相似度)"""
    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks = []          # 存原始文本
        self.vectors = []         # 存对应向量

    def add(self, texts):
        """添加文档"""
        for t in texts:
            self.chunks.append(t)
            self.vectors.append(self.embedder.encode(t))

    def search(self, query, top_k=2):
        """检索最相关的 top_k 个 chunk"""
        q_vec = self.embedder.encode(query)
        sims = [cosine_similarity(q_vec, v) for v in self.vectors]
        # 按相似度降序排
        ranked = sorted(zip(sims, self.chunks), key=lambda x: -x[0])
        return ranked[:top_k]

# ====== 知识库 (模拟企业文档) ======
knowledge_base = [
    "Qwen2.5 是阿里通义千问推出的开源大语言模型系列。",
    "LoRA 是一种参数高效的微调方法，只训练少量参数。",
    "Transformer 架构由 Google 在 2017 年提出，是 LLM 的基础。",
    "RAG 技术通过检索外部知识库来增强大模型的回答能力。",
    "DeepSeek 是深度求索推出的开源 MoE 架构大模型。",
    "注意力机制让模型能关注序列中重要的部分。",
]

# 构建索引
embedder = MiniEmbedder(dim=128)
db = MiniVectorDB(embedder)
db.add(knowledge_base)
print(f"  知识库已建立: {len(knowledge_base)} 条文档")

# 提问并检索
queries = [
    "通义千问是什么?",
    "怎么微调模型?",
    "怎么减少大模型的幻觉?",
]
for q in queries:
    print(f"\n  问题: {q}")
    results = db.search(q, top_k=2)
    for score, chunk in results:
        print(f"    [{score:.3f}] {chunk}")

# ============================================================
# 8. 把检索结果拼成 Prompt
# ============================================================
print("\n【8】增强生成: 把检索结果喂给 LLM")
print("-" * 60)

def build_rag_prompt(query, retrieved_chunks):
    """构造 RAG prompt 模板"""
    context = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(retrieved_chunks))
    prompt = f"""请根据以下参考资料回答问题。如果资料中没有答案，请说明。

【参考资料】
{context}

【问题】
{query}

【回答】"""
    return prompt

# 用上面第一个问题的检索结果
query = queries[0]
top_chunks = [c for _, c in db.search(query, top_k=2)]
prompt = build_rag_prompt(query, top_chunks)
print(prompt)
print("-" * 60)
print("  → 把这个 prompt 发给任意 LLM，就能得到基于知识库的回答")

# ============================================================
# 9. 生产级 RAG 框架对比
# ============================================================
print("\n【9】生产级 RAG 框架与组件")
print("-" * 65)
components = [
    ("编排框架",    "LangChain / LlamaIndex",    "整套 RAG 流程编排"),
    ("向量数据库",  "Chroma / FAISS / Milvus",   "存储+检索 embedding"),
    ("Embedding",   "BGE-M3 / Qwen3-Emb / OpenAI","文本转向量"),
    ("重排 Rerank", "bge-reranker / cohere",      "二次精排，提升精度"),
    ("文档解析",    "Unstructured / PyMuPDF",     "PDF/Word → 文本"),
    ("评估",        "RAGAS / TruLens",            "评估 RAG 质量"),
]
for cat, tools, use in components:
    print(f"  · {cat:14s} {tools:32s} {use}")

# ============================================================
# 10. RAG 进阶技巧
# ============================================================
print("\n【10】RAG 进阶技巧 (提升效果)")
advanced = [
    ("HyDE 假设文档",     "先让 LLM 生成一个'假答案'，再用它去检索 (效果更好)"),
    ("Multi-Query",       "让 LLM 把问题改写成多个版本，分别检索再合并"),
    ("Rerank 重排",       "先召回 top-20，再用精排模型选 top-3"),
    ("Hybrid Search",     "向量检索 + 关键词检索(BM25) 混合，互补"),
    ("Parent-Child",      "检索小 chunk，返回它所属的大 chunk (上下文完整)"),
    ("Self-RAG",          "让模型自己判断要不要检索、检索结果相不相关"),
]
for i, (name, desc) in enumerate(advanced, 1):
    print(f"  {i}. {name:18s} {desc}")

# ============================================================
# 11. RAG 常见问题
# ============================================================
print("\n【11】RAG 常见问题排查")
issues = [
    ("检索不准",          "chunk 太长/太短 → 调整切分; 或换更好的 embedding"),
    ("答非所问",          "top-k 太多噪声 → 加 rerank; 或缩小 k"),
    ("模型不引用来源",    "prompt 加约束: '必须引用[编号]'"),
    ("中文检索效果差",    "换成中文 embedding (BGE / Qwen3-Emb)"),
    ("向量库太大",        "用量化 (PQ/SQ); 或换 Milvus 等专用库"),
    ("实时性要求高",      "用 ANN 索引 (HNSW / IVF) 替代暴力搜索"),
]
for issue, solution in issues:
    print(f"  · {issue:18s} → {solution}")

print("\n" + "=" * 60)
print("第7课小结")
print("=" * 60)
print("""
  [OK] RAG 解决: 幻觉 / 时效 / 私有知识 / 可溯源
  [OK] 完整流程: Load → Split → Embed → Retrieve → Generate
  [OK] 切分: chunk_size 200~500, overlap 50~100
  [OK] 相似度: 余弦相似度最常用
  [OK] 本课手写了一个可运行的 MiniRAG (numpy 即可跑)
  [OK] 生产栈: LangChain + Chroma + BGE + LLM

  恭喜！你已完成 LLM 学习路线全部 7 课 🎉
  下一步建议: 选一个真实场景 (客服/知识库/代码助手) 完整实现一遍。
""")

# ============================================================
# 练习 (可选)
# ============================================================
# 1. 把 MiniRAG 的 embedder 换成真实的 sentence-transformers，对比效果
# 2. 用 LangChain + Chroma 搭建一个 PDF 问答系统
# 3. 思考: 为什么有时候"直接问 LLM"反而比"RAG"效果好？
#    (提示: 知识库质量差 / 通用问题 / 模型已经会了)
# 4. 进阶: 实现一个带 rerank 的两阶段 RAG
