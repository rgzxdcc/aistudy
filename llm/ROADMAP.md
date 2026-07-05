# 大语言模型 (LLM) 学习路线图

> 目标：从理论到实战，系统掌握大语言模型的核心原理与应用开发能力。
>
> 前置基础：建议先完成 `ml/` 机器学习模块与 `dl/` 深度学习基础。

## 学习阶段

### 模块 1 · LLM 通识与发展史 `01_LLM通识与发展史.py`
建立对大模型的整体认知，避免"只见树木不见森林"：
- 从 n-gram、Word2Vec 到 Transformer 的演进
- 三大预训练范式：编码器(BERT) / 解码器(GPT) / 编码-解码(T5)
- 涌现能力、Scaling Law、上下文学习(ICL)
- 中文开源生态：Qwen、ChatGLM、DeepSeek、Baichuan

### 模块 2 · Tokenizer 分词原理 `02_Tokenizer分词原理.py`
理解模型"看到"的文本是什么形态：
- 字符 / 词 / 子词(subword) 三种粒度
- BPE、WordPiece、Unigram 算法原理
- SentencePiece 与中文分词特点
- token id、特殊 token、padding 与 attention mask

### 模块 3 · Transformer 与注意力机制 `03_Transformer与注意力.py`
LLM 的"心脏"——注意力机制的原理与实现：
- Self-Attention：Q/K/V 与缩放点积
- Multi-Head Attention：多视角关注
- 位置编码：绝对 / 相对 / RoPE
- 前馈网络、残差连接、LayerNorm
- 手写一个最小 Transformer 块

### 模块 4 · 预训练模型实战 `04_预训练模型实战.py`
使用 HuggingFace 生态加载与调用预训练模型：
- pipeline 一行代码完成 NLP 任务
- AutoModel / AutoTokenizer 自动加载
- 文本生成原理：自回归解码、采样策略(temperature / top-k / top-p)
- 中文情感分析、文本分类、文本生成实战

### 模块 5 · Prompt 提示工程 `05_Prompt提示工程.py`
不训练参数，仅靠"会说话"就能激发模型能力：
- Zero-shot / Few-shot / Chain-of-Thought
- 角色设定、格式约束、示例设计
- 输出结构化：JSON 输出、ReAct 模式
- 接入 OpenAI 兼容 API 的标准调用方式

### 模块 6 · 模型微调 LoRA `06_模型微调LoRA.py`
让开源模型适配垂直领域的高效方法：
- 全参微调 vs 参数高效微调 (PEFT)
- LoRA 原理：低秩矩阵分解
- 指令数据集格式 (Alpaca / ShareGPT)
- 使用 peft / transformers 训练流程
- 量化基础：4bit / 8bit (QLoRA)

### 模块 7 · RAG 检索增强生成 `07_RAG检索增强.py`
让 LLM 基于私有知识准确回答：
- 为什么需要 RAG：幻觉问题与知识时效
- 文本切分 (chunking) 策略
- 向量嵌入 (Embedding) 与相似度检索
- 召回 → 重排 → 生成 完整链路
- 最小 RAG 系统搭建实战

### 模块 8 · Agent 智能体 `08_Agent智能体.py`
让 LLM 从"只会说"进化为"会做事"的智能体：
- Agent 五大组件：感知 / 大脑 / 记忆 / 工具 / 行动
- ReAct 范式：Thought → Action → Observation 循环
- Function Calling：结构化工具调用
- 记忆系统：短期 / 长期 / 工作记忆
- 规划策略与多 Agent 协作
- 手写可运行的 ReActAgent (无需 API)

---

## 学习建议

1. **按顺序学**：1→3 是理论地基，4→8 是工程实战，环环相扣。
2. **跑通示例**：每课都设计了可运行的最小代码，先跑通再读源码。
3. **理论结合实践**：注释中【核心概念】标记的段落务必理解，代码只是落地。
4. **配套环境**：建议使用 GPU 环境 (Colab / 本地显卡) 跑大模型，CPU 仅适合演示。

## 约定

- 每个知识点以「带详细中文注释的示例代码」形式呈现，可边读边运行。
- 文件命名：`NN_主题.py`，如 `01_LLM通识与发展史.py`。
- 大模型相关依赖：`transformers`、`peft`、`sentence-transformers`、`accelerate`、`bitsandbytes`。

## 依赖安装

```bash
pip install transformers sentencepiece accelerate
pip install peft bitsandbytes            # 微调课用
pip install sentence-transformers faiss-cpu   # RAG 课用
pip install langchain langchain-openai        # Agent 课用
```
