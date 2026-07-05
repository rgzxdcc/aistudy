# -*- coding: utf-8 -*-
"""
模块8 第5课：Prompt 提示工程
================================
本课目标：
  1. 理解为什么"会写 prompt"是 LLM 时代的核心技能
  2. 掌握 Zero-shot / Few-shot / CoT 三大范式
  3. 学会结构化输出 (JSON / ReAct)
  4. 用 OpenAI 兼容 API 调用任意大模型

Prompt Engineering (提示工程) 的本质：
    通过精心设计的"输入文本"，引导大模型产出符合预期的输出。
    不改模型参数，只改输入——成本最低、效果立竿见影。

【一句话】"Ask in the right way, and the model will answer better."
"""

import os
import sys
import json

# Windows 控制台默认 GBK 编码，强制改成 UTF-8 以支持特殊符号
sys.stdout.reconfigure(encoding="utf-8")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("第5课：Prompt 提示工程")
print("=" * 60)

# ============================================================
# 1. 为什么 Prompt 如此重要？
# ============================================================
# 【核心概念】同一个模型，不同 prompt → 截然不同的输出质量
#   LLM 是"概率续写器"，它倾向于生成与你 prompt 风格一致的文本。
#   prompt 模糊 → 输出随机；prompt 精准 → 输出可控。
print("\n【1】Prompt 为什么重要？(对比示例)")
print("-" * 60)
bad_good = [
    ("❌ 差的 Prompt", "写一首诗"),
    ("✅ 好的 Prompt", "请用五言绝句形式写一首关于秋天的诗，要求押韵，包含'落叶'意象"),
]
for label, prompt in bad_good:
    print(f"  {label}:")
    print(f"      {prompt}")
print("-" * 60)
print("  → 模型不知道你想要什么风格/主题/长度，只能瞎猜")

# ============================================================
# 2. Prompt 的基本结构
# ============================================================
# 【核心概念】一个完整的生产级 prompt 通常包含 4 个角色块：
print("\n【2】Prompt 的标准结构")
print("-" * 60)
print("""
  ┌─────────────────────────────────────┐
  │ 1. 系统指令 (System)                │  设定角色、规则、约束
  │    "你是一个严谨的中文翻译助手"      │
  ├─────────────────────────────────────┤
  │ 2. 上下文 (Context)                 │  提供背景知识
  │    "以下是产品说明书: ..."           │
  ├─────────────────────────────────────┤
  │ 3. 示例 (Examples / Few-shot)       │  展示期望的输入输出
  │    "输入: hello → 输出: 你好"        │
  ├─────────────────────────────────────┤
  │ 4. 用户问题 (User)                  │  实际任务
  │    "请翻译: Good morning"            │
  └─────────────────────────────────────┘
""")
print("-" * 60)

# ============================================================
# 3. 三大 Prompt 范式
# ============================================================
print("\n【3】三大 Prompt 范式")

# ----- 3.1 Zero-shot -----
# 【核心概念】Zero-shot: 不给示例，直接问。依赖模型预训练能力。
print("\n  【3.1】Zero-shot (零样本)")
zero_shot_prompt = """请判断以下评论的情感倾向(正面/负面)：
评论：这个手机拍照清晰，电池耐用。
情感："""
print(f"  Prompt:\n{zero_shot_prompt}")
print("  优点: 简单直接  缺点: 复杂任务容易跑偏")

# ----- 3.2 Few-shot -----
# 【核心概念】Few-shot: 给几个示例，让模型"依葫芦画瓢"
#   这是 In-Context Learning 的核心，无需训练就能学新任务
print("\n  【3.2】Few-shot (少样本)")
few_shot_prompt = """请判断评论的情感倾向(正面/负面)：

评论：这家餐厅服务态度很差。 → 负面
评论：电影特效太棒了！      → 正面
评论：物流速度还可以。      → 正面

评论：这个手机拍照清晰，电池耐用。 → """
print(f"  Prompt:\n{few_shot_prompt}")
print("  优点: 准确率显著提升  缺点: 占用 token，示例要精心设计")

# ----- 3.3 Chain-of-Thought (CoT) -----
# 【核心概念】CoT: 让模型"一步一步思考"，对推理类任务提升巨大
#   关键短语: "Let's think step by step" / "请一步一步推理"
print("\n  【3.3】Chain-of-Thought (思维链)")
cot_prompt = """问题：小明有 5 个苹果，给了小红 2 个，又买了 3 个，请问现在有几个？
请一步一步推理后给出答案。"""
print(f"  Prompt:\n{cot_prompt}")
print("  模型输出 (示意):")
print("""    步骤1: 小明原有 5 个苹果
    步骤2: 给小红 2 个 → 5 - 2 = 3 个
    步骤3: 又买 3 个 → 3 + 3 = 6 个
    答案: 6 个""")
print("  → 数学/逻辑/代码任务，CoT 几乎是必杀技")

# ============================================================
# 4. Prompt 设计的 6 个实用技巧
# ============================================================
print("\n【4】Prompt 设计 6 大技巧")
techniques = [
    ("角色设定",   "开头明确身份: '你是一位有10年经验的Python工程师'"),
    ("任务明确",   "动词开头: '请总结' / '请翻译' / '请列出'"),
    ("格式约束",   "明确输出格式: '用 JSON 返回' / '不超过 100 字'"),
    ("分隔符",     "用 ''' 或 <tag> 分隔指令和内容，避免混淆"),
    ("反向示例",   '告诉模型"不要做什么"同样重要'),
    ("自检机制",   "要求模型检查自己的答案: '请确认答案合理后输出'"),
]
for i, (name, tip) in enumerate(techniques, 1):
    print(f"  {i}. {name:8s}  {tip}")

# ============================================================
# 5. 结构化输出：让模型吐 JSON
# ============================================================
# 【核心概念】生产环境最常用的技巧：让模型输出机器可解析的结构化数据
print("\n【5】结构化输出: JSON")
print("-" * 60)
json_prompt = """请从以下文本中提取信息，并以 JSON 格式返回：
文本：张三，1990年生，目前在阿里巴巴担任高级算法工程师。

要求：
1. 只返回 JSON，不要任何额外文字
2. 字段: name(姓名), birth_year(出生年份), company(公司), position(职位)

输出格式示例：
{"name": "李四", "birth_year": 1985, "company": "腾讯", "position": "产品经理"}
"""
print(json_prompt)
print("-" * 60)

# 模拟模型返回的 JSON (实际由 LLM 生成)
mock_response = '{"name": "张三", "birth_year": 1990, "company": "阿里巴巴", "position": "高级算法工程师"}'
print("  模型返回 (示意):")
print(f"    {mock_response}")
parsed = json.loads(mock_response)
print(f"  解析后可直接使用: {parsed['name']} 在 {parsed['company']}")
print("  → 配合 JSON 模式 / function calling 更稳定")

# ============================================================
# 6. ReAct 模式：让模型"行动"
# ============================================================
# 【核心概念】ReAct = Reasoning + Acting
#   让模型交替"思考"和"调用工具"，适合 Agent 场景
print("\n【6】ReAct 模式 (推理+行动)")
print("-" * 60)
print("""
  Thought: 我需要查询今天北京的天气
  Action: search_weather
  Action Input: {"city": "北京", "date": "今天"}
  Observation: 北京今天 25°C，晴
  Thought: 我已经获取到天气信息，可以回答用户了
  Answer: 北京今天天气晴朗，气温 25°C。
""")
print("-" * 60)
print("  → 这是 LangChain / AutoGPT 等 Agent 框架的基础范式")

# ============================================================
# 7. 调用 OpenAI 兼容 API (通用模板)
# ============================================================
# 【核心概念】OpenAI 的 API 格式已成为事实标准
#   几乎所有厂商 (Qwen/DeepSeek/GLM/Moonshot) 都提供 OpenAI 兼容接口
#   只需改 base_url 和 api_key 即可切换模型
print("\n【7】OpenAI 兼容 API 调用模板")
print("-" * 60)
print("""
  # 安装: pip install openai
  from openai import OpenAI

  client = OpenAI(
      api_key="sk-xxx",                              # 你的 API Key
      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云
      # base_url="https://api.deepseek.com",          # DeepSeek
      # base_url="https://api.openai.com/v1",         # OpenAI 官方
  )

  response = client.chat.completions.create(
      model="qwen-plus",         # 模型名
      messages=[
          {"role": "system", "content": "你是一位幽默的助手"},
          {"role": "user", "content": "讲个程序员笑话"},
      ],
      temperature=0.7,
      max_tokens=200,
  )
  print(response.choices[0].message.content)
""")
print("-" * 60)

# ============================================================
# 8. 常用国内大模型 API 速查
# ============================================================
print("\n【8】国内主流大模型 API 速查")
print(f"  {'厂商':10s} {'base_url':45s} {'代表模型'}")
print("  " + "-" * 75)
api_providers = [
    ("阿里通义", "dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus / qwen-max"),
    ("DeepSeek", "api.deepseek.com",                            "deepseek-chat / deepseek-coder"),
    ("智谱GLM",  "open.bigmodel.cn/api/paas/v4",                "glm-4 / glm-4-flash"),
    ("Moonshot", "api.moonshot.cn",                             "moonshot-v1-8k / 32k"),
    ("百度文心", "qianfan.baidubce.com",                        "ernie-bot-4 / ernie-speed"),
    ("OpenAI",   "api.openai.com/v1",                           "gpt-4o / gpt-4o-mini"),
]
for org, url, model in api_providers:
    print(f"  {org:10s} {url:45s} {model}")

# ============================================================
# 9. Prompt 调试实战 (本地模拟)
# ============================================================
# 由于环境可能没有 API Key，这里用一个"模拟模型"演示完整调用流程
print("\n【9】Prompt 调试流程演示 (本地模拟)")
print("-" * 60)

def mock_chat(messages, **kwargs):
    """模拟一个 LLM 的响应 (实际项目中替换为真实 API 调用)"""
    last_msg = messages[-1]["content"]
    # 简单规则模拟: 包含"翻译"就翻译，包含"情感"就分析
    if "翻译" in last_msg:
        return "Good morning 的中文翻译是: 早上好"
    elif "情感" in last_msg or "正面" in last_msg:
        if "差" in last_msg or "糟糕" in last_msg:
            return "负面"
        return "正面"
    elif "JSON" in last_msg:
        return '{"result": "模拟结果"}'
    return "这是一个模拟回复 (实际项目请替换为真实 API)"

# 测试不同 prompt
test_cases = [
    ("Zero-shot 情感分析", [
        {"role": "user", "content": "判断情感(正面/负面): 这个产品太糟糕了"}
    ]),
    ("翻译任务", [
        {"role": "system", "content": "你是翻译助手"},
        {"role": "user", "content": "请翻译: Good morning"}
    ]),
]
for name, msgs in test_cases:
    print(f"\n  测试: {name}")
    for m in msgs:
        print(f"    [{m['role']}] {m['content']}")
    reply = mock_chat(msgs)
    print(f"    [assistant] {reply}")

print("\n" + "=" * 60)
print("第5课小结")
print("=" * 60)
print("""
  [OK] Prompt 决定输出质量: 越具体、越清晰 → 越可控
  [OK] 标准结构: System + Context + Examples + User
  [OK] 三大范式: Zero-shot / Few-shot / CoT (推理任务必备)
  [OK] 结构化输出: 让模型吐 JSON，便于程序处理
  [OK] ReAct: Reasoning + Action，Agent 框架的基础
  [OK] OpenAI 兼容 API: 一套代码切换所有主流模型

  下一课: LoRA 微调 —— 让开源模型变成你的"专属助手"。
""")

# ============================================================
# 练习 (可选)
# ============================================================
# 1. 写一个 prompt，让模型把任意中文新闻转成 3 个要点 + 情感标签
# 2. 对比 Zero-shot 和 Few-shot 在某个任务上的效果差异 (用真实 API)
# 3. 思考题: 如果模型总是输出格式不对的 JSON，该怎么办？
#    (提示: ① 用 JSON Mode ② 加更强约束 ③ 用 function calling ④ 后处理修复)
