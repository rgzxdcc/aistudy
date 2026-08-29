# -*- coding: utf-8 -*-
"""
模块8 第8课：Agent 智能体
================================
本课目标：
  1. 理解什么是 Agent，它与普通 LLM 调用的本质区别
  2. 掌握 Agent 的五大核心组件: 感知 / 大脑 / 记忆 / 工具 / 行动
  3. 手写一个可运行的 ReAct Agent (无需 API Key)
  4. 了解 Function Calling / 规划 / 多 Agent 协作

Agent = LLM + 工具调用 + 循环决策
    普通 LLM 调用: 用户问 → 模型答 (一问一答，到此为止)
    Agent:         用户问 → 模型思考 → 调工具 → 观察 → 再思考 → ... → 最终答
    它能"自主完成任务"，而不是只回答问题。

【一句话】"LLM 是大脑，Agent 是有手有脚、能使用工具的 LLM"
"""

import os
import sys
import json
import re
import math
from datetime import datetime

# Windows 控制台默认 GBK 编码，强制改成 UTF-8 以支持特殊符号
sys.stdout.reconfigure(encoding="utf-8")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("第8课：Agent 智能体")
print("=" * 60)

# ============================================================
# 1. 为什么需要 Agent？
# ============================================================
# 【核心概念】LLM 的局限 → Agent 的价值
print("\n【1】LLM 的局限 vs Agent 的解药")
print("-" * 70)
limitations = [
    ("知识截止",    "训练后的信息不知道",
     "Agent 可调用搜索引擎/数据库 → 实时获取"),
    ("不会算术",    "大数字计算经常出错",
     "Agent 调用计算器工具 → 精确无误"),
    ("无法行动",    "只能输出文本，不能改变世界",
     "Agent 调用 API → 发邮件/写文件/执行代码"),
    ("长任务易跑偏", "复杂多步任务一次生成易错",
     "Agent 分步规划+反馈循环 → 持续修正"),
    ("无状态",      "每次对话独立，记不住历史",
     "Agent 有记忆系统 → 跨会话连续"),
]
for pain, desc, cure in limitations:
    print(f"  · {pain:14s} {desc:28s}")
    print(f"    {'→ Agent:':14s} {cure}")
print("-" * 70)

# ============================================================
# 2. Agent 的五大核心组件
# ============================================================
# 【核心概念】一个完整 Agent 的标准架构
print("\n【2】Agent 五大核心组件")
print("-" * 70)
components = [
    ("感知 Perception",  "接收用户输入、环境信号、工具返回结果"),
    ("大脑 Brain",        "LLM 作为推理引擎，负责思考与决策"),
    ("记忆 Memory",       "短期(对话历史) + 长期(向量库/知识库)"),
    ("工具 Tools",        "搜索、计算器、代码执行、API、数据库"),
    ("行动 Action",       "调用工具、生成回复、修改环境"),
]
for name, desc in components:
    print(f"  · {name:20s}  {desc}")
print("-" * 70)

print("""
  架构图:
                    ┌─────────────┐
                    │   用户输入  │  ← 感知
                    └──────┬──────┘
                           ▼
    ┌──────────┐    ┌─────────────┐    ┌──────────┐
    │  记忆    │ ←→ │  LLM 大脑   │ ←→ │  工具集  │
    │ Memory   │    │ (思考/决策) │    │  Tools   │
    └──────────┘    └──────┬──────┘    └────┬─────┘
                           │                │
                           └─────←──────────┘
                           ▼
                    ┌─────────────┐
                    │  输出/行动  │  ← 最终回复或调用工具
                    └─────────────┘
""")

# ============================================================
# 3. ReAct 范式 —— Agent 的核心运行机制
# ============================================================
# 【核心概念】ReAct = Reasoning + Acting (推理 + 行动)
#   让 LLM 交替输出"思考"和"动作"，形成闭环:
#
#   Thought  → 我需要做什么
#   Action   → 调用某个工具
#   Observation → 工具返回的结果
#   Thought  → 根据结果再思考
#   ... (循环直到能回答)
#   Answer   → 最终回复
print("\n【3】ReAct 范式 (Agent 的核心循环)")
print("-" * 70)
print("""
  标准流程 (直到产出 Final Answer):

    用户问题
        ↓
    ┌────────────────────────────────┐
    │ Thought 1: 我需要查询天气       │
    │ Action 1: search_weather(...)  │
    │ Observation 1: 北京 25°C 晴    │
    │                                │
    │ Thought 2: 还需要知道用户日程  │
    │ Action 2: get_calendar(...)    │
    │ Observation 2: 下午3点有会议   │
    │                                │
    │ Thought 3: 信息已足够，回答    │
    │ Final Answer: ...              │
    └────────────────────────────────┘
""")
print("-" * 70)

# ============================================================
# 4. 定义工具集 —— Agent 的"双手"
# ============================================================
# 【动手实践】先定义几个简单工具，Agent 后面会调用它们
print('\n【4】定义工具集 (Agent 的"双手")')
print("-" * 70)

# 工具用 dict 描述，方便 Agent 理解"有什么工具可用"
TOOLS = {}

def register_tool(name, description, func):
    """注册一个工具到全局工具集"""
    TOOLS[name] = {"description": description, "func": func}

# ----- 工具 1: 计算器 -----
def calculator(expression: str) -> str:
    """安全的计算器: 支持基本算术运算"""
    try:
        # 只允许数字和基本运算符 (防止代码注入)
        if not re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', expression):
            return "错误: 表达式包含非法字符"
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"

register_tool(
    "calculator",
    "数学计算器，输入算术表达式(如 '2+3*4')返回结果",
    calculator,
)

# ----- 工具 2: 时间查询 -----
def get_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

register_tool(
    "get_time",
    "获取当前日期和时间",
    get_time,
)

# ----- 工具 3: 模拟天气查询 -----
def search_weather(city: str) -> str:
    """模拟天气查询 (实际项目会调用真实天气 API)"""
    fake_data = {
        "北京": "25°C，晴",
        "上海": "28°C，多云",
        "广州": "32°C，雷阵雨",
        "深圳": "30°C，阴",
    }
    return fake_data.get(city, f"{city}: 暂无天气数据")

register_tool(
    "search_weather",
    "查询指定城市的天气，参数: city(城市名)",
    search_weather,
)

# ----- 工具 4: 模拟知识库检索 -----
def search_kb(query: str) -> str:
    """模拟 RAG 检索 (第7课的内容)"""
    kb = {
        "退货": "7天内无理由退货，请联系客服。",
        "发票": "电子发票会在订单完成后24小时内发送。",
        "配送": "一线城市次日达，其他城市2-3天。",
    }
    for key, val in kb.items():
        if key in query:
            return val
    return f"未找到关于 '{query}' 的信息"

register_tool(
    "search_kb",
    "查询知识库(客服FAQ)，参数: query(查询关键词)",
    search_kb,
)

# 打印工具清单
print(f"  已注册 {len(TOOLS)} 个工具:")
for name, info in TOOLS.items():
    print(f"    · {name:15s}  {info['description']}")

# ============================================================
# 5. 手写一个 ReAct Agent (核心！可运行)
# ============================================================
# 【动手实践】完整的 Agent 循环，无需任何外部 API
#   用"规则路由"模拟 LLM 的思考过程，方便演示 Agent 机制
print("\n【5】手写 ReAct Agent (无需 API，立即可跑)")
print("-" * 70)

class ReActAgent:
    """一个最小但完整的 ReAct Agent

    生产环境中 run_step 里的"决策"应该由 LLM 完成
    (解析 LLM 输出的 Thought/Action)，这里用规则演示流程。
    """

    def __init__(self, tools, max_steps=5):
        self.tools = tools
        self.max_steps = max_steps
        self.memory = []          # 短期记忆: 记录整个推理过程

    def _log(self, role, content):
        """记录到记忆并打印"""
        self.memory.append({"role": role, "content": content})
        print(f"    [{role:11s}] {content}")

    def _decide_action(self, question, observations):
        """决策模块: 决定下一步该做什么

        【生产实现】这里应该:
          1. 构造 prompt (包含问题、历史、工具说明)
          2. 调用 LLM
          3. 解析输出的 Thought / Action
        这里用规则演示，实际项目替换为 LLM 调用即可。
        """
        # 已经有足够信息时，准备最终回答
        if len(observations) >= 1:
            return "FINISH", None

        q = question.lower()
        # 规则 1: 计算任务
        if any(c in question for c in "+-*/") and "计算" in q or \
           any(c in question for c in "+-*/"):
            expr = re.search(r'[\d\s\+\-\*\/\(\)\.]+', question)
            if expr:
                return "calculator", expr.group().strip()
        # 规则 2: 时间
        if "时间" in q or "几点" in q or "日期" in q:
            return "get_time", None
        # 规则 3: 天气
        for city in ["北京", "上海", "广州", "深圳"]:
            if city in question:
                return "search_weather", city
        # 规则 4: 知识库
        for kw in ["退货", "发票", "配送"]:
            if kw in question:
                return "search_kb", kw

        return "FINISH", None

    def _execute_tool(self, tool_name, arg):
        """执行工具调用"""
        if tool_name not in self.tools:
            return f"工具 {tool_name} 不存在"
        func = self.tools[tool_name]["func"]
        if arg is None:
            return str(func())
        return str(func(arg))

    def run(self, question):
        """Agent 主循环: ReAct 的核心"""
        print(f"\n  ▶ 用户问题: {question}")
        self._log("User", question)

        observations = []
        for step in range(1, self.max_steps + 1):
            # ① Thought: 决定下一步
            action, arg = self._decide_action(question, observations)

            if action == "FINISH":
                self._log("Thought", "信息已足够，准备最终回答")
                break

            self._log("Thought", f"我需要调用工具 {action}({arg})")

            # ② Action: 调用工具
            self._log("Action", f"{action}({arg})")

            # ③ Observation: 获取工具返回
            result = self._execute_tool(action, arg)
            observations.append(result)
            self._log("Observation", result)

        # ④ Final Answer
        answer = self._generate_answer(question, observations)
        self._log("Answer", answer)
        return answer

    def _generate_answer(self, question, observations):
        """整合所有观察，生成最终回答"""
        if not observations:
            return "抱歉，我无法回答这个问题。"
        return "根据查询: " + "; ".join(observations)

# 测试 Agent
agent = ReActAgent(TOOLS, max_steps=5)

print("\n  ----- 测试 1: 数学计算 -----")
agent.run("请计算 123 + 456 * 2")

print("\n  ----- 测试 2: 天气查询 -----")
agent.run("北京今天天气怎么样?")

print("\n  ----- 测试 3: 客服知识库 -----")
agent.run("我想了解退货政策")

print("\n  ----- 测试 4: 当前时间 -----")
agent.run("现在几点了?")

# ============================================================
# 6. Function Calling (OpenAI 标准化的工具调用)
# ============================================================
# 【核心概念】OpenAI 的 Function Calling 把 ReAct 的"解析文本"变成"结构化输出"
#   模型直接输出 JSON: {"name": "工具名", "arguments": {...}}
#   比解析 "Action: xxx" 文本稳定得多
print("\n【6】Function Calling (结构化工具调用)")
print("-" * 70)
print("""
  # 定义工具的 schema (告诉模型有什么工具可用)
  tools = [
      {
          "type": "function",
          "function": {
              "name": "get_weather",
              "description": "查询指定城市的天气",
              "parameters": {
                  "type": "object",
                  "properties": {
                      "city": {"type": "string", "description": "城市名"}
                  },
                  "required": ["city"]
              }
          }
      }
  ]

  # 调用 LLM 时传入 tools
  response = client.chat.completions.create(
      model="qwen-plus",
      messages=[{"role": "user", "content": "北京天气如何?"}],
      tools=tools,
  )

  # 模型直接返回结构化的工具调用 (而非文本!)
  tool_call = response.choices[0].message.tool_calls[0]
  # tool_call.function.name == "get_weather"
  # tool_call.function.arguments == '{"city": "北京"}'
  args = json.loads(tool_call.function.arguments)

  # 执行工具 → 把结果回传给模型 → 生成最终回答
  result = get_weather(**args)
  messages.extend([
      response.choices[0].message,
      {"role": "tool", "tool_call_id": tool_call.id, "content": result}
  ])
  final = client.chat.completions.create(model="qwen-plus", messages=messages)
""")
print("-" * 70)
print("  【对比】Function Calling vs ReAct 文本解析:")
print("    ReAct:  模型输出 'Action: get_weather(city=北京)' → 用正则解析 (易错)")
print("    FC:     模型输出 {name, arguments} JSON → 直接结构化 (稳定)")

# ============================================================
# 7. 记忆系统 (Memory)
# ============================================================
# 【核心概念】Agent 需要"记住过去"才能处理连续任务
print("\n【7】记忆系统 (短期 + 长期)")
print("-" * 70)
memory_types = [
    ("短期记忆 Short-term", "当前对话的上下文历史",
     "实现: messages 列表，超长时截断或摘要"),
    ("长期记忆 Long-term", "跨会话的用户偏好、事实",
     "实现: 向量数据库 (参考第7课 RAG)"),
    ("工作记忆 Working",   "当前任务的关键中间变量",
     "实现: Agent 类的 self.state / scratchpad"),
]
print(f"  {'类型':22s} {'存什么':30s} {'怎么实现'}")
print("  " + "-" * 75)
for m, what, how in memory_types:
    print(f"  {m:22s} {what:30s} {how}")
print("-" * 75)

# 演示简单的对话记忆
print("\n  简单对话记忆示例:")
class SimpleChatMemory:
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history

    def add(self, role, content):
        self.history.append({"role": role, "content": content})
        # 超过容量，移除最旧的 (FIFO)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_context(self):
        """返回模型能看到的上下文"""
        return self.history

memory = SimpleChatMemory(max_history=4)
memory.add("user", "我叫张三")
memory.add("assistant", "你好，张三!")
memory.add("user", "我喜欢Python")
memory.add("assistant", "Python 是门好语言!")
# 第 5 条会挤掉第 1 条
memory.add("user", "我几岁了?")
print(f"    记忆容量: {len(memory.history)} 条 (上限 4)")
print(f"    上下文: {memory.get_context()}")
print("  → 有记忆后，Agent 能回答 '我叫什么名字' 等指代问题")

# ============================================================
# 8. 规划策略 (Planning)
# ============================================================
# 【核心概念】复杂任务需要先"规划"再"执行"
print("\n【8】规划策略 (应对复杂任务)")
print("-" * 70)
strategies = [
    ("CoT (Chain-of-Thought)",
     "线性分步思考",
     "简单任务: A→B→C→结论"),
    ("ToT (Tree-of-Thoughts)",
     "树状探索多种可能",
     "每步生成多个候选，评估后选最优分支"),
    ("ReWOO",
     "先规划全部步骤再执行",
     "减少 token 消耗，避免中途幻觉"),
    ("Plan-and-Execute",
     "Planner 出计划 → Executor 逐步执行",
     "LangChain 主推的 Agent 范式"),
    ("Reflection 反思",
     "执行后自我评估、必要时重做",
     "类似人类的'做完检查一遍'"),
]
print(f"  {'策略':28s} {'核心思想':24s} {'适用场景'}")
print("  " + "-" * 75)
for name, idea, scene in strategies:
    print(f"  {name:28s} {idea:24s} {scene}")

# 演示一个简单的 Plan-and-Execute
print("\n  Plan-and-Execute 示例 (写一篇技术报告):")
plan = [
    "1. 搜索相关资料 (调用 search)",
    "2. 整理要点 (LLM 处理)",
    "3. 写初稿     (LLM 生成)",
    "4. 检查修正   (Reflection)",
    "5. 输出报告   (final)",
]
for step in plan:
    print(f"    {step}")
print("  → Planner 一次性出计划，Executor 逐步执行，避免走一步看一步")

# ============================================================
# 9. 多 Agent 协作 (Multi-Agent)
# ============================================================
# 【核心概念】把不同角色的 Agent 组合起来，模拟人类团队
print("\n【9】多 Agent 协作 (团队作战)")
print("-" * 70)
print("""
  场景: 软件开发流程

    ┌──────────────┐
    │ 产品经理 Agent │  ← 需求分析、写 PRD
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ 程序员 Agent │  ← 写代码
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ 测试员 Agent │  ← 写测试用例、找 Bug
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ 审查员 Agent │  ← Code Review
    └──────────────┘

  代表项目:
    · AutoGPT     第一个火爆的开源 Agent (单 Agent 自主循环)
    · MetaGPT     模拟软件公司多角色协作
    · CrewAI      轻量级多 Agent 编排框架
    · AutoGen     微软出品，对话式多 Agent
    · ChatDev     用 Agent 模拟完整软件开发
""")

# ============================================================
# 10. Agent 完整骨架代码 (LangChain 风格)
# ============================================================
print("\n【10】生产级 Agent 骨架 (LangChain 风格)")
print("-" * 70)
print("""
  # pip install langchain langchain-openai
  from langchain.agents import create_tool_calling_agent, AgentExecutor
  from langchain_openai import ChatOpenAI
  from langchain.tools import Tool
  from langchain import hub

  # 1. 定义工具
  def calculator(expr):
      return str(eval(expr))
  tools = [
      Tool(name="calculator", func=calculator, description="数学计算"),
      Tool(name="search", func=lambda q: "...", description="搜索"),
  ]

  # 2. 创建 LLM
  llm = ChatOpenAI(model="qwen-plus", temperature=0)

  # 3. 拉取官方 prompt 模板 (也可以自定义)
  prompt = hub.pull("hwchase17/openai-functions-agent")

  # 4. 创建 Agent
  agent = create_tool_calling_agent(llm, tools, prompt)
  executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

  # 5. 运行!
  result = executor.invoke({"input": "25乘以13等于多少?"})
  print(result["output"])
""")
print("-" * 70)

# ============================================================
# 11. Agent 设计模式速查
# ============================================================
print("\n【11】Agent 设计模式速查")
patterns = [
    ("ReAct",          "思考-行动-观察循环",            "通用、最经典"),
    ("Function Call",  "LLM 直接输出工具调用 JSON",     "OpenAI/Qwen 都支持"),
    ("Plan-Execute",   "先规划再执行",                  "复杂长任务"),
    ("Reflection",     "自我反思后修正",                "代码生成、写作"),
    ("Multi-Agent",    "多角色分工协作",                "软件开发、研究"),
    ("Autonomous",     "给定目标后自主跑到完成",        "AutoGPT 类"),
]
print(f"  {'模式':18s} {'核心思想':30s} {'适用'}")
print("  " + "-" * 70)
for p, idea, scene in patterns:
    print(f"  {p:18s} {idea:30s} {scene}")

# ============================================================
# 12. Agent 落地的关键挑战
# ============================================================
print("\n【12】Agent 落地的关键挑战")
challenges = [
    ("可靠性",    "LLM 可能调用不存在的工具、参数错误",
     "用 Function Calling + 严格 schema 校验"),
    ("死循环",    "Agent 陷入重复调用同一工具",
     "限制 max_steps + 检测重复 Action"),
    ("成本",      "每步都调用 LLM，token 消耗大",
     "用小模型做路由、缓存相似查询"),
    ("延迟",      "多步串行 → 用户等待时间长",
     "Stream 输出 + 并行工具调用"),
    ("幻觉",      "Agent 编造工具结果",
     "强制要求工具返回，禁止 LLM 自填"),
    ("评估难",    "如何评价 Agent 做得好不好?",
     "用 AgentBench / AgentBench 类基准"),
]
print(f"  {'问题':12s} {'现象':35s} {'对策'}")
print("  " + "-" * 75)
for issue, symptom, cure in challenges:
    print(f"  {issue:12s} {symptom:35s} {cure}")

# ============================================================
# 13. 学习路径建议
# ============================================================
print("\n【13】Agent 学习路径")
print("-" * 60)
roadmap = [
    ("第1步", "理解 ReAct 原理",          "本课已实现 → 阅读源码"),
    ("第2步", "用 Function Calling",      "调用 Qwen/OpenAI 的 tools 参数"),
    ("第3步", "上手 LangChain Agent",     "跑通 create_tool_calling_agent"),
    ("第4步", "实现垂直场景 Agent",       "如客服/数据分析/代码助手"),
    ("第5步", "多 Agent 协作",            "学习 CrewAI / AutoGen"),
    ("第6步", "Agent 评估与优化",         "AgentBench / 自建评估集"),
]
for step, topic, action in roadmap:
    print(f"  {step} · {topic:22s} → {action}")
print("-" * 60)

# ============================================================
# 14. 主流 Agent 框架对比
# ============================================================
print("\n【14】主流 Agent 框架对比")
print("-" * 70)
frameworks = [
    ("LangChain",     "通用 LLM 应用框架", "Python/JS",  "生态最全，学习曲线中等"),
    ("LlamaIndex",    "RAG + Agent",       "Python",     "数据连接强，RAG 优秀"),
    ("AutoGen",       "多 Agent 对话",     "Python",     "微软出品，对话式协作"),
    ("CrewAI",        "角色化多 Agent",    "Python",     "轻量、易上手"),
    ("MetaGPT",       "模拟软件公司",      "Python",     "中文社区，多角色"),
    ("OpenAI Assistants", "托管 Agent",    "API",        "免运维，但绑定 OpenAI"),
    ("Dify",          "低代码 Agent 平台", "Python",     "可视化编排"),
]
print(f"  {'框架':18s} {'定位':22s} {'语言':10s} {'特点'}")
print("  " + "-" * 75)
for f, pos, lang, feat in frameworks:
    print(f"  {f:18s} {pos:22s} {lang:10s} {feat}")

print("\n" + "=" * 60)
print("第8课小结")
print("=" * 60)
print("""
  [OK] Agent = LLM + 工具 + 循环决策 (能"自主完成任务")
  [OK] 五大组件: 感知 / 大脑(LLM) / 记忆 / 工具 / 行动
  [OK] ReAct: Thought → Action → Observation 循环
  [OK] Function Calling: 结构化工具调用，比文本解析稳定
  [OK] 记忆: 短期(对话) + 长期(向量库) + 工作记忆
  [OK] 规划: CoT 线性 / ToT 树状 / Plan-Execute
  [OK] 多 Agent: 角色分工，模拟人类团队
  [OK] 本课手写了一个可运行的 ReActAgent (无需 API)

  恭喜！你已完成 LLM + Agent 全部 8 课 🎉
  下一步: 选一个真实场景 (客服/数据分析/个人助手) 完整落地。
""")

# ============================================================
# 练习 (可选)
# ============================================================
# 1. 给 ReActAgent 添加一个新工具 (如 send_email)，并测试调用
# 2. 把 _decide_action 替换为真实 LLM 调用 (用第4课的 transformers 或第5课的 API)
# 3. 实现一个"客服 Agent": 接入 RAG 知识库 + 调用人工坐席工具
# 4. 思考: 为什么 AutoGPT 在生产环境很少用？(提示: 可靠性/成本/可控性)
# 5. 进阶: 用 LangChain 实现一个能查数据库的 Agent (Text-to-SQL)
