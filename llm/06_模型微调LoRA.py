# -*- coding: utf-8 -*-
"""
模块8 第6课：模型微调 LoRA
================================
本课目标：
  1. 理解为什么需要微调，以及全参微调的痛点
  2. 掌握 LoRA 的低秩分解原理 (用 numpy 演示)
  3. 了解 PEFT 库与指令数据集格式
  4. 跑通 QLoRA 完整训练流程模板

【一句话】预训练模型是"通才"，微调让它成为你专属的"专才"。
"""

import os
import sys
import numpy as np

# Windows 控制台默认 GBK 编码，强制改成 UTF-8 以支持特殊符号
sys.stdout.reconfigure(encoding="utf-8")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
np.random.seed(42)

print("=" * 60)
print("第6课：模型微调 LoRA")
print("=" * 60)

# ============================================================
# 1. 为什么需要微调？
# ============================================================
# 【核心概念】预训练 → 通用能力; 微调 → 专属能力
#   预训练模型懂"人类语言"，但不懂你的业务 (客服/医疗/法律...)
#   微调 = 用少量领域数据，让模型适配具体任务
print("\n【1】微调的必要性")
stages = [
    ("预训练 Pre-training",   "海量无标注文本 (TB级)", "学会语言通用规律"),
    ("指令微调 SFT",          "万级对话数据",          "学会听指令、对话"),
    ("RLHF / DPO",            "人类偏好数据",          "对齐人类价值观"),
    ("领域微调",              "你自己的业务数据",      "成为垂直领域专家"),
]
print(f"  {'阶段':22s} {'数据':22s} {'目标'}")
print("  " + "-" * 60)
for stage, data, goal in stages:
    print(f"  {stage:22s} {data:22s} {goal}")

# ============================================================
# 2. 全参微调 vs 参数高效微调 (PEFT)
# ============================================================
# 【核心概念】全参微调的痛点
#   一个 7B 模型 = 70 亿参数，全参微调需要:
#     - 优化器状态 (Adam: 2x 参数大小)
#     - 梯度 (1x)
#     - 模型本身 (1x)
#   → 总共需要 ~28B 参数的显存 (约 56GB fp16) → 普通人玩不起
print("\n【2】全参微调 vs PEFT")
print("-" * 60)
print(f"  {'方法':18s} {'可训练参数':12s} {'显存需求':12s} {'效果'}")
print("-" * 60)
methods = [
    ("Full Fine-tuning", "100%",       "极高",       "最好"),
    ("Prefix Tuning",    "<1%",        "低",         "中"),
    ("P-Tuning v2",      "<1%",        "低",         "中"),
    ("LoRA",             "0.1~1%",     "低",         "接近全参"),
    ("QLoRA (4bit+LoRA)","0.1~1%",     "极低",       "接近全参"),
]
for m, params, mem, effect in methods:
    print(f"  {m:18s} {params:12s} {mem:12s} {effect}")
print("-" * 60)
print("  【结论】LoRA / QLoRA 是当前个人开发者的首选方案")

# ============================================================
# 3. LoRA 原理：低秩矩阵分解 (核心！)
# ============================================================
# 【核心概念】LoRA (Low-Rank Adaptation) 的核心思想：
#   "模型微调时的权重变化 ΔW 是低秩的"
#
#   原始:  Y = W · X           W: d×d  (巨大的矩阵)
#   微调后: Y' = (W + ΔW) · X
#
#   LoRA 把 ΔW 分解成两个小矩阵的乘积:
#       ΔW = B · A              A: d×r,  B: r×d  (r 远小于 d)
#
#   训练时: W 冻结不变，只训练 A 和 B → 参数量从 d² 降到 2dr
print("\n【3】LoRA 原理：低秩矩阵分解")
print("-" * 60)

# 用 numpy 直观演示 LoRA 的参数节省
d = 4096         # 模型隐藏维度 (7B 模型典型值)
r = 8            # LoRA 秩 (常用 8/16/32)
full_params = d * d
lora_params = 2 * d * r
print(f"  原始权重 W: {d}×{d} = {full_params:,} 个参数")
print(f"  LoRA A:     {d}×{r} = {d*r:,}")
print(f"  LoRA B:     {r}×{d} = {d*r:,}")
print(f"  LoRA 总参数:        {lora_params:,}")
print(f"  参数比例: {lora_params/full_params*100:.3f}%  (节省 {(1-lora_params/full_params)*100:.3f}%)")

# 实际矩阵计算演示
print(f"\n  矩阵运算演示 (用小维度 d=8, r=2):")
d_small, r_small = 8, 2
W = np.random.randn(d_small, d_small) * 0.1     # 原始冻结权重 (d, d)
A = np.random.randn(r_small, d_small) * 0.01    # LoRA A: (r, d) 训练
B = np.zeros((d_small, r_small))                 # LoRA B: (d, r) 初始化为0
X = np.random.randn(d_small, 1)                  # 输入 (d, 1)

# 原始前向
Y_orig = W @ X
# LoRA 前向 (ΔW = B @ A, 形状 (d,r)@(r,d)=(d,d))
Y_lora = (W + B @ A) @ X
# 注意: B 初始为 0，所以 ΔW = 0，前向输出与原始一致 (训练起点不动)
print(f"    初始时 (B=0): 输出差异 = {np.abs(Y_orig - Y_lora).max():.6f}  (应为 0)")

# 模拟训练后 B 变化
B = np.random.randn(d_small, r_small) * 0.05
Y_lora_trained = (W + B @ A) @ X
print(f"    训练后:       输出差异 = {np.abs(Y_orig - Y_lora_trained).max():.6f}  (已改变)")
print("  → LoRA 通过学习 A、B 来模拟全参数微调的效果")

# ============================================================
# 4. LoRA 应用在哪些层？
# ============================================================
# 【核心概念】Transformer 中有多个线性层，LoRA 通常加在注意力层
#   原论文: 只加在 Q、V 上效果就好
#   实践中: Q、K、V、O 全加 + FFN 也能加 → 效果更好但参数多
print("\n【4】LoRA 加在 Transformer 的哪些层？")
print("-" * 60)
target_modules = [
    ("q_proj, v_proj",         "LoRA 原论文方案",        "最省参数，够用"),
    ("q_proj, k_proj, v_proj, o_proj", "注意力全加",     "效果更好"),
    ("+ gate_proj, up_proj, down_proj", "加上 FFN",     "效果最佳，参数多"),
]
for modules, name, note in target_modules:
    print(f"  · {modules:35s} → {note}")
print("-" * 60)

# ============================================================
# 5. 指令数据集格式
# ============================================================
# 【核心概念】SFT (Supervised Fine-Tuning) 需要有标注的对话数据
#   主流格式: Alpaca / ShareGPT
print("\n【5】指令数据集格式")
print("-" * 60)
print("""
  【Alpaca 格式】(单轮对话)
  {
    "instruction": "把下面句子翻译成英文",
    "input":       "今天天气真好",
    "output":      "The weather is nice today."
  }

  【ShareGPT 格式】(多轮对话)
  {
    "conversations": [
      {"from": "human",  "value": "你好"},
      {"from": "gpt",    "value": "你好！有什么可以帮你？"},
      {"from": "human",  "value": "解释一下 Transformer"},
      {"from": "gpt",    "value": "Transformer 是..."}
    ]
  }
""")
print("-" * 60)

# ============================================================
# 6. 量化基础 (为 QLoRA 做铺垫)
# ============================================================
# 【核心概念】把 fp16 (16bit) 压成 4bit，显存需求降 4 倍
#   ① NF4: NormalFloat 4-bit，QLoRA 论文提出，最优
#   ② 推理时反量化回 fp16 做计算
#   ③ 配合 LoRA → QLoRA: 7B 模型 6GB 显存就能微调！
print("\n【6】量化基础 (4bit/8bit)")
quantization = [
    ("FP32",  "32 bit",  "原始精度，最大显存"),
    ("FP16",  "16 bit",  "训练默认，半精度"),
    ("INT8",  "8 bit",   "显存减半，精度损失小"),
    ("NF4",   "4 bit",   "QLoRA 专用，显存 1/4，精度可接受"),
]
print(f"  {'格式':8s} {'位数':8s} {'说明'}")
for fmt, bits, desc in quantization:
    print(f"  {fmt:8s} {bits:8s} {desc}")

# 模拟量化过程 (直观感受精度损失)
print("\n  量化精度损失演示:")
original = np.array([0.123, -0.456, 0.789, -0.012], dtype=np.float32)
scale = original.max() / 127          # INT8 缩放因子
quantized = np.round(original / scale).astype(np.int8)
dequantized = quantized * scale
print(f"    原始:    {original}")
print(f"    反量化:  {dequantized}")
print(f"    误差:    {np.abs(original - dequantized).max():.6f}  (可接受)")

# ============================================================
# 7. 完整 LoRA 训练代码模板 (HuggingFace PEFT)
# ============================================================
print("\n【7】QLoRA 训练完整代码模板")
print("-" * 60)
print("""
  # pip install peft transformers trl bitsandbytes accelerate
  import torch
  from datasets import Dataset
  from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
  from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
  from trl import SFTTrainer

  # 1. 加载 tokenizer
  model_name = "Qwen/Qwen2-0.5B"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  tokenizer.pad_token = tokenizer.eos_token

  # 2. 4bit 量化加载模型 (省显存)
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      load_in_4bit=True,            # 4bit 量化
      device_map="auto",
      torch_dtype=torch.float16,
  )
  model = prepare_model_for_kbit_training(model)

  # 3. 配置 LoRA
  lora_config = LoraConfig(
      r=8,                                          # 秩 (常用 8/16)
      lora_alpha=16,                                # 缩放因子 (一般 = 2r)
      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
      lora_dropout=0.05,
      bias="none",
      task_type="CAUSAL_LM",
  )
  model = get_peft_model(model, lora_config)
  model.print_trainable_parameters()
  # 输出: trainable params: 1,234,567 || all params: 500,000,000 || 0.25%

  # 4. 准备数据集 (Alpaca 格式)
  data = [
      {"instruction": "翻译成英文", "input": "你好", "output": "Hello"},
      # ... 你的训练数据
  ]
  def format_prompt(sample):
      return f"指令: {sample['instruction']}\\n输入: {sample['input']}\\n输出: {sample['output']}"
  dataset = Dataset.from_list(data).map(lambda x: {"text": format_prompt(x)})

  # 5. 训练
  trainer = SFTTrainer(
      model=model,
      train_dataset=dataset,
      peft_config=lora_config,
      formatting_func=lambda x: x["text"],
      args=TrainingArguments(
          output_dir="./lora_output",
          num_train_epochs=3,
          per_device_train_batch_size=4,
          learning_rate=2e-4,        # LoRA 学习率比全参大 (1e-4 ~ 5e-4)
          save_steps=100,
          logging_steps=10,
      ),
  )
  trainer.train()

  # 6. 保存 LoRA 权重 (只有几十 MB!)
  model.save_pretrained("./my_lora_adapter")
""")
print("-" * 60)

# ============================================================
# 8. LoRA 权重的加载与合并
# ============================================================
print("\n【8】使用训练好的 LoRA")
print("-" * 60)
print("""
  方式一: 单独加载 adapter (推荐，灵活)
  ───────────────────────────────────────
  from peft import PeftModel
  base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
  model = PeftModel.from_pretrained(base, "./my_lora_adapter")

  方式二: 合并到基础模型 (推理更快，但不可逆)
  ───────────────────────────────────────
  model = model.merge_and_unload()
  model.save_pretrained("./my_merged_model")

  方式三: 多 LoRA 切换 (一个底座 + 多个 adapter)
  ───────────────────────────────────────
  # 适合: 同一个底座服务多个垂直场景
  model.load_adapter("./lora_customer_service", adapter_name="客服")
  model.load_adapter("./lora_medical", adapter_name="医疗")
  model.set_adapter("客服")    # 一键切换
""")
print("-" * 60)

# ============================================================
# 9. 微调常见陷阱
# ============================================================
print("\n【9】微调常见陷阱")
pitfalls = [
    ("学习率太大",         "LoRA 推荐 1e-4~5e-4，全参 1e-5~5e-5"),
    ("数据量太少",         "至少 100 条高质量数据，少了容易过拟合"),
    ("数据质量差",         "1000 条垃圾 < 100 条精标 (质量 > 数量)"),
    ("忘加 chat_template", "对话模型必须套用模板，否则学不到对话能力"),
    ("LoRA r 太大",        "r=64 通常足够，过大反而过拟合"),
    ("没保存底座",         "LoRA 权重脱离底座没用，部署时需要两者一起"),
]
for issue, solution in pitfalls:
    print(f"  · {issue:22s} → {solution}")

print("\n" + "=" * 60)
print("第6课小结")
print("=" * 60)
print("""
  [OK] 微调目的: 让通用模型变成专属模型
  [OK] LoRA 核心: ΔW = B·A，参数从 d² 降到 2dr
  [OK] PEFT 库: LoraConfig + get_peft_model 一键启用
  [OK] QLoRA = 4bit 量化 + LoRA，6GB 显存微调 7B
  [OK] 数据格式: Alpaca(单轮) / ShareGPT(多轮)
  [OK] 学习率: LoRA 用 2e-4 (比全参大一个数量级)

  下一课: RAG 检索增强 —— 让模型基于私有知识回答。
""")

# ============================================================
# 练习 (可选)
# ============================================================
# 1. 把 LoRA 的 r 从 8 改成 64，对比参数比例和理论容量
# 2. 准备 50 条你领域的数据 (Alpaca 格式)，跑通 QLoRA 训练
# 3. 思考: 为什么 LoRA 的学习率比全参微调大 10 倍？
#    (提示: 可训练参数少，需要更大的步长才能学到东西)
