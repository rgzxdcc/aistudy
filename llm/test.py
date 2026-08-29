from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ==================== 模型切换 ====================
# 日常练习用 Qwen（小、快），需要对照书中效果时切换为 Phi-3
# MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"      # 约 1GB，首次运行自动下载
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  # 约 7.6GB，书中指定模型
# ==================================================

# 设备选择：优先 Apple GPU（mps），不支持时回退 CPU
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
)
model = model.to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 创建流水线
generator = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    return_full_text = False,
    max_new_tokens = 500,
    do_sample=False
)

# 构造对话；pipeline 会自动应用模型的对话模板（无需手动 apply_chat_template）
# messages = [{"role": "user", "content": "Create a funny joke about chickens."}]
messages = [{"role": "user", "content": "如皋有哪些特色美食？"}]
# 生成输出
output = generator(messages)
print(output[0]['generated_text'])
