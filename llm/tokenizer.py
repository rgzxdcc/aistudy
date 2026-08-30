from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ==================== 模型切换 ====================
# 日常练习用 Qwen（小、快），需要对照书中效果时切换为 Phi-3
# MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"      # 约 1GB，首次运行自动下载
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  # 约 7.6GB，书中指定模型
# ==================================================

# 设备选择：优先 Apple GPU（mps），不支持时回退 CPU
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

# 创建提示词
prompt = "Write an email apologizing to Sarah for the tragic gradening mishap. Explain how it happened.<|assistant|>"
# 对输入提示词进行分词
input_ids = tokenizer(prompt,return_tensors='pt').input_ids.to(DEVICE)

# 生成文本
generations_output = model.generate(input_ids=input_ids, max_new_tokens=20)

# 打印输出
print(tokenizer.decode(generations_output[0]))
print(input_ids)
for id in input_ids[0]:
    print(tokenizer.decode(id))
