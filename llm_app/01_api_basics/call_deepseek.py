import os
from dotenv import load_dotenv
from openai import OpenAI

# 若设置了deepseek，优先用
load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY", "ollama"),
                base_url=os.getenv("DEEPSEEK_BASE_URL", "http://localhost:11434/v1"))

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "system", "content":"You are a helpful assistant"},
        {"role": "user", "content":"用一段话解释什么是RAG"},
    ],
    temperature=1.5,
    max_tokens = 500
)

print(response.choices[0].message.content)


# # 流式输出
# for chunk in stream:
#     delta = chunk.choices[0].delta.content or ""
#     print(delta, end="", flush=True)
# print()
