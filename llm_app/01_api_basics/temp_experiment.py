import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY", "ollama"), 
                base_url=os.getenv("DEEPSEEK_BASE_URL", "http://localhost:11434/v1"))

def ask(prompt, temp, n=3, **extra):
    for i in range(n):
        r = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content":prompt}],
            temperature=temp,
            max_tokens=1000,
            extra_body={"thinking": {"type": "disabled"}},
            **extra
        )
        print(f"--- T={temp} 第{i+1}次 ---")
        msg = r.choices[0].message
        # print("思考：", getattr(msg, "reasoning_content", None))
        print("回答：",msg.content,  "\n")


# # 实验一，答案明确的对话，观察temperature影响
# ask("中国首都在哪里", 0, n=2)
# ask("中国首都在哪里", 1.5, n=2)

# 实验二，开放式答案，调整temperature，观察结果
ask("给一个卖手冲咖啡的小店起3个店名", 0)
ask("给一个卖手冲咖啡的小店起3个店名", 0.7)
ask("给一个卖手冲咖啡的小店起3个店名", 1.5)
