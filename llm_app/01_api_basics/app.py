import os
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
from openai import OpenAI
from fastapi import FastAPI
import openai
from pydantic import BaseModel

# 将ollama的网址改为配置读取，解决在Docker中访问错误问题
client_local = OpenAI(api_key="ollama", base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))

load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("DEEPSEEK_BASE_URL"))

class ChatRequest(BaseModel):
    messages: list

app = FastAPI()

# 调用deepseek进行输出
@app.post("/chat")
def chat(req: ChatRequest):
    response = client.chat.completions.create(
        model = "deepseek-v4-flash",
        messages = req.messages
    )
    return {"reply": response.choices[0].message.content}

# 调用本地模型（qwen3:8b）进行输出
@app.post("/chat/local")
def chat_local(req : ChatRequest):
    response = client_local.chat.completions.create(
        model = "qwen3:8b",
        messages = req.messages
    )
    return {"reply": response.choices[0].message.content}

# 流式输出
@app.post("/chat_stream")
async def chat_stream(req: ChatRequest):
    def stream():
        stream = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages = req.messages,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return StreamingResponse(stream(), media_type="text/event-stream")

# Docker重新烧制命令
# Docker build -t llm-chat .

# Docker停止旧容器
# Docker ps
# Docker stop <容器ID>

# Docker启动新容器（跑云端模型）
# docker run --rm --env-file /Users/xuwen/Documents/dcc_study/aistudy/.env -p 8000:8000 llm-chat

# Docker跑本地模型运行命令
# docker run --rm --env-file /Users/xuwen/Documents/dcc_study/aistudy/.env \
#   -e OLLAMA_BASE_URL=http://host.docker.internal:11434/v1 \
#   -p 8000:8000 llm-chat

# 终端连接验证： chat
# curl -X POST http://127.0.0.1:8000/chat \
#   -H "Content-Type: application/json" \
#   -d '{"messages":[{"role":"user","content":"用一句话解释什么是Docker"}]}'

# 终端连接验证： chat_stream
# curl -N -X POST http://127.0.0.1:8000/chat_stream \
#   -H "Content-Type: application/json" \
#   -d '{"messages":[{"role":"user","content":"从1数到100"}]}'

# 终端连接验证： chat_local
# curl -X POST http://127.0.0.1:8000/chat/local \
#   -H "Content-Type: application/json" \
#   -d '{"messages":[{"role":"user","content":"用一句话解释什么是Docker"}]}'