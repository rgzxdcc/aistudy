import os
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
from openai import OpenAI
from fastapi import FastAPI
import openai
from pydantic import BaseModel


client_local = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")

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