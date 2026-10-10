import os
os.environ["GRPC_VERBOSITY"] = "ERROR" # 屏蔽gRPC提示信息
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from embedding import recursive_chunking
from dotenv import load_dotenv
from openai import OpenAI

COLLECTION_NAME = "rag_collection"
DB_FILENAME = "milvus_rag.db"
VECTOR_DIM = 512

def rag_pipeline(query, top_k = 3):
    # 加载本地向量数据库
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    db_filename = os.path.join(cur_dir, DB_FILENAME)
    client = MilvusClient(db_filename)
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    # 加载collection，若不存在则新建
    if client.has_collection(COLLECTION_NAME):
        client.load_collection(COLLECTION_NAME)
    else:
        client.create_collection(COLLECTION_NAME, dimension=VECTOR_DIM, metric_type="COSINE")
        file_path = os.path.join(cur_dir, "2026-09-26.md")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        texts = recursive_chunking(text, chunk_size=200)
        vectors = model.encode(texts)
        data = [{"id": i, "vector": vector, "text":chunk} 
            for i, (vector, chunk) in enumerate(zip(vectors, texts))]
        client.insert(COLLECTION_NAME, data)
    
    # 组织查询
    q_vector = model.encode(query)
    results = client.search(
        COLLECTION_NAME,
        data = [q_vector],
        limit = top_k,
        output_fields=["text"]
    )

    # 打印向量数据库查询结果，生成上下文
    context = []
    for rank, result in enumerate(results[0]):
        print(f"   Top {rank+1}: ID={result['id']}, 相似度={result['distance']:.4f}, 文本={result['entity']['text']}")
        context.append(result["entity"]["text"])
    context_str = "\n\n".join(f"[{i+1}] {c}" for i,c in enumerate(context))

    # 将问题与查询结果均送入LLM，进行最终结果生成
    # 使用 deepseek
    load_dotenv()
    llm_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY", "ollama"),
                base_url=os.getenv("DEEPSEEK_BASE_URL", "http://localhost:11434/v1"))

    q_context = f"""基于以下参考资料回答问题。请严格根据参考资料内容作答，如果资料中没有提及，请明确回答“参考资料中未提及相关信息”，禁止捏造事实。

    【参考资料】：
    {context_str}

    【用户问题】：
    {query}

    【回答】："""

    response = llm_client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content":"You are a helpful assistant"},
            {"role": "user", "content":q_context},
        ],
        extra_body={"thinking":{"type":"disabled"}}
    )
    print("final query result from llm: ", response.choices[0].message.content)
    return response.choices[0].message.content

# ================================ 指定执行 ================================
if __name__=="__main__":
    rag_pipeline("大模型的上下文长度是指什么")
    # rag_pipeline("今天天气怎么样")