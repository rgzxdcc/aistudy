import os
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from embedding import recursive_chunking

COLLECTION_NAME = "rag_collection"
DB_FILENAME = "milvus_rag.db"
VECTOR_DIM = 512

# 创建向量数据库，并查询
def create_milvus():

    # 初始化客户端
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    db_filename = os.path.join(cur_dir, DB_FILENAME)
    client = MilvusClient(db_filename)

    # 创建collection
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    client.create_collection(COLLECTION_NAME, dimension=VECTOR_DIM, metric_type="COSINE")
    print(client.describe_collection(COLLECTION_NAME))

    # 插入数据
    file_path = os.path.join(cur_dir, "2026-09-26.md")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    texts = recursive_chunking(text, chunk_size=200)
    print("len of texts", len(texts))

    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    vectors = model.encode(texts)

    print("len of vectors", len(vectors))

    data = [{"id": i, "vector": vector, "text":chunk} for i, (vector, chunk) in enumerate(zip(vectors, texts))]
    client.insert(COLLECTION_NAME, data)

    # 检索查询
    query = "大模型的上下文长度是指什么"
    q_vector = model.encode(query)
    results = client.search(
        COLLECTION_NAME,
        data = [q_vector],
        limit = 3,
        output_fields=["text"]
    )

    # 打印结果
    for rank, result in enumerate(results[0]):
        print(f"   Top {rank+1}: ID={result['id']}, 相似度={result['distance']:.4f}, 文本={result['entity']['text']}")







# ================================ 指定执行 ================================
if __name__=="__main__":
    create_milvus()