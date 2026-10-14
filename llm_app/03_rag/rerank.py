import os
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from embedding import recursive_chunking
from hybrid_search import hybrid_search
from pipeline import create_emb, load_milvus, create_llm, generate


# ================================ 重排 ================================
# 定义重排检索函数
def rerank_search(reranker, query, candidates, top_k=3):
    # 组织重排向量，计算得分
    pairs = [[query, item["text"]] for item in candidates]
    scores = reranker.predict(pairs)

    # 组织rerank后排序结果
    id_scores = {}
    id_texts = {}
    for item,score in zip(candidates, scores):
        id_scores[item["id"]] = score
        id_texts[item["id"]] = item["text"]
    sorted_scores = sorted(id_scores.items(), key=lambda item:item[1], reverse=True)[:top_k]

    results = [{"id":id, "score":score, "text":id_texts[id]} for id,score in sorted_scores]
    return results

# 测试重排
def test_rerank_search(query, top_k=3):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "doc_texts.md")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chunks = recursive_chunking(text, chunk_size=200)
    corpus_tokens = [jieba.lcut(chunk) for chunk in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    emb = create_emb()
    milvus = load_milvus(emb)
    reranker = CrossEncoder("BAAI/bge-reranker-base")

    # 执行混合搜索
    candidates = hybrid_search(bm25, chunks, emb, milvus, query, top_k=10, retrieve_k = 10)

    # 执行检索重排
    results = rerank_search(reranker, query, candidates, top_k=3)
    # 打印
    for rank, result in enumerate(results):
        print(f"   Top {rank+1}: ID={result['id']}, 相关性分数={result['score']:.4f}, 文本={result['text']}")

    # 组织上下文
    context = "\n\n".join(f"[{i+1}] {item['text']}" for i,item in enumerate(results))

    # 将问题与查询结果均送入LLM，进行最终结果生成
    model = create_llm()
    final_result = generate(model, query, context)
    print("Rerank search final query result from llm: ", final_result)
    return final_result

# ================================ 指定执行 ================================
if __name__=="__main__":
    query = "Flash Attention 怎么增量算 softmax？"
    test_rerank_search(query, top_k=3)