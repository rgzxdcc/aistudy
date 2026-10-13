from collections import defaultdict
import os
import jieba
from rank_bm25 import BM25Okapi
from chunking import recursive_chunking
from pipeline import COLLECTION_NAME, create_emb, \
     load_milvus, create_llm, generate


# ================================ 关键词查询 ================================
def sparse_search(bm25, chunks, query, top_k=10):
    tokenized_query = jieba.lcut(query)
    scores = bm25.get_scores(tokenized_query)

    id_scores = {id:score for id,score in enumerate(scores) if score > 0}
    sorted_ids = sorted(id_scores, key=lambda i:id_scores[i], reverse=True)[:top_k]

    # 组织查询结果
    results = [{"id":id, "score":id_scores[id], "text":chunks[id]} for id in sorted_ids]
    
    return results

# 测试关键词检索（稀疏检索）
def test_sparse_search(query):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "doc_texts.md")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    split_texts = recursive_chunking(text, chunk_size=200)

    corpus_tokens = [jieba.lcut(chunk) for chunk in split_texts]
    bm25 = BM25Okapi(corpus_tokens)
    results = sparse_search(bm25, split_texts, query)
    [print(result) for result in results]

# ================================ RRF ================================
def rrf_fusion(sparse_results, dense_results, top_k=3, k=60):
    rrf_scores = defaultdict(float)
    id_texts = {}
    for rank, result in enumerate(sparse_results, start=1):
        rrf_scores[result['id']] += 1 / (rank + k)
        id_texts[result['id']] = result['text']
    for rank, result in enumerate(dense_results, start=1):
        rrf_scores[result['id']] += 1 / (rank + k)
        id_texts[result['id']] = result['text']
    
    # 按分数降序排序
    sorted_scores = sorted(rrf_scores.items(), key=lambda item:item[1], reverse=True)

    # 组织输出
    final_top_k = [{"id":id, "score":score, "text":id_texts[id]} 
        for id, score in sorted_scores[:top_k]]

    return final_top_k

# ================================ 向量查询 ================================
def dense_search(emb, milvus, query, top_k=10):
    q_query = emb.encode(query)
    results = milvus.search(
        COLLECTION_NAME,
        data=[q_query],
        limit=top_k,
        output_fields=['text']
    )[0]

    # 组织输出
    final_top_k = [{"id":result['id'], "score":result['distance'], "text":result['entity']['text']} 
        for result in results]

    return final_top_k

# 测试向量查询
def test_dense_search(query):
    emb = create_emb()
    milvus = load_milvus(emb)

    results = dense_search(emb, milvus, query)
    [print(result) for result in results]

# ================================ 混合查询 ================================
def hybrid_search(bm25, chunks, emb, milvus, query, top_k=3, retrieve_k = 9):
    sparse_res = sparse_search(bm25, chunks, query, retrieve_k)
    dense_res = dense_search(emb, milvus, query, retrieve_k)
    hybrid_res = rrf_fusion(sparse_res, dense_res, top_k)
    return hybrid_res

# 测试混合查询
def test_hybrid_search(query):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "doc_texts.md")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chunks = recursive_chunking(text, chunk_size=200)
    corpus_tokens = [jieba.lcut(chunk) for chunk in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    emb = create_emb()
    milvus = load_milvus(emb)

    # 执行搜索
    results = hybrid_search(bm25, chunks, emb, milvus, query, top_k=3, retrieve_k = 9)
    for rank, result in enumerate(results):
        print(f"   Top {rank+1}: ID={result['id']}, RRF分数={result['score']:.4f}, 文本={result['text']}")

    # 组织上下文
    context = "\n\n".join(f"[{i+1}] {item['text']}" for i,item in enumerate(results))

    # 将问题与查询结果均送入LLM，进行最终结果生成
    model = create_llm()
    final_result = generate(model, query, context)
    print("Hybrid search final query result from llm: ", final_result)
    return final_result   


# ================================ 指定执行 ================================
if __name__=="__main__":
    # query = "bge-small-zh-v1.5 的向量维度是多少？"
    query = "Flash Attention 怎么增量算 softmax？"
    # test_sparse_search(query)
    # test_dense_search(query)
    test_hybrid_search(query)