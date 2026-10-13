import os
os.environ["GRPC_VERBOSITY"] = "ERROR" # 屏蔽gRPC提示信息
from pipeline import create_emb, load_milvus, \
    create_llm, rag_pipeline, generate

COLLECTION_NAME = "rag_collection"

# 重写查询问题
def query_rewrite(model, query)->list[str]:
    prompt = f""" 提示词重写格式如下 
        【输入】：
        {query}
        
        【输出】：
        严格输出3行不同角度但语义一致的查询语句"""
    response = model.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content":"You are a helpful assistant"},
            {"role": "user", "content":prompt},
        ] 
    )
    queries = response.choices[0].message.content
    print("rewrite queries from llm: ", queries)
    queries_list = [line.strip() for line in queries.splitlines() if line.strip()]
    return queries_list


# 多路检索融合
def Multi_Retrieve(embbed, milvus, queries, top_k = 3):
    emb_queries = embbed.encode(queries).tolist()
    print("emb_queries shape", len(emb_queries))

    # 多路检索
    # 注意：不可改回下方注释的批量检索，当前环境 milvus-lite 对 nq>1 的批量查询存在 segfault 缺陷
    # results = milvus.search(
    #     COLLECTION_NAME,
    #     emb_queries,
    #     limit = top_k,
    #     output_fields=["text"])
    results = []
    for q in emb_queries:
        res = milvus.search(
            COLLECTION_NAME,
            data=[q],
            limit = top_k,
            output_fields=["text"])
        results.append(res[0])

    # 将检索结果去重合并：同一id保留最高相似度，最后按照相似度降序排序
    best = {}
    for result_q in results:
        for result in result_q:
            if result['id'] not in best or result['distance'] > best[result['id']]['distance']:
                best[result['id']] = result
    final_result = sorted(best.values(), key=lambda r:r['distance'], reverse=True)[:top_k]
    return final_result

# 重写查询后的rag
def rag_rewrite(embbed, milvus, model, query, top_k = 3):
    queries = query_rewrite(model, query)
    queries.append(query)

    # 执行多路检索
    results = Multi_Retrieve(embbed, milvus, queries, top_k)

    # 打印查询结果，生成上下文
    context = []
    for rank, result in enumerate(results):
        print(f"   Top {rank+1}: ID={result['id']}, 相似度={result['distance']:.4f}, 文本={result['entity']['text']}")
        context.append(result["entity"]["text"])
    context_str = "\n\n".join(f"[{i+1}] {c}" for i,c in enumerate(context))

    # 将问题与查询结果均送入LLM，进行最终结果生成
    final_result = generate(model, query, context_str)
    # print("RAG rewrite final query result from llm: ", final_result)
    return final_result


# 对比原生查询语句与重写查询语句结果
def compare_rag(query):
    embbed = create_emb()
    milvus = load_milvus(embbed)
    model = create_llm()
    top_k = 3

    # 简单检索管道查询
    rag_pipeline(embbed, milvus, model, query, top_k)
    # 重写查询后检索
    rag_rewrite(embbed, milvus, model, query, top_k)

# ================================ 指定执行 ================================
if __name__=="__main__":
    # query_rewrite("上下文长度是什么")
    compare_rag("上下文长度是什么")