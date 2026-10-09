import os
import numpy as np
from sentence_transformers import SentenceTransformer
from chunking import recursive_chunking

# 计算余弦相似度
def calc_cos_similar(v1:np.ndarray, v2:np.ndarray):
    dot = np.dot(v1, v2)
    l_v1 = np.linalg.norm(v1)
    l_v2 = np.linalg.norm(v2)
    if l_v1 == 0 or l_v2 == 0:
        return 0.0
    
    return float(dot / (l_v1 * l_v2))


# 实现 top-k 检索排序
def retrieve_rank(query, texts, top_k):

    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    q_emb = model.encode(query)

    # 遍历文档块，计算相似度
    scores = []
    t_embs = model.encode(texts)
    for t_emb in t_embs:
        score = calc_cos_similar(q_emb, t_emb)
        scores.append(score)

    # 选择 top-k ，并打印
    ranked = sorted(enumerate(scores), key=lambda pair:pair[1], reverse=True)
    for rank, (idx, score) in enumerate(ranked[:top_k], 1):
        print(f"top{rank} (chunk{idx}, score={score:.4f}): {texts[idx]}")

# 检验top-k检索排序
def test_retrieve_topk():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "2026-09-26.md")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    split_texts = recursive_chunking(text, chunk_size=200)
    retrieve_rank("大模型的上下文长度是指什么", split_texts, 3)


# ================================ 指定执行 ================================
if __name__=="__main__":
    test_retrieve_topk()
    



