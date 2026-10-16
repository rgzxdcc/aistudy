import json
import os
import re
import pickle
import random
from typing import Any
from rank_bm25 import BM25Okapi
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from sentence_transformers import SentenceTransformer

# ================================ 对单个文档分块 ================================
# 提取文档中的元数据以及内容文本（去除元数据）
def extract_metadata_and_body(md_text:str, file_path:str)->tuple[dict, str]:
    """ 解析 markdown， 返回元数据以及文本内容（剔除元数据）
        
    Args:
        md_text: markdown 原文
        file_path: 文档路径
    """
    file_name = os.path.basename(file_path)
    base_name = os.path.splitext(file_name)[0]

    meta = {
        "doc_id":base_name,
        "title":"",
        "summary":"",
        "file_path":file_path        
    }
    clean_body = md_text

    # 匹配 Frontmatter(---...---)
    fm_match = re.search(r"^---\s*\n(.*?)\n---\s*\n", md_text, re.DOTALL)
    if fm_match:
        fm_text = fm_match.group(1)
        clean_body = md_text[fm_match.end():]

        # 提取 title
        title_match = re.search(r"^title:\s*[\"']?(.*?)[\"']?$", fm_text, re.MULTILINE)
        if title_match:
            meta["title"] = title_match.group(1).strip()

        # 提取 summary
        summary_match = re.search(r"^summary:\s*[\"']?(.*?)[\"']?$", fm_text, re.MULTILINE)
        if summary_match:
            meta["summary"] = summary_match.group(1).strip()

    # 如果 meta_data 中没有 title，从正文中找第一个标题
    if not meta["title"]:
        h1_match = re.search(r"^#\s+(.+)$", clean_body, re.MULTILINE)
        if h1_match:
            meta["title"] = h1_match.group(1).strip()
        else:
            # 如果还没有，就用文件名
            meta["title"] = base_name.replace("_", " ").title()
    
    return meta, clean_body



# 分层解析markdown文档，包括：
# ① 提取原数据 ② 保持代码块完整 ③ 保留标题层级路径 ④ 生成唯一标识符（doc_id#chunk_id）
def parse_markdown_with_hierarchy(md_text:str, file_path:str)->list[dict[str, Any]]:
    """ 分层解析 markdown， 返回带元数据的文本块列表
    
    Args:
        md_text: markdown 原文
        file_path: 文档路径
    """

    # 1. 先提取元数据以及文本内容块（content）
    doc_meta, body_text = extract_metadata_and_body(md_text, file_path)

    # 2. 初始化各记录变量

    # 当前所处标题，后续读取到标题后更新
    current_headers = {1:doc_meta.get("title", ""), 2:"", 3:""}
    # 当前累积的 chunk 文本缓冲
    chunks = []
    current_chunk_lines = []
    current_length = 0
    MAX_CHUNK_SIZE = 600
    MIN_MERGE_SIZE = 200
    in_code_block = False
    chunk_counter = 0

    # 对每个 chunk 进行封装
    def emit_chunk(lines_buffer, headers_snapshot):
        nonlocal chunk_counter
        content = "\n".join(lines_buffer).strip()
        if not content:
            return None

        # 构建面包屑路径：title > first_head > second_head
        breadcrumb = " > ".join([h for h in headers_snapshot.values() if h])

        # 将面包屑前缀注入内容头部，赋予该 Chunk 上下文信息
        full_content = f"[{breadcrumb}]\n{content}"

        chunk_data = {
            "chunk_id":f"{doc_meta['doc_id']}#c{chunk_counter:03d}",
            "doc_id":doc_meta["doc_id"],
            "title":doc_meta.get("title", ""),
            "breadcrumb":breadcrumb,
            "content":full_content,
            "raw_content":content,
            "length":len(full_content)
        }
        chunk_counter += 1
        return chunk_data
    
    # 3. 遍历文本行，做文本切分
    lines = body_text.splitlines()
    for line in lines:
        stripped = line.strip()

        # (A) 状态判断：是否进入/离开代码块
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            current_chunk_lines.append(line)
            current_length += len(line)
            continue

        # (B) 如果在代码块内部：不切断，无脑追加
        if in_code_block:
            current_chunk_lines.append(line)
            current_length += len(line)
            continue

        # (C) 状态判断：是否是标题行（如#，##，###）
        header_match = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if header_match:
            level = len(header_match.group(1))
            header_title = header_match.group(2).strip()

            # 如果遇到新标题，并且当前缓冲区已有一些内容，即使长度小于 MAX_CHUNK_SIZE，可直接切分
            if current_length >= MIN_MERGE_SIZE:
                chunk = emit_chunk(current_chunk_lines, current_headers)
                if chunk:
                    chunks.append(chunk)
                current_chunk_lines = []
                current_length = 0

            # 更新标题层级
            current_headers[level] = header_title
            # 清理比当前级别更深的子标题
            for l in range(level+1, 4):
                current_headers[l] = ""

            current_chunk_lines.append(line)
            current_length += len(line)
            continue

        # (D) 普通正文行，按长度与自然段落累积
        current_chunk_lines.append(line)
        current_length += len(line)

        # 超过最大长度，且遇到空行（自然段落结束），进行优雅切分
        if current_length >= MAX_CHUNK_SIZE and stripped == "":
            chunk = emit_chunk(current_chunk_lines, current_headers)
            if chunk:
                chunks.append(chunk)
            current_chunk_lines = []
            current_length = 0

    # 4. 收尾最后一个 chunk
    if current_chunk_lines:
        chunk = emit_chunk(current_chunk_lines, current_headers)
        if chunk:
            chunks.append(chunk)

    return chunks

# ================================ 创建稠密索引、稀疏索引库 ================================
# 创建向量数据库的构建、加载
COLLECTION_NAME = "professional_collection"
DB_FILENAME = "milvus_rag.db"
VECTOR_DIM = 384
def build_and_save_milvus(chunks:list[dict], output_dir:str):
    # 初始化客户端
    db_filename = os.path.join(output_dir, DB_FILENAME)
    client = MilvusClient(db_filename)

    # 创建collection
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    schema = CollectionSchema([
        FieldSchema("id", DataType.VARCHAR, max_length=128, is_primary=True),
        FieldSchema("vector", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)],
        enable_dynamic_field=True)
    client.create_collection(COLLECTION_NAME, schema=schema, metric_type="COSINE")

    # 生成向量：此处用的 content，带面包屑上下文信息
    contents = [chunk["content"] for chunk in chunks]
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    vectors = model.encode(contents)

    # 插入数据
    data = [{"id": chunk["chunk_id"], "vector": vector, "text":chunk["content"], "meta":chunk} 
        for vector, chunk in zip(vectors, chunks)]
    client.insert(COLLECTION_NAME, data)

# 英文分词：用基础正则提取单词、下划线、字母数字，全部转小写
def tokenize_en(text:str)->list[str]:
    return re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())

# 创建稀疏索引落盘
def build_and_save_bm25(chunks:list[dict], output_dir:str):
    print("开始构建 BM25 稀疏索引")

    # 1. 批量对所有 chunk 文本进行分词
    tokenized_corpus = [tokenize_en(chunk["content"]) for chunk in chunks]

    # 构建 BM25 模型
    bm25 = BM25Okapi(tokenized_corpus)

    # 如果输入目录不存在，则创建
    os.makedirs(output_dir, exist_ok=True)

    # 将模型存入磁盘文件
    bm25_path = os.path.join(output_dir, "bm25_model.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    
    # 将元数据以及文本数据存入文件
    meta_path = os.path.join(output_dir, "chunks_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print(f"BM25 索引已持久化保存至：{bm25_path}")
    print(f"chunks 元数据已持久化保存至：{meta_path}")

# 分别执行稠密向量入库与稀疏向量落盘
def process_data_ingest():
    # 读取目录下所有文档
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(cur_dir, "data/milvus_docs")
    doc_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                doc_list.append(os.path.join(root, file))

    # 划分所有文本块
    all_chunks = []
    for doc_path in doc_list:
        with open(doc_path, 'r', encoding="utf-8") as f:
            text = f.read()
        chunks = parse_markdown_with_hierarchy(text, doc_path)
        all_chunks.extend(chunks)

    # # 输出统计数据
    # print("数据总文档数： ", len(doc_list))
    # print("总文本块(chunk)数： ", len(all_chunks))
    # all_length = 0
    # for chunk in all_chunks:
    #     all_length += chunk["length"]
    # print("文本块(chunk)平均长度： ", float(all_length)/len(all_chunks))
    # indices = random.sample(range(0, len(all_chunks)), 3)
    # for index in indices:
    #     print(f"文本块chunk({index})内容： ", all_chunks[index])

    # 稠密向量入库
    build_and_save_milvus(all_chunks, cur_dir)

    # 稀疏向量落盘
    build_and_save_bm25(all_chunks, os.path.join(cur_dir, "data"))



# ================================ 构建双路查询管道 ================================
# 创建稀疏索引加载器，用于快速加载离线稀疏落盘文件
class SparseIndexManager:
    _instance = None

    def __new__(cls, *args, **kwargs)->"SparseIndexManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, data_dir:str) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        with open(os.path.join(data_dir, "bm25_model.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)
        with open(os.path.join(data_dir, "chunks_metadata.json"), "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def search(self, query:str, top_k=10):
        tokens = tokenize_en(query)
        scores = self.bm25.get_scores(tokens)

        # 挑选top_k
        top_indices = sorted(range(len(scores)), key=lambda i:scores[i], reverse=True)[:top_k]

        # 组织文本块
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "id":self.chunks[idx]["chunk_id"],
                    "score":scores[idx],
                    "text":self.chunks[idx]["content"],
                    "metadata":self.chunks[idx]
                })
        return results
        
        
# 创建 Advanced RAG Pipeline
class AdvancedRAGPipeline:
    def __init__(self) -> None:
        pass

# ================================ 测试查询 ================================
def test_Advanced_RAG(query, top_k):
    pass


# ================================ 指定执行 ================================
if __name__=="__main__":
    # process_data_ingest()
    test_Advanced_RAG()