import os

# 固定窗口切块，考虑重叠
def window_chunking(text, chunk_size, overlap):
    split_text = []
    for i in range(0, len(text), chunk_size - overlap):
        split_text.append(text[i:i+chunk_size])
    
    return split_text

# 测试固定窗口切块
def test_window_chunk():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "2026-09-26.md")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    split_texts = window_chunking(text, chunk_size=200, overlap=40)
    print(f"Window split texts length: {len(split_texts)}")
    for text in split_texts:
        print(text, "\n")

# 按照标点层级递归切块
def recursive_chunking(text, chunk_size, separators=["\n\n", "\n", "。", "！", "？", " ", ""]):
    def real_recursive(text, i_sep):
        split_texts = []
        sep = separators[i_sep]
        texts = list(texts) if sep == "" else text.split(sep)
        for split_text in texts:
            if len(split_text) > chunk_size and (i_sep + 1) < len(separators):
                split_texts.extend(real_recursive(split_text, i_sep + 1))
            else:
                split_texts.append(split_text)
        return split_texts

    # 递归划分
    split_texts = real_recursive(text, 0)

    # 将短小片段合并
    texts = []
    i = 0
    while(i < len(split_texts)):
        j = 0
        merge_len = len(split_texts[i])
        while(merge_len < chunk_size and i + j + 1 < len(split_texts)):
            j += 1
            merge_len += len(split_texts[i+j])
        
        merged = ''.join(split_texts[i:i+j+1])
        texts.append(merged)
        i = i + j + 1

    # 返回合并结果
    return texts
 
# 测试递归切块
def test_recursive_chunk():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "2026-09-26.md")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    split_texts = recursive_chunking(text, chunk_size=200)
    print(f"Recursive split texts length: {len(split_texts)}")
    for text in split_texts:
        print(text, "\n")






# ================================ 指定执行 ================================
if __name__=="__main__":
    test_window_chunk()
    test_recursive_chunk()