# 测试tiktoken环境
from numpy import result_type
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
text = ("Hello, do you like tea?")
integers = tokenizer.encode(text)
print(integers)
print(tiktoken.__version__)

# 下载the-verdict.txt
import urllib.request
url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
file_path = "llm_app/02_llm_from_scratch/the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

# 读取the-verdict.txt
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])

# 测试正则表达式
import re
# text = "Hello, word. This, is a test."
# result = re.split(r'(\s)', text)
# print (result)
# result = re.split(r'([,.]|\s)', text)
# result=[item for item in result if item.strip()]
# print(result)

# 改进分词方法：添加其他标点符号处理，包括双破折号
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item for item in result if item.strip()]
print(result)

# 将该分词器用于the-verdict.txt全文
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])