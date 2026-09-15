import os
import re
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# ================================ 文本分词 ================================

# 读取the-verdict.txt
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "the-verdict.txt")
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character in the-verdict.txt:", len(raw_text))

# 使用简易分词器对the-verdict.txt内容进行词元分割
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item for item in preprocessed if item.strip()]
print("Total number of word in the-verdict.txt:", len(preprocessed))
# print(preprocessed[:30])


# ================================ token->ID ================================
# 提问：词汇表构建流程

# 按照字母顺序，剔除重复词元
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(f"origin vaocab_size: {vocab_size}")

# 创建词汇表，打印前50条示例
vocab = {token:id for id, token in enumerate(all_words)}
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i >= 50:
#         break

# 实现简单文本分词器
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[str] for str in preprocessed]
        return ids 

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # 移除标点符号前的空格
        text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)
        return text

# 测试简单分词器
def test_simpleTokenizerV1():
    tokenizer = SimpleTokenizerV1(vocab)
    text = """"It's the last he painted, you know." Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(ids)

    text = tokenizer.decode(ids)
    print(text)

    # 测试词元在训练集之外的分词器表现
    text = "Hello, do you like tea?"
    ids = tokenizer.encode(text) # KeyError: 'Hello'
    print(ids)


# ================================ 特殊词元 ================================

# 添加<|unk|>、<|endoftext|>
all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {str:id for id, str in enumerate(all_tokens)}
print("extend vocab_size: ", len(vocab))

# for i, item in enumerate(list(vocab.items())[-5:]):
#     print(item)

# 更新分词器(支持未识别词元处理)
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[str] for str in preprocessed]
        return ids 

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # 移除标点符号前的空格
        text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)
        return text

# 测试分词器
def test_simpleTokenizerV2():
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of palace."
    text = "<|endoftext|> ".join((text1, text2))
    print(text)

    tokenizer = SimpleTokenizerV2(vocab)
    ids = tokenizer.encode(text)
    print(ids)

    text = tokenizer.decode(ids)
    print(text)


# ================================ BPE ================================
# 测试BPE分词器
# 提问：BPE如何处理未知词元
def test_BPE():
    print("version if tiktoken: " + tiktoken.__version__)
    tokenizer = tiktoken.get_encoding("gpt2")
    text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.")
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(ids)
    strings = tokenizer.decode(ids)
    print(strings)

# practice 2.1 ：拆分"Akwirw ier"，并打印
def practice2_1():
    tokenizer = tiktoken.get_encoding("gpt2")
    unknown_text = "Akwirw ier"
    unkonwn_ids = tokenizer.encode(unknown_text)
    print(unkonwn_ids)

    for id in unkonwn_ids:
        token = tokenizer.decode([id])
        print(token)
    decode_strings = tokenizer.decode(unkonwn_ids)
    print(decode_strings)

# ================================ 数据采样 ================================
def data_preprocessing():
    tokenizer = tiktoken.get_encoding("gpt2")
    enc_text = tokenizer.encode(raw_text)
    print("token size of verdict.txt in BPE: ", len(enc_text))

    # 剔除前50个元素，原因？
    enc_sample = enc_text[50:]

    # 组织输入-输出雏形
    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size+1]
    print(f"x: {x}")
    print(f"y:      {y}")

    # 通过箭头表示大模型预测过程
    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        # print(context, "---->", desired)
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

# 封装DataSet
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        length = len(token_ids)
        for i in range(0, length - max_length, stride):
            input = token_ids[i:i+max_length]
            output = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input))
            self.target_ids.append(torch.tensor(output))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

# 根据dataset创建dataloader
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers)

    return dataloader

# 测试dataloader
def test_dataloader():
    dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
    data_iter = iter(dataloader)
    first_branch = next(data_iter)
    print(first_branch)
    second_branch = next(data_iter)
    print(second_branch)
        
# practice 2.2：体验不同步幅与上下文长度的dataloader
def practice2_2():
    dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=2, stride=2, shuffle=False)
    data_iter = iter(dataloader)
    first_branch = next(data_iter)
    print("max_length=2, stride=2, first_branch: ", first_branch)
    second_branch = next(data_iter)
    print("max_length=2, stride=2, second_branch: ", second_branch)

    dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=8, stride=2, shuffle=False)
    data_iter = iter(dataloader)
    first_branch = next(data_iter)
    print("max_length=8, stride=2, first_branch: ", first_branch)
    second_branch = next(data_iter)
    print("max_length=8, stride=2, second_branch: ", second_branch)

# 体会设置批次带来的变化
def test_batchset():
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
    data_iter = iter(dataloader)
    first_branch = next(data_iter)
    print("batch_size=8, max_length=4, stride=4, first_branch: \n", first_branch)
    second_branch = next(data_iter)
    print("batch_size=8, max_length=4, stride=4, second_branch: \n", second_branch)


# ================================ embedding ================================
# 创建embedding层，获取由id->vec
def embedding():
    input_ids = torch.tensor([2, 3, 5, 1])
    vocab_size = 6
    output_dim = 3

    torch.manual_seed(123)
    embedding = torch.nn.Embedding(vocab_size, output_dim)

    # 打印权重
    print(embedding.weight)

    # 输出指定id对应嵌入
    print(embedding(torch.tensor([3])))

    # 输出输入对应输出
    print(embedding(input_ids))

# 位置编码
# 提问：为什么需要位置编码
def position_embedding():

    # 创建dataloader
    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    max_length = 4
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
    dataiter = iter(dataloader)
    inputs, targets = next(dataiter)
    print("Token ids: \n", inputs)
    print("\n Inputs shape: \n", inputs.shape)

    # 计算embedding
    token_embedding = token_embedding_layer(inputs)
    print("token embedding shape: ", token_embedding.shape)

    # 创建位置嵌入层
    context_length = max_length
    position_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    position_embedding = position_embedding_layer(torch.arange(context_length))
    print("position embedding shape: ", position_embedding.shape)

    # 计算总嵌入
    input_embedding = token_embedding + position_embedding
    print("input embedding shape: ", input_embedding.shape)

# ================================ 指定执行 ================================
if __name__=="__main__":
    # test_dataloader()
    # practice2_2()
    # test_batchset()
    # embedding()
    position_embedding()