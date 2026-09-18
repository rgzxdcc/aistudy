import torch
import tiktoken
from model import GPTModel
from model import generate_text_simple
from embedding import create_dataloader_v1
import os
import torch.nn as nn
from device import common_device
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # 词汇表大小
    "context_length":256,  # 上下文长度
    "emb_dim":768,          # 嵌入维度
    "n_heads":12,           # 注意力头数量
    "n_layers":12,          # 层数
    "drop_rate":0.1,        # dropout率
    "qkv_bias":False        # 查询-键-值偏置
}

# 定义全局model、tokenizer
# torch.manual_seed(123)
# model = GPTModel(GPT_CONFIG_124M)
tokenizer = tiktoken.get_encoding("gpt2")

# ================================ generate and evaluate ================================
def test_generate_shape():
    
    # 打印模型所有参数名：张量形状
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
    model.eval()

# 封装id-text互相转换便捷函数
# 提问 1：text_to_token_ids()、token_ids_to_text()函数输入、输出的维度
def text_to_token_ids(text, tokenizer: tiktoken.Encoding):
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(ids).unsqueeze(0)

def token_ids_to_text(ids: torch.Tensor, tokenizer:tiktoken.Encoding):
    flat = ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def test_ids_text():
    start_context = "Every effort moves you"

    token_ids = generate_text_simple(
        model = model,
        idx = text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    idx = text_to_token_ids(start_context, tokenizer)
    output_text = token_ids_to_text(token_ids, tokenizer)
    print("shape of input token ids shape: ", idx.shape)
    print("shape of output token ids shape: ", token_ids.shape)
    print("output text: ", output_text)

# 逐步计算损失
def calc_loss_by_step():
    inputs = torch.tensor([[16833, 3626, 6100], # ["every effort moves",
                           [40, 1107, 588]])    #  "I realy like"]
    targets = torch.tensor([[3626, 6100, 345], # ["effort moves you",
                           [1107, 588, 11311]])    #  "realy like chocolate"]                       

    # 使用模型计算得分
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
    
    # 计算概率
    probas = torch.softmax(logits, dim=-1)
    print(probas.shape)

    # 由概率得到id
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print(token_ids.shape)
    print(token_ids)

    # 将词元id转化为文本
    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

    # 打印正确词元位置概率值（训练目标是使此位置的概率值最高）
    # 提问 2：此处的target_probas_1为什么是一维张量（与one-hot联动理解）
    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 1: ", target_probas_1)
    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 2: ", target_probas_2)

    # 对概率分数取对数
    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print("log_probas: ", log_probas)

    avg_log_probas = torch.mean(log_probas)
    print("avg_log_probas: ", avg_log_probas)

    # 交叉熵损失
    neg_avg_log_probas = avg_log_probas * -1
    print("neg_avg_log_probas: ", neg_avg_log_probas)

    # 提问 3： 怎么理解交叉熵损失计算过程

    # 使用torch.nn.functional.cross_entropy()
    print("logits shape: ", logits.shape)
    print("targets shape: ", targets.shape)
    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    print("flattened logits shape: ", logits_flat.shape)
    print("flattened targets shape: ", targets_flat.shape)
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print("loss: ", loss)

    # 计算困惑度
    perplexity = torch.exp(loss)
    print("perplexity: ", perplexity)

# 组件训练批次进行损失计算
def construct_batch_calc_loss():

    # 仍然加载短篇小说The Verdict
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "the-verdict.txt")
    with open(file_path, 'r', encoding='utf-8') as f:
        text_data = f.read()
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    # print("total characters: ", total_characters)
    # print("total tokens: ", total_tokens)

    # 划分训练数据与验证数据
    # 提问：划分两种数据的意义
    train_radio = 0.9
    split_idx = int(len(text_data) * train_radio)
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    # 各自创建数据加载器
    torch.manual_seed(123)
    train_loader = create_dataloader_v1(
        train_data,
        batch_size = 2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride = GPT_CONFIG_124M["context_length"],
        shuffle = True,
        drop_last = True,
        num_workers=0
    )
    val_loader = create_dataloader_v1(
        val_data,
        batch_size = 2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride = GPT_CONFIG_124M["context_length"],
        shuffle = False,
        drop_last = False,
        num_workers=0
    )

    # print("train loader: ")
    # for x,y in train_loader:
    #     print(x.shape, y.shape)
    # print("val loader: ")
    # for x,y in val_loader:
    #     print(x.shape, y.shape)

    return train_loader, val_loader

# 定义给定批次（单个批次）的交叉熵计算
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

# 计算加载器中所有批次损失
def calc_loss_loader(data_loader, model, device, num_batches=None):
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    # 按批次求损失，最后返回整个数据集的平均损失作为最终结果
    total_loss = 0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    loss = total_loss / num_batches
    return loss

# 测试损失计算函数
def test_calc_loss_loader():
    device = common_device
    model.to(device)
    train_loader, val_loader = construct_batch_calc_loss()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    print("train_loader loss: ", train_loss)
    print("val_loader loss: ", val_loss)

# ================================ train model ================================ 
# 定义预训练大模型主函数
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, 
                        eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_token_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # 遍历所有轮次
    for epoch in range(num_epochs):
        model.train()

        # 遍历所有批次
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            # 以一定频次对模型当前训练水平做验证
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")

        # 一个轮次后根据输入打印一个输出
        generate_and_print_sample(model, tokenizer, device, start_context)
    
    return train_losses, val_losses, track_token_seen

# 评估模型损失
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss

# 打印模型生成文本
def generate_and_print_sample(model:GPTModel, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model, encoded, 50, context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

# 正式训练！
def pre_train():
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(common_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0004, weight_decay=0.1)
    num_epochs = 10
    train_loader, val_loader = construct_batch_calc_loss()

    # 开启训练循环
    train_losses, val_losses, tokens_seen = train_model_simple(model, train_loader,
        val_loader, optimizer, common_device, num_epochs,
        eval_freq=5, eval_iter=5, start_context="Every effort moves you", tokenizer=tokenizer)

    # # 绘制损失曲线
    # epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    # plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    # 返回训练好的模型
    return model

# 绘制损失曲线
def plot_losses(epoches_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epoches_seen, train_losses, label="Training loss")
    ax1.plot(epoches_seen, val_losses, linestyle='-', label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

# ================================ 选择策略 ================================
def test_gen_after_pretrain():
    model = pre_train()

    model.to("cpu")
    model.eval()

    token_ids = generate_text_simple(
        model=model,
        idx = text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"])
    print("output text: \n", token_ids_to_text(token_ids, tokenizer))

# 制作简易词汇表
def create_inverse_vocab():
    vocab = {
        "closer":0,
        "every":1,
        "effort":2,
        "forward":3,
        "inches":4,
        "moves":5,
        "pizza":6,
        "towards":7,
        "you":8
    }
    inverse_vocab = {v:k for k,v in vocab.items()}
    return inverse_vocab

# 测试温度缩放
def test_temperature():
    inverse_vocab = create_inverse_vocab()
    # 贪心解码
    next_token_logits = torch.tensor(
        [4.51, 0.89, -1.9, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
        )
    probas = torch.softmax(next_token_logits, dim=-1)
    next_token_id = torch.argmax(probas).item()
    print("argmax: ", inverse_vocab[next_token_id])

    # 概率采样
    torch.manual_seed(123)
    next_token_id = torch.multinomial(probas, num_samples=1).item()
    print("multinomial: ", inverse_vocab[next_token_id])

    # 重复采样测试
    print_sampled_tokens(probas)

    # 温度缩放测试
    temperatures = [1, 0.1, 5]
    scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
    x = torch.arange(len(inverse_vocab))
    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, T in enumerate(temperatures):
        rects = ax.bar(x+i*bar_width, scaled_probas[i],
            bar_width, label=f'Temperature {T}')
    ax.set_ylabel('Probability', rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(inverse_vocab.values())
    ax.legend()
    plt.tight_layout()
    plt.show()

# 采样1000次，打印采样概率
def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1000)]
    sample_ids = torch.bincount(torch.tensor(sample))
    for i,freq in enumerate(sample_ids):
        print(f"{create_inverse_vocab()[i]} * {freq}, freq: {(freq / 1000):.3f}")

# 温度缩放
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

# practice5.1: 打印温度缩放采样频率
def practice5_1():
    next_token_logits = torch.tensor(
        [4.51, 0.89, -1.9, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
        )
    temperatures = [1, 0.1, 5]
    scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
    for i,probas in enumerate(scaled_probas):
        print(f"Temperature {temperatures[i]} :")
        print_sampled_tokens(probas)
        print("\n")

    # 找到pizza的正确概率(仅在temperature为5时不为0)
    probas_pizza = scaled_probas[2][6]
    print(probas_pizza)

# top-k采样
def top_k_sample():
    next_token_logits = torch.tensor(
        [4.51, 0.89, -1.9, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
        )

    # 找出top_k个最大值
    top_k = 3
    top_logits, top_pos = torch.topk(next_token_logits, top_k)
    print("top logits: ", top_logits)
    print("top pos: ", top_pos)

    # 将其余值提华为-inf
    new_logits = torch.where(
        condition = next_token_logits < top_logits[-1],
        input = torch.tensor(float('-inf')),
        other = next_token_logits
    )
    print(new_logits)

    # 转换为概率分数
    probas_topk = torch.softmax(new_logits, dim=-1)
    print(probas_topk)

# 应用温度缩放与top-k采样修改generate()类
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        
        # 使用top_k过滤
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            logits = torch.where(
                condition = logits < top_logits[:, -1],
                input = torch.tensor(float('-inf')).to(logits.device),
                other = logits
            )
        
        # 应用温度放缩
        if temperature > 0:
            logits = logits / temperature
            probas = torch.softmax(logits, dim=-1)
            idx_text = torch.multinomial(probas, 1)
        else:
            idx_text = torch.argmax(logits, dim=-1, keepdim=True)
        
        # 如果遇到下一个词元为结束符，则停止生成
        if eos_id is not None and torch.all(idx_text == eos_id):
            break

        # 将生成词元添加到上下文最后
        idx = torch.cat((idx, idx_text), dim=1)

    return idx
# 测试generate函数
def test_generate():
    model = pre_train()
    model.to("cpu")
    model.eval()

    torch.manual_seed(123)
    token_ids = generate(
        model=model,
        idx = text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=15,
        context_size = GPT_CONFIG_124M["context_length"],
        temperature=1.4,
        top_k=25,
    )
    text_out = token_ids_to_text(token_ids, tokenizer)

    print("text output: ", text_out)

# practice 5.2
# 适合低温度和Top-k设置场景：用于严肃场景下，如生成技术性文件，编写代码数学公式等
# 适合高温度和Top-k设置场景：用于创造场景，如写作，生成文本报告，出方案等

# practice 5.3
# 强制generate表现出确定性行为（禁用随机采样）：将temperature设置为不大于1，将top-k设置为1

# ================================ 指定执行 ================================
if __name__=="__main__":
    # test_generate_shape()
    # test_ids_text()
    # construct_batch_calc_loss()
    # test_calc_loss_loader()
    # pre_train()
    # test_gen_after_pretrain()
    # test_temperature()
    # practice5_1()
    # top_k_sample()
    test_generate()