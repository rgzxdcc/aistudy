import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
import os, sys
sys.path.insert( 0 , os.path.dirname(os.path.abspath(__file__)))
from attention import MultiHeadAttention, batch



# ================================ GPT架构 ================================
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # 词汇表大小
    "context_length":1024,  # 上下文长度
    "emb_dim":768,          # 嵌入维度
    "n_heads":12,           # 注意力头数量
    "n_layers":12,          # 层数
    "drop_rate":0.1,        # dropout率
    "qkv_bias":False        # 查询-键-值偏置
}

# 提问 1：qkv_bias有什么作用

# 大模型框架
class DummyGPTModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[ DummyTransformerBlocks(cfg) 
            for _ in range (cfg["n_layers"]) ]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits

class DummyTransformerBlocks(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5) -> None:
        super().__init__()

    def forward(self, x):
        return x

# 测试大模型数据流
def test_GPTModel():
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    text1 = "Every effort moves you"
    text2 = "Every day holds a"

    # 分词得到词元ID序列
    batch.append(torch.tensor(tokenizer.encode(text1)))
    batch.append(torch.tensor(tokenizer.encode(text2)))
    print(batch)
    batch = torch.stack(batch, dim=0)
    print(batch)

    # 创建大模型
    d_gpt = DummyGPTModel(GPT_CONFIG_124M)
    logits = d_gpt(batch)
    print(logits.shape)


# ================================ layer norm ================================
# 提问 2：层归一化与批归一化有什么区别

def test_norm():
    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    out = layer(batch_example)
    print(out)

    # 计算均值、方差
    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)
    print("mean: ", mean)
    print("var:", var)

    # 归一化
    out_norm = (out - mean) / torch.sqrt(var)
    mean_norm = out_norm.mean(dim=-1, keepdim=True)
    var_norm = out_norm.var(dim=-1, keepdim=True)
    torch.set_printoptions(sci_mode=False)
    print("out_norm:", out_norm)
    print("mean norm: ", mean_norm)
    print("var norm:", var_norm)

# 实现层归一化
class LayerNorm(nn.Module):
    def __init__(self, emb_dim) -> None:
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / (torch.sqrt(var + self.eps))
        return self.scale * norm_x + self.shift

# 测试层归一化
def test_layerNorm():
    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)

    ln = LayerNorm(5)
    out = ln(batch_example)
    mean_out = out.mean(dim=-1, keepdim=True)
    var_out = out.var(dim=-1, keepdim=True, unbiased=False)
    torch.set_printoptions(sci_mode=False)
    print("mean norm: ", mean_out)
    print("var norm:", var_out)

# ================================ GELU前馈层 ================================
# 提问 3：GELU相较于RELU的区别
class GELU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        y = 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
            ))
        return y

# 对比RELU与GELU
def cmp_gelu_relu():
    gelu, relu = GELU(), nn.ReLU()

    x = torch.linspace(-3, 3, 100)
    y_gelu, y_relu = gelu(x), relu(x)
    plt.figure(figsize=(8, 3))
    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "RELU"]), 1):
        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.title(f"{label} avtivation function")
        plt.xlabel("x")
        plt.ylabel(f"{label}(x)")
        plt.grid(True)
    plt.tight_layout()
    plt.show()

# 提问 4：前馈层先扩展维度再恢复的意义

# 创建前馈神经网络层
class FeedForward(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
                GELU(),
                nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
    
    def forward(self, x):
        return self.layers(x)

# 测试前馈网络层输出
def test_feedforward():
    ffn = FeedForward(GPT_CONFIG_124M)
    x = torch.randn(2, 3, 768)
    out = ffn(x)
    print(out.shape)

# ================================ 快捷连接（残差连接） ================================
# 测试残差连接作用
class ExampleNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shorcut) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
            ])
        self.use_shortcut = use_shorcut

    def forward(self, x):
        for layer in self.layers:
            layer_out = layer(x)
            if self.use_shortcut and x.shape == layer_out.shape:
                x = layer_out + x
            else:
                x = layer_out
        return x

# 提问 5：残差连接作用

# 测试残差连接使用效果
def test_residual_connection():
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1.0, 0., -1.0]])
    torch.manual_seed(123)
    model_without_shortcut = ExampleNeuralNetwork(layer_sizes, use_shorcut=False)
    

    # 封装反向传播打印梯度函数
    def print_gradients(model, x):
        output = model(x)
        target = torch.tensor([[0.0]])

        loss = nn.MSELoss()
        loss = loss(output, target)
        loss.backward()

        for name, param in model.named_parameters():
            if 'weight' in name:
                weight_mean = param.grad.abs().mean().item()
                print(f"{name} has gradient mean of {weight_mean}") 
    
    # 对比有连接与无连接区别
    print_gradients(model_without_shortcut, sample_input)
    torch.manual_seed(123)
    model_with_shortcut = ExampleNeuralNetwork(layer_sizes, use_shorcut=True)
    print_gradients(model_with_shortcut, sample_input)

# ================================ transfomer block ================================
# 构建Transformer
class TransformerBlock(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.atten = MultiHeadAttention(
            cfg["emb_dim"],
            cfg["emb_dim"],
            cfg["context_length"],
            cfg["drop_rate"],
            cfg["n_heads"],
            cfg["qkv_bias"])
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.ff = FeedForward(cfg)
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        bak_x = x
        x = self.norm1(x)
        x = self.atten(x)
        x = self.dropout(x)
        x += bak_x

        bak_x = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)

        return x + bak_x
# 提问 6：Transformer核心思想

# 测试TransformBlock
def test_TransformerBlock():
    input = torch.randn(2, 3, 768)
    tb = TransformerBlock(GPT_CONFIG_124M)
    output = tb(input)

    print("shape of input: ", input.shape)
    print("shape of output: ", output.shape)

# ================================ 实现GPT ================================
class GPTModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.tfb = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x):
        batch_size, num_tokens = x.shape
        tok_embeds = self.tok_emb(x)
        pos_embeds = self.pos_emb(torch.arange(num_tokens, device=x.device))
        x = tok_embeds + pos_embeds
        x = self.dropout(x)
        x = self.tfb(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits

# 测试GPTModel生成维度
def test_GPTModel():
    tokenizer = tiktoken.get_encoding("gpt2")

    # 提问 7：text1，text2如果长度不同，有什么结果
    batch = []
    text1 = "Every effort moves you"
    text2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(text1)))
    batch.append(torch.tensor(tokenizer.encode(text2)))
    batch = torch.stack(batch, dim=0)

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print("batch: ", batch)
    print("shape of logits: ", logits.shape)
    print(logits)

    # 计算模型总参数量
    total_params = sum([p.numel() for p in model.parameters()])
    print(f"Total number of parameters: {total_params}")

    # 计算权重占用内存
    total_size_byte = total_params * 4 # 假设权重为32位浮点数
    total_size_mb = total_size_byte / 1024 / 1024
    print(f"Total size of parameters: {total_size_mb} Mb")


    # 计算共享权重参数量
    print("token embedding layer weight shape: ", model.tok_emb.weight.shape)
    print("output head layer weight shape: ", model.out_head.weight.shape)

    # 减去共享权重重新计算参数总量
    total_params_gpt2 = total_params - sum([p.numel() for p in model.out_head.parameters()])
    print(f"Total number of GPT-2 parameters: {total_params_gpt2}")

# practice4.1
def practice4_1():
    # 计算前馈网络层与注意力层参数量
    tfb = TransformerBlock(GPT_CONFIG_124M)
    num_ff = sum([p.numel() for p in tfb.ff.parameters()])
    num_atten = sum([p.numel() for p in tfb.atten.parameters()])
    print(f"Total number of FeedForward parameters: {num_ff}")
    print(f"Total number of Attention parameters: {num_atten}")

    dim = GPT_CONFIG_124M["emb_dim"]
    num_ff = (4 * dim * dim) * 2 + (4 * dim + dim)
    num_atten = (dim * dim) * 3 + (dim * dim + dim)
    print(f"Total number of FeedForward parameters: {num_ff}")
    print(f"Total number of Attention parameters: {num_atten}")

    # 提问 8：为什么手动计算参数量与通过函数获得不一致

# practice4.2
def practice4_2():

    gpt_config_medium = GPT_CONFIG_124M.copy()
    gpt_config_medium.update({
        "emb_dim":1024,
        "n_heads":16,
        "n_layers":24
    })
    gpt_config_large = GPT_CONFIG_124M.copy()
    gpt_config_large.update({
        "emb_dim":1280,
        "n_heads":20,
        "n_layers":36
    })
    gpt_config_xl = GPT_CONFIG_124M.copy()
    gpt_config_xl.update({
        "emb_dim":1600,
        "n_heads":25,
        "n_layers":48
    })
    gpt_configs = [gpt_config_medium, gpt_config_large, gpt_config_xl]
    for config in gpt_configs:
        model = GPTModel(config)
        total_params = sum([p.numel() for p in model.parameters()])
        print(f"Total number of parameters: {total_params}")

# ================================ 生成文本 ================================ 
# 循环生成词元
def generate_text_simple(model:GPTModel, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # 上下文长度限制
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        ids = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, ids), dim=1)

    return idx

# 测试模型输出
def test_generate():
    tokenizer = tiktoken.get_encoding("gpt2")
    start_text = "Hello, I am"
    encoded = tokenizer.encode(start_text)
    print("encoded: ", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor shape: ", encoded_tensor.shape)

    # 开始输出
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    out = generate_text_simple(model, encoded_tensor, 6, GPT_CONFIG_124M["context_length"])
    print("output: ", out)
    print("output length: ", len(out[0]))

    # ID转为文本输出
    out_text = tokenizer.decode(out.squeeze(0).tolist())
    print(out_text)

# practice4.3：GPTModel中使用三种不同的Dropout
def practice4_3():
    pass
    # 做如下修改
    config_new = GPT_CONFIG_124M.copy()
    config_new.update({
        "drop_rate_atten": 0.1,
        "drop_rate_shortcut": 0.1
    })
    model = GPTModel(config_new)

    # 嵌入层Dropout入参仍然使用：cfg["drop_rate"]
    # 修改 TransformerBlock:__init__()
    # self.atten = MultiHeadAttention(
    #         cfg["emb_dim"],
    #         cfg["emb_dim"],
    #         cfg["context_length"],
    #         cfg["drop_rate_atten"],
    #         cfg["n_heads"],
    #         cfg["qkv_bias"])
    # self.dropout = nn.Dropout(cfg["drop_rate_shortcut"])
    


# ================================ 指定执行 ================================
if __name__=="__main__":
    # test_GPTModel()
    # test_norm()
    # test_layerNorm()
    # cmp_gelu_relu()
    # test_feedforward()
    # test_residual_connection()
    # test_TransformerBlock()
    # test_GPTModel()
    # practice4_1()
    # practice4_2()
    test_generate()