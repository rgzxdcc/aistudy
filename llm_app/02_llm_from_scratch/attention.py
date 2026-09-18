import torch
from torch._inductor.ir import InputsKernel
import torch.nn as nn

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64],
     [0.22, 0.58, 0.33],
     [0.77, 0.25, 0.10],
     [0.05, 0.80, 0.55]]
)
# ================================ simple-attention ================================
def attention_from_scratch():

    # 计算第二个词元与每个词元的注意力分数
    # 提问 1： 点积与张量乘法的区别
    query = inputs[1]
    atten_score2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        atten_score2[i] = torch.dot(x_i, query)    
    print(atten_score2)

    # 注意力分数归一化
    atten_weight_2_temp = atten_score2 / atten_score2.sum()
    print("atten_weights: ", atten_weight_2_temp)
    print("sum: ", atten_weight_2_temp.sum())

    # 原生softmax归一化
    def softmax_naive(x):
        return torch.exp(x) / torch.exp(x).sum(dim=0)
    
    atten_weight_2_naive = softmax_naive(atten_score2)
    print("atten_weights by naive softmax: ", atten_weight_2_naive)
    print("sum by naive softmax: ", atten_weight_2_naive.sum())

    # 使用torch提供的softmax进行归一化
    atten_weight_2 = torch.softmax(atten_score2, dim=0)
    print("atten_weights by torch softmax: ", atten_weight_2)
    print("sum by torch softmax: ", atten_weight_2.sum())

    # 计算得到上下文向量context_vec_2
    query = inputs[1]
    context_vec_2 = torch.zeros_like(query)
    for i, x_i in enumerate(inputs):
        context_vec_2 += atten_weight_2[i] * x_i
    print(context_vec_2)

    # 计算所有输入的注意力权重
    atten_scores = torch.empty(inputs.shape[0], inputs.shape[0])
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            atten_scores[i, j] = torch.dot(x_i, x_j)
    print(atten_scores)
    # 优化为矩阵计算
    atten_scores = inputs @ inputs.T
    print(atten_scores)
    # 归一化
    atten_weights = torch.softmax(atten_scores, dim=-1)
    print(atten_weights)
    # 验证归一化结果
    atten_sum = atten_weights.sum(dim=-1)
    print(atten_sum)

    # 根据注意力权重，计算所有上下文向量
    context_vecs = atten_weights @ inputs
    print(context_vecs)
    
# ================================ self-attention ================================
def self_attention():
    
    # 提问 2：权重参数与注意力权重区别
    x_2 = inputs[1]
    d_in = inputs.shape[1]
    d_out = 2

    # 声明qkv三个权重矩阵
    torch.manual_seed(123)
    W_q = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_k = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_v = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    # 计算查询向量、k向量，v向量
    # 提问 3：为什么要引入q、k、v三个矩阵
    query_2 = x_2 @ W_q
    print(query_2)

    # 计算注意力分数
    keys = inputs @ W_k
    values = inputs @ W_v
    atten_score_2 = query_2 @ keys.T
    print(atten_score_2)

    # 计算注意力权重
    # 提问 4：为什么要除以√d_k
    d_k = keys.shape[-1]
    atten_weight_2 = torch.softmax(atten_score_2 / d_k**0.5, dim=-1)
    print(atten_weight_2)

    # 生成上下文向量
    context_vec_2 = atten_weight_2 @ values
    print(context_vec_2)


# 实现一个简单的自注意力类
class SimpleAttention_V1(nn.Module):
    def __init__(self, d_in, d_out) -> None:
        super().__init__()
        self.W_q = nn.Parameter(torch.rand(d_in, d_out))
        self.W_k = nn.Parameter(torch.rand(d_in, d_out))
        self.W_v = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        querys = x @ self.W_q
        keys = x @ self.W_k
        values = x @ self.W_v

        # 计算注意力权重，生成上下文矩阵
        atten_scores = querys @ keys.T
        d_k = keys.shape[-1]
        atten_weights = torch.softmax(atten_scores / d_k**0.5, dim=-1)
        context_vecs = atten_weights @ values

        return context_vecs

# 测试attention_v1
def test_simpleAttentionV1():
    torch.manual_seed(123)
    attention = SimpleAttention_V1(d_in=inputs.shape[-1], d_out=2)
    contexts = attention(inputs)
    print(contexts)

# 优化自注意力类
class SimpleAttention_V2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False) -> None:
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        querys = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        # 计算注意力权重，生成上下文矩阵
        atten_scores = querys @ keys.T
        d_k = keys.shape[-1]
        atten_weights = torch.softmax(atten_scores / d_k**0.5, dim=-1)
        context_vecs = atten_weights @ values

        return context_vecs

# 测试attention_v2
def test_simpleAttentionV2():
    torch.manual_seed(789)
    attention = SimpleAttention_V2(d_in=inputs.shape[-1], d_out=2)
    contexts = attention(inputs)
    print(contexts)


# practice 3.1
def practice3_1():
    torch.manual_seed(789)
    attention = SimpleAttention_V2(d_in=inputs.shape[-1], d_out=2)
    contexts = attention(inputs)
    print(contexts)

    # 将SimpleAttention_V2的权重应用到SimpleAttention_V1中
    attention1 = SimpleAttention_V1(d_in=inputs.shape[-1], d_out=2)
    attention1.W_q = nn.Parameter(attention.W_q.weight.T)
    attention1.W_k = nn.Parameter(attention.W_k.weight.T)
    attention1.W_v = nn.Parameter(attention.W_v.weight.T)
    contexts = attention1(inputs)
    print(contexts)

# ================================ 因果掩码 ================================ 
# 提问 5：因果掩码的必要性

def causal_mask():

    # 计算权重
    torch.manual_seed(789)
    attention = SimpleAttention_V2(d_in=inputs.shape[-1], d_out=2)
    query = attention.W_q(inputs)
    keys = attention.W_k(inputs)
    atten_scores = query @ keys.T
    atten_weights = torch.softmax(atten_scores / keys.shape[-1]**0.5, dim=-1)
    print(atten_weights)

    # 创建对角线掩码
    context_length = atten_weights.shape[0]
    mask_simple = torch.tril(torch.ones(context_length, context_length))
    print(mask_simple)

    # 应用掩码
    mask_weights = atten_weights * mask_simple
    print(mask_weights)
    row_sums = mask_weights.sum(dim=-1, keepdim=True)
    print(row_sums)
    mask_weights_norm = mask_weights / row_sums
    print(mask_weights_norm)

    # 优化掩码生成
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = atten_scores.masked_fill(mask.bool(), -torch.inf)
    print(masked)
    atten_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
    print(atten_weights)

    # dropout掩码
    torch.manual_seed(123)
    dropout = nn.Dropout(0.5)
    example = torch.ones(6, 6)
    print(dropout(example))

    torch.manual_seed(123)
    print(dropout(atten_weights))

# 创建因果注意力类
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False) -> None:
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        querys = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        # 计算权重
        atten_scores = querys @ keys.transpose(1, 2)
        atten_scores.masked_fill_(self.mask[:num_tokens, :num_tokens].bool(), -torch.inf)
        atten_weights = torch.softmax(atten_scores / keys.shape[-1]**0.5, dim=-1)
        atten_weights = self.dropout(atten_weights)

        # 生成上下文
        context_vec = atten_weights @ values
        return context_vec


# 创建三维张量
batch = torch.stack((inputs, inputs), dim=0)
# print(batch.shape)

# 测试因果注意力
def test_causalAttention():
    torch.manual_seed(123)
    d_in = batch.shape[-1]
    context_length = batch.shape[1]
    ca = CausalAttention(d_in, 2, context_length, 0.0)
    context_vecs = ca(batch)
    print("shape of context_vecs: ", context_vecs.shape)


# ================================ 多头注意力 ===============================
# 提问 6：为什么需要扩展多头注意力

# 创建多头注意力容器：串联使用多个掩码注意力
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

# 测试多头注意力容器
def test_MultiHeadAttentionWrapper():
    b, context_length, d_in = batch.shape
    
    torch.manual_seed(123)
    d_out, num_heads = 2, 2
    mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0, num_heads)
    context_vecs = mha(batch)
    print("context_vecs shape: ", context_vecs.shape)
    print("context_vecs: ", context_vecs)

# practice3.2: 调整输入参数，使得上下文输出嵌入为2维：维持num_heads=2不变
def practice3_2():
    b, context_length, d_in = batch.shape
    
    torch.manual_seed(123)
    d_out, num_heads = 1, 2
    mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0, num_heads)
    context_vecs = mha(batch)
    print("context_vecs shape: ", context_vecs.shape)
    print("context_vecs: ", context_vecs)

# 创建多头注意力类
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False) -> None:
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_q = nn.Linear(d_in, d_out, qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # [b, num_tokens, d_out]
        q = self.W_q(x) 
        k = self.W_k(x)
        v = self.W_v(x)

        # [b, num_tokens, num_heads, head_dim]
        q = q.view(b, num_tokens, self.num_heads, self.head_dim) 
        k = k.view(b, num_tokens, self.num_heads, self.head_dim)
        v = v.view(b, num_tokens, self.num_heads, self.head_dim)

        # [b, num_heads, num_tokens, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # 提问7： 为何此处要做转置?

        # 计算注意力权重
        attend_score = q @ k.transpose(2, 3) # [b, num_heads, num_tokens, num_tokens]
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attend_score.masked_fill_(mask_bool, -torch.inf)
        attend_weights = torch.softmax(attend_score / k.shape[-1]**0.5, dim=-1)
        attend_weights = self.dropout(attend_weights)

        # 计算上下文张量
        context_vecs = attend_weights @ v # [b, num_heads, num_tokens, head_dim]
        context_vecs = context_vecs.transpose(1, 2) # [b, num_tokens, num_heads, head_dim]
        context_vecs = context_vecs.contiguous().view(b, num_tokens, self.d_out) # [b, num_tokens, d_out]
        context_vecs = self.out_proj(context_vecs)
        
        return context_vecs

# 测试多头注意力
def test_MHA():
    torch.manual_seed(123)
    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiHeadAttention(d_in, 2, context_length, 0.0, 2)
    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs shape: ", context_vecs.shape)

# practice3.3: 初始化GPT2大小的多头注意力
def practice3_3():
    num_heads = 12
    d_in = 768
    d_out = 768
    context_length = 1024

    inputs = torch.rand(2, context_length, d_in)
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.1, num_heads)
    context_vecs = mha(inputs)
    print("context_vecs shape: ", context_vecs.shape)
    

# ================================ 指定执行 ================================
if __name__=="__main__":
    # attention_from_scratch()
    self_attention()
    # test_simpleAttentionV1()
    # test_simpleAttentionV2()
    # practice3_1()
    # causal_mask()
    # test_causalAttention()
    # test_MultiHeadAttentionWrapper()
    # practice3_2()
    # test_MHA()
    # practice3_3()

# ================================ 重写MHA ================================
class MultiHeadAttention_Rewrite(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False) -> None:
        super().__init__()

        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(d_out, d_out)
        self.register_buffer('mask', 
            torch.triu(torch.ones(context_length, context_length), diagonal=1))
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # [b, num_tokens, d_out]
        querys = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # view + transpose
        querys = querys.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        # [b, num_heads, num_tokens, head_dim]
        querys = querys.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # calc attention
        # [b, num_heads, num_tokens, num_tokens]
        atten_scores = querys @ keys.transpose(2, 3)
        masked_scores = atten_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        atten_weights = torch.softmax(masked_scores / keys.shape[-1]**0.5, dim=-1)
        atten_weights = self.dropout(atten_weights)

        # generate context
        # [b, num_heads, num_tokens, head_dim] -> [b, num_tokens, num_heads, head_dim]
        context_vecs = (atten_weights @ values).transpose(1, 2)
        # [b, num_heads, num_tokens, head_dim] -> [b, num_tokens, d_out]
        context_vecs = context_vecs.contiguous().view(b, num_tokens, self.d_out)
        context_vecs = self.proj_out(context_vecs)

        return context_vecs

