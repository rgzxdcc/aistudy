# PyTorch 最小集（手写 GPT 专用速查表）

> 服务对象：《从零构建大模型》第 2–4 章手敲。范围纪律：不学 CNN/GPU/分布式。
> 用法：9.7 突击时过一遍；W3 起每天收工把当天新见的 API 追加进对应块。

## A. 张量操作（第2章起全程）

| API | 干什么 | 我的理解/易错点（自己填） |
|---|---|---|
| `torch.tensor([...])` | 从数据建张量 | |
| `torch.randn(a,b)` | 标准正态随机（权重初始化用） | |
| `torch.manual_seed(123)` | 固定随机种子（可复现） | |
| `.shape / .reshape / .view` | 形状查看与变换 | |
| `.unsqueeze(dim)` / `.squeeze()` | 升维/降维（attention 里加 batch 维） | |
| `.transpose(a,b)` / `.permute()` | 交换维度（多头注意力会频繁用） | |
| `A @ B` / `torch.matmul` | 矩阵乘（attention 的核心运算） | |
| `x.sum(dim=...)` / `.mean(dim=...)` | 按维求和/均值（loss 计算常用） | |
| 索引切片、广播 | 与 NumPy 几乎一致 | |
| `.detach().numpy()` | 转出画图/打印 | |

**类比**：torch.Tensor ≈ np.ndarray + 设备(.to("mps")=M2加速) + 求导。

## B. 自动求导（第5章训练循环）

| API | 干什么 | 我的理解/易错点 |
|---|---|---|
| `requires_grad=True` | 告诉 torch 记录这个张量的运算 | |
| `loss.backward()` | 沿运算图反向算出所有 .grad | |
| `param.grad` | 看到的梯度就是鱼书里手写的反向传播结果 | |
| `with torch.no_grad():` | 关梯度（生成文本时必须：省内存且不污染计算图） | |

## C. nn.Module 体系（第3–4章主角）

| 写法/API | 干什么 | 我的理解/易错点 |
|---|---|---|
| `class Net(nn.Module):` | 惯例：__init__ 里定义子模块，forward() 里写数据流 | |
| `nn.Embedding(vocab, dim)` | 词 id → 向量（第2章词嵌入层） | |
| `nn.Linear(in, out, bias=)` | 全连接（QKV 投影、输出头全是它） | |
| `nn.LayerNorm(dim)` | 层归一化（第4章 GPT 部件） | |
| `nn.Dropout(p)` | 随机丢弃（训练时开，eval 时自动关） | |
| `nn.GELU()` | GPT 用的激活函数 | |
| `model.parameters()` | 把所有可训练参数交给优化器 | |
| `model.eval()` / `model.train()` | 切换 dropout 行为（生成前记得 eval） | |

## D. 训练循环范式（第2章 DataLoader + 第5章预训练）

```python
# 每本书每个模型都是这个骨架，背下结构即可
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for epoch in range(n):
    for x, y in train_loader:        # 自定义 Dataset + DataLoader 产出 batch
        optimizer.zero_grad()        # 清上一轮梯度
        logits = model(x)            # 前向
        loss = loss_fn(logits, y)    # 交叉熵
        loss.backward()              # 反向
        optimizer.step()             # 更新参数
```

## E. 当天新增（W3 起每天追加）

- （示例）9.14：`torch.utils.data.Dataset` 的 `__len__/__getitem__` 约定 —— ……

> 检验：合书能写出"结构 + 数据流形状变化"，API 名可查；说不出形状从哪变到哪 = 没懂。
