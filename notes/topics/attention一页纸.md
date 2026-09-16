# Attention 一页纸

> 目标：W3 结束时，这页纸能支撑你在白板上完整画出并讲清 attention。
> 配套：《从零构建大模型》第 3 章 + 手写代码 llm\_app 内对应实现

## 一、公式与每一步的含义

$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

| 步骤 | 矩阵运算                    | 在干嘛      | 形状变化                                       |
| -- | ----------------------- | -------- | ------------------------------------------ |
| 1  | Q @ K.transpose         | 计算注意力分数  | \[b, num\_heads, num\_tokens, num\_tokens] |
| 2  | Q @ K.transpose / √d\_k | 除以 √d\_k | \[b, num\_heads, num\_tokens, num\_tokens] |
| 3  | softmax                 | 归一化      | \[b, num\_heads, num\_tokens, num\_tokens] |
| 4  | ·V                      | 计算上下文    | \[b, num\_heads, num\_tokens, head\_dim]   |

## 二、演进链（书里的推进路线，每步为什么不够→下一步加了什么）

simple attention → 带可训练权重的 self-attention → causal attention → dropout → 多头

1. simple attention: 使用简易自注意力，权重计算自词元嵌入点积
2. self-attention：添加可学习的投影矩阵来增加模型的学习能力。q/k分离，让词元能够作为查询方与被查询方，计算两两相关性。相较于点积计算权重有很大提升
3. causal attention：为了对齐应用场景，将当前输入之后的上下文屏蔽，有助于模型真正学习到从左到右预测，专注于生成下一个单词
4. dropout：深度学习中的随机抛弃神经元策略，能够减小模型学习的过拟合风险
5. 多头注意力：单头注意力的学习内容有限。类似CNN的滤波器，多头注意力能够挖掘上下文更深层次的语法关系，词元联系等，提高模型的表现力

## 三、面试高频三问（自答）

1. 为什么要除以 √d\_k？
    - 当上下文向量维度较高时，计算出的注意力分数会很大。输入过大时，softmax进入饱和区，梯度接近0，导致该部分模型学不到东西。
    - 除以√d_k后可维持方差仍然为1，不会破坏学习数值的均衡

2. causal mask 为什么是生成模型的必需品？
     -  不加掩码，每一个上下文向量都会根据全部输入计算，然而实际模型应用时，他看不到当前输入之后的输入
     -  为了对齐应用场景，将当前输入之后的上下文屏蔽，有助于模型真正学习到从左到右预测，专注于生成下一个单词
 
3. 多头注意力比单头好在哪？
     - 多头注意力能够挖掘上下文更深层次的语法关系，词元联系等，提高模型的表现力

## 四、我的手写实现要点（代码位置 + 踩过的坑）

1. 代码见 llm\_app/02\_llm\_from\_scratch/attention.py : class MultiHeadAttention
2. 踩过的坑：

- contiguous函数拼写错误
- 因果掩码矩阵忘记做num\_tokens裁剪

