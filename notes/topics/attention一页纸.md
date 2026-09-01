# Attention 一页纸

> 目标：W3 结束时，这页纸能支撑你在白板上完整画出并讲清 attention。
> 配套：《从零构建大模型》第 3 章 + 手写代码 llm_app 内对应实现

## 一、公式与每一步的含义

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| 步骤 | 矩阵运算 | 在干嘛 | 形状变化 |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |

## 二、演进链（书里的推进路线，每步为什么不够→下一步加了什么）

simple attention → 带可训练权重的 self-attention → causal attention → dropout → 多头

## 三、面试高频三问（自答）

1. 为什么要除以 √d_k？
2. causal mask 为什么是生成模型的必需品？
3. 多头注意力比单头好在哪？

## 四、我的手写实现要点（代码位置 + 踩过的坑）

-
