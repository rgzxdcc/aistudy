# 从 0 到 1 训练 GPT 我做了什么

> 完成节点：W4（9.26）｜用途：简历"手写 GPT"板块的讲解底稿
> 代码位置：aistudy/llm_app/（W3–W4 产出）

## 一、背景与目标
（为什么手写：目标岗位要求理解大模型原理；手写 = 把"看过"变成"证明会"）

## 二、做了什么（按数据流写完整链路）
- [ ] 分词：BPE 词表训练 + encode/decode
- [ ] 输入：token embedding + position embedding
- [ ] 主体：N × Transformer block（各组件）
- [ ] 输出：线性头 + softmax
- [ ] 训练：cross-entropy + AdamW 训练循环
- [ ] 生成：temperature / top-k 解码
- [ ] 加载开源权重后能生成连贯文本

## 三、技术决策及依据
（例：为什么 position embedding 用可学习矩阵而不是 RoPE——写你当时的真实理由）

## 四、踩坑与解决（2–3 个真实案例）

## 五、结果
（参数量、生成示例前后对比、训练 loss 曲线截图）

## 六、如果重做会怎么改

## 七、预答面试追问（写 5 个可能被问的问题+答案要点）
1.
2.
3.
4.
5.
