import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import preprocess, create_contexts_target, convert_one_hot
from simple_cbow import SimpleCBOW


window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)

contexts, target = create_contexts_target(corpus, window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)
trainer.fit(contexts, target, max_epoch, batch_size)


def predict(context_words):
    """给定上下文词列表（如 ['you', 'goodbye']），返回中间词的概率分布"""
    ids = [word_to_id[w] for w in context_words]
    onehots = np.eye(vocab_size)[ids]
    ctx = onehots.reshape(1, len(ids), vocab_size)

    h0 = model.in_layer0.forward(ctx[:, 0])
    h1 = model.in_layer1.forward(ctx[:, 1])
    h = (h0 + h1) * 0.5
    score = model.out_layer.forward(h)

    score = score - score.max(axis=1, keepdims=True)
    exp = np.exp(score)
    prob = exp / exp.sum(axis=1, keepdims=True)
    return prob[0]


print('\n=== 词汇表 ===')
for wid, w in id_to_word.items():
    print(f'  {wid}: {w}')

print('\n=== 预测测试 ===')
for ctx_words in [['you', 'goodbye'], ['i', 'hello'], ['you', 'i']]:
    prob = predict(ctx_words)
    print(f'\n上下文 {ctx_words} → 预测中间词概率:')
    for wid in range(vocab_size):
        print(f'  {id_to_word[wid]:10s}: {prob[wid]:.4f}')
    top = prob.argmax()
    print(f'  → 最高概率: {id_to_word[top]} ({prob[top]:.4f})')