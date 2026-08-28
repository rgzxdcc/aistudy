import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 上级 dlp

import numpy as np
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq

# 读入数据条
(x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')

# 将每个输入序列的顺序翻转（seq2seq 实验中常用）
x_train = x_train[:, ::-1]
x_test = x_test[:, ::-1]
# 正常训练正确率：7.84%，翻转输入后，准确率：54.44%

char_to_id, id_to_char = sequence.get_vocab()

# 设定超参数
vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 128
batch_size = 128
max_epoch = 25
max_grad = 5.0

# 生成模型/优化器/训练器
# model = Seq2seq(vocab_size, wordvec_size, hidden_size)
# 使用PeekySeq2seq + 反向输入后，准确率：
model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# 学习评估
acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(model, question, correct, id_to_char, verbose)

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('val acc %.3f%%' % (acc * 100))