import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))         # 本目录 05_RNN，用于导入 simple_rnnlm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 上级 dlp，用于导入 common/dataset

from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from dataset import ptb
from simple_rnnlm import SimpleRnnlm

# 设定超参数
batch_size = 10
wordvec_size = 100
hidden_size = 100
time_size = 5
lr = 0.1
max_epoch = 100

# 读入训练数据
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)
xs = corpus[:-1]
ts = corpus[1:]

# 生成模型
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimier = SGD(lr)
trainer = RnnlmTrainer(model, optimier)

trainer.fit(xs, ts, max_epoch, batch_size, time_size)
trainer.plot()