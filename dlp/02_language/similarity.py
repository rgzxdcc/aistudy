import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.util import preprocess, create_to_matrix, cos_similarity


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_to_matrix(corpus, vocab_size)

c0 = C[word_to_id['you']]   # you的单词向量
c1 = C[word_to_id['i']]     # i 的单词向量
print(cos_similarity(c0, c1)) # 0.7071067691154799相似度