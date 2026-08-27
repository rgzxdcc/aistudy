import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from common.functions import softmax

T, H = 5, 4
hs = np.random.randn(T, H)

a = np.array([0.8, 0.1, 0.1, 0.0, 0.0])
ar = a.reshape(5, 1).repeat(4, axis=1)
print(ar.shape)

t = hs * ar
print(t.shape)

c = np.sum(t, axis=0)
print(c.shape)

print(hs)
print(c)

s = softmax(c)
print(s)


