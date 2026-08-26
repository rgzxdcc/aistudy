import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from common.util import clip_grads

dW1 = np.random.rand(3, 3) * 10
dW2 = np.random.rand(3, 3) * 10
grads = [dW1, dW2]
max_norms = 5.0

print (dW1)
print (dW2)

clip_grads(grads, max_norms)
clipW1, clipW2 = grads
print(clipW1)
print(clipW2)