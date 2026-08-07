import cupy as cp
x = cp.arange(6).reshape(2, 3).astype('f')
print(x)
x = x.sum(axis=1)
print(x)