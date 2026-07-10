import numpy as np

# 计算梯度
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    # np.nditer 能逐元素遍历多维数组,并提供 multi_index
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (h * 2)
        x[idx] = tmp_val

        it.iternext()

    return grad

# 梯度下降法，得到x的极小值
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x
