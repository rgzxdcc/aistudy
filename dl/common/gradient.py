import numpy as np

# 计算梯度
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh2 - fxh1) / (h * 2)
        x[idx] = tmp_val

    return grad

# 梯度下降法，得到x的极小值
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in step_num:
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x
