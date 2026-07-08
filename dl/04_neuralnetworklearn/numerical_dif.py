def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

# 测试函数
def function_1(x):
    return 0.01 * x **2 + 0.1 * x

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.0, 20.0, 0.1) # 0.0 到 20.0 之间 0.1 步长
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
#plt.show()

numerical_diff(function_1, 5.0)
print(f"numerical_diff(function_1, 5.0) = {numerical_diff(function_1, 5.0):.14f}")
numerical_diff(function_1, 10.0)
print(f"numerical_diff(function_1, 10.0) = {numerical_diff(function_1, 10.0):.14f}")

def function_2(x):
    return x[0] ** 2 + x[1] ** 2

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x) 

        #f(x - h)
        x[idx] = tmp_val - h
        fxh2 = f(x) 

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad


def gradient_descnet(f, init_x, lr=0.01, step_num=1000):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

init_x = np.array([-3.0, 4.0])
gradient_descnet(function_2, init_x, lr = 0.1, step_num=100)
print(init_x)

init_x = np.array([-3.0, 4.0])
gradient_descnet(function_2, init_x, lr = 10, step_num=100)
print(init_x)

init_x = np.array([-3.0, 4.0])
gradient_descnet(function_2, init_x, lr = 1e-10, step_num=100)
print(init_x)     