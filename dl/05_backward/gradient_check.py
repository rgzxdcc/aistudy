import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
_dl = os.path.dirname(_here)              # dl/
sys.path.append(_dl)                      # 让 from common.layers import * 生效
sys.path.append(os.path.join(_dl, 'common'))  # 让 common/ 内部的 from functions import * 生效
import numpy as np
from mnist import load_mnist
from twolvlnet import TwoLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 求各个权重的绝对误差的平均值
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))