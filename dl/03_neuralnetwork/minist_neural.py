# 1*784 784*100 100*50 50*10

import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dl.mnist import load_mnist

def getData():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    import pickle
    with open("dl/sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network

# create sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# create softmax function
def softmax(x):
    x = x - np.max(x)  # 防止 exp 溢出
    return np.exp(x) / np.sum(np.exp(x))

# create predict function
def predict(network, x):
    W1,W2,W3 = network["W1"], network["W2"], network["W3"]
    b1,b2,b3 = network["b1"], network["b2"], network["b3"]
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2)
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3)
    y = softmax(a3)
    return y

x_test, t_test = getData()
network = init_network()

accuracy_cnt = 0
for i in range(len(x_test)):
    y = predict(network, x_test[i])
    p = np.argmax(y)
    if p == t_test[i]:
        accuracy_cnt += 1
accuracy = float(accuracy_cnt) / len(x_test)
print("accuracy:", accuracy)
print("accuracy accuracy={:.0f}%".format(accuracy * 100))
print("accuracy_cnt:", accuracy_cnt)
print("x_test.shape:", x_test.shape)
y = predict(network, x_test[0])
print(y)

import PIL.Image as Image
recimg = x_test[0].reshape(28, 28)
img = Image.fromarray(np.uint8(recimg * 255))
img.show()