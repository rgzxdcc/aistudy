import numpy as np

def softmax(x):
    exp_a = np.exp(x)
    sun_exp_a = np.sum(exp_a)
    y = exp_a / sun_exp_a

    return y

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))