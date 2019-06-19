import numpy as np
import nn.utils

def sigmoid(x):
    x = np.where(x >= -500, x, -500)
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    s = sigmoid(x)
    elementwise = np.array(np.multiply(s, 1 - s))
    return nn.utils.diagnalize(elementwise)

sigmoid.d = d_sigmoid
del d_sigmoid

def linear(x):
    return x

def d_linear(x):
    return np.eye(x.shape[0])

linear.d = d_linear
del d_linear