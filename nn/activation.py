import numpy as np
import nn.utils

def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    s = sigmoid(x)
    elementwise = np.array(np.multiply(s, 1 - s))
    return nn.utils.diagnalize(elementwise)

sigmoid.d = d_sigmoid
del d_sigmoid