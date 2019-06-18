import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    s = sigmoid(x)
    elementwise = np.array(np.multiply(s, 1 - s))
    return np.matrix(np.diag(elementwise.squeeze()))

sigmoid.d = d_sigmoid
del d_sigmoid