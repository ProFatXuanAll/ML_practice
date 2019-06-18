import numpy as np

def square(predict_y, actual_y):
    return np.square(predict_y - actual_y)

def d_square(predict_y, actual_y):
    elementwise = np.array(0.5 * (predict_y - actual_y))
    return np.matrix(np.diag(elementwise.squeeze()))

square.d = d_square
del d_square

def cross_entropy(predict_y, actual_y):
    pass
