import numpy as np
import nn
import nn.utils
class InputLayer:
    def __init__(self, dim=0, activation=None):
        if type(dim) != int:
            raise TypeError('`dim` must be `int` type')
        elif dim <= 0:
            raise ValueError('`dim` must be greater than zero')

        self._dim = dim

    @property
    def input_dim(self):
        return self._dim

    @property
    def output_dim(self):
        return self._dim

    def predict(self, x):
        return np.matrix(x)

    def back_propagation(self, dL_dy):
        return dL_dy

class FullyConnectedLayer:
    def __init__(self,
                 input_dim=0,
                 output_dim=0,
                 weights=None,
                 biases=None,
                 activation=None):

        if type(input_dim) != int:
            raise TypeError('`input_dim` must be `int` type')
        if type(output_dim) != int:
            raise TypeError('`output_dim` must be `int` type')
        elif input_dim <= 0:
            raise ValueError('`input_dim` must be greater than zero')
        elif output_dim <= 0:
            raise ValueError('`output_dim` must be greater than zero')
        elif activation not in nn.ALL_ACTIVATION_FUNCTION_TYPE:
            raise ValueError('invalid activation function')

        if weights is None:
            self._w = np.matrix(np.random.random((output_dim, input_dim)))
        elif weights.shape != (output_dim, input_dim):
            raise ValueError('`weights` shape inconsist of `(output_dim, input_dim)`')
        else:
            self._w = np.matrix(weights)

        if biases is None:
            self._b = np.matrix(np.random.random((output_dim, 1)))
        elif biases.shape != ((output_dim, 1)):
            raise ValueError('`bias` shape inconsist of `(output_dim, 1)`')
        else:
            self._b = np.matrix(biases)

        self._f = activation
        self._x = np.matrix(np.zeros((input_dim, 1)))
        self._wxb = np.matrix(np.zeros((output_dim, 1)))
        self._y = np.matrix(np.zeros((output_dim, 1)))

    @property
    def weight(self):
        return self._w

    @weight.setter
    def weight(self, new_weight):
        if self._w.shape != new_weight.shape:
            raise ValueError('shape inconsistent of new weight')
        self._w = np.matrix(new_weight)

    @property
    def bias(self):
        return self._b

    @bias.setter
    def bias(self, new_bias):
        if self._b.shape != new_bias.shape:
            raise ValueError('shape inconsistent of new bias')
        self._b = np.matrix(new_bias)

    @property
    def input_dim(self):
        return self._w.shape[1]

    @property
    def output_dim(self):
        return self._w.shape[0]

    def predict(self, x):
        self._x = np.matrix(x)
        self._wxb = self._w.dot(self._x) + self._b
        self._y = self._f(self._wxb)
        return self._y

    def back_propagation(self, dL_dy):
        # dL_dw = dL_dy * dy_dwxb * dwxb_dw
        # dL_db = dL_dy * dy_dwxb * dwxb_db
        # dL_dx = dL_dy * dy_dx

        dy_dwxb = self._f.d(self._wxb)
        dL_dwxb = dL_dy.dot(dy_dwxb)
        dwxb_dw = self._x.T.repeat(self.output_dim, axis=0)
        dL_dw = dL_dwxb.dot(dwxb_dw)
        dL_db = dL_dwxb.dot(np.matrix(np.ones((self.output_dim, 1))))
        dL_dx = np.sum(dL_dy.T.dot(self._w), axis=0) # y1*w11+y2*w21+y3*w31 & y1*w12+y2*w22+y3*w32 & y=mx1 & w=mxn

        self._w = self._w - dL_dw
        self._b = self._b - dL_db

        return nn.utils.diagnalize(dL_dx.T)