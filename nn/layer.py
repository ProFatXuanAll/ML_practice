import numpy as np
import nn
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

    def forward_pass(self, x):
        return np.matrix(x)

class FullyConnectedLayer:
    def __init__(self, input_dim=0, output_dim=0, weights=None, bias=None, activation=None):
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

        self._w = np.matrix(np.random.random((output_dim, input_dim)))
        self._b = np.matrix(np.random.random((output_dim, 1)))
        self._f = activation
        self._x = np.matrix(np.zeros((input_dim, 1)))

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

    def forward_pass(self, x):
        self._x = np.matrix(x)
        return self._f(self._w.dot(self._x) + self._b)