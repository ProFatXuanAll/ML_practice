import numpy as np
import nn
import nn.layer

class Model:
    def __init__(self, units=0):
        if type(units) != int:
            raise TypeError('`units` must be `int` type')
        elif units <= 0:
            raise ValueError('`units` must be greater than zero')
        self._layers = [nn.layer.InputLayer(units)]

    @property
    def layers(self):
        return self._layers

    def add(self,
            layer=None,
            units=0,
            weights=None,
            biases=None,
            activation=None):

        if layer not in nn.ALL_LAYERS_TYPE:
            raise RuntimeError('invalid layer')
        elif type(units) != int:
            raise TypeError('`units` must be `int` type')
        elif units <= 0:
            raise ValueError('`units` must be greater than zero')
        elif activation not in nn.ALL_ACTIVATION_FUNCTION_TYPE:
            raise ValueError('invalid activation function')

        input_dim = self._layers[-1].output_dim

        self._layers.append(layer(input_dim=input_dim,
                                  output_dim=units,
                                  weights=weights,
                                  biases=biases,
                                  activation=activation))

    def compile(self, loss=None):
        if loss not in nn.ALL_LOSS_FUNCTION_TYPE:
            raise RuntimeError('invalid loss function')

        self._l = loss

    def forward_pass(self, x):
        y = np.matrix(x)
        for layer in self._layers:
            y = layer.forward_pass(y)
        return y

    def compute_loss(self, x, y):
        return self._l(self.forward_pass(x), y)

    def backward_pass(self, x, y):
        dL_dy = self._l.d(self.forward_pass(x), y)

        for layer in reversed(self._layers):
            dL_dy = layer.backward_pass(dL_dy)