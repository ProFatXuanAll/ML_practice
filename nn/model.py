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

    def predict(self, x):
        y = np.matrix(x)
        for layer in self._layers:
            y = layer.predict(y)
        return y

    def compute_loss(self, xlist, ylist):
        e = 0
        for x, y in zip(xlist, ylist):
            x = np.matrix(x).T
            y = np.matrix(y).T
            e += self._l(self.predict(x), y)

        return e / len(ylist)

    def back_propagation(self, x, y):
        dL_dy = self._l.d(self.predict(x), y)
        for layer in reversed(self._layers):
            dL_dy = layer.back_propagation(dL_dy)
        return dL_dy

    def fit(self, xlist, ylist):
        for i in range(len(xlist)):
            x = np.matrix(xlist[i]).T
            y = np.matrix(ylist[i]).T
            self.back_propagation(x, y)
