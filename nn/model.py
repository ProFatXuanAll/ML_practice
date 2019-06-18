import numpy as np
import nn
import nn.layer

class Model:
    def __init__(self, units=0):
        if type(units) != int:
            raise TypeError('`units` must be `int` type')
        elif units <= 0:
            raise ValueError('`units` must be greater than zero')
        self.layers = [nn.layer.InputLayer(units)]

    def add(self, layer=None, units=0):
        if layer not in nn.ALL_LAYERS_TYPE:
            raise RuntimeError('invalid layer')
        elif type(units) != int:
            raise TypeError('`units` must be `int` type')
        elif units <= 0:
            raise ValueError('`units` must be greater than zero')

        input_dim = self.layers[-1].output_dim
        self.layers.append(layer(input_dim=input_dim, output_dim=units))
