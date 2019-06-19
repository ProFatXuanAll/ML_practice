import nn.layer
import nn.activation
import nn.loss

ALL_ACTIVATION_FUNCTION_TYPE = [
    nn.activation.sigmoid,
    nn.activation.linear,
]

ALL_LOSS_FUNCTION_TYPE = [
    nn.loss.square,
]

ALL_LAYERS_TYPE = [
    nn.layer.InputLayer,
    nn.layer.FullyConnectedLayer,
]