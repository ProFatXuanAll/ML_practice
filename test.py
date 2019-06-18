import numpy as np
import nn
import nn.activation
import nn.layer
import nn.model

try:
    x = np.matrix(np.ones((2,1)))

    m1 = nn.model.Model(units=2)
    m1.add(layer=nn.layer.FullyConnectedLayer, units=4, activation=nn.activation.sigmoid)

    layers1 = m1.layers
    layers1[1].weight = np.ones((4,2))
    layers1[1].bias = np.ones((4,1))

    print(m1.forward_pass(x))

    m1_w = np.matrix(np.ones((4,2)))
    m1_b = np.matrix(np.ones((4,1)))
    y1 = nn.activation.sigmoid(m1_w.dot(x) + m1_b)

    print(y1)

    m2 = nn.model.Model(units=4)
    m2.add(layer=nn.layer.FullyConnectedLayer, units=3, activation=nn.activation.sigmoid)

    layers2 = m2.layers
    layers2[1].weight = np.ones((3,4))
    layers2[1].bias = np.ones((3,1))

    print(m2.forward_pass(y1))

    m2_w = np.matrix(np.ones((3,4)))
    m2_b = np.matrix(np.ones((3,1)))
    y2 = nn.activation.sigmoid(m2_w.dot(y1) + m2_b)

    print(y2)

    m3 = nn.model.Model(units=3)
    m3.add(layer=nn.layer.FullyConnectedLayer, units=5, activation=nn.activation.sigmoid)

    layers3 = m3.layers
    layers3[1].weight = np.ones((5,3))
    layers3[1].bias = np.ones((5,1))

    print(m3.forward_pass(y2))

    m3_w = np.matrix(np.ones((5,3)))
    m3_b = np.matrix(np.ones((5,1)))
    y3 = nn.activation.sigmoid(m3_w.dot(y2) + m3_b)

    print(y3)

    m = nn.model.Model(units=2)
    m.add(layer=nn.layer.FullyConnectedLayer, units=4, activation=nn.activation.sigmoid)
    m.add(layer=nn.layer.FullyConnectedLayer, units=3, activation=nn.activation.sigmoid)
    m.add(layer=nn.layer.FullyConnectedLayer, units=5, activation=nn.activation.sigmoid)

    layers = m.layers
    layers[1].weight = np.ones((4,2))
    layers[1].bias = np.ones((4,1))
    layers[2].weight = np.ones((3,4))
    layers[2].bias = np.ones((3,1))
    layers[3].weight = np.ones((5,3))
    layers[3].bias = np.ones((5,1))

    print(m.forward_pass(x))

except Exception as e:
    print(str(e))

print('pass test')