import numpy as np
import nn
import nn.activation
import nn.layer
import nn.model

try:
    x = np.matrix(np.ones((2,1)))


    m1_w = np.matrix(np.ones((4,2)))
    m1_b = np.matrix(np.ones((4,1)))
    y1 = nn.activation.sigmoid(m1_w.dot(x) + m1_b)

    m1 = nn.model.Model(units=2)
    m1.add(layer=nn.layer.FullyConnectedLayer,
           units=4,
           weights=m1_w,
           biases=m1_b,
           activation=nn.activation.sigmoid)

    print('m1 forward pass')
    print(m1.forward_pass(x))
    print('m1 direct calculate')
    print(y1)
    print('-------------------')

    m2_w = np.matrix(np.ones((3,4)))
    m2_b = np.matrix(np.ones((3,1)))
    y2 = nn.activation.sigmoid(m2_w.dot(y1) + m2_b)

    m2 = nn.model.Model(units=4)
    m2.add(layer=nn.layer.FullyConnectedLayer,
           units=3,
           weights=m2_w,
           biases=m2_b,
           activation=nn.activation.sigmoid)

    print('m2 forward pass')
    print(m2.forward_pass(y1))
    print('m2 direct calculate')
    print(y2)
    print('-------------------')

    m3_w = np.matrix(np.ones((5,3)))
    m3_b = np.matrix(np.ones((5,1)))
    y3 = nn.activation.sigmoid(m3_w.dot(y2) + m3_b)

    m3 = nn.model.Model(units=3)
    m3.add(layer=nn.layer.FullyConnectedLayer,
           units=5,
           weights=m3_w,
           biases=m3_b,
           activation=nn.activation.sigmoid)

    print('m3 forward pass')
    print(m3.forward_pass(y2))
    print('m3 direct calculate')
    print(y3)
    print('-------------------')

    m = nn.model.Model(units=2)
    m.add(layer=nn.layer.FullyConnectedLayer,
          units=4,
          weights=m1_w,
          biases=m1_b,
          activation=nn.activation.sigmoid)
    m.add(layer=nn.layer.FullyConnectedLayer,
          units=3,
          weights=m2_w,
          biases=m2_b,
          activation=nn.activation.sigmoid)
    m.add(layer=nn.layer.FullyConnectedLayer,
          units=5,
          weights=m3_w,
          biases=m3_b,
          activation=nn.activation.sigmoid)

    print('m forward pass')
    print(m.forward_pass(x))
    print('m direct calculate')
    print(y3)
    print('-------------------')

except Exception as e:
    print(str(e))

print('pass test')