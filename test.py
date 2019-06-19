import numpy as np
import nn
import nn.activation
import nn.layer
import nn.loss
import nn.model
import matplotlib.pyplot as plt

try:
    m = nn.model.Model(units=1)
    # m.add(layer=nn.layer.FullyConnectedLayer,
    #       units=10,
    #       activation=nn.activation.sigmoid)
    # m.add(layer=nn.layer.FullyConnectedLayer,
    #       units=10,
    #       activation=nn.activation.sigmoid)
    m.add(layer=nn.layer.FullyConnectedLayer,
          units=10,
          activation=nn.activation.linear)
    m.add(layer=nn.layer.FullyConnectedLayer,
          units=10,
          activation=nn.activation.linear)
    m.add(layer=nn.layer.FullyConnectedLayer,
          units=10,
          activation=nn.activation.linear)
    m.add(layer=nn.layer.FullyConnectedLayer,
          units=10,
          activation=nn.activation.linear)
    m.add(layer=nn.layer.FullyConnectedLayer,
          units=10,
          activation=nn.activation.linear)
    m.add(layer=nn.layer.FullyConnectedLayer,
          units=1,
          activation=nn.activation.linear)
    m.compile(loss=nn.loss.square)

    # xlist = np.linspace(-100, 100, 200)
    # ylist = np.cos(xlist)
    xlist = np.linspace(-100, 100, 200)
    ylist = -1 * xlist

    for epoch in range(10):
        m.fit(xlist, ylist)
        print('epoch {}, loss: {}'.format(epoch, m.compute_loss(xlist, ylist)))

    plt.plot(xlist,
             ylist,
             label='actual train')
    plt.plot(xlist,
             [np.sum(m.predict(x)) for x in xlist],
             label='predict train')
    plt.legend()
    plt.show()

except Exception as e:
    print(str(e))

print('pass test')