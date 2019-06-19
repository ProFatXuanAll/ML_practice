import numpy as np
import nn
import nn.activation
import nn.layer
import nn.loss
import nn.model
import matplotlib.pyplot as plt

try:
    # m = nn.model.Model(units=5)
    # m.add(layer=nn.layer.FullyConnectedLayer,
    #       units=4,
    #       activation=nn.activation.sigmoid)
    # m.add(layer=nn.layer.FullyConnectedLayer,
    #       units=3,
    #       activation=nn.activation.sigmoid)
    # m.add(layer=nn.layer.FullyConnectedLayer,
    #       units=2,
    #       activation=nn.activation.sigmoid)
    # m.add(layer=nn.layer.FullyConnectedLayer,
    #       units=1,
    #       activation=nn.activation.sigmoid)
    # m.compile(loss=nn.loss.square)

    # target_function = lambda x1, x2, x3, x4, x5: (x1+x2)*x3*(x4+x5)
    # xlist = np.random.random((1, 5))
    # ylist = np.array([[target_function(*x)] for x in xlist])

    m = nn.model.Model(units=1)
    m.add(layer=nn.layer.FullyConnectedLayer,
          units=10,
          activation=nn.activation.sigmoid)
    m.add(layer=nn.layer.FullyConnectedLayer,
          units=10,
          activation=nn.activation.sigmoid)
    m.add(layer=nn.layer.FullyConnectedLayer,
          units=10,
          activation=nn.activation.sigmoid)
    m.add(layer=nn.layer.FullyConnectedLayer,
          units=10,
          activation=nn.activation.sigmoid)
    m.add(layer=nn.layer.FullyConnectedLayer,
          units=1,
          activation=nn.activation.sigmoid)
    m.compile(loss=nn.loss.square)

    # xlist = np.linspace(-50,50,1000)
    # ylist = np.square(xlist)
    xlist = np.linspace(-5,5,10)
    ylist = np.square(xlist)

    for epoch in range(100):
        m.fit(xlist, ylist)
        print('epoch {}, loss: {}'.format(epoch, m.compute_loss(xlist, ylist)))

    plt.plot(np.linspace(-50,50,1000),
             np.square(np.linspace(-50,50,1000)),
             label='square')
    plt.plot(np.linspace(-50,50,1000),
             [np.sum(m.predict(x)) for x in np.linspace(-50,50,1000)],
             label='test')
    plt.legend()
    plt.show()

except Exception as e:
    print(str(e))

print('pass test')