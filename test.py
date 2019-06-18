import numpy as np
import nn
import nn.layer
import nn.model

###########################
# test argument input_dim
###########################
try:
    nn.layer.FullyConnectedLayer()
except ValueError:
    pass
except Exception:
    raise SyntaxError('failed to catch `input_dim` ValueError')
try:
    nn.layer.FullyConnectedLayer(input_dim=-1)
except ValueError:
    pass
except Exception:
    raise SyntaxError('failed to catch `input_dim` ValueError')

try:
    f = nn.layer.FullyConnectedLayer(input_dim=2, output_dim=3)
    print('old weight')
    print(f.weight)
    f.weight = np.array((1,2,3,4,5,6)).reshape((3,2))
    print('new weight')
    print(f.weight)

    print('old bias')
    print(f.bias)
    f.bias = np.array((7,8,9)).reshape(3,1)
    print('new bias')
    print(f.bias)

    m = nn.model.Model(units=2)
    m.add(layer=nn.layer.FullyConnectedLayer, units=4)
    m.add(layer=nn.layer.FullyConnectedLayer, units=3)
except Exception as e:
    print(str(e))

print('pass test')