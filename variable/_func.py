from variable._util import coerce
from variable._var import Variable


def relu(var: Variable) -> Variable:
    return coerce(var).max(0)


def sigmoid(var: Variable) -> Variable:
    return 1 / (1 + coerce(var).exp())
