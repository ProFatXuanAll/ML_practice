import numpy as np

def diagnalize(x):
    return np.matrix(np.diag(np.array(x.T)[0]))