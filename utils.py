import numpy as np  # linear algebra



def addOnes(a):
    return np.concatenate([np.ones((a.shape[0], 1)), a], axis=1)
