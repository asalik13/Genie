import numpy as np
from utils import addOnes


def relu(self, input):
    return np.maximum(input@self.weights.T, 0)


def sigmoid(self, input):
    inputMat = input@self.weights.T
    
    activ = _sigmoid(inputMat)
    grad = activ * (1 - activ)
    return activ, grad


def _sigmoid(input):
    activ =  1.0 / (1.0 + np.exp(-input))
    activ[activ == 1] = 0.9999
    activ[activ == 0] = 0.0001
    return activ


def softmax(self, input):
    self.trainable = False
    result = []
    for i in input:
        result.append(np.exp(
            i) / np.sum(np.exp(i)))

    return np.array(result)


def tanh(self, input):
    x = input@self.weights.T
    t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return t


def getActiv(self, input):
    activations = {
        'tanh': tanh,
        'softmax': softmax,
        'sigmoid': sigmoid,
        'relu': relu
    }

    return activations.get(input, "Invalid activation")
