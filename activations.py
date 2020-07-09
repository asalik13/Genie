import numpy as np
from utils import addOnes


def relu(self, input):
    return np.maximum(addOnes(input)@self.weights.T, 0)


def sigmoid(self, input):
    inputMat = addOnes(input)@self.weights.T
    activ = _sigmoid(inputMat)
    grad = activ * (1 - activ)
    return activ, grad


def _sigmoid(input):
    return 1 / (1 + np.exp(-1 * input))


def softmax(self, input):
    input = sigmoid(self, input)
    result = []
    for i in input:
        result.append(np.exp(
            i) / np.sum(np.exp(i)))

    return np.array(result)


def tanh(self, input):
    x = addOnes(input)@self.weights.T
    t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return t
