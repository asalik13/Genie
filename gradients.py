import numpy as np
from utils import addOnes
from activations import sigmoid

def relu(self, input):
    return np.maximum(addOnes(input)@self.weights.T, 0)


def sigmoid(self, input):
    return 1 / (1 + np.exp(-1 * (addOnes(input)@self.weights.T)))


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
