import numpy as np
from utils import addOnes


class Dense:
    def __init__(self, units, activation, epsilon_init=0.12):
        self.shape = units
        self.epsilon_init = epsilon_init
        self.type = 'Dense'

        if activation == 'relu':
            self.activate = lambda input: np.maximum(
                addOnes(input)@self.weights.T, 0)
        elif activation == 'sigmoid':
            self.activate = lambda input: 1 / \
                (1 + np.exp(-1 * (addOnes(input)@self.weights.T)))
        elif activation == 'softmax':
            self.activate = lambda input:np.exp(input[0]) / np.sum(np.exp(input[0]), axis=0)

    def compile(self, prev):
        self.weights = np.random.rand(
            self.shape, prev + 1) * 2 * self.epsilon_init - self.epsilon_init
        return self.shape

    def forward(self, input):
        return self.activate(input)


class Flatten:
    def __init__(self, shape):
        size = 1
        for dim in shape:
            size *= dim

        self.activate = lambda input: input.reshape(1, -1)
        self.shape = size
        self.type = 'Flatten'

    def compile(self, prev):
        return self.shape

    def forward(self, input):
        return self.activate(input)
