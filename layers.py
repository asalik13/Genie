import numpy as np


class Dense:
    def __init__(self, units, activation, epsilon_init=0.12):
        self.units = units
        self.epsilon_init = epsilon_init
        if activation == 'relu':
            self.activate = lambda input: np.maximum(self.weights * input, 0)
        elif activation == 'sigmoid':
            self.activate = lambda input: 1 / \
                (1 + np.exp(self.weights * input))

    def compile(self, prev):
        self.weights = np.random.rand(
            self.units, 1 + prev) * 2 * self.epsilon_init - self.epsilon_init
        return self.weights, self.units+1

    def forward(self, input):
        return self.activate(input)


class Flatten:
    def __init__(self, shape):
        size = 1
        for dim in shape:
            size *= dim

        self.activate = lambda input: np.ndarray.flatten(input)
        self.shape = size

    def compile(self):
        return self.size

    def forward(self, input):
        return self.activate(input)
