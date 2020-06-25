import numpy as np
from activations import relu, sigmoid, softmax


class Dense:
    def __init__(self, units, activation, epsilon_init=0.12):
        self.shape = units
        self.epsilon_init = epsilon_init
        self.type = 'Dense'
        self.trainable = True

        if activation == 'relu':
            self.activate = lambda input: relu(self, input)
        elif activation == 'sigmoid':
            self.activate = lambda input: sigmoid(self, input)
        elif activation == 'softmax':
            self.activate = lambda input: softmax(self, input)

    def compile(self, prev):
        self.weights = np.random.rand(
            self.shape, prev + 1) * 2 * self.epsilon_init - self.epsilon_init
        return self.shape


class Flatten:
    def __init__(self, input_shape):
        size = 1
        for dim in input_shape:
            size *= dim

        self.activate = lambda input: input.reshape(input.shape[0],-1)
        self.shape = size
        self.type = 'Flatten'
        self.trainable = False

    def compile(self, prev):
        return self.shape

    def forward(self, input):
        return self.activate(input)
