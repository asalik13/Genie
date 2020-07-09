import numpy as np
from activations import relu, sigmoid, softmax, tanh


class Dense:
    def __init__(self, units, activation, epsilon_init=0.12):
        self.shape = units
        self.epsilon_init = epsilon_init
        self.type = 'Dense'
        self.trainable = True
        self.activation = activation
        self.last_layer = False

    def activate(self, input):
        activation = self.activation
        output = None
        if activation == 'relu':
            output = relu(self, input)
        elif activation == 'sigmoid':
            output,_ = sigmoid(self, input)
        elif activation == 'tanh':
            output = tanh(self, input)
        elif activation == 'softmax':
            output = softmax(self, input)
        return output

    def compile(self, prev):

        self.weights = np.random.rand(
            self.shape, prev + 1)

        return self.shape


class Flatten:
    def __init__(self, input_shape):
        size = 1
        for dim in input_shape:
            size *= dim
        self.shape = size
        self.type = 'Flatten'
        self.trainable = False

    def activate(self, input):
        return input.reshape(input.shape[0], -1)

    def compile(self, prev):
        return self.shape

    def forward(self, input):
        return self.activate(input)
