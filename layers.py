import numpy as np
from utils import addOnes
from activations import getActiv


class Dense:
    def __init__(self, units, activation, epsilon_init=0.12):
        self.shape = units
        self.epsilon_init = epsilon_init
        self.type = 'Dense'
        self.trainable = True
        self.activation = activation
        self.last_layer = False

    def activate(self, input):
        activationType = self.activation
        output = None
        output, grad = getActiv(activationType)(self.weights,input)
        if self.last_layer is False:
            output = addOnes(output)

        if self.trainable:
            self.grad = grad
        return output

    def compile(self, prev):

        self.weights = np.random.uniform(
            low=-1, high=1, size=(self.shape, prev + 1))

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
        output = input.reshape(input.shape[0], -1)
        if self.last_layer is False:
            output = addOnes(output)
        return output

    def compile(self, prev):
        return self.shape

    def forward(self, input):
        return self.activate(input)
