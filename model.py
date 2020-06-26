from utils import addOnes
from loss import getLoss
import numpy as np


class Model:
    def __init__(self):
        self.layers = []
        self.weights = []
        self.final = None

    def addLayer(self, layer):
        self.layers.append(layer)
        # print("Added layer " + layer.type)
        # print("Layer shape " + str(layer.shape))

    def compile(self, loss):
        prev = None
        for layer in self.layers:
            prev = layer.compile(prev)
        self.loss = lambda y, lambda_ = 0.0: getLoss(loss)(self, y, lambda_)

    def feedforward(self,input):
        passThrough = input
        for layer in self.layers:
            passThrough = layer.activate(passThrough)
        self.final = passThrough
        return(self.final)

    def setWeights(self, flattened_weights):
        prevSize = 0
        for layer in self.layers:
            if layer.trainable:

                size = 1
                shape = layer.weights.shape
                for dim in shape:
                    size *= dim
                size += prevSize
                layer.weights = flattened_weights[prevSize:size].reshape(
                    *shape)
                prevSize = size

    def getWeights(self):
        flattened_weights = np.array([])
        for layer in self.layers:
            if layer.trainable:
                flattened_weights = np.concatenate(
                    [flattened_weights, layer.weights.ravel()])
        return flattened_weights
