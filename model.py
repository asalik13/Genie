from loss import getLoss
import numpy as np
from optimizer import GA


class Model:
    def __init__(self):
        self.layers = []
        self.weights = []
        self.final = None
        self.optimizer = GA(self.individual, self.fitness)

    def addLayer(self, layer):
        self.layers.append(layer)
        # print("Added layer " + layer.type)
        # print("Layer shape " + str(layer.shape))

    def compile(self, loss):
        self.lossType = loss
        prev = None
        for layer in self.layers:
            prev = layer.compile(prev)

    def loss(self, y, lambda_=0.0):
        return getLoss(self.lossType)(self, y, lambda_)

    def setInput(self, input):
        self.input = input

    def feedforward(self):
        passThrough = self.input
        for layer in self.layers:
            passThrough = layer.activate(passThrough)
        self.final = passThrough
        return(self.final)

    def setWeights(self, flattened_weights):
        prevSize = 0
        flattened_weights = np.array(flattened_weights)
        for layer in self.layers:
            if layer.trainable:

                size = 1
                shape = layer.weights.shape

                for dim in shape:
                    size *= dim
                size += prevSize

                layer.weights = np.array(
                    flattened_weights[prevSize:size]).reshape(*shape)
                prevSize = size

    def getWeights(self):
        flattened_weights = []
        for layer in self.layers:
            if layer.trainable:
                flattened_weights.extend(layer.weights.ravel())
        return flattened_weights

    def individual(self):
        self.compile(loss='binary_cross_entropy')
        weights = self.getWeights()
        return weights

    def fitness(self, individual, target):
        self.setWeights(individual)
        self.feedforward()
        return self.loss(target)

    def train(self, popSize, y):
        p = self.optimizer.population(popSize)

        prevGrade = self.optimizer.grade(p, y)

        for i in range(1000):
            print(prevGrade, self.optimizer.fitness(p[0], y))

            p = self.optimizer.evolve(p, y)
            p = p[:popSize]
            newGrade = self.optimizer.grade(p, y)
            # asteroid = abs(prevGrade - newGrade)
            # p += self.optimizer.population(int(popSize / asteroid))

            if newGrade - prevGrade < 0.00001:
                p = p[:500] + self.optimizer.population(popSize-500)
            prevGrade = newGrade
