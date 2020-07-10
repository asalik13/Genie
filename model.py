from loss import getLoss
import numpy as np
from utils import addOnes
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
            layer.last_layer = False
        self.layers[-1].last_layer = True


    def loss(self, y, lambda_=0.0):
        return getLoss(self.lossType)(self, y, lambda_)

    def setInput(self, input):
        self.input = input

    def feedforward(self):
        passThrough = self.input
        for layer in self.layers:
            passThrough = layer.activate(passThrough)

        self.final = passThrough
        return self.final

    def backpropagate(self, target):
        deltas = []
        trainableLayers = list(reversed([layer for layer in self.layers if layer.trainable]))
        currdelta = self.final - target

        deltas.append(currdelta)

        for layer in trainableLayers:
            print(currdelta.shape)
            currdelta = currdelta@(layer.weights[:, 1:])
            deltas.append(currdelta)

        deltas.reverse()

        Deltas = []




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

    def cost(self, target):
        self.feedforward()
        J = self.loss(target)

    def individual(self, _):
        self.compile(loss='binary_cross_entropy')
        weights = self.getWeights()
        return weights

    def fitness(self, individual, target):
        self.setWeights(individual)
        self.feedforward()
        return self.loss(target)

    def train(self, popSize, y, epochs=1000):
        p = self.optimizer.population(popSize)

        prevGrade = self.optimizer.grade(p, y)

        for i in range(epochs):
            print(prevGrade, len(p), self.optimizer.fitness(p[0], y))
            p = self.optimizer.evolve(p, y)
            p = p[:popSize]
            newGrade = self.optimizer.grade(p, y)
            asteroid = abs(prevGrade - newGrade) + 0.5
            p += self.optimizer.population(int(popSize / asteroid))

            if newGrade - prevGrade < 0.00001:
                p += self.optimizer.population(5 * popSize)
            prevGrade = newGrade
