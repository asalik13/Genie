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
        self.lossType = loss
        prev = None
        for layer in self.layers:
            prev = layer.compile(prev)
            layer.last_layer = False
        self.layers[-1].last_layer = True

    def loss(self, predicted, y, lambda_=0.0):
        return getLoss(self.lossType)(self, predicted, y, lambda_)

    def setInput(self, input):
        self.input = input

    def feedforward(self):
        passThrough = self.batch
        for layer in self.layers:
            layer.passThrough = passThrough
            passThrough = layer.activate(passThrough)
        self.final = passThrough
        return self.final

    def predict(self, input):
        prevInput = self.batch
        self.batch = input
        self.feedforward()
        output = self.final
        self.batch = prevInput
        return output

    def backpropagate(self, target, lambda_=0.0):
        """
        Returns gradients of trainable layers by backpropagation.
        """
        deltas = []
        trainableLayers = list(
            reversed([layer for layer in self.layers if layer.trainable]))
        currdelta = self.final - target

        for i in range(len(trainableLayers)):
            deltas.append(currdelta)
            currdelta = currdelta@(trainableLayers[i].weights[:, 1:])
            if(i + 1 < len(trainableLayers)):
                currdelta *= (trainableLayers[i + 1].grad)
        Deltas = []
        for delta, layer in zip(deltas, trainableLayers):
            delta = delta.T@layer.passThrough / target.shape[0]
            delta[:, 1:] = delta[:, 1:] + \
                (lambda_ / target.shape[0]) * layer.weights[:, 1:]
            Deltas.append(delta)
        Deltas.reverse()
        Gradients = []
        for delta in Deltas:
            Gradients.extend(delta.flatten())
        return np.array(Gradients)

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

    def cost(self, input, target, lambda_=0.0):
        self.feedforward()
        predicted = self.predict(input)
        J = self.loss(predicted,target)
        accuracy = np.argmax(target, axis=1) == np.argmax(predicted, axis=1)
        accuracy = np.where(accuracy == 1)
        accuracy = len(accuracy[0]) / target.shape[0] * 100
        grad = self.backpropagate(target, lambda_)
        return J, grad, accuracy

    def costFunction(self, input, target, p):
        self.setWeights(p)
        return self.cost(input, target, lambda_=0.03)

    def addOptimizer(self, opt):
        self.opt = opt
        self.opt.model = self
        self.train = self.opt.train
