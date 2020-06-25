from utils import addOnes
from loss import getLoss


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
        self.loss = lambda y,lambda_ = 0.0: getLoss(loss)(self,y,lambda_)

    def feedforward(self, input):
        passThrough = input
        for layer in self.layers:
            passThrough = layer.activate(passThrough)
        self.final = passThrough
        return(self.final)

    def getWeights(self):
        weights = []
        for layer in self.layers:
            if(layer.trainable):
                weights.append(layer.weights)
        return weights
