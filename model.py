from utils import addOnes


class Model:
    def __init__(self):
        self.layers = []
        self.weights = []

    def addLayer(self, layer):
        self.layers.append(layer)
        # print("Added layer " + layer.type)
        # print("Layer shape " + str(layer.shape))

    def compile(self):
        prev = None
        for layer in self.layers:
            prev = layer.compile(prev)

    def feedforward(self, input):
        print(addOnes(input).shape)
        passThrough = input
        for layer in self.layers:
            passThrough = layer.activate(passThrough)
        print(passThrough)
