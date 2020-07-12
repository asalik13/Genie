from model import Model
from layers import Dense, Flatten
from scipy import optimize
import numpy as np
from activations import _sigmoid
from utils import addOnes
import math

if __name__ == '__main__':
    my_model = Model()
    my_model.addLayer(Flatten(input_shape=(28, 28)))
    my_model.addLayer(Dense(units=32, activation='sigmoid'))
    my_model.addLayer(Dense(units=5, activation='sigmoid'))
    my_model.compile(loss='binary_cross_entropy')
    my_model.setInput(np.random.uniform(-1, 1, size=(1000, 28, 28)))
    y = np.random.uniform(size = (1000, 5))
    my_model.feedforward()
    X = my_model.input
    w = my_model.getWeights()
    my_model.feedforward()
    grad = my_model.backpropagate(y)
    initY = my_model.final
    def costFunction(p):
        my_model.setWeights(p)
        return my_model.cost(y)
    options = {'maxiter': 100}

    res = optimize.minimize(costFunction,
                            w,
                            jac=True,
                            method='TNC',
                            options=options
                            )

    my_model.setWeights(res.x)
    my_model.feedforward()
    a = [(inY[0], predY[0], y[0]) for inY, predY, y in zip(initY, my_model.final, y)]
    for x in a:
        print(x)
