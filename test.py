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
    my_model.addLayer(Dense(units=50, activation='sigmoid'))
    my_model.addLayer(Dense(units=5, activation='sigmoid'))
    my_model.addLayer(Dense(units=5, activation='softmax'))
    my_model.compile(loss='binary_cross_entropy')
    my_model.setInput(np.random.uniform(-1, 1, size=(10000, 28, 28)))
    y = np.random.randint(low=0, high=4, size=(10000))
    y = np.eye(5)[y.reshape(-1)]
    X = my_model.input
    w = my_model.getWeights()

    def costFunction(p):
        my_model.setWeights(p)
        return my_model.cost(y)
    options = {'maxiter': 1000}

    res = optimize.minimize(costFunction,
                            w,
                            jac=True,
                            method='TNC',
                            options=options
                            )

    my_model.setWeights(res.x)
    my_model.feedforward()


accuracy = np.argmax(y, axis=1) == np.argmax(my_model.final, axis=1)
accuracy = np.where(accuracy == 1)
print(len(accuracy[0]) / y.shape[0] * 100, '%')
