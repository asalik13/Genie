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
    my_model.addLayer(Dense(units=128, activation='sigmoid'))
    my_model.addLayer(Dense(units=128, activation='sigmoid'))
    my_model.compile(loss='binary_cross_entropy')
    my_model.setInput(np.random.uniform(-0.5, 0.5, size=(100, 28, 28)))
    y = np.zeros((100, 128))
    my_model.feedforward()
    X = my_model.input
    trainableLayers = [layer for layer in my_model.layers if layer.trainable]
    Theta = [layer.weights for layer in trainableLayers]
    a1 = trainableLayers[0].passThrough
    a2 = addOnes(_sigmoid(a1@Theta[0].T))
    a3 = _sigmoid(a2@Theta[1].T)
    lambda_ = 0.0
    m = 1
    d3 = a3 - y
    d2 = d3@Theta[1][:, 1:] * \
        _sigmoid(a1@Theta[0].T) * (1 - _sigmoid(a1@Theta[0].T))
    D1 = d2.T@a1/y.shape[0]
    D2 = d3.T@a2/y.shape[0]
    my_model.feedforward()
    grad = my_model.backpropagate(y)
    #print(a2 == trainableLayers[1].passThrough)
    #print(d3.shape,d2.shape)
    D = []
    D.extend(D1.flatten())
    D.extend(D2.flatten())
    D = np.array(D)
    print(grad[np.fabs(D-grad)>0.0001])
    #print(len(grad))
    w = my_model.getWeights()
    print(len(w))
'''

    def gradientCheck(epsilon=1e-7):
        w = np.array(my_model.getWeights())
        grad = np.array(my_model.cost(y)[1])
        wp = w + epsilon
        wn = w - epsilon
        my_model.setWeights(wp)
        gradp = np.array(my_model.cost(y)[1])
        my_model.setWeights(wn)
        gradn = np.array(my_model.cost(y)[1])
        approx = (gradp - gradn) / (2 * epsilon)
        num = np.linalg.norm(grad - approx)
        den = np.linalg.norm(grad) + np.linalg.norm(approx)
        print(gradp[0], grad[0])
        return num / den
    print(gradientCheck())
    my_model.feedforward()
    grad = my_model.backpropagate(y)

    w = my_model.getWeights()
    my_model.feedforward()
    initY = my_model.final

    def costFunction(p):
        my_model.setWeights(p)
        return my_model.cost(y)
    options = {'maxiter': 1000}

    res = optimize.minimize(costFunction,
                            w,
                            jac=True,
                            method='TNC',
                            )

    my_model.setWeights(res.x)
    my_model.feedforward()
    print(np.sum(my_model.final - initY))
'''
