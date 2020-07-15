import numpy as np


def relu(weights, input):
    inputMat = input@weights.T
    activ = np.maximum(inputMat, 0)
    grad = np.copy(activ)
    grad[grad <= 0] = 0
    grad[grad > 0] = 1
    return activ, grad


def sigmoid(weights, input):
    inputMat = input@weights.T

    activ = _sigmoid(inputMat)
    grad = activ * (1 - activ)
    return activ, grad


def _sigmoid(input):
    activ = 1.0 / (1.0 + np.exp(-input))
    activ[activ == 1] = 0.9999
    activ[activ == 0] = 0.0001
    return activ


def softmax(weights, inp):
    inputMat = inp@weights.T
    result = []
    for i in inputMat:
        result.append(np.exp(
            i) / np.sum(np.exp(i)))
    result = np.array(result)
    grad = result * (1 - result)

    return result, grad


def tanh(weights, input):
    x = input@weights.T
    t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    grad = 1 - np.square(t)
    return t, grad


def getActiv(input):
    activations = {
        'tanh': tanh,
        'softmax': softmax,
        'sigmoid': sigmoid,
        'relu': relu
    }

    return activations.get(input, "Invalid activation")
