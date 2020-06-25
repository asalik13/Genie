import numpy as np  # linear algebra


def generatePopulation(length, min, max, count):
    # generate initial normalised population
    population = np.random.rand(length, count) * (max - min) + min
    return population


def addOnes(a):
    return np.concatenate([np.ones((a.shape[0], 1)), a], axis=1)
