import numpy as np  # linear algebra
import bitstring


def generatePopulation(length, min, max, count):
    # generate initial normalised population
    population = np.random.rand(length, count) * (max - min) + min
    return population
