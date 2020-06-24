import numpy as np  # linear algebra
import bitstring


def binary_map(x):
    def toBinary(num):  # normalised decimal => binary 64 bit ndarray
        string = bitstring.BitArray(float=num, length=64).bin
        return np.fromstring(string, 'u1') - ord('0')
    return np.array(list(map(toBinary, x)))


def float_map(x):
    def toFloat(narray):  # binary 64 bit ndarray => normalised decimal
        string = ''.join([str(x) for x in narray])
        bin = bitstring.BitArray(bin=string)
        return bin.float
    return np.array(list(map(toFloat, x)))


def generatePopulation(length, min, max, count):  # generate initial population (bin)

    def generateIndividual(shape):  # Create a member of the population
        return np.random.rand(shape)

    population = []
    for i in range(count):
        individual = binary_map(generateIndividual(length) * (max - min) + min)
        population.append(individual)
    return np.array(population)


def populationData(population):  # (bin)=>(float)
    data = []
    for individual in population:
        data.append(float_map(individual.reshape(-1, 64)))
    return np.array(data)


init = generatePopulation(5, 0, 10, 10)
data = populationData(init)
