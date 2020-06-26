from utils import generatePopulation
from random import uniform, randint
from functools import reduce
from operator import add
import numpy as np


class GA:
    def __init__(self, individual, fitness):
        self.individual = individual
        self.fitness = fitness

    def population(self, count):
        """
        Create a number of individuals (i.e. a population).

        count: the number of individuals in the populatioen
        """
        return [self.individual() for x in range(count)]

    def grade(self, pop, target):
        'Find average fitness for a population.'
        summed = reduce(add, (self.fitness(x, target) for x in pop))
        return summed / (len(pop) * 1.0)

    def mutate(individual):
        """
        Function that mutates parents according to input probability
        Returns mutated individual
        """
        pos_to_mutate = randint(0, len(individual) - 1)
        mutated = np.copy(individual)
        mutated[pos_to_mutate] = uniform(
            min(individual), max(individual))
        return mutated
