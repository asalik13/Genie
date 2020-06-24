from utils import generatePopulation
from random import randint, random
import numpy as np


class Optimizer:
    def fitness(individual, targets):
        fitness = 0
        """
        Function for determining fitness of individual,
        similar to a loss function, lower is better
        """
        return fitness

    def grade(population, targets):
        grade = 0
        """
        Function for determining overall fitness of the population
        Heplful for verbose tracking
        """
        return grade

    def mutate(individual):
        """
        Function that mutates parents according to input probability
        Returns mutated individual
        """
        pos_to_mutate = randint(0, len(individual) - 1)
        mutated = np.copy(individual)
        mutated[pos_to_mutate] = randint(
            min(individual), max(individual))
        return mutated
