from random import uniform, randint, random, sample
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

        count: the number of individuals in the population
        """
        pop = [self.individual()for x in range(count)]
        return pop

    def grade(self, pop, target):
        'Find average fitness for a population.'
        summed = reduce(add, (self.fitness(x, target) for x in pop))
        return summed / (len(pop) * 1.0)

    def mutate(self, individual):
        """
        Function that mutates parents according to input probability
        Returns mutated individual
        """

        for i in range(randint(1, len(individual))):
            pos_to_mutate = randint(0, len(individual) - 1)
            mutated = np.copy(individual)
            mutated[pos_to_mutate] = uniform(-2, 2)
        return mutated

    def evolve(self, pop, target):
        retain = 0.2
        random_select = 0.1
        mutate = 0.3
        graded = [(self.fitness(x, target), x) for x in pop]
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0])]
        retain_length = int(len(graded) * retain)
        parents = graded
        # randomly add other individuals to
        # promote genetic diversity
        for individual in graded[retain_length:]:
            if random_select > random():
                parents.append(individual)
        # mutate some individuals
        for individual in parents:
            if mutate > random():
                individual = self.mutate(individual)
        # crossover parents to create children
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []
        childLength = len(parents[0])
        possibleNumParents = [i for i in range(
            1, min(childLength + 1, parents_length)) if childLength % i == 0]
        possibleNumParents = possibleNumParents[:-1]
        while len(children) < desired_length:
            numParents = sample(possibleNumParents, 1)[0]
            newChildParents = sample(parents, numParents)
            newChild = []
            start = 0
            division = int(childLength / numParents)
            for newParent in newChildParents:
                newChild += newParent[start:division]
                start = division
                division += division

            children.append(newChild)
        parents.extend(children)
        for i in parents:
            if(np.array(i).size == 0):
                print("meh")
                exit(1)
        parents = [(self.fitness(x, target), x) for x in parents]
        parents = [x[1] for x in sorted(parents, key=lambda x: x[0])]
        return parents
