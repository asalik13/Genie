from random import uniform, randint, random, sample
from functools import reduce
from operator import add
import numpy as np
from multiprocessing import Pool
from progressbar import progressbar, ProgressBar


class GA:

    def individual(self, _, loss='binary_cross_entropy'):
        '''
        Compiles model and all of its layers with random initial weights
        '''
        self.model.compile(loss)
        weights = self.model.getWeights()
        return weights

    def fitness(self, individual, target):
        return self.model.loss(target)

    def population(self, count):
        """
        Create a number of individuals (i.e. a population).

        count: the number of individuals in the population
        """
        with Pool(2) as p:
            pop = p.map(self.individual, range(count))
            p.terminate()
            return pop

    def f(self, x):
        return self.fitness(x, self.target)

    def grade(self, pop, target):
        'Find average fitness for a population.'
        self.target = target
        with Pool(2) as p:
            mapped = p.map(self.f, pop)
            summed = reduce(add, mapped)
            return summed / (len(pop) * 1.0)
            p.terminate()

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

    def evolve(self, pop, target, retain=0.2, random_select=0.1, mutate=0.7):
        graded = [(self.fitness(x, target), x) for x in pop]
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0])]
        retain_length = int(len(graded) * retain)
        parents = graded[:retain_length]
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
        possibleNumParents = possibleNumParents
        print(possibleNumParents)
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
        parents = [(self.fitness(x, target), x) for x in parents]
        parents = [x[1] for x in sorted(parents, key=lambda x: x[0])]
        return parents

    def train(self, input, y, popSize, epochs=1000):
        p = self.population(popSize)
        self.model.setInput(input)

        prevGrade = self.grade(p, y)

        for i in range(epochs):
            print(prevGrade, len(p), self.fitness(p[0], y))
            p = self.evolve(p, y)
            print('evolved')
            p = p[:popSize]
            newGrade = self.grade(p, y)
            if newGrade - prevGrade < 0.00001:
                p = p[:int(popSize * 2 / 100)] + \
                    self.population(int(popSize * 98 / 100))
            prevGrade = newGrade


class Adam:
    def __init__(self, b1=0.5, b2=0.9, a=0.03, e=10e-8):
        self.b1, self.b2, self.a, self.e = b1, b2, a, e

    def train(self, input, target, batch_size=60, epochs=500):
        m = 0
        v = 0
        t = 0
        i = 0
        w = self.model.getWeights()
        accuracy = 0
        self.model.setInput(input)
        input = np.copy(input)

        for i in range(epochs):
            print('epoch: ', i)
            batches = np.array_split(input, batch_size)
            for batch in batches:
                self.model.batch = batch
                t += 1
                J, grad, accuracy = self.model.costFunction(input, target, w)
                m = self.b1 * m + (1 - self.b1) * grad
                v = self.b2 * v + (1 - self.b2) * np.square(grad)
                m, v = m / (1 - self.b1**t), v / (1 - self.b2**t)
                w -= self.a * m / (np.sqrt(v) + self.e)
                print(accuracy)
            i += 1

        return accuracy
