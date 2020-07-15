from model import Model
from layers import Dense, Flatten
from scipy import optimize
import numpy as np
from activations import _sigmoid
from utils import addOnes
import math
from optimizer import Adam, GA

my_model = Model()
my_model.addLayer(Flatten(input_shape=(1, 1)))
my_model.addLayer(Dense(units=10, activation='sigmoid'))
my_model.addLayer(Dense(units=5, activation='sigmoid'))
my_model.addLayer(Dense(units=5, activation='softmax'))
my_model.addOptimizer(GA())
my_model.compile(loss='binary_cross_entropy')


X = np.random.uniform(-1, 1, size=(100, 1, 1))
y = np.random.randint(low=0, high=4, size=(100))
y = np.eye(5)[y.reshape(-1)]

# +
w = []
h2 = my_model.train(X,y, popSize= 100, epochs=20)
