from model import Model
from layers import Dense, Flatten
import numpy as np
import math

my_model = Model()
my_model.addLayer(Flatten(input_shape=(1, 1)))
my_model.addLayer(Dense(units=10, activation='tanh'))
my_model.compile(loss='binary_cross_entropy')
my_model.setInput(np.random.rand(100, 1, 1))
y = np.random.rand(100, 10)
my_model.train(1000,y)
