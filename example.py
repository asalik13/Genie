from model import Model
from layers import Dense, Flatten
import numpy as np
from utils import addOnes
my_model = Model()
my_model.addLayer(Flatten(shape=(1, 3)))
my_model.addLayer(Dense(units=1, activation='sigmoid'))
my_model.addLayer(Dense(units=10, activation='sigmoid'))
my_model.addLayer(Dense(units=100, activation='sigmoid'))
my_model.addLayer(Dense(units=2, activation='sigmoid'))
my_model.compile()
my_model.feedforward(np.random.rand(1,3))
