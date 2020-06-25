from model import Model
from layers import Dense, Flatten
import numpy as np
from utils import addOnes
my_model = Model()
my_model.addLayer(Flatten(input_shape=(150, 150)))
my_model.addLayer(Dense(units=10, activation='relu'))
my_model.addLayer(Dense(units=1, activation='sigmoid'))
my_model.compile(loss = 'binary_cross_entropy')
data = my_model.feedforward(np.random.rand(4,150,150))
