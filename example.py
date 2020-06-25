from model import Model
from layers import Dense, Flatten
import numpy as np
from utils import addOnes
my_model = Model()
my_model.addLayer(Flatten(shape=(150, 150)))
my_model.addLayer(Dense(units=1, activation='relu'))
my_model.addLayer(Dense(units=10, activation='relu'))
my_model.addLayer(Dense(units=512, activation='sigmoid'))
my_model.addLayer(Dense(units=4, activation='sigmoid'))
my_model.addLayer(Dense(units=4, activation='softmax'))

my_model.compile()
print(my_model.feedforward(np.random.rand(150,150)))
