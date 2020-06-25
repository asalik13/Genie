from model import Model
from layers import Dense, Flatten
import numpy as np
from utils import addOnes
my_model = Model()
my_model.addLayer(Flatten(shape=(150, 150)))
my_model.addLayer(Dense(units=1024, activation='sigmoid'))
my_model.addLayer(Dense(units=2, activation='softmax'))
my_model.compile()
data = my_model.feedforward(np.random.rand(50,150,150))
print(data)
