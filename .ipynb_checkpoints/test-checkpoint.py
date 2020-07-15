from model import Model
from layers import Dense, Flatten
from scipy import optimize
import numpy as np
from activations import _sigmoid
from utils import addOnes
import math
from optimizer import Adam

my_model = Model()
my_model.addLayer(Flatten(input_shape=(28, 28)))
my_model.addLayer(Dense(units=50, activation='sigmoid'))
my_model.addLayer(Dense(units=5, activation='sigmoid'))
my_model.addLayer(Dense(units=5, activation='softmax'))
my_model.addOptimizer(Adam())
my_model.compile(loss='binary_cross_entropy')


X = np.random.uniform(-1, 1, size=(1000, 28, 28))
y = np.random.randint(low=0, high=4, size=(1000))
y = np.eye(5)[y.reshape(-1)]

# +
h = []
for i in range(10):
    h1 = my_model.train(X, y, epochs=20)
    h2 = my_model.opt.trainR(X,y,epochs=20)
    h.append(h2-h1)
print(np.sum(h)/len(h))
    

# -


