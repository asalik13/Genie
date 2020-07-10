from model import Model
from layers import Dense, Flatten
import numpy as np

if __name__ == '__main__':
    my_model = Model()
    my_model.addLayer(Flatten(input_shape=(1, 1)))
    my_model.addLayer(Dense(units=28, activation='sigmoid'))
    my_model.addLayer(Dense(units=10, activation='sigmoid'))
    my_model.compile(loss='binary_cross_entropy')
    my_model.setInput(np.random.rand(100, 1, 1))
    my_model.feedforward()
    y = np.random.rand(100,10)
    my_model.backpropagate(y)
