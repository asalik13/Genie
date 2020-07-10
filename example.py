from model import Model
from layers import Dense, Flatten
import numpy as np

if __name__ == '__main__':
    my_model = Model()
    my_model.addLayer(Flatten(input_shape=(28, 28)))
    my_model.addLayer(Dense(units=256, activation='sigmoid'))
    my_model.addLayer(Dense(units=512, activation='sigmoid'))
    my_model.addLayer(Dense(units=1, activation='sigmoid'))
    my_model.compile(loss='binary_cross_entropy')
    my_model.setInput(np.random.rand(10000, 28, 28))
    my_model.feedforward()
    y = np.random.rand(10000, 1)
    my_model.backpropagate(y)
