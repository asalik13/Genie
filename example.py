from model import Model
from layers import Dense, Flatten


my_model = Model()
my_model.addLayer(Flatten(input_shape=(150, 150)))
my_model.addLayer(Dense(units=1024, activation='relu'))
my_model.addLayer(Dense(units=5, activation='softmax'))
my_model.compile(loss='binary_cross_entropy')
flattened_weights = my_model.getWeights()
my_model.setWeights(flattened_weights)
print(flattened_weights.shape)
