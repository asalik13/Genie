from model import Model
from layers import Dense, Flatten

my_model = Model()
my_model.addLayer(Flatten(shape=(28, 28)))
my_model.addLayer(Dense(units=512, activation='relu'))
my_model.addLayer(Dense(units=1024, activation='sigmoid'))
my_model.addLayer(Dense(units=10, activation='sigmoid'))

my_model.compile()
