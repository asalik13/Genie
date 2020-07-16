# +
import matplotlib.pyplot as plt
import matplotlib as mpl
from model import Model
from layers import Dense, Flatten
from scipy import optimize
from optimizer import Adam
import numpy as np
import pandas as pd
import pickle

train = pd.read_csv('train.csv')
train.head()
X = train.drop('label', axis=1)
Y = train['label'].values
X = X.values.reshape((len(X), 28, 28))
# -

my_model = Model()
my_model.addLayer(Flatten(input_shape=(28, 28)))
my_model.addLayer(Dense(units=256, activation='sigmoid'))
my_model.addLayer(Dense(units=10, activation='sigmoid'))
my_model.addLayer(Dense(units=10, activation='softmax'))
my_model.addOptimizer(Adam())
my_model.compile(loss='binary_cross_entropy')

'''with open('mnist_weights.pkl', 'rb') as f:
    weights = pickle.load(f)
    my_model.setWeights(weights)
'''

y = np.eye(10)[Y.reshape(-1)]
accuracy = my_model.train(X, y, epochs=30)
# +
# %matplotlib inline
fig = plt.figure(figsize=(9, 13))
columns = 7
rows = 7


# ax enables access to manipulate each of subplots
ax = []

for i in range(columns * rows):
    img = X[i]
    inputs = np.array([X[i]])
    output = my_model.predict(inputs)
    output = np.argmax(output, axis=1)
    # create subplot and append to ax
    ax.append(fig.add_subplot(rows, columns, i + 1))
    ax[-1].set_title(str(output[0]))  # set title
    plt.imshow(img, alpha=1)


plt.show()  # finally, render the plot

# -
'''
test = pd.read_csv('test.csv')
X = test.drop('label', axis=1)
Y = test['label'].values
X = X.values.reshape((-1, 28, 28))


with open('mnist_weights.pkl', 'wb') as f:
    pickle.dump(my_model.getWeights(), f)

with open('mnist_weights.pkl', 'rb') as f:
    weights = pickle.load(f)
'''
