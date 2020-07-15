# +
import matplotlib.pyplot as plt
import matplotlib as mpl
from model import Model
from layers import Dense, Flatten
from scipy import optimize
from optimizer import Adam
import numpy as np
import pandas as pd

train = pd.read_csv('train.csv')
train.head()
X = train.drop('label', axis=1)
Y = train['label'].values
X = X.values.reshape((len(X), 28, 28))
# -

my_model = Model()
my_model.addLayer(Flatten(input_shape=(28, 28)))
my_model.addLayer(Dense(units=100, activation='sigmoid'))
my_model.addLayer(Dense(units=10, activation='sigmoid'))
my_model.addLayer(Dense(units=10, activation='softmax'))
my_model.compile(loss='binary_cross_entropy')

my_model.setInput(X)
Y = np.eye(10)[Y.reshape(-1)]


# +
w = my_model.getWeights()


def costFunction(p):
    my_model.setWeights(p)
    return my_model.cost(Y, lambda_=0.03)


opt = Adam()
opt.train(costFunction, w)

'''
options = {'maxiter': 1000}

res = optimize.minimize(costFunction,
                        w,
                        jac=True,
                        method='TNC',
                        options=options
                        )

my_model.setWeights(res.x)
my_model.feedforward()
# -

my_model.setInput(X)
my_model.feedforward()
accuracy = Y == np.argmax(my_model.final, axis=1)
accuracy = np.where(accuracy == 1)
print(len(accuracy[0]) / Y.shape[0] * 100, '%')

# +
# %matplotlib inline
mpl.rcParams['text.color'] = 'white'
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

test = pd.read_csv('test.csv')
X = test.drop('label', axis=1)
Y = test['label'].values
X = X.values.reshape((-1, 28, 28))

import pickle
with open('mnist_weights.pkl', 'wb') as f:
    pickle.dump(my_model.getWeights(), f)

with open('mnist_weights.pkl', 'rb') as f:
    weights = pickle.load(f)
'''
