# Genie
Genie is a project made for independent research related to applying genetic algorithm on a model's weights to optimize training (matrix multiplication is expensive!). 
If the amount of abstraction in TensorFlow or Keras holds you from understanding what's behind the hood, 
this might be the right place for you. (I basically built it to understand what is actually happening)

Genie's API was purposely kept very similar to that of Tensorflow/Keras, it does lack in a lot of features, but it gives an understanding of multiple things ranging from 

1. The model structure and how it's created when you use model.addLayer() (in model.py) and all the different types of Layers. (in layers.py)
2. How the input gets processed through the different layers, and the way activation functions help in feed forward as well as calculating gradients. You can find these activations in activations.py 
3. The backpropogation algorithm forms the root for most optimization algorithms. Most places explain it with a couple layers but this is how it can be applied to as many layers as you want. 
   You can find model.backpropagate(target) directly in model.py
4. The optimization algorithms. Genie contains 2 optimizers as of now. Adam and a custom one based on genetic optimizers. You can find these in optimizer.py.


Sidenote: If the genetic algorithm seems different to what you have seen before, it's because it uses a multi parent crossover technique for inheritance of genomes that I came upon while researching. 
The other different thing in GA is something I like to call the asteroid affect. Whenever the optimizer gets stuck in a local minima,
98% of individuals are wiped out and new individuals (according to how stuck the optimizer is) are introduced.  

# Demo
handwritten.py contains code for the MNIST database.You'd need to download the .csv files from https://www.kaggle.com/oddrationale/mnist-in-csv, unzip it and change the paths accordingly. 
Feel free to experiment with the number of neurons/layers and run the example with python handwritten.py. I achieved ~98% accuracy in little over 20 epochs!

# Current Work
The current stuff I am working on is to create a version of Adam that chooses new betas in every iteration as well as adding Convolution and Pooling layers. 
