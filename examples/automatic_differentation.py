# Just disables the warning, doesn't enable AVX/FMA
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


### Defining a model using subclassing ###

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

### Gradient computation with GradientTape ###
# y = x^2
# Example: x = 3.0
x = tf.Variable(3.0)
# Initiate the gradient tape
with tf.GradientTape() as tape:
    # Define the function
    y = x * x

# Access the gradient -- derivative of y with respect to x
dy_dx = tape.gradient(y, x)

assert dy_dx.numpy() == 6.0

### Function minimization with automatic differentiation and SGD ###
x = tf.Variable([tf.random.normal([1])])
print("Initializing x={}".format(x.numpy()))

learning_rate = 1e-2 # learning rate for SGD
history = []
# Define the target value
x_f = 4

# We will run SGD for a number of iterations. At each iteration, we compute the loss,
#   compute the derivative of the loss with respect to x, and perform the SGD update.
for i in range(500):
    with tf.GradientTape() as tape:
        '''TODO: define the loss as described above'''
        loss = (x - x_f)**2

    # loss minimization using gradient tape
    grad = tape.gradient(loss, x) # compute the derivative of the loss with respect to x
    new_x = x - learning_rate*grad # sgd update
    x.assign(new_x)
    history.append(x.numpy()[0])

# Plot the evolution of x as we optimize towards x_f!
plt.plot(history)
plt.plot([0, 500], [x_f, x_f])
plt.legend(('Predicted', 'True'))
plt.xlabel('Iteration')
plt.ylabel('x value')