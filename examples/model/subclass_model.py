# Just disables the warning, doesn't enable AVX/FMA
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


### Defining a model using subclassing ###

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class SubclassModel(tf.keras.Model):

    # In __init__, we define the Model's layers
    def __init__(self, n_output_nodes):
        super(SubclassModel, self).__init__()
        '''TODO: Our model consists of a single Dense layer. Define this layer.'''
        self.dense_layer = Dense(n_output_nodes, activation="sigmoid")

    # In the call function, we define the Model's forward pass.
    def call(self, inputs):
        return self.dense_layer(inputs)


n_output_nodes = 3
model = SubclassModel(n_output_nodes)
x_input = tf.constant([[1, 2.]], shape=(1, 2))
print(model.call(x_input))


class IdentityModel(tf.keras.Model):

    # As before, in __init__ we define the Model's layers
    # Since our desired behavior involves the forward pass, this part is unchanged
    def __init__(self, n_output_nodes):
        super(IdentityModel, self).__init__()
        self.dense_layer = Dense(n_output_nodes, activation="sigmoid")

    '''TODO: Implement the behavior where the network outputs the input, unchanged, 
          under control of the isidentity argument.'''
    def call(self, inputs, isidentity=False):
        x = self.dense_layer(inputs)
        if isidentity:
            return inputs
        return x


n_output_nodes = 3
model = IdentityModel(n_output_nodes)

x_input = tf.constant([[1, 2.]], shape=(1, 2))
'''TODO: pass the input into the model and call with and without the input identity option.'''
out_activate = model.call(x_input)
out_identity = model.call(x_input, isidentity=True)
print("Network output with activation: {}; network identity output: {}".format(out_activate.numpy(), out_identity.numpy()))