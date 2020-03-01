# Just disables the warning, doesn't enable AVX/FMA
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

### 1.1

sport = tf.constant("Tennis", tf.string)
number = tf.constant(1.41421356237, tf.float64)

print("`sport` is a {}-d Tensor".format(tf.rank(sport).numpy()))
print("`number` is a {}-d Tensor".format(tf.rank(number).numpy()))

sports = tf.constant(["Tennis", "Basketball"], tf.string)
numbers = tf.constant([3.141592, 1.414213, 2.71821], tf.float64)

print("'sports' is a {}-d Tensor with shape: {}".format(tf.rank(sports).numpy(), tf.shape(sports)))
print("'numbers' is a {}-d Tensor with shape: {}".format(tf.rank(numbers).numpy(), tf.shape(numbers)))

### Defining higher-order Tensors ###

'''TODO: Define a 2-d Tensor'''
matrix = tf.constant(
    ["Basketball", "Tennis", "Chess", "Football", "Ping Pong", "Formula 1", "Hockey", "Baseball", "Quidditch", "Sport x", "Sport y", "Sport z"], shape=[3, 4])

assert isinstance(matrix, tf.Tensor), "matrix must be a tf Tensor object"
assert tf.rank(matrix).numpy() == 2

'''TODO: Define a 4-d Tensor.'''
# Use tf.zeros to initialize a 4-d Tensor of zeros with size 10 x 256 x 256 x 3.
#   You can think of this as 10 images where each image is RGB 256 x 256.
images = tf.zeros([10, 256, 256, 3])

assert isinstance(images, tf.Tensor), "matrix must be a tf Tensor object"
assert tf.rank(images).numpy() == 4, "matrix must be of rank 4"
assert tf.shape(images).numpy().tolist() == [10, 256, 256, 3], "matrix is incorrect shape"

row_vector = matrix[1]
column_vector = matrix[:, 2]
scalar = matrix[1, 2]

print("'row_vector': {}".format(row_vector.numpy()))
print("'column_vector': {}".format(column_vector.numpy()))
print("'scalar': {}".format(scalar.numpy()))

###1.2
# Create the nodes in the graph, and initialize values
a = tf.constant(15)
b = tf.constant(61)

# Add them!
c1 = tf.add(a, b)
c2 = a + b  # TensorFlow overrides the "+" operation so that it is able to act on Tensors

print(c1)
print(c2)

### Defining Tensor computations ###

# Construct a simple computation function
'''TODO: Define the operation for c, d, e (use tf.add, tf.subtract, tf.multiply).'''


### Defining Tensor computations ###

# Construct a simple computation function
def func(a, b):
    '''TODO: Define the operation for c, d, e (use tf.add, tf.subtract, tf.multiply).'''
    c = tf.add(a, b)
    d = tf.subtract(b, 1)
    e = tf.multiply(c, d)
    return e


# Consider example values for a,b
a, b = 1.5, 2.5
# Execute the computation
e_out = func(a, b)
print(e_out)