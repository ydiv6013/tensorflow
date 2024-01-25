import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

x = tf.constant ([[-7,-10,-1.5]])
print(x)
y = tf.constant ([[-25,-12,-0.5]])
print(x)
# Get the absolute value
print(tf.abs(x))
# Get the minimum value
print("minimum",tf.minimum(x,y))
# Get the maximum value
print("maximum",tf.maximum(x,y))

# random tensor with value 0,100 

A = tf.constant(np.random.randint(0,100,size=50))

print(A)

print(tf.reduce_min(A))
print(tf.reduce_max(A))
print(tf.reduce_mean(A))
print(tf.reduce_sum(A))


# standard deviation

print("std ",tf.math.reduce_std(tf.cast(A,dtype=tf.float32)))

# varience

print("std ",tf.math.reduce_variance(tf.cast(A,dtype=tf.float32)))

