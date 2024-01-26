import tensorflow as tf
import numpy as np

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



# Positional maximum and minimum

tf.random.set_seed(42)

M = tf.random.uniform(shape=[50])

print(M)

# find the positional maximum
print(tf.argmax(M),M[tf.argmax(M)])

# find the positional minimum
print(tf.argmin(M),M[tf.argmin(M)])


# Squeesing a tensors means remove all single dimensions
tf.random.set_seed(42)
N = tf.constant(tf.random.uniform(shape=([50])),shape=([1,1,1,1,1,50]))

print(N,N.shape,N.ndim)

N_squeesed = tf.squeeze(N)
print(N_squeesed,N_squeesed.shape,N_squeesed.ndim)


# Square root , log,square

S = tf.cast(tf.range(1,10),dtype=tf.float32)
T = tf.range(1,20)

# Square
print(tf.square(S))

# Square root
print(tf.sqrt(S))

# log
print(tf.math.log(S))


# convert numpy array to tensor

np = np.arange(1,101)
NP = tf.constant(np,shape=([10,5,2]))

print(NP)

# tensor to numpy array

NP = NP.numpy

print(NP)

print(tf.config.list_physical_devices)
print(tf.config.list_logical_devices)