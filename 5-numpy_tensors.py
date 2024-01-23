import tensorflow as tf
import numpy as np

# create a tensor of all ones

print(tf.ones([10,5],tf.int32))
print(tf.ones([5,5]))

# create a tensor of all zeros

print(tf.zeros([10,5],tf.int32))
print(tf.zeros([5,5]))


# convert numpy array to tensor
# capital for matrix and small letter for vector

array = np.arange(1,25)
print(array,array.ndim,array.dtype)

A = tf.constant(array)
print(A,A.ndim)


A = tf.constant(array,shape=(4,3,2))
print(A)
print(A.ndim)