import tensorflow as tf

print(tf.__version__)

# create a  Scaler
# Scalar (0th-order tensor): A single numerical value, often denoted by lowercase letters
# example a = 7 
a = tf.constant(7)
b = tf.constant(10)
print(a,b)
# check the dimension of the tensor
print(a.ndim)

# create a vector 
vector = tf.constant([10,10])

print(vector)

# check dimension of the vector
print(vector.ndim)

# create a matrix(more then 1 dimension)
matrix = tf.constant([[10,7],[7,10]])
print(matrix)

# check the dimension of the matrix
print(matrix.ndim)

# create a tensor
tensor = tf.constant([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(tensor)

print(tensor.ndim)

# create a tensor from From Python list

l  = [1,2,3,4] # list
print(tf.constant(l))
t =(1,2,3,4) # tuple
print(tf.constant(t))