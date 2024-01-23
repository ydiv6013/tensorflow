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
# Vector (1st-order tensor) :1-dimensional array, 
vector = tf.constant([10,10])
a = tf.constant([1,2,3,4,5,6])
print("Vector",a,a.ndim)
print(vector)
# check dimension of the vector
print(vector.ndim)

# create a matrix(more then 1 dimension) : Matrix (2nd-order tensor)
matrix = tf.constant([[1,2],[3,4]])
a = tf.constant([[1,2,3],[3,4,5]])
print(a)
print(matrix)

# check the dimension of the matrix
print(a.ndim,matrix.ndim)

# create a tensor : (n-order tensor)
tensor = tf.constant([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(tensor)

print(tensor.ndim)

# create a tensor from From Python list

l  = [1,2,3,4] # list
print(tf.constant(l))
t =(1,2,3,4) # tuple
print(tf.constant(t))