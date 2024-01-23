import tensorflow as tf

# rank 4 tensor or 4 dimensional tensor
A = tf.ones(shape=[4,5,4,5])
print(A)
print(A.ndim)
print(A[0])
print(A[1])
print(A[:,3])


print("datatype :",A.dtype)
print("rank or dimension : " ,A.ndim)
print("Size ",tf.size(A))

print("shape of tensor :",A.shape)
print("Shape of 1st element: ",A.shape[0])
print("Shape of last element: ",A.shape[-1])


# tensor indexing

print(A[:,:,:,:])
print(A.shape)

print(A[:2,:1,:3,:2])
print(A.shape)


# add an extra additional dimension

B = tf.constant([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(B)

tf.expand_dims(B, axis=-1) # -1 means last index

print(B)

tf.expand_dims(B, axis=0) # 0 means last index

print(B)