import tensorflow as tf

# create a tensor using tf.variable

changable_tensor = tf.Variable([10,7])
unchangable_tensor = tf.constant([10,7])

print('\n',changable_tensor,'\n',unchangable_tensor)

# change the value of the elements in tensor

changable_tensor[0].assign(7)

print('\n',changable_tensor)

# additon
changable_tensor.assign_add([2,2])
print('\n',changable_tensor)
# substraction
changable_tensor.assign_sub([2,2])
print('\n',changable_tensor)
# reshape
tensor = tf.Variable([1,2,3,4])
print(tensor)
tf.reshape(tensor, shape=(2,2))
print(tensor)

# create a two tensors

tensor1 = tf.Variable([3,5])
tensor2 = tf.Variable([6,8])

print("add",tensor1 + tensor2)
print("sub" ,tensor1-tensor2)
print("mul",tensor1*tensor2)
print("div",tensor1/tensor2)