import tensorflow as tf

# one hot encoding

data = [0,1,2,3,4,5,6,7]

depth = len(data)

print(depth)

print(tf.one_hot(data,depth=depth))
print(tf.one_hot(data,depth=depth,on_value=5,off_value=2))