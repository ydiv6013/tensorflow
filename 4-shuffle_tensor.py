import tensorflow as tf

# create a  tensor

not_shuffled = tf.constant([[10,11],[12,13],[14,15]])

print(not_shuffled)

# shuffel tensor
#tf.random.set_seed(42) # global level seeds
print(tf.random.shuffle(not_shuffled,seed=55)) # operation level seeds
print(tf.random.shuffle(not_shuffled,seed=40)) # operation level seeds