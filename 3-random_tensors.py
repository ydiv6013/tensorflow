import tensorflow as tf

# create two random tensor (same tensor)

random1 = tf.random.Generator.from_seed(13) # set seed for reproducibility
random1 = random1.normal(shape =(3,2))
print(random1)

random2 = tf.random.Generator.from_seed(12) # seed = any value
random2 = random2.normal(shape =(3,2))
print(random2)

print(random1 , random2, random1 == random2 )

random3 = tf.random.uniform(shape=(5,5),minval=-5,maxval=5,seed=10)
tf.random.set_seed(15)
random4 = tf.random.uniform(shape=(5,5),minval= -10,maxval=10)


print(random3,random4,random3 == random4)