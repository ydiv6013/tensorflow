import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist

# Note : Nueral network tends to prefer data in numerical formate  as well as scalled/(normalised)
# numbers in between 0 and 1

# Multiclass classification : problems have more than 2 classes or lables ie. tshirt,shirt,pants
# we are using a tensorflow inbuilt dataset known as fashion_mnist.
#https://www.tensorflow.org/datasets/catalog/fashion_mnist
# https://github.com/zalandoresearch/fashion-mnist
"""
Label	Description
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot
"""
# Dataset Acquisition
(train_data,train_label),(test_data,test_label)=fashion_mnist.load_data()


print("data: ",train_data[0])
print("label:",train_label[0])

# check the shape of the data
print(train_data[0].shape)
print(train_label[0].shape)

# plot the single data
data_index = np.random.randint(1,len(train_data))
print(data_index)
plt.imshow(train_data[data_index])
plt.show()

# making a list of list classes
class_lables = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]


# --------------------------Scaling and normalisation --------------------------------
# Neural network prefers data to be scalled or normalised

# check the min and max value of the data
train_min =train_data.min()
train_max=train_data.max()
test_min =test_data.min()
test_max = test_data.max()

print(train_min,train_max,test_min,test_max)
# Normalize pixel values to be between 0 and 1
# we can do thi with training and testing data by dividing with maximum .
train_data_norm = train_data / train_max
test_data_norm = test_data / test_max

print(train_data_norm[0])
print(test_data_norm[0])

# check the min and max of the scaled data
train_data_norm_min = train_data_norm.min()
train_data_norm_max = train_data_norm.max()

print(train_data_norm_min,train_data_norm_max)

#---------------------------------- Neural network---------------------------------

# set random seed
tf.random.set_seed(42)
# creating a model
multiclass_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(4,activation="relu"),
    keras.layers.Dense(4,activation="relu"),
    keras.layers.Dense(10,activation="softmax") # 10 oputput class for classification
])

# https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy
#Use this crossentropy loss function when there are two or more label classes. 
#We expect labels to be provided in a one_hot representation. 
#If you want to provide labels as integers, please use SparseCategoricalCrossentropy loss.

# one hot encode the label data
test_label = tf.one_hot(test_label,depth=10)
train_label = tf.one_hot(train_label,depth=10)
# compile the model
multiclass_model.compile(loss=keras.losses.CategoricalCrossentropy(),
                         optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
                         metrics=["accuracy"]
)
# fit the model

#validation_split=0.2 means that 20% of the training data will be used for validation during training.
#If you have a separate validation dataset (test_data and test_label), you can use the validation_data parameter instead:
norm_history = multiclass_model.fit(train_data_norm,
                               train_label,
                               epochs=100,
                               validation_data=(test_data_norm,test_label)
)

# not normlised 
non_history = multiclass_model.fit(train_data,
                               train_label,
                               epochs=100,
                               validation_data=(test_data,test_label))



#Normalised ploting : Nueral net performs better with normalised data 
pd.DataFrame(norm_history.history).plot(title="normalised data")
pd.DataFrame(non_history.history).plot(title="non normalise data")
plt.show()
 


