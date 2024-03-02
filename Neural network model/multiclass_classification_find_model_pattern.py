from audioop import bias
from matplotlib import axis
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import cv2

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
#plt.imshow(train_data[data_index])
#plt.show()

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

#---------------------------------- Neural network find model pattern---------------------------------

# set random seed
tf.random.set_seed(42)
# creating a model
multiclass_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(4,activation="relu"),
    keras.layers.Dense(4,activation="relu"),
    keras.layers.Dense(10,activation="softmax") # 10 oputput class for classification
])

# find the layeras of the model

pattern = multiclass_model.layers
print(pattern)

# get the pattern of the layers
weights ,biases = multiclass_model.layers[1].get_weights()

print(weights)
print(weights.shape)
print(biases)
print(biases.shape)

model_summary = multiclass_model.summary()

print(model_summary)
# finding the ideal learning rate

# create a learnig rate call back 
lr_schedular = keras.callbacks.LearningRateScheduler(lambda epochs : 1e-3 * 10 **(epochs/20))

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
n_epochs = 5
#validation_split=0.2 means that 20% of the training data will be used for validation during training.
#If you have a separate validation dataset (test_data and test_label), you can use the validation_data parameter instead:
norm_history = multiclass_model.fit(train_data_norm,
                               train_label,
                               epochs=n_epochs,
                               validation_data=(test_data_norm,test_label),
                               callbacks=lr_schedular
)
"""
# not normlised 
non_history = multiclass_model.fit(train_data,
                               train_label,
                               epochs=100,
                               validation_data=(test_data,test_label),
                               callbacks=lr_schedular
                               )
"""


#Normalised ploting : Nueral net performs better with normalised data 
pd.DataFrame(norm_history.history).plot(title="normalised data")
#pd.DataFrame(non_history.history).plot(title="non normalise data")
plt.show()
 

# plot the learnig rate decay curve

lrs = 1e-3 * (10 ** (tf.range(n_epochs)/20))
plt.semilogx(lrs,norm_history.history["loss"])
plt.xlabel("Learnig rate")
plt.ylabel("Loss")
plt.title("Finding the ideal learning rate")
#plt.show()

# Note : From the graph loswst point on the curve would be your ideal learning rate
print("Ideal learning rate",10 **-3) 


# -----------------prediction 
y_pred = multiclass_model.predict(test_data_norm)

print("y_pred (normalised)",y_pred)

# Convert y_pred to discrete labels
y_pred_labels_discreate = tf.argmax(y_pred, axis=1)
print("y pred labels \n",y_pred_labels_discreate)
# Convert test_data to discrete labels
test_labels_discrete = tf.argmax(test_label, axis=1)
print("test labels \n",test_labels_discrete)

# convert all the prediction into interger numbers
print("y_pred \n",y_pred.argmax(axis=1))


# make prediction on random images to check model performance

img = cv2.imread("/Users/yogesh/pythoncode/Tensorflow/tensorflow/Neural network model/image_data_set/sandals.jpeg")
# Convert the image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# resize the original  image the 28*28
img = cv2.resize(img_gray,dsize=(28,28))
# normalise the resized image pixel values to be in between 0 and 1
img_min =img.min()
img_max=img.max()
img_norm = img/img_max
print(img_norm)

# reshape the normalised image
img_norm_reshape = img.reshape(1,28,28)

# make a prediction on reshaped normalised image 
prediction = multiclass_model.predict(img_norm_reshape)

print(prediction.argmax(axis=1))


# -----------------evaluate the model
loss,accuracy = multiclass_model.evaluate(test_data_norm,test_label)

# create a confusion matrix
cm = confusion_matrix(y_true=test_labels_discrete,y_pred=y_pred_labels_discreate)

print(cm)

display = ConfusionMatrixDisplay(cm)
display.plot()
plt.show()

