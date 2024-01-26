import tensorflow as tf
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model



X = tf.cast(tf.constant(np.arange(-100,100,4)),dtype=tf.float32)
# Reshape X to have 2 dimensions
X = tf.reshape(X, (-1, 1))
y = tf.cast(tf.constant(np.arange(5,500,10)),dtype=tf.float32)

#Set Random seed
tf.random.set_seed(42)

# create a model
model = keras.Sequential(name="my_sequential_model")
model.add(layers.Input(shape=(1,)))
model.add(layers.Dense(50,activation=None , name ="Input_layer_1"))
model.add(layers.Dense(100,activation=None , name ="Input_layer_2"))
model.add(layers.Dense(1, name = "Output_layer"))

# compile the model

model.compile(loss=keras.losses.mae,
              optimizer=keras.optimizers.Adam(learning_rate=0.1),
              metrics=keras.metrics.mean_absolute_error)

# split the dataset into train and test set
X_train = X[:40]
X_test = X[40:]
y_train = y[:40]
y_test = y[40:]

print(X_train,X_test)
# fit the model
model.fit(X_train,y_train,epochs=100)


#predict the model to unseen data
y_pred = model.predict(np.array([100.0]))
print(y_pred)
# predict to test data
y_pred_test = model.predict(X_test)
print(y_pred_test)
# Model Summary
print(model.summary())


# evaluate the model
print(model.evaluate(X_test,y_test))

# calculate the mean absolute error
print(tf._metrics.mean_absolute_error(y_test,tf.squeeze(y_pred_test)))

# calculate the mean square error
print(tf._metrics.mean_squared_error(y_test,tf.squeeze(y_pred_test)))


# plot the data
plt.scatter(X_train,y_train,c="red",label ="Training")
plt.scatter(X_test,y_test,c="green",label ="Testing")
plt.scatter(X_test,y_pred_test,c="blue",label ="X_test Pediction")
plt.legend()
plt.show()

# save model
model.save("/Users/yogesh/pythoncode/Tensorflow/tensorflow/model1.h5",overwrite=False)

# load the saved model
saved_model = keras.models.load_model("/Users/yogesh/pythoncode/Tensorflow/tensorflow/model1.h5")

# predict using saved models

saved_y_pred_test = saved_model.predict(X_test)

print(saved_y_pred_test)
