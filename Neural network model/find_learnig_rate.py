from filecmp import cmp
import keras
import tensorflow as tf
import pandas as pd
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np


# find the ideal learnig rate

# make a dasta 
X,y = make_circles(n_samples=1000,noise=0.03,random_state=42)

print(X,y )

# create a dataset
dataset = pd.DataFrame({"X0" : X[:,0],"X1" : X[:,1],"Labels" : y})

print(dataset)

# check the shape of the dataset
print(X.shape)
print(y.shape)

# Visulise the dataset

"""plt.scatter(X[:,0],X[:,1],c=y,cmap="RdYlBu")
plt.plot
plt.show()"""

# creating a trainig and testing set
X_train = X[:800]
X_test = X[800:]

y_train = y[:800]
y_test = y[800:]


print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# create a model
tf.random.set_seed(42)

model = keras.Sequential([
    keras.layers.Dense(4,activation="relu"),
    keras.layers.Dense(4,activation="relu"),
    keras.layers.Dense(1,activation="sigmoid")
])

# compile the model
model.compile(loss=keras.losses.BinaryCrossentropy(),
              optimizer=keras.optimizers.legacy.Adam(learning_rate=0.01),
              metrics=["accuracy"]
)
# crate a learning rate callback
lr_scheduler = keras.callbacks.LearningRateScheduler(lambda epoch : 1e-4 * 10**(epoch/20))

# fit the model using lr schedular 

history = model.fit(X_train,
                    y_train,
                    epochs=100,
                    callbacks=[lr_scheduler])

print(history.history)

# plot the history

pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()

# plot the learning rate vs loss
lrs = 1e-4 * (10 ** (tf.range(100)/20))
plt.semilogx(lrs,history.history["loss"])
plt.xlabel("Learning rate")
plt.ylabel("Loss")
plt.title("Learning rate Vs loss")
plt.show()

# evolute the model
loss,accuracy = model.evaluate(X_test,y_test)
print(f"Model loss on test set : {loss} ")
print(f"model accuracy on test set : {(accuracy*100):.2f} %")



