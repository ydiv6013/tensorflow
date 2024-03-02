import tensorflow as tf
import keras
import matplotlib.pylab as plt
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

# set a random seed
seed = tf.random.set_seed(42)

# pre process the data (scaling/normalisation : image pixel values inbetween 0 & 1)
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range= 0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1./255)

# setup paths to our data directories
train_dir = "/Users/yogesh/pythoncode/datasets/pizza-samosa/small/train"
test_dir ="/Users/yogesh/pythoncode/datasets/pizza-samosa/small/test"
pred_dir = "/Users/yogesh/pythoncode/datasets/pizza-samosa/pred"




# import data from directories and turn it into batches
train_data = train_datagen.flow_from_directory(
    directory=train_dir,
    batch_size=32,
    target_size=(224,224),
    class_mode="binary",
)

valid_data = valid_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode="binary",
)
# create a model 

model1 = keras.Sequential([
    keras.layers.Flatten(input_shape =(224,224,3)),
    keras.layers.Dense(4,activation="relu"),
    keras.layers.Dense(4,activation="relu"),
    keras.layers.Dense(1,activation="sigmoid")

])
# model summary
print(model1.summary())

model2 = keras.Sequential([
    keras.layers.Flatten(input_shape =(224,224,3)),
    keras.layers.Dense(100,activation="relu"),
    keras.layers.Dense(100,activation="relu"),
     keras.layers.Dense(100,activation="relu"),
    keras.layers.Dense(1,activation="sigmoid")

])
# compile the model
model2.compile(loss=keras.losses.BinaryCrossentropy(),
              optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=["accuracy"])

model3 = keras.Sequential([
    keras.layers.Flatten(input_shape =(224,224,3)),
    keras.layers.Dense(100,activation="relu"),
    keras.layers.Dense(100,activation="relu"),
    keras.layers.Dense(100,activation="relu"),
    keras.layers.Dense(100,activation="relu"),
    keras.layers.Dense(1,activation="sigmoid")

])
# compile the model
model3.compile(loss=keras.losses.BinaryCrossentropy(),
              optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=["accuracy"])
# model summary
print(model3.summary())

# fit the model
history3 = model3.fit(train_data,
                    epochs=50,
                    steps_per_epoch=len(train_data),
                    validation_data=valid_data,
                    validation_steps=len(valid_data))

# Non Cnn model not works well as compare to CNN model

# Evaluate the model
# plot the training cureves
history3_df=pd.DataFrame(history3.history)
history3_df.plot(figsize=(10,7))
plt.show()

# plot the validation and training curves
def plot_loss_curves (history):
    """
    returns loss curves for training and validation metrics.
    """
    loss = history3.history["loss"]
    val_loss = history3.history["val_loss"]
    accuracy = history3.history["accuracy"]
    val_accuracy = history3.history["val_accuracy"]
    
    no_epochs = range(len(loss)) # gives us no of epochs

    # plot loss
    plt.plot(no_epochs,loss,label = "training loss")
    plt.plot(no_epochs,val_loss,label="Validation loss")
    plt.title("loss")
    plt.xlabel("epochs")
    plt.legend()
    plt.show()

    # plot accuracy
    plt.plot(no_epochs,accuracy,label = "training accuracy")
    plt.plot(no_epochs,val_accuracy,label="Validation accuracy")
    plt.title("accuracy")
    plt.xlabel("epochs")
    plt.legend()
    plt.show()

# plo the loss and accuracy of model curves
plot_loss_curves(history3_df)