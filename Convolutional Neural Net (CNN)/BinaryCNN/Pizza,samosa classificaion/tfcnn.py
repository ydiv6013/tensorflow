import tensorflow as tf
import keras 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


# set the random seed
tf.random.set_seed(42)

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

# build a CNN model (same as the tiny VGG)
conv_layer1 = keras.layers.Conv2D(filters=5,
                        kernel_size=3,
                        activation="relu",
                        strides=(1,1),
                        input_shape =(224,224,3))
conv_layer2 = keras.layers.Conv2D(filters=5,
                        kernel_size=3,
                        activation="relu")
maxpool_layer1 =keras.layers.MaxPool2D(pool_size=2,padding="valid")
conv_layer3 = keras.layers.Conv2D(filters=5,
                        kernel_size=3,
                        activation="relu")
conv_layer4 = keras.layers.Conv2D(filters=5,
                        kernel_size=3,
                        activation="relu")
maxpool_layer2 = keras.layers.MaxPool2D(2)
flatten_layer = keras.layers.Flatten()
dense_layer = keras.layers.Dense(1,activation="sigmoid")


model = keras.Sequential([
        conv_layer1,
        conv_layer2,
        maxpool_layer1,
        conv_layer3,
        conv_layer4,
        maxpool_layer2,
        flatten_layer,
        dense_layer 
        ])

# compile the model

model.compile(loss=keras.losses.BinaryCrossentropy(),
            optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
            metrics=['accuracy'])

# fit the model
model_history = model.fit(train_data,
                    epochs=1,
                    steps_per_epoch=len(train_data),
                    validation_data=valid_data,
                    validation_steps=len(valid_data))

# make a prediction on unseen image
def preprocess_unseen_img (image_path):
    # Load and preprocess a single image

    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the pixel values to be between 0 and 1
    
    return img_array

image_path = "/Users/yogesh/pythoncode/datasets/pizza-samosa/pred/samosa248447.jpg"
normalised_img = preprocess_unseen_img(image_path)
# Make predictions
prediction = model.predict(normalised_img)

# The 'prediction' variable now contains the model's output for the input image
print("Predictions:", prediction)
print(prediction[0][0])

if prediction[0][0] >= 0.4 :
    print("Samosa")
    # plot the image and predicted class
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.title(f"Prediction is : Samosa")
    plt.show()
else :
    print("Pizza")
     # plot the image and predicted class
    img =plt.imread(image_path)
    plt.imshow(img)
    plt.title(f"Prediction is : Pizza")
    plt.show()


# Non Cnn model not works well as compare to CNN model

# Evaluate the model
# plot the training cureves
history_df=pd.DataFrame(model_history.history)
history_df.plot(figsize=(10,7))
plt.show()

# plot the validation and training curves
def plot_loss_curves (model_history):
    """
    returns loss curves for training and validation metrics.
    """
    loss = model_history.history["loss"]
    val_loss = model_history.history["val_loss"]
    accuracy = model_history.history["accuracy"]
    val_accuracy = model_history.history["val_accuracy"]
    
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
plot_loss_curves(model_history)