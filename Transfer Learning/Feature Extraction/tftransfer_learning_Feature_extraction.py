import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
import datetime as dt
import os
from keras.callbacks import TensorBoard
import tensorflow_hub  as hub
import matplotlib.pyplot as plt

# Types of Transfer Learning
"""
*"As is"  : Transfer learning using existing model without changes 
            e.g  Use imagenet model on ImageNet classes ,none of your own
"Feature Extraction " : Transfer learning using the pre learned patterns of an existing model
            e.g EfficientNetV0 trained on Imagenet, adjust the output layer for 
            own problem
"Fine Tuning" : Use the pre learned patterns of existring model and fine tune many of all of the 
            underlying layers

"""

# pre process the data

IMAGE_SHAPE = (224,224)
BATCH_SIZE = 32

train_dir = "/Users/yogesh/pythoncode/datasets/10_food_classes_10_percent/train"
test_dir = "/Users/yogesh/pythoncode/datasets/10_food_classes_10_percent/test"

train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)

print("Training Images \n")
train_data = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=IMAGE_SHAPE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

print("Testing images \n")
test_data = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size= IMAGE_SHAPE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# settinf up callbacks
# TensorBoard callbacks
# ModelCheckpoint callbacks
# EarlyStopping callbacks

# Create a TensorBoard callbacks 

def create_tensorboard_callback(dir_name,experiment_name) :
    """
    tracking Experiments using tensorboard callback .

    """
    log_dir = os.path.join(dir_name,experiment_name,dt.datetime.now().strftime("%D%M%Y-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    print(f"Saving Tensorboard log files ro : {log_dir}")

    return tensorboard_callback

# creating models using TensorFlow Hub


# create a function to create a model from url

def create_model(model_url,num_classes):
    """
    Takes tensorflow hub url and create a Keras Sequential model with it

    model_url(str) : A tensorflow Hub model urls
    num_classes(int) : Number of output neurons on the output layer,
            no of target classes

    Returns :
    It will returns a Seuqntial model.
    """
    # download the pre trained models and save as keras layers.
    feature_extracted_layer = hub.KerasLayer(model_url,
                                            trainable= False, # Freeze the already learned patterns
                                            name ="feature_extraction_layer",
                                            input_shape = (224,224,3))
    # create a model

    model = keras.Sequential([
        feature_extracted_layer,
        keras.layers.Dense(num_classes,activation="softmax",name= "output_layer")

    ])
    return model

# create a Resnet_50 Model using Transfer Learning

resnet50_url = "https://www.kaggle.com/models/tensorflow/resnet-50/frameworks/TensorFlow2/variations/classification/versions/1"

resnet_model50 = create_model(resnet50_url,
                              num_classes=10)

resnet_model50.summary()

# compile the model

resnet_model50.compile(loss=keras.losses.CategoricalCrossentropy(),
                       optimizer=keras.optimizers.Adam(learning_rate=0.001),
                       metrics="accuracy")

# fit the model

model_history = resnet_model50.fit(
                train_data,
                epochs=2,
                steps_per_epoch= len(train_data),
                validation_data=test_data,
                validation_steps=len(test_data),
                callbacks=[create_tensorboard_callback(dir_name="tensorflow_hub_logs",
                                                       experiment_name="resnet50")]
)

# evaluate the model

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

plot_loss_curves(model_history)


# to View the tensorboard Hub logs use the below command in terminal
# tensorboard --logdir=/Users/yogesh/pythoncode/Tensorflow/tensorflow_hub_logs
# open this link to view :     http://localhost:6006/