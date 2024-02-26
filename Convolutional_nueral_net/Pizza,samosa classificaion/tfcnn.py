import tensorflow as tf
import keras 
from keras.preprocessing.image import ImageDataGenerator


# set the random seed
tf.random.set_seed(42)

# pre process the data (scaling/normalisation : image pixel values inbetween 0 & 1)
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range= 0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255)

# setup paths to our data directories
train_dir = "/Users/yogesh/pythoncode/datasets/pizza-samosa/train"
test_dir ="/Users/yogesh/pythoncode/datasets/pizza-samosa/test"

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
history = model.fit(train_data,
                    epochs=15,
                    steps_per_epoch=len(train_data),
                    validation_data=valid_data,
                    validation_steps=len(valid_data))