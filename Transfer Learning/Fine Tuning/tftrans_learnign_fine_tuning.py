import tensorflow as tf
import keras
from keras.preprocessing import image_dataset_from_directory
from keras.callbacks import ModelCheckpoint
from keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt



# pre process the data

IMAGE_SHAPE = (224,224)
BATCH_SIZE = 32

train_dir = "/Users/yogesh/pythoncode/datasets/101_food_classes_10_percent/train"
test_dir = "/Users/yogesh/pythoncode/datasets/101_food_classes_10_percent/test"

train_data = image_dataset_from_directory(train_dir,label_mode="categorical",image_size=(224,224))
test_data = image_dataset_from_directory(test_dir,
                                         label_mode="categorical",
                                         image_size=(224,224),
                                         shuffle=False) # Keep it False,It will not change the order of the image ,same as in dataset

print(train_data)
print(test_data)
#---------------------------------------------------------
# Train Big dog model with trnasfer learning on 10% of 101 food classes

# create a checkpoint callback
checkpoint_path = train_dir
checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                      save_weights_only=True,
                                      monitor="Val_accuracy",
                                      save_best_only=True)

# Data Augmentation layer 
data_augmentation = keras.Sequential([
  preprocessing.RandomFlip("horizontal"),
  preprocessing.RandomRotation(0.2),
  preprocessing.RandomHeight(0.2),
  preprocessing.RandomWidth(0.2),
  preprocessing.RandomZoom(0.2),

],name="data_augmentation")
# Build a Headless(no top layers)

# setup the base model and freeze its layers (this will extract features)
base_model = keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False # this will freeze the top layers

# set up model architecture with trainable top layers
inputs = keras.layers.Input(shape=(224,224,3),name="input_layer")
x = data_augmentation(inputs)# augment images (only happens during training phase)
x = base_model(x, training = False) #it will put the base model in inference mode so weights which need to stay frozez,stay frozen.
x = keras.layers.GlobalAveragePooling2D(name ="gloabal_avg_pool_layer")(x)

outputs = keras.layers.Dense(len(train_data.class_names),activation="softmax")(x)

model = keras.Model(inputs,outputs)

print(model.summary())

# compile model
model.compile(loss=keras.losses.CategoricalCrossentropy(),
              optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=["accuracy"])
# feature extraction 
model_history = model.fit(train_data,
                          epochs=5,
                          validation_data=test_data,
                          validation_steps=int(0.15* len(test_data)),
                          callbacks=[checkpoint_callback]) # validate on only 15% of test data
                          
# evaluate model on the whole test dataset

fine_tunig_results = model.evaluate(test_data)

print(fine_tunig_results)

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


# Fine tuning in the case of overfitting : working well on train data but overperform on unseen or testdata
#we can achieve not overfit using fine tuning

# step 1. Unfreeze all the top layers in base model
base_model.trainable = True
# refreeze the every layer except the last 5
for layers in base_model.layers[:-5]:
    layers.trainable = False
# recompile the model with lower learning rate (best practice to lower the learning rate when fine tuning)
    
fine_tune_model1 = base_model.compile(loss=keras.losses.CategoricalCrossentropy(),
                                      optimizer=keras.optimizers.Adam(learning_rate=0.0001),# learning rates lowered by 10x
                                      metrics=["accuracy"])

# find the trainable layers
for layers in base_model.layers :
    print(layers.name,layers.trainable)

# check which layers are trainable in our base model

for layer_number,layer in enumerate(model.layers[2].layers):
    print(layer_number,layer.name,layer.trainable)

# we can continue unfreezing another 5 layers and so on.

# fine tune for 5 more epochs
model_history_fine_tune = model.fit(train_data,
                          epochs=10,
                          validation_data=test_data,
                          validation_steps=int(0.15* len(test_data)),
                          initial_epoch=model_history.epoch[-1]) # validate on only 15% of test data


# continue fine tuning till you get required models


# save fine tuned model
model_path = "/Users/yogesh/pythoncode/Tensorflow/tensorflow/Transfer Learning/Fine Tuning/saved_models"
model.save(model_path)

# load and evaluate asved model
loaded_model = keras.models.load_model(model_path)

# Evaluate loaded model and compare performance to pre-saved models
loaded_model_results =loaded_model.evaluate(test_data)

print(loaded_model_results)

#make predictions
pred_probs = loaded_model.predict(test_data,verbose=1)

# how many prediction probabilities are there?
print(len(pred_probs))
# shape of predictions
print(len(pred_probs.shape))
print("prod pro sample 0 looks like \n ",pred_probs[0])
print(pred_probs[:10])
print("No of prediction probbalities for sample 0",len(pred_probs[0]))
print("sum of all pred probs for sample  0 : ",sum(pred_probs[0]))

print("The class for highest prediction probabilities by model for sample 0 \n", tf.argmax(pred_probs[0]))

print("All the class names in dataset: \n",test_data.class_names)
print("Class name of the sample 25 ",test_data.class_names[25])

#get the pred classes of each label
pred_classes = pred_probs.argmax(axis=1)

print(pred_classes)
