import keras
from keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import os
import random


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


model_path = "/Users/yogesh/pythoncode/Tensorflow/tensorflow/Transfer Learning/Fine Tuning/saved_models"

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

#print(pred_classes)

# to get our test labels we need to unravel our test_data BatchDataset
#print(test_data) # its a BatchDataset

y_labels = []

for images,labels in test_data.unbatch():
    y_labels.append(labels.numpy().argmax())#

print(len(y_labels))
print(y_labels)
#----------------------------------------------
# Evaluating our models prediction

# accuracy : (True Positive + True Negative) / Total Predictions
sklearn_accuracy = metrics.accuracy_score(y_true=y_labels,
                          y_pred=pred_classes)

print("Accuracy is : ",sklearn_accuracy)

# compare the both true and pred labels accuracy

print(np.isclose(loaded_model_results[1],sklearn_accuracy))

# confusion matrix
"""
cm =  metrics.confusion_matrix(y_pred=pred_classes,y_true=y_labels).ravel()
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[False,True])
cm_display.plot()
plt.show()"""

# precision : True Positive / (True Positive + False Positive)
precision = metrics.precision_score(y_pred=pred_classes,y_true=y_labels,average='weighted')

print("Precision is : ",precision)

# recall : True Positive / (True Positive + False Negative)

recall = metrics.recall_score(y_pred=pred_classes,y_true=y_labels,average='weighted')

print("Recall is : ",recall)
# F1 score : F-score is the "harmonic mean" of precision and sensitivity
# f1 = 2 * ((Precision * Sensitivity) / (Precision + Sensitivity))

f1_score = metrics.f1_score(y_pred=pred_classes,y_true=y_labels,average='weighted')

print("F1 Score is : ",f1_score)

# classificatio report
report =metrics.classification_report(y_pred=pred_classes,y_true=y_labels)
report_dictionary =metrics.classification_report(y_pred=pred_classes,y_true=y_labels,output_dict=True)

print(report)
print(report_dictionary)



print(pd.DataFrame(report_dictionary))

#--------------------------------------------------------------
# make a prediction on test images

def load_preprocess_image(filename,img_shape = 224,scale = True):
    # read image
    img = tf.io.read_file(filename=filename)
    # decode image into tensor
    img = tf.io.decode_image(img,channels=3)
    # resize the image
    img = tf.image.resize(img,size=[img_shape,img_shape])
    # scale ? Yes /no
    if scale :
        # rescale the image (all the value between 0 and 1)
        return img/255.
    else :
        return img

# make prediction on random images
plt.figure(figsize=(20,10))

for i in range(100):
    # choose a random image from a random class
    class_name = random.choice(test_data.class_names)
    filename = random.choice(os.listdir(os.path.join(test_dir,class_name)))
    filepath = os.path.join(test_dir,class_name,filename)
    # load the images and make predictions
    img=load_preprocess_image(filepath,scale=False)
    img_expanded = tf.expand_dims(img,axis=0) # here BatchDataset image shape in (None,224,224,3),need to expand dim of img
    pred_probs = loaded_model.predict(img_expanded) # get prediction probs array
    pred_classes = test_data.class_names[pred_probs.argmax()] # gets highest probs index
    
    plt.subplot(4,5,i+1)
    plt.subplots_adjust(top=0.9)
    plt.imshow(img/255.)
    if class_name == pred_classes :
        title_color = "g"
    else :
        title_color = "r"
    plt.title(f"actual : {class_name} , pred : {pred_classes} , prob : {pred_probs.max():.2f}",c=title_color)
    plt.axis(False)
    plt.show()


# Finding the most wrong predictions ( Highest prediction probabilities but wrong lable)
# e.g Data issue : model is right,lable is wrong
# confusing classes  : more diverse data
    
#step 1. get all the images file paths in the test dataset

test_filepaths = []

for filepath in test_data.list_files("/Users/yogesh/pythoncode/datasets/101_food_classes_10_percent/test/*/*",
                                     shuffle = False):
    test_filepaths.append(filepath.numpy())

print(test_filepaths)
#step 2: create a dataframe for different params for each of test images
pred_dataframe = pd.DataFrame({"img_path" : test_filepaths,
                              "y_true": y_labels,
                              "y_pred" : pred_classes,
                              "pred_conf" : pred_probs.max(axis=1),
                              "y_true_classname": [test_data.class_names[i] for i in y_labels], 
                              "y_pred_classname" : [test_data.class_names[i] for i in pred_classes]})
print(pred_dataframe)

#stpe 3: Finds the wrong prediction

pred_dataframe["pred_correct"]= pred_dataframe["y_true"]== pred_dataframe["y_pred"]
pred_dataframe.head()

#step 4: sort the dataframes to wrong prediction at the top
print((pred_dataframe["pred_correct"] == False))