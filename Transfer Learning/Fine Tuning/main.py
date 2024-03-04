
import os
import tensorflow as tf
import random
from keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

train_dir = "/Users/yogesh/pythoncode/datasets/101_food_classes_10_percent/train"
test_dir = "/Users/yogesh/pythoncode/datasets/101_food_classes_10_percent/test"

train_data = image_dataset_from_directory(train_dir,label_mode="categorical",image_size=(224,224))
test_data = image_dataset_from_directory(test_dir,
                                         label_mode="categorical",
                                         image_size=(224,224),
                                         shuffle=False) # Keep it False,It will not change the order of the image ,same as in dataset



#step 1. get all the images file paths in the test dataset

test_filepaths = []

for filepath in test_data.list_files("/Users/yogesh/pythoncode/datasets/101_food_classes_10_percent/test/*/*",
                                     shuffle = False):
    test_filepaths.append(filepath.numpy())

print(test_filepaths)
