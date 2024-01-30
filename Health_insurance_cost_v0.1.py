import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

insurance_data = pd.read_csv("/Users/yogesh/pythoncode/Tensorflow/tensorflow/insurance.csv")

print(insurance_data)

# one hot encoding using pandas
data = pd.get_dummies(insurance_data,dtype=float)
print(data) 

# independent variable or features

X = data.drop("charges",axis=1)
print(X)
# Dependent variable or Outcome
y = data["charges"]
print(y)

# creating a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Creating the model
tf.random.set_seed(42)

insurance_model = keras.Sequential(name="insurance_model")
insurance_model.add(layers.Dense(32,name ="input_layer"))
insurance_model.add(layers.Dense(1,name="output_layer"))

# compile the model

insurance_model.compile(loss=keras.losses.mae,
                        optimizer=keras.optimizers.Adam(learning_rate=0.01),
                        metrics=keras.metrics.mae)


# fit the model
insurance_model.fit(X_train,y_train,epochs=100)

# evaluate the model
print(insurance_model.evaluate(X_test,y_test))

print(y_test.mean(),y_test.median())

# make a prediction
y_pred_test = insurance_model.predict(X_test)
print(y_pred_test)
