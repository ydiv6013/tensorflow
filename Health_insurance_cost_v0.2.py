import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split



# To (try) improve model we will do two experiment
# 1.add an extra layer with more hidden units
# 2. Train for longer

insurance_data = pd.read_csv("/Users/yogesh/pythoncode/Tensorflow/tensorflow/insurance.csv")

print(insurance_data)

# one hot encoding using pandas
# one hot encoding convert the string values (categorical data )to numeric values
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

insurance_model= keras.Sequential()
insurance_model.add(layers.Dense(100))
insurance_model.add(layers.Dense(10))
insurance_model.add(layers.Dense(1))

# compile the model
insurance_model.compile(
    loss=keras.losses.mae,
    optimizer=keras.optimizers.Adam(learning_rate=0.1),
    metrics=keras.metrics.mae
)

# fit the model
history =insurance_model.fit(X_train,y_train,epochs=100)

# evaluate the model
print(insurance_model.evaluate(X_test,y_test))

print("Mean Value :",y_test.mean(),"\n Median Value:",y_test.median())

# make a prediction
y_pred_test = insurance_model.predict(X_test)
print(y_pred_test)

# plot the history (known as loss curve or training curve)
pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()