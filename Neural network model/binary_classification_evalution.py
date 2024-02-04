from cProfile import label
import dis
from filecmp import cmp
import keras
import tensorflow as tf
import pandas as pd
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

# Neural network to classify data points as red or blue (binary data) non linear

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

# set the random seed
tf.random.set_seed(42)

# create a model

binary_model = keras.Sequential([
    keras.layers.Dense(4,activation="relu"),
    keras.layers.Dense(4,activation="relu"),
    keras.layers.Dense(1,activation="sigmoid")
])
# compile the model

binary_model.compile(loss=keras.losses.BinaryCrossentropy(),
                     optimizer=keras.optimizers.legacy.Adam(learning_rate=0.01),
                     metrics=['accuracy'])

# fit the model

history = binary_model.fit(X_train,y_train,epochs=25)
# Evalute the model
print(binary_model.evaluate(X,y))


# Make a prediction
y_pred = binary_model.predict([[0.442208,-0.896723]])
y_pred_set = binary_model.predict(X_test)


#  y_pred output will be  the values in prediction probability form.. which is a standard output from sigmoid or softmax

print(y_pred) 
print(y_pred_set)

# convert prediction  probability to binary formate

if y_pred >= 0.5 :
     print(f"{y_pred} Label : {tf.round(y_pred)} ", 1)
else :
     print(f"{y_pred} label : {tf.round(y_pred)} ", 0)

# using tensorflow round
print(tf.round(y_pred_set))

# to visulise the predicion  create a function 'plot_decision_boundary()

def plot_decision_boundary(model, X, y):
        """
        Plots the decision boundary created by a model predicting on X.
        This function was inspired by two resources:
        1. https://cs231n.github.io/neural-networks-case-study/
        2. https://github.com/madewithml/basics/blob/master/notebooks/09_Multilayer_Perceptrons/09_TF_Multilayer_Perceptrons.ipynb 
        """
        # Define the axis boundaries of the plot and create a meshgrid
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # Create X value (we're going to make predictions on these)
        x_in = np.c_[xx.ravel(), yy.ravel()] # stack 2D arrays together

        # Make predictions
        y_pred = model.predict(x_in)

        # Check for multi-class
        if len(y_pred[0]) > 1:
            print("doing multiclass classification")
            # We have to reshape our prediction to get them ready for plotting
            y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
        else:
            print("doing binary classification")
            y_pred = np.round(y_pred).reshape(xx.shape)
        
        # Plot the decision boundary
        plt.contourf(xx, yy, y_pred, cmap="RdYlBu", alpha=0.7)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap="RdYlBu")
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.show()

plot_decision_boundary(binary_model,X,y)
plot_decision_boundary(binary_model,X_test,y_test)


#plot the history (known as loss curve or training curve)
pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()



# evolute the model
loss,accuracy = binary_model.evaluate(X_test,y_test)
print(f"Model loss on test set : {loss} ")
print(f"model accuracy on test set : {(accuracy*100):.2f} %")


# create a confusion matrix
cm = confusion_matrix(y_true=y_test,y_pred=tf.round(y_pred_set))

print(cm)

display = ConfusionMatrixDisplay(cm)
display.plot()
plt.show()