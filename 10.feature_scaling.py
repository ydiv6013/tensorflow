from socket import IPV6_JOIN_GROUP
from matplotlib import axis
from numpy import matrix
import pandas as pd
import tensorflow as tf
import keras 
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler , OneHotEncoder ,StandardScaler
from sklearn.model_selection import train_test_split


"""
Normalization and standardization are techniques used in data preprocessing to 
scale or transform the features of a dataset. These techniques are often applied 
to ensure that the data has consistent and comparable scales, which can be important 
for certain machine learning algorithms.

1. **Normalization:**
   - **Objective:** Scaling the features of a dataset to a specific range, typically between 0 and 1.
   - **Formula:** \( X_{\text{normalized}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}} \)
   - **Purpose:** Ensures that all features have the same scale, making the data comparable.

   ```python
   from sklearn.preprocessing import MinMaxScaler

   # Create a MinMaxScaler object
   scaler = MinMaxScaler()

   # Fit and transform the data
   X_normalized = scaler.fit_transform(X)
   ```

2. **Standardization:**
   - **Objective:** Transforming the features of a dataset to have a mean of 0 and a standard deviation of 1.
   - **Formula:** \( X_{\text{standardized}} = \frac{X - \text{mean}(X)}{\text{std}(X)} \)
   - **Purpose:** Standardization makes the features have similar scales, but it also ensures that the distribution of the features has a mean of 0 and a standard deviation of 1.

   ```python
   from sklearn.preprocessing import StandardScaler

   # Create a StandardScaler object
   scaler = StandardScaler()

   # Fit and transform the data
   X_standardized = scaler.fit_transform(X)
   ```

**Key Differences:**
- Normalization scales the data to a specific range (typically 0 to 1), while standardization transforms the data to have a mean of 0 and a standard deviation of 1.
- Normalization is sensitive to outliers, as the range is determined by the minimum and maximum values in the dataset. Standardization is less affected by outliers.
- Normalization is often more suitable when the data and features have a bounded range, while standardization may be preferred when the data has a more Gaussian distribution.

The choice between normalization and standardization depends on the characteristics of the data and the requirements of the machine learning algorithm being used.



"""
data = pd.read_csv("/Users/yogesh/pythoncode/Tensorflow/tensorflow/insurance.csv")

print(data)
# Feature scaling (normalisation)
# create a column transformer
ct = make_column_transformer(
    (MinMaxScaler(),["age","bmi","children"]),
    (OneHotEncoder(handle_unknown="ignore"),["sex","smoker","region"])
)

# Feature scaling (standardisation)
# create a column transformer
sc = make_column_transformer(
    (StandardScaler(),["age","bmi","children"]),
    (OneHotEncoder(handle_unknown="ignore"),["sex","smoker","region"])
)

# create X and y

X = data.drop("charges",axis=1)
y = data["charges"]

print(X)
print(y)

# build train and test set

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# fit and transform column transformer to train and test set with normalisation and onehot encoding

X_train_norm = ct.fit_transform(X_train)
X_test_norm = ct.fit_transform(X_test)

print(X_train_norm)
print(X_test_norm)

# fit and transform column transformer to train and test set with standardisation and onehot encoding
X_train_stand= sc.fit_transform(X_train)
X_test_stand = sc.fit_transform(X_test)

print(X_train_stand)
print(X_test_stand)


# Build a neural network to fit on feature scakled data

# Creating the model
tf.random.set_seed(42)
insurance_model = keras.Sequential([
    keras.layers.Dense(100),
    keras.layers.Dense(10),
    keras.layers.Dense(1)
])
# compile model
insurance_model.compile(
    loss=keras.losses.mae,
    optimizer=keras.optimizers.Adam(learning_rate=0.1),
    metrics=keras.metrics.mae
)

#fit the model
insurance_model.fit(X_train_norm,y_train,epochs=100)

#evalute the model
print(insurance_model.evaluate(X_test_norm,y_test)) 

# make a prediction
y_pred = insurance_model.predict(X_test_norm) 
print(y_pred)