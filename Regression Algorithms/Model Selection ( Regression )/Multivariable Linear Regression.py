# Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Importing Dataset
dataset = pd.read_csv ('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the Dataset into Training and Testing Sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training the SLR Model on the Training Set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set Results
y_pred = regressor.predict(x_test)

np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

from sklearn.metrics import r2_score
print (r2_score(y_test, y_pred))
