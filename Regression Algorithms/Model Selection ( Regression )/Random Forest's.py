# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset
dataset = pd.read_csv ('Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the Dataset into Training and Testing Sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training the Random Forest ( Whole Dataset )
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor (n_estimators=50, random_state=0)
regressor.fit (x_train, y_train)

# Predicting the Test set Results
y_pred = regressor.predict(x_test)

np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

from sklearn.metrics import r2_score
print (r2_score(y_test, y_pred))
