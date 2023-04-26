# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing Dataset
dataset = pd.read_csv ('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the Dataset into Training and Testing Sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training the Polynomial Regression Model ( On Whole Dataset )
from sklearn.preprocessing import PolynomialFeatures
from  sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x_train)
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y_train)

y_pred = lin_reg.predict (poly_reg.transform (x_test))
np.set_printoptions (precision = 2)

print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

from sklearn.metrics import r2_score
print (r2_score(y_test, y_pred))
