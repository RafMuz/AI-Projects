# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing Dataset
dataset = pd.read_csv ('Position_Salaries_2.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Linear Regression Model ( On Whole Dataset )
from  sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Training the Polynomial Regression Model ( On Whole Dataset )
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Visualizing the Linear Regression Results
'''plt.scatter(x, y, color='red')
plt.plot(x,lin_reg.predict(x), color='blue')

plt.title('True or False ( Linear Regression )')
plt.xlabel('Position Level')
plt.ylabel('salary')

plt.show()'''

# Visualizing the Polynomial Regression Results
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg_2.predict(x_poly), color='blue')

plt.title('True or False ( Polynomial Regression )')
plt.xlabel('Position Level')
plt.ylabel('Salary')

plt.show()

# Predicting a new result with linear regression
# print(lin_reg.predict([[20.5]]))

# Predicting a new result with polynomial regression
print(lin_reg_2.predict(poly_reg.fit_transform([[20.5]])))