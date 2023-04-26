# Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Importing Dataset
dataset = pd.read_csv ('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# Splitting the Dataset into Training and Testing Sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# Training the SLR Model on the Training Set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set Results
y_pred = regressor.predict(x_test)

# Visuilising Training set Results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs. Exp (Training Set)')
plt.xlabel('Exp')
plt.ylabel('Salary')
plt.show()


# Visuilizing Test set Results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs. Exp (Testing Set)')
plt.xlabel('Exp')
plt.ylabel('Salary')
plt.show()

