# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv ('Datasets/Position_Salaries.csv')

# x = dataset.iloc[:, 1:-1].values
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print(dataset)
print(x)
print(y)


# Training the Decicion Tree's ( On the Whole Dataset )
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor (random_state = 0)
regressor.fit (x, y)

# Pridicting the New Results
print (regressor.predict ([[6.5]]))

# Visuilizing the Results

X_Grid = np.arange (min (x), max (x), 0.01)
X_Grid = X_Grid.reshape ((len (X_Grid), 1))

plt.scatter (x, y, color = 'red')
plt.plot  (X_Grid, regressor.predict (X_Grid), color = 'blue')
plt.title('Decision Tree Regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

