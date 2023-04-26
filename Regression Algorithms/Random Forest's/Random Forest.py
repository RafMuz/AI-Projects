# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset
dataset = pd.read_csv ('Position_Salaries.csv')

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Random Forest ( Whole Dataset )
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor (n_estimators=50, random_state=0)
regressor.fit (x, y)

# Predicting the new Result
print (regressor.predict ([[6.5]]))

# Visuilizing the Random Forest Results ( in Higher Resulution)

# Making X have more datapoints
X_grid = np.arange (min (x), max (x), 0.01)
X_grid = X_grid.reshape ((len (X_grid), 1))

plt.scatter (x, y, color = 'red')
plt.plot (X_grid, regressor.predict (X_grid), color = 'blue')
plt.title ('Truth or False')
plt.xlabel ('Position Level')
plt.ylabel ('Salary')

plt.show ()

