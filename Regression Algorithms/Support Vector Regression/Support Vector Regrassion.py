# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv ('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print (x)
# print (y)

y = y.reshape (len (y), 1)

print (y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler ()
sc_y = StandardScaler ()

x = sc_x.fit_transform (x)
y = sc_y.fit_transform (y)

print (x)
print (y)

# Training Support Vector Regression model ( On whole Dataset )
from sklearn.svm import SVR

y = y.reshape (len (y))

regressor = SVR (kernel = 'rbf')
regressor.fit (x, y)

# Predicting the Results
y_result = sc_y.inverse_transform (regressor.predict (sc_x.transform ([[6.5]])))
print (y_result)


# Making X have more datapoints
X_grid = np.linspace (min (sc_x.inverse_transform (x)), max (sc_x.inverse_transform (x)), 40)
X_grid = X_grid.reshape ((len (X_grid), 1))
print (X_grid)

'''# Predicting the Results with X_Grid
sc_x.fit (X_grid)
print (sc_y.inverse_transform (regressor.predict (sc_x.transform ([[6.5]]))))
'''

# Visualising the SVR results

plt.scatter (sc_x.inverse_transform (x), sc_y.inverse_transform (y), color = 'red')
plt.plot (X_grid, sc_y.inverse_transform (regressor.predict(sc_x.transform (X_grid))), color = 'blue')
plt.title ('Truth or False (SVR)')
plt.xlabel ('Position level')
plt.ylabel ('Salary')
plt.show ()
