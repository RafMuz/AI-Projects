# Importing Libraris
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Reading the Dataset
file_location = 'Social_Network_Ads.csv'
dataset = pd.read_csv (file_location)

x = dataset.iloc [:, :-1].values
y = dataset.iloc [:, -1].values

# Splitting into Testing and Training Sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size = 0.25, random_state = 10)

# Normalizing the Data
from sklearn.preprocessing import StandardScaler

sc = StandardScaler ()

x_train = sc.fit_transform (x_train)
x_test = sc.fit_transform (x_test)

# Training the Decicion Tree Model Clasification ( on training set )
from sklearn.tree import DecisionTreeClassifier

classify = DecisionTreeClassifier (criterion = 'entropy', random_state = 0)
classify.fit (x_train, y_train)

# Predicting a New Result
print (classify.predict (sc.fit_transform([[30, 87000]])))

# Predicting the Test Set Results
y_pred = classify.predict (x_test)

print (np.concatenate ((y_pred.reshape (len(y_pred), 1), y_test.reshape (len (y_test), 1)), 1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix (y_test, y_pred)

print (cm)
print (accuracy_score (y_test, y_pred))

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(x_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classify.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Naive Bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(x_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classify.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()