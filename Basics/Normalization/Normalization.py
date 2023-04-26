import numpy as np
import pandas as panda

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


Data = np.array (

       [[-100, 29],
       [-52, 64],
       [98, 10],
       [12, 18]] )


Normalized_Data = StandardScaler ().fit_transform (Data)
#Normalized_Data = MinMaxScaler (feature_range = (-1, 1)).fit_transform (Data)

print ("\nThis is the data: \n")
print (panda.DataFrame (Data))

print ("\n\nThis is the Normalized Data: \n")
print (panda.DataFrame (Normalized_Data))
print ()
