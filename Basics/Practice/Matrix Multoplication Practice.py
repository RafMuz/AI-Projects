import numpy as np
import pandas as panda

Matrix_1 = np.array ([[1],
                      [2],
                      [3]])

Matrix_2 = np.array ([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9],
                     [10, 11, 12]])

print ('Matrix 1 is:\n\n{0}'.format (panda.DataFrame (Matrix_1)))
print ('\nMatrix 2 is:\n\n{0}'.format(panda.DataFrame (Matrix_2)))

Matrix_1 = Matrix_1.T
Matrix_2 = Matrix_2.T

Matrix_3 = np.dot (Matrix_1, Matrix_2)

print ('\nMatrix 1 x Matrix 2 is:\n\n{0}'.format (panda.DataFrame (Matrix_3)))