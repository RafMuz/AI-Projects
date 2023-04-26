import numpy as np
import pandas as panda


# Setup the Matrices
Matrix1 = np.array ([[1, 2],
                    [36, 4]])

Matrix2 = np.array ([[2, 4],
                    [63, 8]])

# Setup the Row's and Coloum's of Matrix 1
Matrix1_Row = {0:0, 1:1}
Matrix1_Coloum = {0:0, 1:1}


# Setup the Row's and Coloum's of Matrix 2
Matrix2_Row = {0:0, 1:1}
Matrix2_Coloum = {0:0, 1:1}


# Ask the User to Input the Values for Matrix 1
for Rows in Matrix1_Row:

    for Coloums in Matrix1_Coloum:

        Matrix1 [Rows, Coloums] = input ("Input a Number for Row {0} and Coloum {1} in  Matrix 1: ".format (Rows, Coloums))


# Show the User the Value's of Matrix 1
print ("\nThis is Matrix 1: \n\n", panda.DataFrame (Matrix1), "\n")




# Ask the User to Input the Values for Matrix 2
for Rows in Matrix2_Row:

    for Coloums in Matrix2_Coloum:

        Matrix2 [Rows, Coloums] = input ("Input a Number for Row {0} and Coloum {1} in  Matrix 2: ".format (Rows, Coloums))


# Show the User the Value's of Matrix 2
print ("\nThis is Matrix 2: \n\n", panda.DataFrame (Matrix2), "\n")




# Take the Transpose of a Matrix if Needed
if Matrix1_Coloum == Matrix2_Row:

    Matrix_1 = Matrix1

else:

    Matrix_1 = Matrix1.T


# Multiply Matrix 1 and Matrix 2 ( using np.dot ) then show the result
print ("\nThis is what Matrix 1 and Matrix 2 multiplied together look's like: \n\n", panda.DataFrame (np.dot (Matrix_1, Matrix2)), "\n")
