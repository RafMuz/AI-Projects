from sklearn.model_selection import train_test_split as Train_Test_Split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from matplotlib import pyplot
import pandas as panda
import numpy as np


# Taking the Path of the Dataset.
Path = "/home/raf_muz/Downloads/Python/Python Project File's 2021/January 2021/Binary Neural Network ( Heart Disease )/Modified Data.csv"

# Loading the Dataset.
Data = panda.read_csv (Path, header = None)
Data_Values = Data.values


# Separating data and targets.
Inputs = Data_Values [:, 0:-1]
Expected_Outputs = Data_Values [:, -1]

# Normalizing the Inputs.
Normalized_Inputs = MinMaxScaler (feature_range = (-1, 1)).fit_transform (Inputs)  # Scaler, feature_range

# Spliting the data into training and testing sets.
Training_Inputs, Validation_Inputs, Training_Outputs, Validation_Outputs = Train_Test_Split (Normalized_Inputs, Expected_Outputs, test_size = 0.2, random_state = 42)  # Shuffle / Random_State



# Setup the model.
model = Sequential()

# Number of Neurons, Activation Function (You can use (Sigmoid, ReLU, Tanh)), Dropout.

model.add (Dense (28, input_dim = Training_Inputs.shape [1], activation = 'relu', kernel_initializer = 'he_uniform'))
model.add(Dropout (0.2))
model.add (Dense (16, activation = 'sigmoid', kernel_initializer = 'he_uniform'))
model.add(Dropout (0.15))
model.add (Dense (6, activation = 'sigmoid', kernel_initializer = 'he_uniform'))
model.add(Dropout (0.1))
model.add (Dense (1, activation = 'sigmoid', kernel_initializer = 'he_uniform'))



# Compliing  the model and setting the optimizer.
op = SGD (lr = 0.1)
model.compile (loss = 'binary_crossentropy', optimizer = op, metrics = ['accuracy'])  # Optimizer, LR.

# Calling fit model function | Which Let's us run the NN and set the epochs & batch size, among other things.
box = model.fit (Training_Inputs, Training_Outputs, epochs = 1000, batch_size = 20, validation_data = (Validation_Inputs, Validation_Outputs))  # epochs, batch_size.


# Capturing the Model Accuracy & Loss ( For Later Use ).
Validation_Loss = box.history ['val_loss']
Validation_Accuracy = box.history ['val_accuracy']
Training_Loss = box.history ['loss']
Training_Accuracy = box.history ['accuracy']


#  Plotting the Graph
# Create 2 Graph's and Select Graph 1.
pyplot.subplot (211)

pyplot.title ('Loss')
pyplot.ylabel ('Value')

# Plot the Neural Network's Loss Variable's.
pyplot.plot (Training_Loss, label = 'Train')
pyplot.plot (Validation_Loss, label = 'Val')
pyplot.legend ()

# Select Graph 2
pyplot.subplot (212)

pyplot.title ('Accuracy')
pyplot.xlabel ('Epochs')
pyplot.ylabel ('Value')

# Plot the Neural Network's Accuracy Variable's.
pyplot.plot (Training_Accuracy, label = 'Train')
pyplot.plot (Validation_Accuracy, label = 'Val')
pyplot.legend ()

#model.save ('Binary NN ( Heart Disease ).h5')

N = np.shape (Data_Values) [0]

x_pred = Normalized_Inputs [0:N, :]
y_pred = (model.predict (x_pred) > 0.5).astype ("int32")

y_pred = y_pred.reshape (N)

Out = Expected_Outputs [0:N]
Out = Out.astype (np.int32)
Out = Out.reshape (N)

Error = 0
x = 0

while x < N:

    if(y_pred[x] != Out[x]):

        Error = Error + 1

        pre = model.predict (x_pred [x:x+1, 0:])
        pred = pre.astype(np.float64) [0]

        print ("Row Number: {0}, Out: {1}, Pred: {2}".format (x, Out [x], round (pred [0], 2)))

    x = x + 1

#print ("Expected Output = {0}, \nPredicted Output= {1}".format (Out, y_pred))
print ("\nThe Data size is: {0} \nAnd it got {1} Wrong".format (N, Error))

a = (1 - Error/ N) * 100
print("The Accuracy is: {0}%\n".format (round (a, 2)))



# Show the Graph.
pyplot.show ()