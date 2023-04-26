from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from matplotlib import pyplot
import pandas as pd



# Taking the Path of the Dataset.
path = "/home/raf_muz/Downloads/Python/Python Project File's 2020/December 2020/Neural Network/prima-indians-diabetes.csv"

# Loading the Dataset.
data = pd.read_csv (path, header = None)
data_values = data.values


# Separating data and targets.
x = data_values [:, :8]
y = data_values [:, 8]

# Normalizing the Inputs.
x_norm = MinMaxScaler (feature_range = (-1, 1)).fit_transform (x)


# Spliting the data into training and testing sets.
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size = 0.2, shuffle=False)


# Setup the model.
model = Sequential()
model.add (Dense (15, input_dim = x_train.shape [1], activation = "relu", kernel_initializer = "he_uniform"))
#model.add(Dropout(0.2))
model.add (Dense (10, activation = 'relu', kernel_initializer = "he_uniform"))
#model.add(Dropout(0.2))
model.add (Dense (8, activation = 'relu', kernel_initializer = "he_uniform"))
model.add(Dropout(0.2))
model.add (Dense (1, activation = 'sigmoid'))


# Compliing  the model and setting the optimizer.
#op = SGD(lr=0.05)
model.compile (loss = "binary_crossentropy", optimizer = 'adam', metrics = ['accuracy'])

# Calling fit model function | Which Let's us run the NN and set the epochs & batch size, among other things.
box = model.fit (x_train, y_train, epochs = 100, batch_size = 20, validation_data = (x_test, y_test))


# Capturing the Model Accuracy & Loss ( For Later Use ).
v_loss = box.history ['val_loss']
v_acc = box.history ['val_accuracy']
t_loss = box.history ['loss']
t_acc = box.history ['accuracy']


#  Plotting the Graph
# Create 2 Graph's and Select Graph 1.
pyplot.subplot (211)
pyplot.title ('Loss')

# Plot the Neural Network's Loss Variable's.
pyplot.plot (t_loss, label='Train')
pyplot.plot (v_loss, label='Val')
pyplot.legend ()

# Select Graph 2
pyplot.subplot (212)
pyplot.title ('Accuracy')

# Plot the Neural Network's Accuracy Variable's.
pyplot.plot (t_acc, label='Train')
pyplot.plot (v_acc, label='Val')
pyplot.legend ()


# Show the Graph.
pyplot.show ()
