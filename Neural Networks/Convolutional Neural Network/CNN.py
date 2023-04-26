from keras.callbacks import ModelCheckpoint
import tensorflow.keras as keras
import tensorflow as tf
#import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(Training_Inputs, Training_Outputs), (Validation_Inputs, Validation_Outputs) = mnist.load_data ()

Training_Inputs = tf.keras.utils.normalize (Training_Inputs, axis = 1)
Validation_Inputs = tf.keras.utils.normalize (Validation_Inputs, axis = 1)

'''
print (Training_Inputs [0])
plt.imshow (Training_Inputs [0], cmap = plt.cm.binary)
plt.show ()

# Reshape the Inputs
Training_Inputs = Training_Inputs.reshape (60000, 28, 28, 1)
Validation_Inputs = Validation_Inputs.reshape (10000, 28, 28, 1)
'''

model = tf.keras.models.Sequential ()

model.add (keras.layers.Flatten ())
model.add (keras.layers.Dense (256, activation = tf.nn.relu, kernel_initializer = 'he_uniform'))
model.add (keras.layers.Dropout (0.2))
model.add (keras.layers.Dense (10, activation = tf.nn.softmax, kernel_initializer = 'he_uniform'))

model.compile (optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'])

#Weights = ModelCheckpoint ('CNN_Weights.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
#Callbacks_List = [Weights]

model.fit (Training_Inputs, Training_Outputs, epochs = 50, batch_size = 96, validation_data = (Validation_Inputs, Validation_Outputs))

model.evaluate (Validation_Inputs, Validation_Outputs)

model.save ('CNN_Model.h5')
