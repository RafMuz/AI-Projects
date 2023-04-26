from keras.models import load_model
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(Training_Inputs, Training_Outputs), (Validation_Inputs, Validation_Outputs) = mnist.load_data ()

Training_Inputs = tf.keras.utils.normalize (Training_Inputs, axis = 1)
Validation_Inputs = tf.keras.utils.normalize (Validation_Inputs, axis = 1)


model = load_model ("CNN_Model.h5")
model.evaluate (Validation_Inputs, Validation_Outputs)
#model.load_weights (weights_path, by_name = True)
