# =============================================================================
# PROBLEM B1
#
# Given two arrays, train a neural network model to match the X to the Y.
# Predict the model with new values of X [-2.0, 10.0]
# We provide the model prediction, do not change the code.
#
# The test infrastructure expects a trained model that accepts
# an input shape of [1]
# Do not use lambda layers in your model.
#
# Please be aware that this is a linear model.
# We will test your model with values in a range as defined in the array to make sure your model is linear.
#
# Desired loss (MSE) < 1e-3
# =============================================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras

class Callback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):

    # Check loss < 1e-09
    if(logs.get('loss') < 1e-09):
      self.model.stop_training = True

# Instantiate class
cb = Callback()

def solution_B1():
    np.random.seed(42)
    tf.random.set_seed(42)
    # DO NOT CHANGE THIS CODE
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=float)
    Y = np.array([5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0], dtype=float)

    # YOUR CODE HERE
    # create FFNN Sequential
    model = keras.Sequential([
        keras.layers.Dense(1, input_shape=(1,)),
    ])

    # compile the model with MSE loss and Adam optimizer
    model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), metrics=['mse'])

    # train the model with X and Y
    model.fit(X, Y, epochs=10000, callbacks=[cb])

    # END YOUR CODE

    print(model.predict([-2.0, 10.0]))
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B1()
    model.save("model_B1.h5")