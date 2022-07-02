# =====================================================================================
# PROBLEM A2
#
# Build a Neural Network Model for Horse or Human Dataset.
# The test will expect it to classify binary classes.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy and validation_accuracy > 83%
# ======================================================================================

import numpy as np
import urllib.request
import zipfile
import tensorflow as tf
import os
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

class Callback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):

    # Check accuracy > 83%
    if(logs.get('val_accuracy') > 0.83) and (logs.get('accuracy') > 0.83):
      self.model.stop_training = True

# Instantiate class
cb = Callback()

def solution_A2():
    os.environ["PYTHONHASHSEED"] = str(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    TRAINING_DIR = 'data/horse-or-human'
    VALIDATION_DIR = 'data/validation-horse-or-human'

    # check if the training and validation directories exist
    if not os.path.exists(TRAINING_DIR):
        data_url_1 = 'https://github.com/dicodingacademy/assets/releases/download/release-horse-or-human/horse-or-human.zip'
        urllib.request.urlretrieve(data_url_1, 'horse-or-human.zip')
        local_file = 'horse-or-human.zip'
        zip_ref = zipfile.ZipFile(local_file, 'r')
        zip_ref.extractall('data/horse-or-human')

    if not os.path.exists(VALIDATION_DIR):
        data_url_2 = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/validation-horse-or-human.zip'
        urllib.request.urlretrieve(data_url_2, 'validation-horse-or-human.zip')
        local_file = 'validation-horse-or-human.zip'
        zip_ref = zipfile.ZipFile(local_file, 'r')
        zip_ref.extractall('data/validation-horse-or-human')
        zip_ref.close()

    train_datagen = ImageDataGenerator(
      rescale=1./255,
      horizontal_flip=True,
        # rotation_range=40,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.2,
        zoom_range=0.2,
      )

    # YOUR IMAGE SIZE SHOULD BE 150x150
    train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')
    
    validation_datagen = ImageDataGenerator(rescale=1/255)

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    model=tf.keras.models.Sequential([
        # YOUR CODE HERE, end with a Neuron Dense, activated by sigmoid
                tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.55),
                tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    
    model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])
    
    # Constant for epochs
    EPOCHS = 20

    # Train the model
    model.fit(
        train_generator,
        epochs=EPOCHS,
        verbose=1,
        validation_data = validation_generator,
        callbacks=[cb])

    return model

# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=solution_A2()
    model.save("model_A2.h5")
