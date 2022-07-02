# ======================================================================================================
# PROBLEM A3
#
# Build a classifier for the Human or Horse Dataset with Transfer Learning.
# The test will expect it to classify binary classes.
# Note that all the layers in the pre-trained model are non-trainable.
# Do not use lambda layers in your model.
#
# The horse-or-human dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
# Inception_v3, pre-trained model used in this problem is developed by Google.
#
# Desired accuracy and validation_accuracy > 97%.
# =======================================================================================================

import numpy as np
import urllib.request
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
import os

class Callback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):

    # Check accuracy > 97%
    if(logs.get('val_accuracy') > 0.97) and (logs.get('accuracy') > 0.97):
      self.model.stop_training = True

# Instantiate class
cb = Callback()

def solution_A3():
    np.random.seed(42)
    tf.random.set_seed(42)
    if not os.path.exists('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'):
        inceptionv3 = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
        urllib.request.urlretrieve(
            inceptionv3, 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
    local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

    pre_trained_model =  InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)
    
    pre_trained_model.load_weights(local_weights_file)

    for layer in pre_trained_model.layers:
        layer.trainable = False

    last_layer = pre_trained_model.get_layer('mixed7')

    last_output = last_layer.output

    if not os.path.exists('horse-or-human.zip') and not os.path.exists('data/horse-or-human'):
        data_url_1 = 'https://github.com/dicodingacademy/assets/releases/download/release-horse-or-human/horse-or-human.zip'
        urllib.request.urlretrieve(data_url_1, 'horse-or-human.zip')
        local_file = 'horse-or-human.zip'
        zip_ref = zipfile.ZipFile(local_file, 'r')
        zip_ref.extractall('data/horse-or-human')
        zip_ref.close()
    
    if not os.path.exists('validation-horse-or-human.zip') and not os.path.exists('data/validation-horse-or-human'):
        data_url_2 = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/validation-horse-or-human.zip'
        urllib.request.urlretrieve(data_url_2, 'validation-horse-or-human.zip')
        local_file = 'validation-horse-or-human.zip'
        zip_ref = zipfile.ZipFile(local_file, 'r')
        zip_ref.extractall('data/validation-horse-or-human')
        zip_ref.close()

    train_dir = 'data/horse-or-human'
    validation_dir = 'data/validation-horse-or-human'

    train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

    # YOUR IMAGE SIZE SHOULD BE 150x150
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=128,
        class_mode='binary')
    
    validation_datagen = ImageDataGenerator(rescale=1/255)

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    # YOUR CODE HERE, BUT END WITH A Neuron Dense, activated by sigmoid
    x = layers.Flatten()(last_output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)  
    x = layers.Dense(1, activation='sigmoid')(x)

    model = Model(pre_trained_model.input, x)

    model.compile(optimizer=RMSprop(lr=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    steps_per_epoch = train_generator.n // train_generator.batch_size
    validation_steps = validation_generator.n // validation_generator.batch_size
    model.fit(
        train_generator,
        validation_data = validation_generator,
        steps_per_epoch = steps_per_epoch,
        epochs = 5,
        validation_steps = validation_steps,
        verbose = 2,
        callbacks=[cb])

    return model

# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=solution_A3()
    model.save("model_A3.h5")
