# ========================================================================================
# PROBLEM B3
#
# Build a CNN based classifier for Rock-Paper-Scissors dataset.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# This is unlabeled data, use ImageDataGenerator to automatically label it.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy AND validation_accuracy > 83%
# ========================================================================================

import urllib.request
import zipfile
import tensorflow as tf
import os
from keras_preprocessing.image import ImageDataGenerator
import shutil

class Callback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):

    # Check accuracy > 83%
    if(logs.get('val_accuracy') > 0.83) and (logs.get('accuracy') > 0.83):
      self.model.stop_training = True

# Instantiate class
cb = Callback()

def solution_B3():
    data_url = 'https://github.com/dicodingacademy/assets/releases/download/release-rps/rps.zip'
    urllib.request.urlretrieve(data_url, 'rps.zip')
    local_file = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/')
    zip_ref.close()

    base_dir = "data/rps/"

    rock_dir = os.path.join(base_dir, 'rock')
    paper_dir = os.path.join(base_dir, 'paper')
    scissors_dir = os.path.join(base_dir, 'scissors')

    # Split the data into train and test sets (0.2 test)
    # Moving the data to the right folder
    train_rock_dir = os.path.join(base_dir, 'train', 'rock')
    train_paper_dir = os.path.join(base_dir, 'train', 'paper')
    train_scissors_dir = os.path.join(base_dir, 'train', 'scissors')
    test_rock_dir = os.path.join(base_dir, 'test', 'rock')
    test_paper_dir = os.path.join(base_dir, 'test', 'paper')
    test_scissors_dir = os.path.join(base_dir, 'test', 'scissors')

    # Create rock_dir
    if not os.path.exists(train_rock_dir):
        os.makedirs(train_rock_dir)
    if not os.path.exists(test_rock_dir):
        os.makedirs(test_rock_dir)
    
    # Create paper_dir
    if not os.path.exists(train_paper_dir):
        os.makedirs(train_paper_dir)
    if not os.path.exists(test_paper_dir):
        os.makedirs(test_paper_dir)
    
    # Create scissors_dir
    if not os.path.exists(train_scissors_dir):
        os.makedirs(train_scissors_dir)
    if not os.path.exists(test_scissors_dir):
        os.makedirs(test_scissors_dir)
    
    # Moving the data to the right folder
    for i,file in enumerate(os.listdir(rock_dir)):
        if i <= 0.8 * len(os.listdir(rock_dir)) and not os.path.exists(os.path.join(train_rock_dir, file)):
            shutil.move(os.path.join(rock_dir, file), train_rock_dir)
        elif not os.path.exists(os.path.join(test_rock_dir, file)):
            shutil.move(os.path.join(rock_dir, file), test_rock_dir)
    for i,file in enumerate(os.listdir(paper_dir)):
        if i <= 0.8 * len(os.listdir(paper_dir)) and not os.path.exists(os.path.join(train_paper_dir, file)):
            shutil.move(os.path.join(paper_dir, file), train_paper_dir)
        elif not os.path.exists(os.path.join(test_paper_dir, file)):
            shutil.move(os.path.join(paper_dir, file), test_paper_dir)
    for i,file in enumerate(os.listdir(scissors_dir)):
        if i <= 0.8 * len(os.listdir(scissors_dir)) and not os.path.exists(os.path.join(train_scissors_dir, file)):
            shutil.move(os.path.join(scissors_dir, file), train_scissors_dir)
        elif not os.path.exists(os.path.join(test_scissors_dir, file)):
            shutil.move(os.path.join(scissors_dir, file), test_scissors_dir)
    
    TRAINING_DIR = os.path.join(base_dir, 'train')
    TEST_DIR = os.path.join(base_dir, 'test')

    training_datagen = ImageDataGenerator(
        rescale=1./255,)

    # YOUR IMAGE SIZE SHOULD BE 150x150
    # Make sure you used "categorical"
    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')
    
    val_datagen = ImageDataGenerator(
        rescale=1./255,)
    
    val_generator = val_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')

    # YOUR CODE HERE, end with 3 Neuron Dense, activated by softmax
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    # COMPILE YOUR MODEL 
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # TRAIN YOUR MODEL (use validation data)
    model.fit(train_generator, epochs=10, validation_data=val_generator,callbacks=[cb])

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=solution_B3()
    model.save("model_B3.h5")
