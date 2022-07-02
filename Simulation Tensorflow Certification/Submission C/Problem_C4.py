# =====================================================================================================
# PROBLEM C4
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
#
# Do not use lambda layers in your model.
#
# Dataset used in this problem is built by Rishabh Misra (https://rishabhmisra.github.io/publications).
#
# Desired accuracy and validation_accuracy > 75%
# =======================================================================================================

from gc import callbacks
import json
import pandas as pd
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')

lemma = WordNetLemmatizer()
sw = stopwords.words('english')

class Callback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):

    # Check accuracy > 75%
    if(logs.get('val_accuracy') > 0.75) and (logs.get('accuracy') > 0.75):
      self.model.stop_training = True

# Instantiate class
cb = Callback()

def solution_C4():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/sarcasm.json'
    urllib.request.urlretrieve(data_url, 'sarcasm.json')
    data = json.load(open('sarcasm.json','r'))
    data = pd.DataFrame(data)

    # data['headline'] = data['headline'].apply(lambda x: x.lower())
    # # remove non alphabet and numeric char
    # data['headline'] = data['headline'].str.replace('[^a-zA-Z0-9\s]', ' ')
    # # remove stopwords
    # data['headline'] = data['headline'].apply(lemma.lemmatize)
    # # remove multispace
    # data['headline'] = data['headline'].str.replace('\s+', ' ')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or test may fail
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000/data.shape[0]

    # sentences = []
    # labels = []
    # YOUR CODE HERE
    X_train, X_test, y_train, y_test = train_test_split(data['headline'], data['is_sarcastic'], train_size=training_size, random_state=42)

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(X_train)

    sequences = tokenizer.texts_to_sequences(X_train)
    padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type, padding=padding_type)

    testing_sequences = tokenizer.texts_to_sequences(X_test)
    testing_padded = pad_sequences(testing_sequences,maxlen=max_length)

    model = tf.keras.Sequential([
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        # tf.keras.layers.Conv1D(64,3,activation='relu'),
        # tf.keras.layers.MaxPooling1D(8),
        # tf.keras.layers.Flatten(),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer="adam",
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    model.fit(padded, y_train, epochs=20, validation_data=(testing_padded, y_test),verbose=1,
    callbacks=[cb])

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C4()
    model.save("model_C4.h5")
