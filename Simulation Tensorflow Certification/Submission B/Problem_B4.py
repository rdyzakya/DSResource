# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.optimizers import Adam, RMSprop

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

    # Check accuracy > 91%
    if(logs.get('val_accuracy') > 0.91) and (logs.get('accuracy') > 0.91):
      self.model.stop_training = True

# Instantiate class
cb = Callback()


def solution_B4():
    bbc = pd.read_csv(
        'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    def remove_sw(text):
        return ' '.join([word for word in text.split() if word not in sw])
    
    bbc['text'] = bbc['text'].apply(remove_sw)
    bbc['text'] = bbc['text'].apply(lemma.lemmatize)
    bbc['text'] = bbc['text'].str.replace('[^\w\s]', '',regex=True)
    bbc['text'] = bbc['text'].str.replace('\d+', '',regex=True)

    # YOUR CODE HERE
    # Using "shuffle=False"
    # Split the data into train and test sets (0.2 test)
    X_train, X_test, y_train, y_test = train_test_split(bbc['text'], bbc['category'], train_size=training_portion, shuffle=False)

    le = Tokenizer()
    le.fit_on_texts(y_train)
    y_train = np.array(le.texts_to_sequences(y_train)) - 1
    y_test = np.array(le.texts_to_sequences(y_test)) - 1
    
    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(X_train)

    sequences = tokenizer.texts_to_sequences(X_train)
    padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type, padding=padding_type)

    testing_sequences = tokenizer.texts_to_sequences(X_test)
    testing_padded = pad_sequences(testing_sequences,maxlen=max_length)

    model = tf.keras.Sequential([
        # YOUR CODE HERE.
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(32, 3, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction
    model.compile(optimizer="adam",
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    # Train your model with "fit" method
    model.fit(padded, y_train, epochs=20, validation_data=(testing_padded, y_test),verbose=2,
                callbacks=[cb])

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")
