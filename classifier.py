"""
Binary classification to detect trams and cars from each other.

Authors:
"""

import numpy as np
import matplotlib.pyplot as plt
from dataloader import load_data
import sounddevice as sd
import threading as th
import librosa as lb
import tensorflow as tf
from keras import backend as K

# load all .wav files and get their dft
train_X, train_Y = load_data('training_data')
test_X, test_Y = load_data('testing_data')

# normalize the data
train_X = train_X / np.max(train_X)
test_X = test_X / np.max(test_X)

train_X = np.abs(np.fft.fft(train_X))
test_X = np.abs(np.fft.fft(test_X))

train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)

val_X = train_X[int(0.8 * train_X.shape[0]):]
val_Y = train_Y[int(0.8 * train_Y.shape[0]):]
train_X = train_X[:int(0.8 * train_X.shape[0])]
train_Y = train_Y[:int(0.8 * train_Y.shape[0])]

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(train_X.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_X, train_Y, epochs=5, validation_data=(val_X, val_Y))

print('Testing...')

test = model.evaluate(test_X, test_Y)

print('Test loss: ', test[0])
print('Test accuracy: ', test[1])
