"""
Binary classification to detect trams and cars from each other.

Authors:
"""

import numpy as np
import matplotlib.pyplot as plt
from dataloader import load_data
import sounddevice as sd
import librosa as lb
import tensorflow as tf
from keras import backend as K

# Loads data and transforms it into MFCCs
def get_data():
    # load all .wav files and get their dft
    train_X, train_Y = load_data('training_data')
    test_X, test_Y = load_data('testing_data')
    val_X, val_Y = load_data('validation_data')

    # normalize the data
    train_X = train_X / np.max(train_X)
    test_X = test_X / np.max(test_X)
    val_X = val_X / np.max(val_X)


    train_X = get_mfccs(train_X)
    test_X = get_mfccs(test_X)
    val_X = get_mfccs(val_X)

    return train_X, train_Y, test_X, test_Y, val_X, val_Y

# Transform data into MFCC
def get_mfccs(data):

    new_data = []

    for i in range(data.shape[0]):
        mfcc = lb.feature.mfcc(y=data[i], sr=44100, n_mfcc=13)
        new_data.append(mfcc)

    return np.array(new_data)

# Train model
def train_model(train_X, train_Y, val_X, val_Y):

    print()
    print('Compiling and training model...')
    print()

    shape_x = train_X.shape[1]
    shape_y = train_X.shape[2]

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(shape_x, shape_y, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(train_X, train_Y, epochs=5, validation_data=(val_X, val_Y))

    print('Done!')

    return model, history

# Test model
def test_model(model, test_X, test_Y):

    print()
    print('Testing...')
    print()

    test = model.evaluate(test_X, test_Y)

    print()

    print('Test loss: ', test[0])
    print('Test accuracy: ', test[1])

# Print accuracy and loss graphs
def get_training_graphs(history):

    plt.figure(figsize=(10, 5))

    # Plot accuracies into one subplot
    plt.subplot(121)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='lower right')

    # Plot losses into one subplot
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.show()


# ---------------------------- AUDIO PROJECT ---------------------------- 

print("\n\n -------------------------------------------------------- \n\n")

# Get data
train_X, train_Y, test_X, test_Y, val_X, val_Y = get_data()

# Compile model
model, history = train_model(train_X, train_Y, val_X, val_Y)

# Test model
test_model(model, test_X, test_Y)

# Get graphs
get_training_graphs(history)

print()
print()