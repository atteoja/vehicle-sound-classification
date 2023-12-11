import numpy as np
import librosa as lb
import sounddevice as sd
import os

# Loads all .wav files from a directory.
# Returns a list of car sounds labeled as 0 and a list of tram sounds labeled as 1.
def load_data(path, sampling_rate=44100):
    """
    Load all .wav files from a directory and get their dft.
    """

    vehicles = np.empty((0, 5*sampling_rate))
    labels = np.empty((0, 1))

    for file in os.listdir(path):
        if file.endswith(".wav"):
            try:
                sound, _ = lb.load(os.path.join(path, file), sr=sampling_rate)

                # if sound is shorter than 5 seconds, pad with zeros
                if len(sound) < 5*sampling_rate:
                    sound = np.pad(sound, (0, 5*sampling_rate - len(sound)), 'constant')

                # take first 5 seconds of the sound
                sound = sound[:5*sampling_rate]

                if file.startswith('auto'):
                    vehicles = np.vstack((vehicles, sound))
                    labels = np.append(labels, 0)

                if file.startswith('ratikka'):
                    vehicles = np.vstack((vehicles, sound))
                    labels = np.append(labels, 1)

                print("File " + file + " loaded.")

            except Exception as e:
                print("File " + file + " could not be loaded.")
                print(e)

    return vehicles, labels

'''
train_X, train_Y = load_data('wavfiles')
print(train_X.shape)
print(train_Y.shape)

#play first 5 sounds in train_X
for i in range(5):
    print(train_Y[i])
    sd.play(train_X[i], 44100)
    sd.wait()
'''