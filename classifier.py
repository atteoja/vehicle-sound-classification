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


# load all .wav files and get their dft
train_X, train_Y = load_data('wavfiles')

print(train_X.shape)
print(train_Y.shape)