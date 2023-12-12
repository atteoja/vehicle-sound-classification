"""
Convert audio files from m4a to wav.
"""

from pydub import AudioSegment
import os

input_directory = 'audiofiles'
output_directory = 'wavfiles'

def convert_m4a_to_wav(input_file, output_file):
    sound = AudioSegment.from_file(input_file, format="m4a")
    sound.export(output_file, format="wav")

for filename in os.listdir(input_directory):
    if filename.endswith(".m4a"):
        input_file_path = os.path.join(input_directory, filename)
        output_file_path = os.path.join(output_directory, os.path.splitext(filename)[0] + '.wav')
        convert_m4a_to_wav(input_file_path, output_file_path)
