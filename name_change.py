import os

folder_name = 'training_data'

for i, filename in enumerate(os.listdir(folder_name)):
    os.rename(folder_name + '/' + filename, folder_name + '/' + 'ratikka_' + str(i + 1) + '.wav')