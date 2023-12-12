import os

folder_name = 'testing_data'

i = 1

for file in os.listdir(folder_name):
    # rename the file to 'ratikka_i.wav'
    print(file, "ratikka_" + str(i) + ".wav")
    os.rename(folder_name + '/' + file, folder_name + '/' + "ratikka_" + str(i) + ".wav")

    i += 1