from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import BatchNormalization, Dropout
from keras import optimizers
import numpy as np
import json

#data load and preprocessing
with open('.\data\output2.json') as json_file:
    json_data = json.load(json_file)
    json_array = []
    print(len(json_data))

    for i in range(len(json_data)):
        json_array.append(json_data[i]["array"])

x_train = []
y_train = []
for i in range(len(json_data)):
    x_train.append(json_array[i][:10])
    y_train.append(json_array[i][10])

x_train = np.array(x_train)

for i in range(len(json_data)):
    if(y_train[i] == 0):
        y_train[i] = [1, 0]
    if(y_train[i] == 1):
        y_train[i] = [0, 1]

y_train = np.array(y_train)

print(y_train)



