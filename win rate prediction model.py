from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization, Dropout
from keras import optimizers
import numpy as np
import json

# data load and preprocessing
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
    if y_train[i] == 0:
        y_train[i] = [1, 0]
    if y_train[i] == 1:
        y_train[i] = [0, 1]

y_train = np.array(y_train)

x_train = x_train / 152

# model
model = Sequential()

model.add(Dense(128, activation='relu', input_dim=10))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(2, activation='softmax'))

rmsprop = optimizers.RMSprop(lr=0.001)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000, batch_size=512)

model.save('model.h5')
