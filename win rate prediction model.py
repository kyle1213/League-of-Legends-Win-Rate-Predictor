from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import BatchNormalization
from keras import optimizers
import numpy as np
import json

#data load and preprocessing
with open('.\data\datas.json') as json_file:
    json_data = json.load(json_file)
    json_array = []
    print(len(json_data))

    for i in range(len(json_data)):
        json_array.append(json_data[i]["array"])

x_train = []
y_train = []
for i in range(len(json_data)):
    x_train.append(json_array[i][:10])
    y_train.append(json_array[i][10:])

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = x_train / 1000


#model
model = Sequential()

model.add(Dense(16, input_dim = 10))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(32))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(32))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(16))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(2))
model.add(Activation('softmax'))

rmsprop = optimizers.RMSprop(lr = 0.00001)
model.compile(optimizer = rmsprop, loss = 'binary_crossentropy', metrics = ['accuracy'])



'''
#minibatch
iters_num = 10
train_size = len(json_data)
batch_size = int(train_size/10)


for i in range(iters_num):
    x_batch = []
    y_batch = []
    for j in range(batch_size):      
        x_batch.append(x_train[j + i * batch_size])
        y_batch.append(y_train[j + i * batch_size])

    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)

    #learning
'''

    

model.fit(x_train, y_train, epochs = 100, batch_size = 128)


model.save('model.h5')






