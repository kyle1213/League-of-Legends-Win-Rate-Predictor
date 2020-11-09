import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import os
import numpy as np

loaded_model = tf.keras.models.load_model('model.h5', custom_objects = {'KerasLayer':hub.KerasLayer})
print("loaded model and weights")
loaded_model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

# data preprocessing
#data = [ 53, 32, 57, 72, 142, 145, 56, 129, 61, 135 ] #0
#data = [ 90, 55, 88, 121, 151, 102, 60, 38, 61, 78 ] #0
#data = [ 8, 60, 129, 72, 93, 115, 95, 105, 121, 135 ] #1
#data = [ 104, 60, 152, 72, 43, 119, 33, 3, 123, 26 ] #1
data = [ 54, 64, 75, 48, 98, 58, 28, 42, 151, 135 ] #0
#data = [58, 28, 42, 151, 135, 54, 64, 75, 48, 98] #1, reverse of above example
data = np.array(data)
data = data/152
data = np.reshape(data, (-1, 10))

print(data.shape)
print(data)


predict = loaded_model.predict(data)

print(predict)

