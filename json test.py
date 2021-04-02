import json
import numpy as np

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

x_train = x_train

print(x_train)
print(y_train)
