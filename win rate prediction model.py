import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import matplotlib.pyplot as plt


with open('./data/output2.json') as json_train_file:
    json_train_data = json.load(json_train_file)
    json_train_array = []
    print(len(json_train_data))

    for i in range(len(json_train_data)):
        json_train_array.append(json_train_data[i]["array"])

x_train = []
y_train = []

for i in range(len(json_train_data)):
    x_train.append(json_train_array[i][:10])
    y_train.append(json_train_array[i][10])

x_train = np.array(x_train)
x_train = np.reshape(x_train, [40180, 2, 5])
x_train = x_train / 152

for i in range(len(json_train_data)):
    if y_train[i] == 0:
        y_train[i] = [1, 0]
    if y_train[i] == 1:
        y_train[i] = [0, 1]

y_train = np.array(y_train)

with open('./data/test.json') as json_test_file:
    json_test_data = json.load(json_test_file)
    json_test_array = []
    print(len(json_test_data))

    for i in range(len(json_test_data)):
        json_test_array.append(json_test_data[i]["array"])

x_test = []
y_test = []

for i in range(len(json_test_data)):
    x_test.append(json_test_array[i][:10])
    y_test.append(json_test_array[i][10])

x_test = np.array(x_test)
x_test = np.reshape(x_test, [75, 2, 5])
x_test = x_test / 152

for i in range(len(json_test_data)):
    if y_test[i] == 0:
        y_test[i] = [1, 0]
    if y_test[i] == 1:
        y_test[i] = [0, 1]

y_test = np.array(y_test)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, _x_train, _y_train):
        self.features = _x_train
        self.labels = _y_train

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

    def __len__(self):
        return len(self.features)


train_loader = torch.utils.data.DataLoader(dataset=Dataset(x_train, y_train), batch_size=1000, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=Dataset(x_test, y_test), batch_size=1000, shuffle=False)


cuda = torch.device('cuda')


class LOL(nn.Module):
    def __init__(self):
        super(LOL, self).__init__()
        self.matrix_weight_1 = torch.nn.Parameter(torch.randn(5, 1).cuda())
        self.matrix_weight_2 = torch.nn.Parameter(torch.randn(2, 1).cuda())
        #use torch.bmm
        self.layer1 = nn.Sequential(nn.Linear(10, 128),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.BatchNorm1d(128),
                                    nn.Linear(128, 256),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.BatchNorm1d(256),
                                    nn.Linear(256, 512),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.BatchNorm1d(512),
                                    nn.Linear(512, 1024),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.BatchNorm1d(1024),
                                    nn.Linear(1024, 1024),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.BatchNorm1d(1024),
                                    nn.Linear(1024, 512),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.BatchNorm1d(512),
                                    nn.Linear(512, 256),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.BatchNorm1d(256),
                                    nn.Linear(256, 2))

    def forward(self, x):
        reshaped_x_1 = torch.matmul(x, self.matrix_weight_1)
        tmp_x = torch.transpose(x, 2, 1)
        reshaped_x_2 = torch.matmul(tmp_x, self.matrix_weight_2)
        reshaped_x_2 = torch.transpose(reshaped_x_2, 2, 1)
        x_1 = x * reshaped_x_1
        x_2 = x * reshaped_x_2
        x = x_1 + x_2
        x = torch.reshape(x, [-1, 10])
        """
        x = torch.reshape(x, [-1, 10])
        reshaped_x_1 = torch.squeeze(reshaped_x_1)
        reshaped_x_2 = torch.squeeze(reshaped_x_2)
        x = torch.cat((x, reshaped_x_1, reshaped_x_2), dim=1)
        """
        x = self.layer1(x)
        return x


model = LOL()
model = model.cuda()

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=5e-4, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
cost = 0

iterations = []
train_losses = []
train_acc = []


for epoch in range(1000):
    model.train()
    correct = 0
    for x, y in train_loader:
        x = x.float()
        y = y.float()
        x = x.to(cuda)
        y = y.to(cuda)
        optimizer.zero_grad()
        hypo = model(x)
        cost = loss(hypo, torch.argmax(y, 1))
        cost.backward()
        optimizer.step()
        scheduler.step()
        prediction = hypo.data.max(1)[1]
        correct += prediction.eq(y.data.max(1)[1]).sum()
    print("Epoch : {:>4} / cost : {:>.9}".format(epoch + 1, cost))
    print("lr : {:>6}".format(scheduler.optimizer.state_dict()['param_groups'][0]['lr']))
    iterations.append(epoch)
    train_losses.append(cost.tolist())
    train_acc.append((100 * correct / len(train_loader.dataset)).tolist())

model.eval()
correct = 0
for data, target in test_loader:
    data = data.float()
    target = target.float()
    data = data.to(cuda)
    target = target.to(cuda)
    output = model(data)
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data.max(1)[1]).sum()

print('Test set: Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))

plt.subplot(121)
plt.plot(range(1, len(iterations)+1), train_losses, 'b--')
plt.subplot(122)
plt.plot(range(1, len(iterations)+1), train_acc, 'b-')
plt.title('loss and accuracy')
plt.show()


"""
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
"""
