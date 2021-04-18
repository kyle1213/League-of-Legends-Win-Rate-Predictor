import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchsummary import summary as summary_
import json
import numpy as np
import matplotlib.pyplot as plt
import math

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
x_train = x_train / 152
pe = np.array([0.2, 0.4, 0.6, 0.8, 1, -0.2, -0.4, -0.6, -0.8, -1])
x_train = x_train + pe
x_train = np.reshape(x_train, (-1, 10, 1))
"""
pe = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
pe = np.tile(pe, (len(x_train), 1))
x_train = np.concatenate((x_train, pe), axis=1)
x_train = np.reshape(x_train, (len(x_train), 2, 10))
x_train = np.swapaxes(x_train, 2, 1)
"""
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
x_test = x_test / 152
pe = np.array([0.2, 0.4, 0.6, 0.8, 1, -0.2, -0.4, -0.6, -0.8, -1])
x_test = x_test + pe
x_test = np.reshape(x_test, (-1, 10, 1))
"""
pe = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
pe = np.tile(pe, (len(x_test), 1))
x_test = np.concatenate((x_test, pe), axis=1)
x_test = np.reshape(x_test, (len(x_test), 2, 10))
x_test = np.swapaxes(x_test, 2, 1)
"""
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
        self.softmax = torch.nn.Softmax(dim=1)
        self.Wq11 = torch.nn.Parameter(torch.randn(1, 16).cuda())
        self.Wk11 = torch.nn.Parameter(torch.randn(1, 16).cuda())
        self.Wv11 = torch.nn.Parameter(torch.randn(1, 16).cuda())
        self.Wq12 = torch.nn.Parameter(torch.randn(1, 16).cuda())
        self.Wk12 = torch.nn.Parameter(torch.randn(1, 16).cuda())
        self.Wv12 = torch.nn.Parameter(torch.randn(1, 16).cuda())
        self.Wo1 = torch.nn.Parameter(torch.randn(16 * 2, 1).cuda())
        self.Wq21 = torch.nn.Parameter(torch.randn(1, 16).cuda())
        self.Wk21 = torch.nn.Parameter(torch.randn(1, 16).cuda())
        self.Wv21 = torch.nn.Parameter(torch.randn(1, 16).cuda())
        self.Wq22 = torch.nn.Parameter(torch.randn(1, 16).cuda())
        self.Wk22 = torch.nn.Parameter(torch.randn(1, 16).cuda())
        self.Wv22 = torch.nn.Parameter(torch.randn(1, 16).cuda())
        self.Wo2 = torch.nn.Parameter(torch.randn(16 * 2, 1).cuda())
        self.layer1 = nn.Sequential(nn.Linear(10, 128),
                                    nn.BatchNorm1d(128),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(128, 10),
                                    nn.BatchNorm1d(10),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(10, 128),
                                    nn.BatchNorm1d(128),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(128, 256),
                                    nn.BatchNorm1d(256),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(256, 2))

    def forward(self, x):
        Q11 = torch.matmul(x, self.Wq11)
        K11 = torch.matmul(x, self.Wk11)
        K11 = torch.transpose(K11, 2, 1)
        V11 = torch.matmul(x, self.Wv11)
        attention1 = torch.bmm(Q11, K11)
        attention1 = self.softmax(attention1/math.sqrt(16))
        attention1 = torch.bmm(attention1, V11)
        Q12 = torch.matmul(x, self.Wq12)
        K12 = torch.matmul(x, self.Wk12)
        K12 = torch.transpose(K12, 2, 1)
        V12 = torch.matmul(x, self.Wv12)
        attention2 = torch.bmm(Q12, K12)
        attention2 = self.softmax(attention2 / math.sqrt(16))
        attention2 = torch.bmm(attention2, V12)
        attention = torch.cat((attention1, attention2), dim=2)
        x = torch.matmul(attention, self.Wo1)
        x = x.squeeze()
        x = self.layer1(x)

        x = torch.reshape(x, (-1, 10, 1))
        Q21 = torch.matmul(x, self.Wq21)
        K21 = torch.matmul(x, self.Wk21)
        K21 = torch.transpose(K21, 2, 1)
        V21 = torch.matmul(x, self.Wv21)
        attention1 = torch.bmm(Q21, K21)
        attention1 = self.softmax(attention1 / math.sqrt(16))
        attention1 = torch.bmm(attention1, V21)
        Q22 = torch.matmul(x, self.Wq22)
        K22 = torch.matmul(x, self.Wk22)
        K22 = torch.transpose(K22, 2, 1)
        V22 = torch.matmul(x, self.Wv22)
        attention2 = torch.bmm(Q22, K22)
        attention2 = self.softmax(attention2 / math.sqrt(16))
        attention2 = torch.bmm(attention2, V22)
        attention = torch.cat((attention1, attention2), dim=2)
        x = torch.matmul(attention, self.Wo2)
        x = x.squeeze()
        x = self.layer2(x)
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

summary_(model, (10, 1), batch_size=1)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, type(param.data), param.size())

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

torch.save(model.state_dict(), './model.pt')
