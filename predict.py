import torch
import torch.nn as nn
import math
import numpy as np

#train
#data = [ 53, 32, 57, 72, 142, 145, 56, 129, 61, 135 ]
#y = [1, 0]
#data = [ 90, 55, 88, 121, 151, 102, 60, 38, 61, 78 ]
#y = [1, 0]
#data = [ 8, 60, 129, 72, 93, 115, 95, 105, 121, 135 ]
#y = [0, 1]
#data = [ 104, 60, 152, 72, 43, 119, 33, 3, 123, 26 ]
#y = [0, 1]
#data = [ 54, 64, 75, 48, 98, 58, 28, 42, 151, 135 ]
#y = [1, 0]
#data = [58, 28, 42, 151, 135, 54, 64, 75, 48, 98] #reverse of above example
#y = [0, 1]
#test
#data = [ 129, 55, 4, 15, 84, 24, 60, 146, 112, 9 ] #1
#y = [0, 1]
#data = [ 54, 5, 89, 61, 137, 17, 115, 38, 148, 149 ] #1
data = [ 132, 118, 8, 48, 137, 50, 131, 38, 148, 149 ]
y = [1, 0]
#data = [ 119, 122, 4, 121, 59, 114, 28, 87, 82, 98 ] #0

data = np.array(data)
data = data/152
data = np.reshape(data, (-1, 10, 1))
data = torch.tensor(data)
y = np.array(y)
y = np.reshape(y, (1, 2))
y = torch.tensor(y)

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
        x = x.squeeze(dim=2)
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
        x = x.squeeze(dim=2)
        x = self.layer2(x)
        return x


model = LOL()
model = model.cuda()

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=5e-4, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
cost = 0

model.load_state_dict(torch.load('./model.pt'))
data = data.float()
data = data.to(cuda)
y = y.float()
y = y.to(cuda)
model.eval()
print(model(data))
print(model(data).data.max(1)[1])
print(y.data.max(1)[1])
if model(data).data.max(1)[1] == 0:
    print("Blue wins")
elif model(data).data.max(1)[1] == 1:
    print("Purple wins")