import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt

xy = np.loadtxt('D:\\CS\Deeplearning\\Pytorch\\PyTorch深度学习实践\\diabetes.csv', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, : -1])
y_data = torch.from_numpy(xy[:, [-1]])


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

criterion = torch.nn.BCELoss(reduction='none')
optimizer = torch.optim.SGD(model.parameters())

for epoch in range(1000):
    # forward
    y_pred = model.forward(x_data)
    loss = criterion(y_pred, y_data)
    print("Epoch = {}, Loss = {}".format(epoch, loss.item()))

    # backward
    optimizer.zero_grad()
    loss.backward()

    # update
    optimizer.step()

# 参数说明
# 第一层的参数：
layer1_weight = model.linear1.weight.data
layer1_bias = model.linear1.bias.data
print("layer1_weight", layer1_weight)
print("layer1_weight.shape", layer1_weight.shape)
print("layer1_bias", layer1_bias)
print("layer1_bias.shape", layer1_bias.shape)