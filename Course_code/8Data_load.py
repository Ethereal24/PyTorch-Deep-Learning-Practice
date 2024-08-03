import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, : -1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset('data/diabetes.csv')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)  # num_workers 多线程


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

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

if __name__ == '__main__':
    epoch_loss_list = []

    for epoch in range(100):
        epoch_loss = 0
        for index, (inputs, labels) in enumerate(train_loader, 0):
            # forward
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print("Epoch = {}, \titeration = {}, \tLoss = {}".format(epoch, index, loss.item()))

            epoch_loss += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()

            # update
            optimizer.step()

        epoch_loss /= len(train_loader)
        epoch_loss_list.append(epoch_loss)

        # if epoch % 10 == 0:
        plt.cla()  # 清除之前的图形
        plt.plot(epoch_loss_list)  # 绘制损失值
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.pause(0.01)  # 暂停一小段时间以更新图形

plt.show()
