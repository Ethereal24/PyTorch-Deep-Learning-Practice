import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# 读取原始数据，并划分训练集和测试集
raw_data = np.loadtxt('data/diabetes.csv', delimiter=',', dtype=np.float32)
X = raw_data[:, :-1]
Y = raw_data[:, [-1]]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
X_test = torch.from_numpy(X_test)
Y_test = torch.from_numpy(Y_test)


class DiabetesDataset(Dataset):
    def __init__(self, data, label):
        self.len = data.shape[0]  # shape(多少行，多少列)
        self.x_data = torch.from_numpy(data)
        self.y_data = torch.from_numpy(label)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


train_dataset = DiabetesDataset(X_train, Y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=0)  # num_workers 多线程


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 2)
        self.linear4 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        return x


model = Model()

# construct loss and optimizer
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# training cycle forward, backward, update

def train(epoch):
    train_loss = 0.0
    count = 0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        count = i

    if epoch % 2000 == 1999:
        print("train loss:", train_loss / count, end=',')


def test():
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_label = torch.where(y_pred >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))
        acc = torch.eq(y_pred_label, Y_test).sum().item() / Y_test.size(0)
        print("test acc:", acc)


# 将训练数据集进行批量处理
if __name__ == '__main__':
    for epoch in range(50000):
        train(epoch)
        if epoch % 2000 == 1999:
            test()
