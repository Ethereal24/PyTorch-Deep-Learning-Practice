import numpy as np
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 准备训练集和测试集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# 定义模型的结构
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)  # 最后一层不作激活函数，过softmax层


model = Net()

# 无需额外设计softmax层，CrossEntropyLoss中包含
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader, 0):
        images_pred = model(images)
        loss = criterion(images_pred, labels)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 300 == 299:
            loss_list.append(running_loss)

            print("Train Epoch: {}, \titeration: {}, \tLoss: {:.4f}".format(epoch, batch_idx, running_loss))

            plt.cla()
            plt.plot(loss_list)
            plt.xlabel('Batch Number')
            plt.ylabel('Loss')
            plt.title("Loss with Batch Number")
            plt.pause(0.01)

            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images_pred = model(images)
            # 第一个位置填写_表示这个值不重要，我们不关心，也就不用再定义一个变量接收
            _, predicted = torch.max(images_pred, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 张量之间的比较运算
    print('Accuracy on test set: %d %% [%d/%d]' % (100 * correct / total, correct, total))


if __name__ == '__main__':
    loss_list = []

    for epoch in range(10):
        train(epoch)
        test()

    plt.show()