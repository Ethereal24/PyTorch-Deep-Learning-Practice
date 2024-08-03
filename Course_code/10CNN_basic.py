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


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # convolution layer 输入的为
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        # Pooling layer
        self.pooling = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # Linear layer
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # flatten data from (n,1,28,28) to (n, 784)
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        return self.fc(x)


model = CNN()

# GPU版本
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# 无需额外设计softmax层，CrossEntropyLoss中包含
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader, 0):
        images_pred = model(images)
        loss = criterion(images_pred, labels)

        # GPU
        # images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 300 == 299:
            loss_list.append(running_loss)

            print("Train Epoch: {}, \titeration: {}, \tLoss: {:.4f}".format(epoch, batch_idx, running_loss / 300))
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
