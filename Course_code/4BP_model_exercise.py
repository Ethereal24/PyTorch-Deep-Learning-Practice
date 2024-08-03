import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = torch.Tensor([1.0])
w1.requires_grad = True
w2 = torch.Tensor([1.0])
w2.requires_grad = True
b = torch.Tensor([1.0])
b.requires_grad = True
alpha = 0.02  # learning rate

epoch_list = []
loss_list = []


def forward(x):
    return w1 * x ** 2 + w2 * x + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


def gradient(x, y):
    return 2 * x * (x * w1 - y)


print("Predict (before training): ", 4, forward(4).item())

for epoch in range(100):
    l = None
    for x, y in zip(x_data, y_data):
        l = loss(x, y)  # l是一个张量，tensor主要是在建立计算图 forward, compute the loss
        l.backward()  # backward,compute grad for Tensor whose requires_grad set to True

        print('\tgrad:', x, y, w1.grad.item(), w2.grad.item(), b.grad.item())

        w1.data = w1.data - alpha * w1.grad.data  # 权重更新时，注意grad也是一个tensor
        w2.data = w2.data - alpha * w2.grad.data
        b.data = b.data - alpha * b.grad.data
        w1.grad.data.zero_()  # 对w的权值进行清零
        w2.grad.data.zero_()  # 对w的权值进行清零
        b.grad.data.zero_()  # 对w的权值进行清零

    print("epoch = ", epoch, "\tloss = ", l.item())  # 取出loss使用l.item，不要直接使用l（l是tensor会构建计算图）

print("Predict (after training): ", 4, forward(4).item())
