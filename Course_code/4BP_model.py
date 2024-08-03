import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])
w.requires_grad = True
alpha = 0.01  # learning rate

epoch_list = []
loss_list = []


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


def gradient(x, y):
    return 2 * x * (x * w - y)


print("Predict (before training): ", 4, forward(4).item())

for epoch in range(100):
    l = None
    for x, y in zip(x_data, y_data):
        l = loss(x, y)  # l是一个张量，tensor主要是在建立计算图 forward, compute the loss
        l.backward()  # backward,compute grad for Tensor whose requires_grad set to True

        print('\tgrad:', x, y, w.grad.item())

        w.data = w.data - 0.01 * w.grad.data  # 权重更新时，注意grad也是一个tensor
        w.grad.data.zero_()  # 对w的权值进行清零

    print('progress:', epoch, l.item())  # 取出loss使用l.item，不要直接使用l（l是tensor会构建计算图）

print("Predict (after training): ", 4, forward(4).item())
