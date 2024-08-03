import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0
alpha = 0.01 # learning rate

def forward(x):
    return x * w

# 使用总体样本的损失值计算为梯度下降方法
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred) ** 2
    return cost / len(xs)


def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


print("Predict (before training): ", 4, forward(4))

for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= alpha * grad_val
    print("Epoch = ", epoch, " w = ", w, " cost = ", cost_val)


print("Predict (after training): ", 4, forward(4))