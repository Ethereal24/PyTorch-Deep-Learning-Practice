import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0
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


print("Predict (before training): ", 4, forward(4))

for epoch in range(100):
    loss_val = 0
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w = w - alpha * grad
        print("\tgrad = ", x, y, grad)
        # 每次样本进行训练后都会计算当前单个样本的loss值，最后计算的样本值作为本轮epoch的最终的loss_val
        loss_val = loss(x, y)

    print("Epoch = ", epoch, "\tw = ", w, "\tLoss =  ", loss_val)
    epoch_list.append(epoch)
    loss_list.append(loss_val)

print("Predict (after training): ", 4, forward(4))

plt.plot(epoch_list, loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
