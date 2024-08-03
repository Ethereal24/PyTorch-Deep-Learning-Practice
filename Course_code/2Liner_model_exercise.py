import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0, 2.0, 3.0]
y_data = [3.0, 5.0, 7.0]


def forward(x):
    return x * w + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


w_list = []
b_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(0.0, 4.1, 0.1):
        print('w = ', w, 'b = ', b)
        loss_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            loss_sum += loss_val
            print('\t', x_val, y_val, y_pred_val, loss_val)
        print('MSE=', loss_sum / 3)
        w_list.append(w)
        b_list.append(b)
        mse_list.append((loss_sum) / 3)

# 绘制图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 绘制线框图
ax.plot(w_list, b_list, mse_list, '-o')
# 设置标签
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Loss')
# 显示图形
plt.show()
