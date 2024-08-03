import torch

# 输入维度，每个字母对应张量维度
batch_size = 1
input_size = 4
hidden_size = 4

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 3, 3]  # hello
y_data = [3, 1, 2, 3, 2]  # ohlol

on_hot_lookup = [[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]

# x_one_hot = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]]
x_one_hot = [on_hot_lookup[x] for x in x_data]

inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
labels = torch.LongTensor(y_data).view(-1, 1)
print(inputs.shape)
print(labels.shape)


class model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)

    def forward(self, inputs, hidden):
        hidden = self.rnncell(inputs, hidden)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)


net = model(input_size, hidden_size, batch_size)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

for epoch in range(100):
    loss = 0
    # 初始化一个全0的hidden统一计算过程
    hidden = net.init_hidden()
    print("Predict string = ", end='')
    for input, label in zip(inputs, labels):
        hidden = net(input, hidden)
        loss += criterion(hidden, label)
        _, idx = hidden.max(dim=1)
        print(idx2char[idx.item()], end='')

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    print(', Epoch %d loss=%.4f' % (epoch + 1, loss.item()))
