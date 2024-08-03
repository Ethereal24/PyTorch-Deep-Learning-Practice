import torch

# 输入维度，每个字母对应张量维度
batch_size = 1
input_size = 4
hidden_size = 4
seq_len = 5
num_layers = 1

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


# 构建模型
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size)

    def forward(self, inputs):
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        out, _ = self.rnn(inputs, hidden)  # 注意维度是(seqLen, batch_size, hidden_size)
        return out.view(-1, self.hidden_size)  # 为了容易计算交叉熵这里调整维度为(seqLen * batch_size, hidden_size)


net = Model(input_size, hidden_size, batch_size)

# 损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

# 训练
for epoch in range(15):
    outputs = net(inputs)
    # print(outputs.shape, labels.shape)
    # 这里的outputs维度是([seqLen * batch_size, hidden]), labels维度是([seqLen])
    loss = criterion(outputs, labels)

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))
