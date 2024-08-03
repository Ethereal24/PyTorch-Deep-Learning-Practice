import torch

# parameters
num_class = 4
input_size = 4
hidden_size = 8
embedding_size = 10
num_layers = 2
batch_size = 1
seq_len = 5

# 准备数据集
idx2char = ['e', 'h', 'l', 'o']
x_data = [[1, 0, 2, 2, 3]]  # (batch, seq_len)
y_data = [3, 1, 2, 3, 2]  # (batch * seq_len)

inputs = torch.LongTensor(x_data)  # Input should be LongTensor: (batchSize, seqLen)
labels = torch.LongTensor(y_data)  # Target should be LongTensor: (batchSize * seqLen)

print("inputs size = ", inputs.size())
print("labels size = ", labels.size())

# 构建模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb = torch.nn.Embedding(input_size, embedding_size)
        self.rnn = torch.nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                                batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_class)

    def forward(self, x):
        hidden = torch.zeros(num_layers, x.size(0), hidden_size)
        x = self.emb(x)  # (batch, seqLen, embeddingSize), torch.Size([1, 5, 10])
        # print(x.size())
        x, _ = self.rnn(x, hidden)  # 输出(𝒃𝒂𝒕𝒄𝒉𝑺𝒊𝒛𝒆, 𝒔𝒆𝒒𝑳𝒆𝒏, hidden_size)
        x = self.fc(x)  # 输出(𝒃𝒂𝒕𝒄𝒉𝑺𝒊𝒛𝒆, 𝒔𝒆𝒒𝑳𝒆𝒏, 𝒏𝒖𝒎𝑪𝒍𝒂𝒔𝒔)
        # reshape to use Cross Entropy: (𝒃𝒂𝒕𝒄𝒉𝑺𝒊𝒛𝒆×𝒔𝒆𝒒𝑳𝒆𝒏, 𝒏𝒖𝒎𝑪𝒍𝒂𝒔𝒔)
        return x.view(-1, num_class)


net = Model()

# 损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

# 训练模型
for epoch in range(15):
    outputs = net(inputs)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))
