import torch

batch_size = 1 # 批处理大小
seq_len = 3 # 序列长度
input_size = 4 # 输入维度
hidden_size = 2 # 隐藏层维度
num_layers = 4  # 隐藏层数量

cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

# (seqLen, batchSize, inputSize)
inputs = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(num_layers, batch_size, hidden_size)
out, hidden = cell(inputs, hidden)

print( 'Output size:', out.shape)
print( 'Output:', out)
print( 'Hidden size: ', hidden.shape)
print( 'Hidden: ', hidden)
