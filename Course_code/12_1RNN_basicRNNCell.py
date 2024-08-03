import numpy as np
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.models.googlenet import Inception

batch_size = 5
seq_len = 3
input_size = 4
hidden_size = 2

cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

dataset = torch.randn(seq_len, batch_size, input_size)
print(dataset)
print(dataset.size())
print('------------------------------')

hidden = torch.zeros(batch_size, hidden_size)
print(hidden)
print(hidden.size())
print('------------------------------')

for idx, input in enumerate(dataset):
    print('=' * 20, idx, '=' * 20)
    print('Input size: ', input.shape)
    hidden = cell(input, hidden)
    print('outputs size: ', hidden.shape)
    print(hidden)
