import torch
from torch import nn

class CNN(nn.Module):

  def __init__(self):
    super().__init__()

    self.conv1 = torch.nn.Conv2d(1, 3, kernel_size=2, stride=2)
    self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=2, stride=2)

    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    self.h1 = nn.Linear(3 * 40 * 30, 64)
    self.h2 = nn.Linear(64, 5)
    self.dropout = nn.Dropout(p=0.16)

    self.relu = nn.ReLU()

  def forward(self, x):
    x = x.float()
    x = x.unsqueeze(1) # Adding channel dimension
    x = self.relu(self.conv1(x))
    x = self.pool(x)
    x = self.dropout(x)
    x = x.view(x.shape[0], -1)
    x = self.relu(self.h1(x))
    x = self.h2(x)
    return x  