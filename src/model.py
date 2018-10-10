import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = self.conv2d(1, 6)
        self.conv2 = self.conv2d(6, 16)
        self.fc1 = self.linear(16*4*4, 120)
        self.fc2 = self.linear(120, 84)
        self.fc3 = self.linear(84, 10)

    def conv2d(self, in_c, out_c):
        return nn.Sequential(
                nn.Conv2d(in_c, out_c, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
                )

    def linear(self, in_c, out_c):
        return nn.Sequential(
                nn.Linear(in_c, out_c),
                nn.ReLU()
                )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 16*4*4)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
