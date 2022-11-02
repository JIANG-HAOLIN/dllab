import torch
import torch.nn as nn
import numpy as np

class MyModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layer0 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=2, stride=1, padding=0),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.BatchNorm2d(64))
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=2, stride=1, padding=0),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.BatchNorm2d(64))
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=2, stride=1, padding=0),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.BatchNorm2d(64))
    def forward(self, x):
        x = self.layer0(x)
        return x




