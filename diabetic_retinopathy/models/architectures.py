import torch
import torch.nn as nn
import numpy as np
import torchvision.models


class MyModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.start_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer0 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=2, stride=1, padding=1),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.BatchNorm2d(64))
        self.layer1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.BatchNorm2d(128))
        self.layer2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=1),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.BatchNorm2d(256))
        self.layer3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=1),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.BatchNorm2d(256))
        self.layer4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=1),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.BatchNorm2d(256))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16384, 1024)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.start_pooling(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.squeeze()
        return x

class Efficient_model_double(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.efficient_model = torchvision.models.efficientnet_b3(pretrained=pretrained)

        self.fc1 = nn.Linear(1536, 2)
        self.fc2 = nn.Linear(1536, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.efficient_model.features(x)
        x = self.efficient_model.avgpool(x)
        x = torch.flatten(x, 1)
        return self.softmax(self.fc1(x)), self.softmax(self.fc2(x))





def get_efficient_model(origin=False, ema=False, reg=False, pretrained=True):
    if origin:
        efficient_model = torchvision.models.efficientnet_b3(pretrained=pretrained)
        efficient_model.classifier = nn.Sequential(nn.Linear(1536, 2), nn.Sigmoid(), nn.Flatten())
        return efficient_model
    if ema:
        efficient_model = Efficient_model_double(pretrained=pretrained)
        for para in efficient_model.parameters():
            para.detach()
        return efficient_model
    if reg:
        efficient_model_reg = torchvision.models.efficientnet_b3(pretrained=pretrained)
        efficient_model_reg.classifier = nn.Sequential(nn.Linear(1536, 1), nn.Flatten())
        return efficient_model_reg
    else:
        efficient_model = Efficient_model_double(pretrained=pretrained)
        return efficient_model



