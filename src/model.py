from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import models


class ModelName(Enum):
    DeepFont = 1
    VGG16 = 2
    efficientnetB3 = 3


class DeepFont(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 58)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 1)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, 1)
        self.conv4 = nn.Conv2d(256, 256, 1)
        self.conv5 = nn.Conv2d(256, 256, 1)

        self.fc6 = nn.Linear(256 * 12 * 12, 4096)
        self.dropout6 = nn.Dropout(0.5)
        self.fc7 = nn.Linear(4096, 4096)
        self.dropout7 = nn.Dropout(0.5)
        self.fc8 = nn.Linear(4096, 28)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool1(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool2(F.relu(self.batch_norm2(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.flatten(x, 1)
        x = self.dropout6(F.relu(self.fc6(x)))
        x = self.dropout7(F.relu(self.fc7(x)))
        x = self.fc8(x)

        return x


def fetch_vgg16() -> nn.Module:
    net = models.vgg16_bn()
    net.features[0] = nn.Conv2d(1, 64, 3, stride=1, padding=1)
    net.classifier[6] = nn.Linear(4096, 28)

    return net


def fetch_efficientnet_b3() -> nn.Module:
    net = models.efficientnet_b3()
    net.features[0] = nn.Conv2d(1, 40, 3, stride=2, padding=1)
    net.classifier[1] = nn.Linear(1536, 28)

    return net
