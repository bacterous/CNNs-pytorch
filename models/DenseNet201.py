# coding:utf8
from .BasicModule import BasicModule
import torch as t
from torch import nn
from torchvision import models


class DenseNet201(BasicModule):
    def __init__(self, num_classes=2):
        super(DenseNet201, self).__init__()
        self.model_name = 'densenet201'

        dense = models.densenet201(pretrained=True)
        self.dense_layer = nn.Sequential(*list(dense.children())[:-2])
        self.classifier = nn.Linear(1920, num_classes)

    def forward(self, x):
        x = self.dense_layer(x)
        return self.classifier(x)