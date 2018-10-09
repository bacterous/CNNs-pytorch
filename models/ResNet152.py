# coding:utf8
from .BasicModule import BasicModule
import torch as t
from torch import nn
from torchvision import models


class ResNet152(BasicModule):
    def __init__(self, num_classes=2):
        super(ResNet152, self).__init__()
        self.model_name = 'resnet152'

        res = models.resnet152(pretrained=False)
        res.load_state_dict(t.load('pretrained_models\\resnet152-b121ed2d.pth'))
        self.res_layer = nn.Sequential(*list(res.children())[:-2])
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.res_layer(x)
        return self.fc(x)