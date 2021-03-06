"""vgg in pytorch
[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
from Pooling.get_pooling import get_pooling


cfg = {
    'A' : [64,     'P', 128,      'P', 256, 256,           'P', 512, 512,           'P', 512, 512,           'P'],
    'B' : [64, 64, 'P', 128, 128, 'P', 256, 256,           'P', 512, 512,           'P', 512, 512,           'P'],
    'D' : [64, 64, 'P', 128, 128, 'P', 256, 256, 256,      'P', 512, 512, 512,      'P', 512, 512, 512,      'P'],
    'E' : [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 512, 512, 512, 512, 'P', 512, 512, 512, 512, 'P']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, __C, batch_norm=False):
    layers = []

    input_channel = 3
    for i,l in enumerate(cfg):
        if l == 'P':

            if __C.pooling == 'lip':
                layers += [get_pooling(__C)(cfg[i-1])]

            else: layers += [get_pooling(__C)(kernel_size=2,stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn(__C):
    return VGG(make_layers(cfg['A'], __C, batch_norm=True))

def vgg13_bn(__C):
    return VGG(make_layers(cfg['B'], __C, batch_norm=True))

def vgg16_bn(__C):
    return VGG(make_layers(cfg['D'], __C, batch_norm=True))

def vgg19_bn(__C):
    return VGG(make_layers(cfg['E'], __C, batch_norm=True))

def vgg11(__C):
    return VGG(make_layers(cfg['A'], __C, batch_norm=False))

def vgg13(__C):
    return VGG(make_layers(cfg['B'], __C, batch_norm=False))

def vgg16(__C):
    return VGG(make_layers(cfg['D'], __C, batch_norm=False))

def vgg19(__C):
    return VGG(make_layers(cfg['E'], __C, batch_norm=False))


