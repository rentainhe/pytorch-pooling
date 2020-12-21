import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride, padding=0, dilation=1):
        super(MaxPool,self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
    def forward(self, x):
        x = F.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation)
        return x

def max():
    return MaxPool