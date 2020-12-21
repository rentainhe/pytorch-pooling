import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride, padding=0):
        super(MaxPool,self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        return x

def max():
    print("You are using Max Pooling Method")
    return MaxPool