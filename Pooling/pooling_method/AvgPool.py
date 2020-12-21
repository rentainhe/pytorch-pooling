import torch
import torch.nn as nn
import torch.nn.functional as F

class AvgPool(nn.Module):
    def __init__(self, kernel_size, stride, padding=0):
        super(AvgPool,self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        return x

def avg():
    print("You are using Avg Pooling Method")
    return AvgPool