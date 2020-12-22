import torch
import torch.nn as nn
import torch.nn.functional as F

class LpPool(nn.Module):
    def __init__(self, kernel_size, stride, padding=0, p=3):
        super(LpPool,self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        assert self.kernel_size == self.stride, "we only support kernel_size=stride now"
        self.padding = padding
        self.p = p

    def lp_pool2d(self,input, p):
        return torch.mean(input ** p, (4, 5)) ** (1 / p)

    def forward(self, x):
        b,c,h,w = x.size()
        kh,kw = self.kernel_size,self.stride
        h = h//kh
        w = w//kw
        x = x.view(b,c,h,kh,w,kw).permute(0,1,2,4,3,5).contiguous()
        x = self.lp_pool2d(x,self.p)
        return x

def Lp():
    print("You are using Lp Pooling Method")
    return LpPool
