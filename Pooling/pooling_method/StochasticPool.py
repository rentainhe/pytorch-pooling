import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

class StochasticPool2d(nn.Module):
    """ Stochastic 2D pooling, where prob(selecting index)~value of the activation
    IM_SIZE should be divisible by 2, not best implementation.
    based off:
    https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598#file-median_pool-py-L5
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=2, stride=2, padding=0, same=False):
        super(StochasticPool2d, self).__init__()
        self.kernel_size = _pair(kernel_size)  # I don't know what this is but it works
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # because multinomial likes to fail on GPU when all values are equal
        # Try randomly sampling without calling the get_random function a million times
        init_size = x.shape

        # x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
        x = x.contiguous().view(-1, 4)
        idx = torch.randint(0, x.shape[1], size=(x.shape[0],)).type(torch.cuda.LongTensor)
        x = x.contiguous().view(-1)
        x = torch.take(x, idx)
        x = x.contiguous().view(init_size[0], init_size[1], int(init_size[2] / 2), int(init_size[3] / 2))
        return x

def stochastic():
    print("You are using Stochatic Pooling Method")
    return StochasticPool2d