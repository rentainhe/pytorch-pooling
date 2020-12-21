import sys
from importlib import import_module

def get_pooling(__C):
    try:
        poolmethod_path = 'Pooling.pooling_method'
        pool_method = getattr(import_module(poolmethod_path), __C.pooling)
        return pool_method()
    except ImportError:
        print('the pool method name you entered is not supported yet')
        sys.exit()

class config:
    def __init__(self):
        self.pooling = 'max'
        self.kernel_size = 2
        self.stride = 2
        self.padding = 0
        self.dilation = 1

c = config()
max = get_pooling(c)
max(kernel_size=1,stride=1)