import sys
from importlib import import_module
import torch

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
        self.pooling = 'lip'

c = config()
pool = get_pooling(c)
p = pool(128)
x = torch.randn(1,128,4,4)
print(p(x))