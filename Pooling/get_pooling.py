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
