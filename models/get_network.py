import os
import sys
import importlib
from importlib import import_module

def get_network(__C):
    try:
        model_path = 'models.net'
        net = getattr(import_module(model_path),__C.model)
        return net()
    except ImportError:
        print('the network name you have entered is not supported yet')
        sys.exit()
