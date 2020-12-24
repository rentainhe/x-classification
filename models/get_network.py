import os
import sys
import importlib
from importlib import import_module
import torch

def get_network(__C):
    try:
        model_path = 'models.net'
        net = getattr(import_module(model_path),__C.model)
        return net()
    except ImportError:
        print('the network name you have entered is not supported yet')
        sys.exit()

class config:
    def __init__(self):
        self.model = 'resnet50'

c = config()
model = get_network(c)
x = torch.randn(1,3,224,224)
print(model(x).size())