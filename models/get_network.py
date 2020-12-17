import os
import sys
import importlib
from importlib import import_module

def get_network(args):
    if args.model == 'resnet18':
        from models.net import resnet18
        net = resnet18()
    elif args.model == 'mobilenet':
        from models.net import mobilenet
        net = mobilenet()
    elif args.model == 'shufflenet':
        from models.net import shufflenet
        net = shufflenet()
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    return net
