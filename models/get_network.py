import os
import sys
import re
import datetime

import numpy
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_network(args):
    if args.model == 'resnet18':
        from models.net.ResNet import resnet18
        net = resnet18()
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    return net