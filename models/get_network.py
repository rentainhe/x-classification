import os
import sys
import importlib
from importlib import import_module
import torch
from torchvision.models import resnet50

def get_network(__C):
    try:
        model_path = 'models.net'
        net = getattr(import_module(model_path),__C.model)
        return net()
    except ImportError:
        print('the network name you have entered is not supported yet')
        sys.exit()

# class config:
#     def __init__(self):
#         self.model = 'our50'
# def count_parameters(net):
#     params = sum([param.nelement() for param in net.parameters() if param.requires_grad])
#     print("Params: %f M" % (params/1000000))
#
# c = config()
# model = get_network(c)
# # model = resnet50(num_classes=100)
# x = torch.randn(1,3,224,224)
# count_parameters(model)
# print(model(x).size())