import os
import sys
import importlib
from importlib import import_module

# def get_network(args):
#     if args.model == 'resnet18':
#         from models.net import resnet18
#         net = resnet18()
#     elif args.model == 'resnet34':
#         from models.net import resnet34
#         net = resnet34()
#     elif args.model == 'resnet50':
#         from models.net import resnet50
#         net = resnet50()
#     elif args.model == 'mobilenet':
#         from models.net import mobilenet
#         net = mobilenet()
#     elif args.model == 'mobilenetv2':
#         from models.net import mobilenetv2
#         net = mobilenetv2()
#     elif args.model == 'shufflenet':
#         from models.net import shufflenet
#         net = shufflenet()
#     else:
#         print('the network name you have entered is not supported yet')
#         sys.exit()
#     return net

def get_network(__C):
    try:
        model_path = 'models.net'
        net = getattr(import_module(model_path),__C.model)
        return net()
    except ImportError:
        print('the network name you have entered is not supported yet')
        sys.exit()
