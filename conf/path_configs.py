import os

class Path_Configs:
    def __init__(self):
        self.init_path()

    def init_path(self):

        self.DATA_PATH = {
            'cifar10' : './data/cifar10',
            'cifar100' : './data/cifar100',
            'imagenet' : './data/imagenet'
        }  # where to store your data

        self.LOG_PATH = './results/log'  # training log

        self.CKPTS_PATH = './ckpts'  # where to store your model

        # if directory doesn't exist, create it
        if 'log' not in os.listdir('./results'):
            os.mkdir('./results/log')

        if 'ckpts' not in os.listdir('./'):
            os.mkdir('./ckpts')