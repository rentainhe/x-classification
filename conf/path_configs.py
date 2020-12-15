import os

class Path_Configs:
    def __init__(self):
        self.init_path()

    def init_path(self):

        self.data_path = {
            'cifar10' : './data/cifar10',
            'cifar100' : './data/cifar100',
            'imagenet' : './data/imagenet'
        }  # where to store your data

        self.result_logfile_path = './results/log'  # training log

        self.ckpts_path = './ckpts'  # where to store your model

        self.tensorboard_file_path = './runs' # where to store tensorboard file

        # if directory doesn't exist, create it
        if 'log' not in os.listdir('./results'):
            os.mkdir('./results/log')

        if 'ckpts' not in os.listdir('./'):
            os.mkdir('./ckpts')

        if 'runs' not in os.listdir('./'):
            os.mkdir('./runs')