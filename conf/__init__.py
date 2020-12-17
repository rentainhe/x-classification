import conf.global_configs as global_configs
from types import MethodType
import os
import torch
import numpy as np
import random

class Settings:
    def __init__(self, settings):

        for attr in dir(settings):
            if not attr.startswith('__'):
                setattr(self,attr,getattr(settings,attr))

    def parse_to_dict(self, args):
        args_dict = {}
        for arg in dir(args):
            if not arg.startswith('_') and not isinstance(getattr(args, arg), MethodType):
                if getattr(args, arg) is not None:
                    args_dict[arg] = getattr(args, arg)
        return args_dict

    def add_args(self, args_dict):
        for arg in args_dict:
            setattr(self, arg, args_dict[arg])

    def training_init(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu
        self.n_gpu = len(self.gpu.split(','))
        self.devices = [ _ for _ in range(self.n_gpu) ]
        torch.set_num_threads(2)

        # fix seed
        torch.manual_seed(self.seed)
        if self.n_gpu < 2:
            torch.cuda.manual_seed(self.seed)
        else:
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Gradient accumulate setup
        assert self.batch_size % self.gradient_accumulation_steps == 0
        self.sub_batch_size = int(self.batch_size / self.gradient_accumulation_steps)
        self.eval_batch_size = int(self.sub_batch_size / 2)

    def path_init(self):
        for attr in dir(self):
            if 'dir' in attr and not attr.startswith('__'):
                if getattr(self,attr) not in os.listdir('./'):
                    os.makedirs(getattr(self, attr))

    def __str__(self):
        # print Hyper Parameters
        settings_str = ''
        for attr in dir(self):
            # 如果不加 attr.startwith('__')会打印出很多额外的参数，是自身自带的一些默认方法和属性
            if not 'np' in attr and not 'random' in attr and not attr.startswith('__') and not isinstance(getattr(self, attr), MethodType):
                settings_str += '{ %-17s }->' % attr + str(getattr(self, attr)) + '\n'
        return settings_str

configs = Settings(global_configs)




