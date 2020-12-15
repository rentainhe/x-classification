from conf.path_configs import Path_Configs
import os
import torch
import random
import numpy as np
from types import MethodType

class Base_Configs(Path_Configs):
    def __init__(self):
        super(Base_Configs, self).__init__()

        # Set Devices
        # If use multi-gpu training, you can set e.g '0,1,2' instead
        self.gpu='0'

        # Set Seed For CPU and GPUs
        self.seed = random.randint(0, 9999999)

        # Version Control
        self.version = str(self.seed)

        # Use checkpoint to resume training
        self.resume = False

        self.ckpt_version = self.version

        self.ckpt_path = None

        self.batch_size = 64

        self.num_workers = 8

        self.pin_mem = True

        self.gradient_accumulation_steps = 1

        self.eval_every_epoch = True

    def str_to_bool(self, args):
        bool_list = [
            'eval_every_epoch',
            'resume',
            'pin_mem',
        ]

        for arg in dir(args):
            if arg in bool_list and getattr(args, arg) is not None:
                setattr(args, arg, getattr(args, arg))

        return args

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

    def process(self):

        # ----------------------- setting devices
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu
        self.N_GPU = len(self.gpu.split(','))
        self.DEVICES = [ _ for _ in range(self.N_GPU)]
        torch.set_num_threads(2)

        # ----------------------- setting seed
        # fix pytorch seed
        torch.manual_seed(self.seed)
        if self.N_GPU < 2:
            torch.cuda.manual_seed(self.seed)
        else:
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True

        # fix numpy seed
        np.random.seed(self.seed)

        # fix random seed
        random.seed(self.seed)

        # ----------- Gradient accumulate setup
        assert self.batch_size % self.gradient_accumulation_steps == 0
        self.sub_batch_size = int(self.batch_size / self.gradient_accumulation_steps)

        # set small eval batch size will reduce gpu memory usage
        self.eval_batch_size = int(self.sub_batch_size / 2)

    def __str__(self):
        # print Hyper Parameters
        config_str = ''
        for attr in dir(self):
            if not attr.startswith('__') and not isinstance(getattr(self, attr), MethodType):
                config_str += '{ %-17s }->' % attr + str(getattr(self, attr)) + '\n'

        return config_str

