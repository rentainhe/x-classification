import numpy as np
import random
from datetime import datetime

# path configs
data_path = {
    'cifar10': './data/cifar10',
    'cifar100': './data/cifar100',
    'imagenet': './data/imagenet'
}
result_log_dir = 'log'
ckpts_dir = 'ckpts'
tensorboard_log_dir = 'runs'

# training configs
epoch = 200
milestones = [60, 120, 160] # 30%, 60%, 80%
save_epoch = 10
gpu = '0'
seed = random.randint(0, 9999999)
version = str(seed)
dataset = 'cifar100'
batch_size = 64

# data configs
mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
num_workers = 8
pin_memory = True
eval_every_epoch = True

# create time
time_now = datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')
