import argparse
from conf import configs
import os
from utils.train_engine import train_engine

def parse_args():
    parser = argparse.ArgumentParser(description='MAC_Classification Args')
    parser.add_argument('--dataset', type=str, choices=['cifar10','cifar100','imagenet'], required=True, help='choose a dataset to learn')
    parser.add_argument('--model', type=str, required=True, help='choose a network')
    parser.add_argument('--gpu', type=str, help="gpu choose, eg. '0,1,2,...' ")
    parser.add_argument('--run', type=str, dest='run_mode',choices=['train','test'])
    parser.add_argument('--seed', type=int, help='fix random seed')
    parser.add_argument('--eval_every_epoch', choices=['True','False'], type=str)
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='warm up training phase')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--lr_decay_rate', type=int, default=0.2, help='learning rate decay')
    parser.add_argument('--resume', action='store_true', default=False, help='resume training')
    parser.add_argument('--warmup_epoch', type=int, default=1, help='warmup epochs')
    parser.add_argument('--epoch', type=int, default=200, help='total epochs')
    parser.add_argument('--ckpt_e', type=int, dest='ckpt_epoch')
    parser.add_argument('--ckpt_v', type=str, dest='ckpt_version')
    parser.add_argument('--label_smoothing', action='store_true', default=False, help='if using label smoothing')
    parser.add_argument('--smoothing', type=int, default=0.1, help='control label smoothing value')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    args_dict = configs.parse_to_dict(args)
    configs.add_args(args_dict)

    configs.path_init()
    configs.training_init()

    print("Hyper parameters:")
    print(configs)

    if configs.run_mode == 'train':
        train_engine(configs)



