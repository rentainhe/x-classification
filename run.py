import argparse, yaml
from conf.base_configs import Base_Configs

def parse_args():
    parser = argparse.ArgumentParser(description='MAC_Classification Args')
    parser.add_argument('--dataset', type=str, choices=['cifar10','cifar100','imagenet'], required=True, help='choose a dataset to learn')
    parser.add_argument('--model', type=str, required=True, help='choose a network')
    parser.add_argument('--gpu', type=str, help="gpu choose, eg. '0,1,2,...' ")
    parser.add_argument('--seed', type=int, help='fix random seed')
    parser.add_argument('--eval_every_epoch', choices=['True','False'], type=str)
    # parser.add_argument('--b', type=int, default=128, help='batch size for dataloader')
    # parser.add_argument('--warm', type=int, default=1, help='warm up training phase')
    # parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # load model config
    cfg_file = "conf/{}/{}.yml".format(args.dataset, args.model)
    with open(cfg_file, 'r') as f:
        yaml_dict = yaml.load(f)

    # load basic config
    config = Base_Configs()
    args = config.str_to_bool(args)
    for arg in args:
        print(arg)
    args_dict = config.parse_to_dict(args)

    config.add_args(args_dict)
    config.process()
    # print(config)
