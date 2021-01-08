import torch
from torchstat import stat
from models.get_network import get_network
import argparse


def count_parameters(net):
    params = sum([param.nelement() for param in net.parameters() if param.requires_grad])
    print("Params: %f M" % (params / 1000000))


def parse_args():
    parser = argparse.ArgumentParser(description='MAC_Classification Args')
    parser.add_argument('--model', type=str, required=True, help='choose a network')
    parser.add_argument('--gpu', type=int, default=0, help="choose a gpu for testing")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_args()

    if device == 'cpu':
        net = get_network(args)
        count_parameters(net)
        stat(net, (3,224,224))
    else:
        with torch.cuda.device(args.gpu):
            net = get_network(args)
            stat(net, (3,224,224))
