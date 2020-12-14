import torch
import torch.nn as nn
from plug_and_play.ResNet.BasicBlock import BasicResidualBlock
from plug_and_play.ResNet.BottleNeck import BottleNeck

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=100):