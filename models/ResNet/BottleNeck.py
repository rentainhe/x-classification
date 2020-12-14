import torch
import torch.nn as nn

class BottleNeck(nn.Module):
    # Residual block for resnet over 50 layers
    # 瓶颈层，因为用了 1*1 的卷积，比较方便改变网络的大小，先降维后升维，将高频噪声消除，并且减少计算量，在深层网络上常用
    expansion=4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion)
        )
        self.shortcut = nn.Sequential()

        # 如果 shortcut 的维度和 residual function 的维度不相等的话，需要用 1*1 的卷积改变一下维度大小
        if stride!=1 or in_channels!=out_channels*BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

residual_block = BottleNeck(64,256,1)
x = torch.randn(1,64,32,32)
print(residual_block(x).size()) # 1,1024,32,32 维度加深4倍