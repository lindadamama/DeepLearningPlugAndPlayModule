import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 两个全连接层
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)

        # Sigmoid 激活函数用于生成权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        # 全局平均池化
        y = self.global_avg_pool(x).view(batch_size, channels)

        # 通过全连接层并生成注意力权重
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)

        # 将权重与输入特征相乘
        return x * y.expand_as(x)


class SENet(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SENet, self).__init__()
        self.se_block = SEBlock(in_channels, reduction)

    def forward(self, x):
        return self.se_block(x)