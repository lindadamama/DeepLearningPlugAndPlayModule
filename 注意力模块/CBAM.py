import torch
import torch.nn as nn


# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化

        # 使用1x1卷积代替全连接层，减少参数量，ratio用于降维
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()  # 激活函数

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))  # 全局平均池化分支
        max_out = self.fc(self.max_pool(x))  # 全局最大池化分支
        out = avg_out + max_out  # 融合两个池化分支
        return self.sigmoid(out)  # 返回通道权重


# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 计算特征图的平均值
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 计算特征图的最大值
        x = torch.cat([avg_out, max_out], dim=1)  # 沿通道维度拼接
        x = self.conv1(x)  # 通过卷积层
        return self.sigmoid(x)  # 返回空间权重


# CBAM Module
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)  # 通道注意力模块
        self.spatial_attention = SpatialAttention(kernel_size)  # 空间注意力模块

    def forward(self, x):
        out = self.channel_attention(x) * x  # 先应用通道注意力
        out = self.spatial_attention(out) * out  # 再应用空间注意力
        return out  # 返回结果