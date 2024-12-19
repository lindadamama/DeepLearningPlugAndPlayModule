import torch
import torch.nn as nn


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
        source: https://github.com/BangguWu/ECANet
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class MFII_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 rla_channel=32, SE=False, ECA_size=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16):
        super(MFII_BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # if groups != 1 or base_width != 64:
        # raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes + rla_channel, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.averagePooling = None
        if downsample is not None and stride != 1:
            self.averagePooling = nn.AvgPool2d((2, 2), stride=(2, 2))

        self.se = None
        if SE:
            self.se = SELayer(planes * self.expansion, reduction)

        self.eca = None
        if ECA_size != None:
            self.eca = eca_layer(planes * self.expansion, int(ECA_size))

    def forward(self, x, h):
        identity = x

        x = torch.cat((x, h), dim=1)  # [8, 96, 56, 56]

        out = self.conv1(x)  # [8, 64, 56, 56]
        out = self.bn1(out)  # [8, 64, 56, 56]
        out = self.relu(out)

        out = self.conv2(out)  # [8, 64, 56, 56]
        out = self.bn2(out)

        if self.se != None:
            out = self.se(out)

        if self.eca != None:
            out = self.eca(out)

        y = out

        if self.downsample is not None:
            identity = self.downsample(identity)
        if self.averagePooling is not None:
            h = self.averagePooling(h)

        out += identity
        out = self.relu(out)

        return out, y, h


class MFII_BasicBlock_half(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 rla_channel=16, SE=False, ECA_size=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16):
        super(MFII_BasicBlock_half, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # if groups != 1 or base_width != 64:
        # raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes + rla_channel, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.averagePooling = None
        if downsample is not None and stride != 1:
            self.averagePooling = nn.AvgPool2d((2, 2), stride=(2, 2))

        self.se = None
        if SE:
            self.se = SELayer(planes * self.expansion, reduction)

        self.eca = None
        if ECA_size != None:
            self.eca = eca_layer(planes * self.expansion, int(ECA_size))

    def forward(self, x, h):
        identity = x

        x = torch.cat((x, h), dim=1)  # [8, 96, 56, 56]

        out = self.conv1(x)  # [8, 64, 56, 56]
        out = self.bn1(out)  # [8, 64, 56, 56]
        out = self.relu(out)

        out = self.conv2(out)  # [8, 64, 56, 56]
        out = self.bn2(out)

        if self.se != None:
            out = self.se(out)

        if self.eca != None:
            out = self.eca(out)

        y = out

        if self.downsample is not None:
            identity = self.downsample(identity)
        if self.averagePooling is not None:
            h = self.averagePooling(h)

        out += identity
        out = self.relu(out)

        return out, y, h