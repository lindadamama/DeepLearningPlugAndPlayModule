import torch
import torch.nn as nn


class BBasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(BBasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)


class DenseLayer(nn.Module):
    def __init__(self, in_C, out_C, down_factor=4, k=4):
        """
        更像是DenseNet的Block，从而构造特征内的密集连接
        """
        super(DenseLayer, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(BBasicConv2d(mid_C * i, mid_C, 3, 1, 1))

        self.fuse = BBasicConv2d(in_C + mid_C, out_C, kernel_size=3, stride=1, padding=1)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        # print(down_feats.shape)
        # print(self.denseblock)
        out_feats = []
        for i in self.denseblock:
            # print(self.denseblock)
            feats = i(torch.cat((*out_feats, down_feats), dim=1))
            # print(feats.shape)
            out_feats.append(feats)

        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)


class GEFM(nn.Module):
    def __init__(self, in_C, out_C):
        super(GEFM, self).__init__()
        self.RGB_K = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.RGB_V = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.Q = BBasicConv2d(in_C, out_C, 3, 1, 1)
        self.INF_K = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.INF_V = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.Second_reduce = BBasicConv2d(in_C, out_C, 3, 1, 1)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        Q = self.Q(torch.cat([x, y], dim=1))
        RGB_K = self.RGB_K(x)
        RGB_V = self.RGB_V(x)
        m_batchsize, C, height, width = RGB_V.size()
        RGB_V = RGB_V.view(m_batchsize, -1, width * height)
        RGB_K = RGB_K.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        RGB_Q = Q.view(m_batchsize, -1, width * height)
        RGB_mask = torch.bmm(RGB_K, RGB_Q)
        RGB_mask = self.softmax(RGB_mask)
        RGB_refine = torch.bmm(RGB_V, RGB_mask.permute(0, 2, 1))
        RGB_refine = RGB_refine.view(m_batchsize, -1, height, width)
        RGB_refine = self.gamma1 * RGB_refine + y

        INF_K = self.INF_K(y)
        INF_V = self.INF_V(y)
        INF_V = INF_V.view(m_batchsize, -1, width * height)
        INF_K = INF_K.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        INF_Q = Q.view(m_batchsize, -1, width * height)
        INF_mask = torch.bmm(INF_K, INF_Q)
        INF_mask = self.softmax(INF_mask)
        INF_refine = torch.bmm(INF_V, INF_mask.permute(0, 2, 1))
        INF_refine = INF_refine.view(m_batchsize, -1, height, width)
        INF_refine = self.gamma2 * INF_refine + x

        out = self.Second_reduce(torch.cat([RGB_refine, INF_refine], dim=1))
        return out


class PSFM(nn.Module):
    def __init__(self, in_C, out_C, cat_C):
        super(PSFM, self).__init__()
        self.RGBobj = DenseLayer(in_C, out_C)
        self.Infobj = DenseLayer(in_C, out_C)
        self.obj_fuse = GEFM(cat_C, out_C)

    def forward(self, rgb, depth):
        rgb_sum = self.RGBobj(rgb)
        Inf_sum = self.Infobj(depth)
        out = self.obj_fuse(rgb_sum, Inf_sum)
        return out