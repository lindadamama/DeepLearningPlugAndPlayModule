import torch
import torch.nn as nn
import torch.nn.functional as F

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=False, bn_layer=True):
        """
        :param in_channels: Number of channels in the input feature map.
        :param inter_channels: Number of channels in the intermediate feature map.
                               If None, it will be set to in_channels // 2.
        :param dimension: The spatial dimensions of the input.
                          2 for 2D (image), 3 for 3D (video).
        :param sub_sample: Whether to apply subsampling to reduce computation.
        :param bn_layer: Whether to add BatchNorm layer after the Non-Local operation.
        """
        super(NonLocalBlock, self).__init__()

        assert dimension in [1, 2, 3], "Only 1D, 2D, and 3D inputs are supported."
        self.dimension = dimension
        self.sub_sample = sub_sample

        # Determine the dimension for 1D, 2D, or 3D
        if dimension == 3:
            self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        elif dimension == 2:
            self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        else:
            self.pool = nn.MaxPool1d(kernel_size=2)

        # Define the intermediate channels size
        if inter_channels is None:
            inter_channels = in_channels // 2
            if inter_channels == 0:
                inter_channels = 1

        self.g = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=1, stride=1, padding=0)

        # theta and phi will be used to compute similarity (dot product)
        self.theta = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=1, stride=1, padding=0)

        # W will transform the output of NonLocal block back to the original dimension
        self.W = nn.Conv2d(in_channels=inter_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)

        # Optionally, include a BatchNorm layer after W
        if bn_layer:
            self.W = nn.Sequential(
                self.W,
                nn.BatchNorm2d(in_channels)
            )

        # Optional subsampling
        if sub_sample:
            self.g = nn.Sequential(self.g, self.pool)
            self.phi = nn.Sequential(self.phi, self.pool)

    def forward(self, x):
        """
        :param x: Input feature map of shape (N, C, H, W) where
                  N is the batch size,
                  C is the number of channels,
                  H is the height,
                  W is the width.
        :return: Output feature map of the same shape as input.
        """
        batch_size, C, H, W = x.size()

        # Apply transformations: theta, phi, and g
        g_x = self.g(x).view(batch_size, -1, H * W)  # g(x) shape: (N, C', H*W)
        g_x = g_x.permute(0, 2, 1)  # g(x) shape: (N, H*W, C')

        theta_x = self.theta(x).view(batch_size, -1, H * W)  # theta(x) shape: (N, C', H*W)
        phi_x = self.phi(x).view(batch_size, -1, H * W)  # phi(x) shape: (N, C', H*W)
        phi_x = phi_x.permute(0, 2, 1)  # phi(x) shape: (N, H*W, C')

        # Compute similarity: theta_x * phi_x^T (matrix multiplication)
        f = torch.matmul(theta_x, phi_x)  # shape: (N, C', C')
        f_div_C = F.softmax(f, dim=-1)  # Apply softmax to normalize similarity

        # Apply attention map to g(x)
        y = torch.matmul(f_div_C, g_x)  # shape: (N, C', H*W)
        y = y.permute(0, 2, 1).contiguous()  # Reshape: (N, H*W, C')
        y = y.view(batch_size, C // 2, H, W)  # Reshape: (N, C', H, W)

        # Transform the output back to original input dimension with W
        W_y = self.W(y)

        # Residual connection: adding input x to the output
        z = W_y + x

        return z


"""
Example of use, inserted into the middle layer of ResNet

from torchvision.models import resnet50

class ResNetWithNonLocal(nn.Module):
    def __init__(self):
        super(ResNetWithNonLocal, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.nonlocal_block = NonLocalBlock(in_channels=256)

    def forward(self, x):
        x = self.resnet.layer1(x)
        x = self.nonlocal_block(x)
        x = self.resnet.layer2(x)
        return x
"""