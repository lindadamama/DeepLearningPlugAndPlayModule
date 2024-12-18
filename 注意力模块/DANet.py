import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels):
        super(PositionAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 8

        # Theta, Phi, G transformations
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)

        # Output transformation
        self.W = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # Theta and Phi feature transformations
        theta = self.theta(x).view(batch_size, self.inter_channels, -1)  # (N, C', H*W)
        phi = self.phi(x).view(batch_size, self.inter_channels, -1)  # (N, C', H*W)
        phi = phi.permute(0, 2, 1)  # Transpose to (N, H*W, C')

        # Compute attention map
        attention = torch.matmul(theta, phi)  # (N, H*W, H*W)
        attention = F.softmax(attention, dim=-1)

        # Apply attention map to g(x)
        g = self.g(x).view(batch_size, self.inter_channels, -1)  # (N, C', H*W)
        g = g.permute(0, 2, 1)  # (N, H*W, C')
        out = torch.matmul(attention, g)  # (N, H*W, C')
        out = out.permute(0, 2, 1).contiguous()  # (N, C', H*W)
        out = out.view(batch_size, self.inter_channels, H, W)  # (N, C', H, W)

        # Final output transformation and residual connection
        out = self.W(out)
        out = out + x

        return out


class ChannelAttentionModule(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_channels):
        super(ChannelAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 8

        # Feature transformations for channel attention
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)

        # Output transformation
        self.W = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # Apply transformations and compute attention
        theta = self.theta(x).view(batch_size, self.inter_channels, -1)  # (N, C', H*W)
        phi = self.phi(x).view(batch_size, self.inter_channels, -1)  # (N, C', H*W)
        theta = theta.permute(0, 2, 1)  # (N, H*W, C')

        # Compute attention map
        attention = torch.matmul(theta, phi)  # (N, H*W, H*W)
        attention = F.softmax(attention, dim=-1)

        # Apply attention map
        out = torch.matmul(phi, attention).view(batch_size, C, H, W)  # (N, C, H, W)

        # Output transformation and residual connection
        out = self.W(out)
        out = out + x

        return out


class DANet(nn.Module):
    """ DANet module """

    def __init__(self, in_channels, out_channels):
        super(DANet, self).__init__()

        # Shared convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Position and Channel attention modules
        self.position_attention = PositionAttentionModule(in_channels)
        self.channel_attention = ChannelAttentionModule(in_channels)

        # Output convolution
        self.output_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)

        # Position and Channel Attention
        pos_out = self.position_attention(x)
        chn_out = self.channel_attention(x)

        # Fusion and output
        fusion = pos_out + chn_out
        out = self.output_conv(fusion)

        return out
