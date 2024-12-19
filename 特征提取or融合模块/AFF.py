import torch
import torch.nn as nn


class AFF(nn.Module):
    def __init__(self, channels=64, reduction=4):
        """
        Attentional Feature Fusion (AFF) module.
        :param channels: Number of input channels.
        :param reduction: Reduction ratio for intermediate channels in attention layers. Default is 4.
        """
        super(AFF, self).__init__()
        intermediate_channels = channels // reduction

        # Local attention layer
        self.local_attention = nn.Sequential(
            nn.Conv2d(channels, intermediate_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # Global attention layer
        self.global_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, intermediate_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # Activation function for attention weight
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_feature, residual_feature):
        """
        Forward pass for the AFF module.
        :param input_feature: First input feature map (tensor of shape N x C x H x W).
        :param residual_feature: Second input feature map (tensor of shape N x C x H x W).
        :return: Output feature map with fused attention applied (same shape as input).
        """
        # Initial fusion by element-wise addition
        combined_feature = input_feature + residual_feature

        # Compute local and global attention
        local_attention = self.local_attention(combined_feature)
        global_attention = self.global_attention(combined_feature)

        # Sum local and global attention, then apply sigmoid for attention weight
        attention_weight = self.sigmoid(local_attention + global_attention)

        # Weighted combination of input and residual features
        output_feature = 2 * input_feature * attention_weight + 2 * residual_feature * (1 - attention_weight)

        return output_feature


"""
使用示例

# 假设输入特征图 input_feature 和 residual_feature 的通道数为 64，尺寸为 32x32
input_feature = torch.randn(1, 64, 32, 32)  # (N, C, H, W)
residual_feature = torch.randn(1, 64, 32, 32)

# 初始化 AFF 模块，指定输入通道数为 64，reduction 比例为 4
aff_module = AFF(channels=64, reduction=4)

# 计算 AFF 模块的输出
output_feature = aff_module(input_feature, residual_feature)

# 输出特征图的形状，应该与输入特征图一致
print(output_feature.shape)  # 输出: torch.Size([1, 64, 32, 32])

"""