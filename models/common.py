"""
共享模块：卷积块、上采样等基础组件。
UNet 和 FCN 共用，避免重复代码
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """双层卷积块：Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""

    def __init__(self, in_channels: int, out_channels: int):
        """
        参数：
            in_channels: 输入通道数
            out_channels: 输出通道数
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class DownSample(nn.Module):
    """下采样：MaxPool 2x2"""

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)

class UpSample(nn.Module):
    """上采样：双线性插值 2x + 1x1卷积调整通道"""

    def __init__(self, in_channels: int, out_channels: int):
        """
        参数：
            in_channels: 输入通道数（通常来自跳跃连接拼接后的2倍通道）
            out_channels: 输出通道数
        """
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """先上采样，再用1x1卷积降通道"""
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x