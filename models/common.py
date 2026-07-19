"""
共享模块：卷积块、上采样等基础组件。
UNet 和 FCN 共用，避免重复代码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _NearestResizeExport(torch.autograd.Function):
    """仅在 ONNX 导出时生成带显式全范围 ROI 的最近邻 Resize。"""

    @staticmethod
    def forward(ctx: object, x: torch.Tensor) -> torch.Tensor:
        """保持与训练阶段相同的 2 倍最近邻上采样数值。"""
        del ctx
        return F.interpolate(x, scale_factor=2.0, mode="nearest")

    @staticmethod
    def symbolic(graph: object, x: object) -> object:
        """导出非空 ROI，避开 ESP-DL 对空可选输入的解析崩溃。"""
        roi = graph.op("Constant", value_t=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32))
        scales = graph.op("Constant", value_t=torch.tensor([1, 1, 2, 2], dtype=torch.float32))
        return graph.op(
            "Resize",
            x,
            roi,
            scales,
            coordinate_transformation_mode_s="asymmetric",
            mode_s="nearest",
            nearest_mode_s="floor",
        )


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
    """上采样：指定插值模式放大 2 倍，再用 1x1 卷积调整通道。"""

    def __init__(self, in_channels: int, out_channels: int, mode: str = "bilinear"):
        """
        参数：
            in_channels: 输入通道数（通常来自跳跃连接拼接后的2倍通道）
            out_channels: 输出通道数
            mode: 上采样模式；旧模型使用 bilinear，部署兼容模型使用 nearest
        """
        super().__init__()
        if mode not in {"bilinear", "nearest"}:
            raise ValueError(f"不支持的上采样模式：{mode}")
        self.mode = mode
        self.up = nn.Upsample(
            scale_factor=2,
            mode=mode,
            align_corners=True if mode == "bilinear" else None,
        )
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """先上采样，再用1x1卷积降通道"""
        if self.mode == "nearest" and torch.onnx.is_in_onnx_export():
            x = _NearestResizeExport.apply(x)
        else:
            x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
