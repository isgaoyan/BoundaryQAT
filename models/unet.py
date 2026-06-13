"""
轻量 UNet 模型：用于瞳孔语义分割。
编码器-解码器结构，4层深度，约 454K 参数。
输入：单通道 IR 眼图 (1×64×64)
输出：瞳孔二值 mask (1×64×64)
"""

import torch
import torch.nn as nn
from models.common import ConvBlock, DownSample, UpSample


class LightUNet(nn.Module):
    """
    轻量 UNet：专为 MCU 部署（INT8 量化）设计。

    架构：
        编码器: 4层 [8, 16, 32, 64] 通道
        瓶颈:   128 通道
        解码器: 4层 [64, 32, 16, 8] 通道 + 跳跃连接
        输出头: 1×1 卷积 + Sigmoid
    参数量: ~454K（INT8 量化后 ~454KB）
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        """
        参数：
            in_channels:  输入图像通道数（IR 眼图为 1）
            out_channels: 输出 mask 通道数（二值分割为 1）
        """
        super().__init__()

        # ── 编码器（Encoder）────────────────────────────
        # 每层: ConvBlock 提取特征 → DownSample 空间减半

        self.enc1 = ConvBlock(in_channels, 8)      # [B, 1, 64, 64] → [B, 8, 64, 64]
        self.down1 = DownSample()                    # [B, 8, 64, 64] → [B, 8, 32, 32]

        self.enc2 = ConvBlock(8, 16)                 # [B, 8, 32, 32] → [B, 16, 32, 32]
        self.down2 = DownSample()                    # [B, 16, 32, 32] → [B, 16, 16, 16]

        self.enc3 = ConvBlock(16, 32)                # [B, 16, 16, 16] → [B, 32, 16, 16]
        self.down3 = DownSample()                    # [B, 32, 16, 16] → [B, 32, 8, 8]

        self.enc4 = ConvBlock(32, 64)                # [B, 32, 8, 8] → [B, 64, 8, 8]
        self.down4 = DownSample()                    # [B, 64, 8, 8] → [B, 64, 4, 4]

        # ── 瓶颈（Bottleneck）───────────────────────────
        # 最深层，不改变空间尺寸，仅扩增通道

        self.bottleneck = ConvBlock(64, 128)         # [B, 64, 4, 4] → [B, 128, 4, 4]

        # ── 解码器（Decoder）────────────────────────────
        # 每层: UpSample 上采样 → 拼接跳跃连接 → ConvBlock 融合

        self.up4 = UpSample(128, 64)                  # [B, 128, 4, 4] → [B, 64, 8, 8]
        self.dec4 = ConvBlock(128, 64)                # 拼接后: [B, 128, 8, 8] → [B, 64, 8, 8]
        #   ↑ 输入 128 通道 = 上采样输出 64 + 跳跃连接 enc4 的 64

        self.up3 = UpSample(64, 32)                   # [B, 64, 8, 8] → [B, 32, 16, 16]
        self.dec3 = ConvBlock(64, 32)                 # [B, 64, 16, 16] → [B, 32, 16, 16]

        self.up2 = UpSample(32, 16)                   # [B, 32, 16, 16] → [B, 16, 32, 32]
        self.dec2 = ConvBlock(32, 16)                 # [B, 32, 32, 32] → [B, 16, 32, 32]

        self.up1 = UpSample(16, 8)                    # [B, 16, 32, 32] → [B, 8, 64, 64]
        self.dec1 = ConvBlock(16, 8)                  # [B, 16, 64, 64] → [B, 8, 64, 64]

        # ── 输出头（Output Head）────────────────────────
        # 1×1 卷积降为单通道 + Sigmoid 输出 [0,1] 概率

        self.output_conv = nn.Conv2d(8, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数：
            x: 输入张量 [B, 1, 64, 64]

        返回：
            分割 mask [B, 1, 64, 64]，每个像素为瞳孔概率 [0, 1]
        """

        # ── 编码路径 ──
        # 每层保存下采样前的特征图作为跳跃连接

        e1 = self.enc1(x)           # [B, 8, 64, 64]
        x = self.down1(e1)          # [B, 8, 32, 32]

        e2 = self.enc2(x)           # [B, 16, 32, 32]
        x = self.down2(e2)          # [B, 16, 16, 16]

        e3 = self.enc3(x)           # [B, 32, 16, 16]
        x = self.down3(e3)          # [B, 32, 8, 8]

        e4 = self.enc4(x)           # [B, 64, 8, 8]
        x = self.down4(e4)          # [B, 64, 4, 4]

        # ── 瓶颈 ──

        x = self.bottleneck(x)      # [B, 128, 4, 4]

        # ── 解码路径 ──
        # 每层: 上采样 → 拼接对应编码层 → ConvBlock 融合

        x = self.up4(x)             # [B, 64, 8, 8]
        x = torch.cat([x, e4], dim=1)  # 拼接跳跃连接 → [B, 128, 8, 8]
        x = self.dec4(x)            # [B, 64, 8, 8]

        x = self.up3(x)             # [B, 32, 16, 16]
        x = torch.cat([x, e3], dim=1)  # [B, 64, 16, 16]
        x = self.dec3(x)            # [B, 32, 16, 16]

        x = self.up2(x)             # [B, 16, 32, 32]
        x = torch.cat([x, e2], dim=1)  # [B, 32, 32, 32]
        x = self.dec2(x)            # [B, 16, 32, 32]

        x = self.up1(x)             # [B, 8, 64, 64]
        x = torch.cat([x, e1], dim=1)  # [B, 16, 64, 64]
        x = self.dec1(x)            # [B, 8, 64, 64]

        # ── 输出头 ──

        x = self.output_conv(x)     # [B, 1, 64, 64]
        x = self.sigmoid(x)         # [0, 1] 概率

        return x

    def count_parameters(self) -> int:
        """统计模型可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── 快速测试入口 ──
if __name__ == "__main__":
    model = LightUNet(in_channels=1, out_channels=1)
    param_count = model.count_parameters()
    print(f"✓ LightUNet 创建成功")
    print(f"  参数量: {param_count:,}")
    print(f"  预期:   ~454,000")

    # 模拟单张 IR 眼图输入
    dummy_input = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"  输入:  {dummy_input.shape}")
    print(f"  输出:  {output.shape}")
    print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")