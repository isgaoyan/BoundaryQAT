"""
模型单元测试：验证 ConvBlock、DownSample、UpSample、LightUNet
的前向传播 shape 正确性、输出范围和参数量。
"""

import pytest
import torch
from models.common import ConvBlock, DownSample, UpSample
from models.unet import LightUNet


# ═══════════════════════════════════════════════════════════════
# ConvBlock 测试
# ═══════════════════════════════════════════════════════════════

class TestConvBlock:
    """ConvBlock 的形状转换和输出行为测试"""

    @pytest.mark.parametrize("in_ch, out_ch, h, w", [
        (1, 8, 64, 64),      # 典型输入：单通道 64×64
        (3, 16, 128, 128),   # RGB 输入
        (8, 8, 32, 32),      # 等通道转换
        (16, 8, 16, 16),     # 通道压缩
    ])
    def test_output_shape(self, in_ch, out_ch, h, w):
        """验证输出尺寸与输入一致（padding=1 保持空间不变）"""
        block = ConvBlock(in_ch, out_ch)
        x = torch.randn(2, in_ch, h, w)
        y = block(x)
        assert y.shape == (2, out_ch, h, w)

    def test_output_not_nan(self):
        """验证输出不含 NaN 或 Inf"""
        block = ConvBlock(3, 16)
        x = torch.randn(4, 3, 64, 64)
        y = block(x)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_train_eval_modes(self):
        """验证 train() 和 eval() 模式不报错"""
        block = ConvBlock(3, 16)
        x = torch.randn(2, 3, 64, 64)
        block.train()
        y_train = block(x)
        block.eval()
        y_eval = block(x)
        # 两种模式应该产生不同结果（BN 行为不同）
        assert not torch.allclose(y_train, y_eval)


# ═══════════════════════════════════════════════════════════════
# DownSample 测试
# ═══════════════════════════════════════════════════════════════

class TestDownSample:
    """DownSample 的形状变换测试"""

    @pytest.mark.parametrize("ch, h, w", [
        (8, 64, 64),
        (16, 128, 128),
        (3, 32, 32),
    ])
    def test_halves_spatial_dims(self, ch, h, w):
        """验证空间尺寸减半，通道数不变"""
        down = DownSample()
        x = torch.randn(2, ch, h, w)
        y = down(x)
        assert y.shape == (2, ch, h // 2, w // 2)


# ═══════════════════════════════════════════════════════════════
# UpSample 测试
# ═══════════════════════════════════════════════════════════════

class TestUpSample:
    """UpSample 的形状变换测试"""

    @pytest.mark.parametrize("in_ch, out_ch, h, w", [
        (128, 64, 4, 4),     # 典型：瓶颈→解码器第一层
        (64, 32, 8, 8),
        (16, 8, 32, 32),
    ])
    def test_doubles_spatial_dims(self, in_ch, out_ch, h, w):
        """验证空间尺寸翻倍，通道数变为目标值"""
        up = UpSample(in_ch, out_ch)
        x = torch.randn(2, in_ch, h, w)
        y = up(x)
        assert y.shape == (2, out_ch, h * 2, w * 2)


# ═══════════════════════════════════════════════════════════════
# LightUNet 测试
# ═══════════════════════════════════════════════════════════════

class TestLightUNet:
    """LightUNet 端到端测试"""

    def test_default_input_output_shape(self):
        """验证默认输入 (1×64×64) 的输出形状"""
        model = LightUNet()
        x = torch.randn(4, 1, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (4, 1, 64, 64)

    def test_batch_size_1(self):
        """验证单样本推理不报错（BN 在 batch=1 时需注意）"""
        model = LightUNet()
        model.eval()  # eval 模式下 BN 用移动平均，batch=1 也能跑
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 1, 64, 64)

    def test_output_range(self):
        """验证输出在 [0, 1] 范围内（Sigmoid 保证）"""
        model = LightUNet()
        model.eval()
        x = torch.randn(2, 1, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.min() >= 0.0
        assert y.max() <= 1.0

    def test_parameter_count(self):
        """验证参数量在设计预期范围内"""
        model = LightUNet()
        count = model.count_parameters()
        # 允许 ±10% 误差（架构微调时不用改测试）
        assert 400_000 <= count <= 510_000, f"参数量 {count:,} 超出预期范围"

    def test_gradient_flow(self):
        """验证梯度能正常反向传播（训练可行性检查）"""
        model = LightUNet()
        model.train()
        x = torch.randn(2, 1, 64, 64, requires_grad=False)
        y = model(x)
        loss = y.mean()
        loss.backward()
        # 检查第一层和最后一层都有梯度
        assert model.enc1.conv1.weight.grad is not None
        assert model.output_conv.weight.grad is not None

    def test_deterministic_inference(self):
        """验证 eval 模式下相同输入产生相同输出（确定性）"""
        model = LightUNet()
        model.eval()
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y1 = model(x)
            y2 = model(x)
        assert torch.allclose(y1, y2)

    def test_different_input_sizes(self):
        """验证 16×16、32×32 都能正确处理（池化次数够就行）"""
        model = LightUNet()
        model.eval()
        for size in [16, 32, 64]:
            x = torch.randn(2, 1, size, size)
            with torch.no_grad():
                y = model(x)
            # 4 次下采样，输出 = 输入 / 16
            expected_size = size
            assert y.shape == (2, 1, expected_size, expected_size), \
                f"输入 {size}×{size} 失败: 输出 {y.shape}"
