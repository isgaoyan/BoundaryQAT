"""瞳孔二值分割 FP32 基线使用的训练损失。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def soft_dice_score(probabilities: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """按样本计算可微分 Soft Dice，并返回批次均值。"""
    if probabilities.shape != targets.shape:
        raise ValueError(f"概率图与真值形状不一致：{probabilities.shape} != {targets.shape}")
    probabilities_flat = probabilities.flatten(start_dim=1)
    targets_flat = targets.flatten(start_dim=1)
    intersection = (probabilities_flat * targets_flat).sum(dim=1)
    denominator = probabilities_flat.sum(dim=1) + targets_flat.sum(dim=1)
    scores = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return scores.mean()


class BCEDiceLoss(nn.Module):
    """将二元交叉熵与 Soft Dice 损失等权相加。"""

    def forward(self, probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算可直接反向传播的 BCE + Dice 标量损失。"""
        if probabilities.shape != targets.shape:
            raise ValueError(f"概率图与真值形状不一致：{probabilities.shape} != {targets.shape}")
        bce = F.binary_cross_entropy(probabilities, targets)
        dice_loss = 1.0 - soft_dice_score(probabilities, targets)
        return bce + dice_loss
