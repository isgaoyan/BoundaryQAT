"""阶段一 BCE+Dice 损失、后处理与主评价指标测试。"""

import math

import numpy as np
import pytest
import torch

from evaluation.segmentation_metrics import (
    binary_dice,
    boundary_iou,
    center_error,
    evaluate_batch,
    probability_to_largest_component,
)
from training.losses import BCEDiceLoss, soft_dice_score


def test_bce_dice_loss_is_small_for_nearly_perfect_prediction() -> None:
    """验证接近真值的概率图具有较小损失和接近 1 的 Soft Dice。"""
    target = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
    probability = torch.tensor([[[[0.01, 0.99], [0.01, 0.99]]]], requires_grad=True)

    loss = BCEDiceLoss()(probability, target)
    loss.backward()

    assert loss.item() < 0.05
    assert soft_dice_score(probability.detach(), target).item() > 0.98
    assert probability.grad is not None


def test_largest_component_removes_smaller_false_positive() -> None:
    """验证后处理仅保留面积最大的八连通预测区域。"""
    probability = np.zeros((8, 8), dtype=np.float32)
    probability[1, 1] = 0.9
    probability[4:7, 4:7] = 0.9

    prediction = probability_to_largest_component(probability, threshold=0.5)

    assert prediction.sum() == 9
    assert not prediction[1, 1]
    assert prediction[5, 5]


def test_identical_masks_have_perfect_overlap_metrics() -> None:
    """验证相同掩码的 Dice 与 Boundary IoU 均为 1。"""
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 5:11] = True

    assert binary_dice(mask, mask) == pytest.approx(1.0)
    assert boundary_iou(mask, mask, width=2) == pytest.approx(1.0)


def test_center_error_uses_euclidean_distance_and_empty_penalty() -> None:
    """验证中心误差使用欧氏距离，空预测按图像对角线计罚。"""
    target = np.zeros((10, 10), dtype=bool)
    prediction = np.zeros((10, 10), dtype=bool)
    target[2, 2] = True
    prediction[5, 6] = True

    error, empty = center_error(prediction, target)
    empty_error, empty_flag = center_error(np.zeros_like(prediction), target)

    assert error == pytest.approx(5.0)
    assert not empty
    assert empty_error == pytest.approx(math.hypot(10, 10))
    assert empty_flag


def test_evaluate_batch_reports_fixed_metric_contract() -> None:
    """验证批次评价同时返回主指标、诊断指标和空预测率。"""
    targets = torch.zeros((2, 1, 16, 16), dtype=torch.float32)
    targets[:, :, 4:12, 4:12] = 1.0
    probabilities = targets.clone()
    probabilities[1].zero_()

    metrics = evaluate_batch(probabilities, targets, threshold=0.5, boundary_width=2)

    assert metrics["sample_count"] == 2.0
    assert metrics["dice"] == pytest.approx(0.5, abs=1e-6)
    assert metrics["empty_prediction_rate"] == pytest.approx(0.5)
    assert set(metrics) == {
        "dice",
        "boundary_iou",
        "boundary_dice",
        "interior_dice",
        "center_mae_pixels",
        "empty_prediction_rate",
        "sample_count",
    }
