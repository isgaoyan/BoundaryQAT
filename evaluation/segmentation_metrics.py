"""固定口径的瞳孔分割后处理、边界质量与中心定位指标。"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy import ndimage


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """只保留八连通二值掩码中面积最大的前景区域。"""
    binary_mask = np.asarray(mask, dtype=bool)
    labels, component_count = ndimage.label(binary_mask, structure=np.ones((3, 3), dtype=np.uint8))
    if component_count == 0:
        return np.zeros_like(binary_mask)
    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    return labels == int(component_sizes.argmax())


def probability_to_largest_component(probability: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """按固定阈值二值化概率图，并只保留最大连通区域。"""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("概率阈值必须位于 [0, 1]。")
    return largest_connected_component(np.asarray(probability) >= threshold)


def binary_dice(prediction: np.ndarray, target: np.ndarray, epsilon: float = 1e-6) -> float:
    """计算两个二值掩码的 Dice，双方均为空时返回 1。"""
    prediction_bool = np.asarray(prediction, dtype=bool)
    target_bool = np.asarray(target, dtype=bool)
    intersection = np.logical_and(prediction_bool, target_bool).sum()
    denominator = prediction_bool.sum() + target_bool.sum()
    return float((2.0 * intersection + epsilon) / (denominator + epsilon))


def inner_boundary(mask: np.ndarray, width: int) -> np.ndarray:
    """提取位于前景内部、宽度固定的边界带。"""
    if width <= 0:
        raise ValueError("边界带宽必须为正整数。")
    binary_mask = np.asarray(mask, dtype=bool)
    eroded = ndimage.binary_erosion(binary_mask, iterations=width, border_value=0)
    return np.logical_and(binary_mask, np.logical_not(eroded))


def boundary_iou(prediction: np.ndarray, target: np.ndarray, width: int = 2, epsilon: float = 1e-6) -> float:
    """计算预测与真值内边界带的交并比。"""
    prediction_boundary = inner_boundary(prediction, width)
    target_boundary = inner_boundary(target, width)
    intersection = np.logical_and(prediction_boundary, target_boundary).sum()
    union = np.logical_or(prediction_boundary, target_boundary).sum()
    return float((intersection + epsilon) / (union + epsilon))


def boundary_dice(prediction: np.ndarray, target: np.ndarray, width: int = 2) -> float:
    """计算预测与真值内边界带的 Dice，作为诊断指标。"""
    return binary_dice(inner_boundary(prediction, width), inner_boundary(target, width))


def interior_dice(prediction: np.ndarray, target: np.ndarray, width: int = 2) -> float:
    """计算去除固定边界带后的内部区域 Dice，作为诊断指标。"""
    prediction_interior = ndimage.binary_erosion(np.asarray(prediction, dtype=bool), iterations=width, border_value=0)
    target_interior = ndimage.binary_erosion(np.asarray(target, dtype=bool), iterations=width, border_value=0)
    return binary_dice(prediction_interior, target_interior)


def foreground_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """以像素中心坐标计算前景质心，空掩码返回 None。"""
    rows, columns = np.nonzero(np.asarray(mask, dtype=bool))
    if len(rows) == 0:
        return None
    return float(columns.mean()), float(rows.mean())


def center_error(prediction: np.ndarray, target: np.ndarray) -> tuple[float, bool]:
    """计算中心欧氏距离；预测为空时按图像对角线计罚并标记失败。"""
    prediction_center = foreground_centroid(prediction)
    target_center = foreground_centroid(target)
    if target_center is None:
        raise ValueError("主评估真值不得为空。")
    if prediction_center is None:
        height, width = np.asarray(target).shape
        return math.hypot(width, height), True
    error = math.hypot(prediction_center[0] - target_center[0], prediction_center[1] - target_center[1])
    return error, False


def evaluate_sample(
    probability: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    boundary_width: int = 2,
) -> tuple[dict[str, float], np.ndarray]:
    """评价单张概率图，并同时返回固定后处理后的预测掩码。"""
    target_bool = np.asarray(target) >= 0.5
    prediction = probability_to_largest_component(probability, threshold)
    error, prediction_empty = center_error(prediction, target_bool)
    metrics = {
        "dice": binary_dice(prediction, target_bool),
        "boundary_iou": boundary_iou(prediction, target_bool, boundary_width),
        "boundary_dice": boundary_dice(prediction, target_bool, boundary_width),
        "interior_dice": interior_dice(prediction, target_bool, boundary_width),
        "center_mae_pixels": error,
        "empty_prediction_rate": float(prediction_empty),
    }
    return metrics, prediction


def evaluate_batch(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    boundary_width: int = 2,
) -> dict[str, float]:
    """对一批概率图执行固定后处理，并返回逐样本平均指标。"""
    if probabilities.shape != targets.shape or probabilities.ndim != 4 or probabilities.shape[1] != 1:
        raise ValueError("指标输入必须是形状一致的 [B, 1, H, W] 张量。")

    probability_arrays = probabilities.detach().cpu().numpy()[:, 0]
    target_arrays = targets.detach().cpu().numpy()[:, 0] >= 0.5
    dice_values: list[float] = []
    boundary_iou_values: list[float] = []
    boundary_dice_values: list[float] = []
    interior_dice_values: list[float] = []
    center_errors: list[float] = []
    empty_predictions = 0

    for probability, target in zip(probability_arrays, target_arrays, strict=True):
        sample_metrics, _ = evaluate_sample(probability, target, threshold, boundary_width)
        dice_values.append(sample_metrics["dice"])
        boundary_iou_values.append(sample_metrics["boundary_iou"])
        boundary_dice_values.append(sample_metrics["boundary_dice"])
        interior_dice_values.append(sample_metrics["interior_dice"])
        center_errors.append(sample_metrics["center_mae_pixels"])
        empty_predictions += int(sample_metrics["empty_prediction_rate"])

    sample_count = len(dice_values)
    return {
        "dice": float(np.mean(dice_values)),
        "boundary_iou": float(np.mean(boundary_iou_values)),
        "boundary_dice": float(np.mean(boundary_dice_values)),
        "interior_dice": float(np.mean(interior_dice_values)),
        "center_mae_pixels": float(np.mean(center_errors)),
        "empty_prediction_rate": empty_predictions / sample_count,
        "sample_count": float(sample_count),
    }
