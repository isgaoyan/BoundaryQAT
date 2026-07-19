"""阶段三边界指标驱动阈值选择测试。"""

from quantization.stage3_boundary_threshold import generate_thresholds, select_candidate


CONSTRAINTS = {
    "max_dice_drop": 0.005,
    "max_center_mae_increase_pixels": 0.5,
    "max_empty_prediction_rate_increase": 0.01,
}


def make_candidate(
    threshold: float,
    dice: float,
    boundary_iou: float,
    center_mae: float,
    empty_rate: float,
) -> dict[str, float]:
    """构造选择器所需的最小候选指标。"""
    return {
        "threshold": threshold,
        "dice": dice,
        "boundary_iou": boundary_iou,
        "center_mae_pixels": center_mae,
        "empty_prediction_rate": empty_rate,
    }


def test_threshold_generation_includes_fixed_standard() -> None:
    """验证搜索区间包含标准阈值且端点稳定。"""
    thresholds = generate_thresholds(0.3, 0.7, 0.01)

    assert thresholds[0] == 0.3
    assert thresholds[-1] == 0.7
    assert 0.5 in thresholds
    assert len(thresholds) == 41


def test_selector_maximizes_boundary_iou_within_constraints() -> None:
    """验证选择器排除越界候选并选择合格的最高边界指标。"""
    candidates = [
        make_candidate(0.5, 0.82, 0.40, 3.0, 0.02),
        make_candidate(0.48, 0.818, 0.43, 3.2, 0.025),
        make_candidate(0.46, 0.80, 0.50, 3.1, 0.02),
    ]

    selected = select_candidate(candidates, 0.5, CONSTRAINTS)

    assert selected["threshold"] == 0.48


def test_selector_keeps_standard_without_boundary_gain() -> None:
    """验证没有真实边界收益时不会为了变化而改变阈值。"""
    candidates = [
        make_candidate(0.5, 0.82, 0.40, 3.0, 0.02),
        make_candidate(0.49, 0.821, 0.39, 2.9, 0.02),
    ]

    selected = select_candidate(candidates, 0.5, CONSTRAINTS)

    assert selected["threshold"] == 0.5
