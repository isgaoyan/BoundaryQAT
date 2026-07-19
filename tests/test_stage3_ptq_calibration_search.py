"""阶段三 PTQ 校准与截断策略选择测试。"""

from quantization.stage3_ptq_calibration_search import candidate_is_eligible, select_candidate


REFERENCE = {
    "dice": 0.82,
    "boundary_iou": 0.40,
    "boundary_dice": 0.53,
    "interior_dice": 0.77,
    "center_mae_pixels": 3.0,
    "empty_prediction_rate": 0.02,
}
CONSTRAINTS = {
    "max_center_mae_increase_pixels": 0.0,
    "max_empty_prediction_rate_increase": 0.0,
}


def make_candidate(name: str, boundary_iou: float, center_mae: float, empty_rate: float) -> dict[str, object]:
    """构造 PTQ 多配置选择器所需候选。"""
    return {
        "candidate": name,
        "activation_algorithm": "kl",
        "calibration_sample_count": 128,
        "threshold": 0.5,
        "dice": 0.82,
        "boundary_iou": boundary_iou,
        "boundary_dice": 0.53,
        "interior_dice": 0.77,
        "center_mae_pixels": center_mae,
        "empty_prediction_rate": empty_rate,
        "sample_count": 450.0,
    }


def test_strict_constraints_reject_any_geometry_regression() -> None:
    """验证中心误差或空预测率任一增加都会被排除。"""
    assert not candidate_is_eligible(make_candidate("center", 0.45, 3.01, 0.02), REFERENCE, CONSTRAINTS)
    assert not candidate_is_eligible(make_candidate("empty", 0.45, 3.0, 0.021), REFERENCE, CONSTRAINTS)


def test_selection_maximizes_boundary_iou_after_strict_filtering() -> None:
    """验证高边界但越界的候选不会击败合格候选。"""
    candidates = [
        make_candidate("baseline", 0.40, 3.0, 0.02),
        make_candidate("eligible", 0.42, 2.9, 0.02),
        make_candidate("rejected", 0.50, 3.1, 0.02),
    ]

    selected = select_candidate(candidates, REFERENCE, CONSTRAINTS)

    assert selected["candidate"] == "eligible"
