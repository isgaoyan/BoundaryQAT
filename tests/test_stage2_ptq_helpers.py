"""阶段二 PTQ 辅助逻辑测试。"""

from quantization.quantize_espdl_ptq import calculate_metric_deltas, to_jsonable


def test_metric_deltas_keep_direction() -> None:
    """验证精度下降和误差上升均保留正确符号。"""
    fp32 = {
        "dice": 0.8,
        "boundary_iou": 0.4,
        "boundary_dice": 0.5,
        "interior_dice": 0.7,
        "center_mae_pixels": 2.0,
        "empty_prediction_rate": 0.01,
    }
    ptq = {
        "dice": 0.78,
        "boundary_iou": 0.35,
        "boundary_dice": 0.46,
        "interior_dice": 0.69,
        "center_mae_pixels": 2.5,
        "empty_prediction_rate": 0.02,
    }

    deltas = calculate_metric_deltas(ptq, fp32)

    assert deltas["dice"] < 0
    assert deltas["center_mae_pixels"] > 0


def test_json_conversion_handles_nested_tuples() -> None:
    """验证层级误差报告中的元组可稳定写入 JSON。"""
    assert to_jsonable({"layer": (0.1, 0.2)}) == {"layer": [0.1, 0.2]}
