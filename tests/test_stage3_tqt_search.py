"""阶段三 TQT 配置构造测试。"""

from quantization.stage3_tqt_search import create_tqt_setting


def test_tqt_setting_preserves_power_of_two_training_configuration() -> None:
    """验证 TQT 设置启用并使用固定训练超参数。"""
    config = {
        "activation_algorithm": "percentile",
        "tqt_learning_rate": 1e-5,
        "tqt_block_size": 4,
        "tqt_int_lambda": 0.01,
        "device": "cuda",
    }

    setting = create_tqt_setting(config, steps=25)

    assert setting.tqt_optimization is True
    assert setting.quantize_activation_setting.calib_algorithm == "percentile"
    assert setting.tqt_optimization_setting.steps == 25
    assert setting.tqt_optimization_setting.int_lambda == 0.01
