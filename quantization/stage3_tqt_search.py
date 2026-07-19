"""搜索 ESP-PPQ TQT 强度与统一后处理阈值，并评价锁定候选。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from quantization.quantize_espdl_ptq import resolve_project_path, write_json
from quantization.stage3_boundary_threshold import (
    METRIC_NAMES,
    build_quantized_graph,
    collect_prediction_cache,
    compare_graph_structure,
    evaluate_cached_predictions,
    generate_thresholds,
)
from quantization.stage3_ptq_calibration_search import (
    candidate_is_eligible,
    load_reference_metrics,
    select_candidate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage3_tqt_search.json"


def parse_args() -> argparse.Namespace:
    """读取 TQT 配置和严格分离的验证搜索或最终测试模式。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("search", "test"), help="search 只读验证集；test 只评价锁定候选。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="TQT 搜索配置。")
    return parser.parse_args()


def create_tqt_setting(config: dict[str, Any], steps: int) -> Any:
    """创建满足 ESP32-S3 二的幂约束的 ESP-PPQ TQT 设置。"""
    from esp_ppq.api import QuantizationSettingFactory

    setting = QuantizationSettingFactory.espdl_setting()
    setting.quantize_activation_setting.calib_algorithm = str(config["activation_algorithm"])
    setting.tqt_optimization = True
    setting.tqt_optimization_setting.steps = steps
    setting.tqt_optimization_setting.lr = float(config["tqt_learning_rate"])
    setting.tqt_optimization_setting.block_size = int(config["tqt_block_size"])
    setting.tqt_optimization_setting.int_lambda = float(config["tqt_int_lambda"])
    setting.tqt_optimization_setting.collecting_device = str(config["device"])
    return setting


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    """按首条记录字段顺序写入完整候选或逐样本指标。"""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def build_tqt_candidate(
    config: dict[str, Any], base_config: dict[str, Any], output_dir: Path, steps: int
) -> tuple[Any, Any, dict[str, Any]]:
    """使用固定训练样本构建指定步数的 TQT 量化图。"""
    candidate_base = dict(base_config)
    candidate_base["device"] = str(config["device"])
    setting = create_tqt_setting(config, steps)
    return build_quantized_graph(
        candidate_base,
        output_dir,
        calibration_path_override=resolve_project_path(config["training_manifest"]),
        setting_override=setting,
    )


def run_search(config: dict[str, Any], base_config: dict[str, Any]) -> None:
    """比较 TQT 强度并仅依据验证集锁定候选。"""
    output_dir = resolve_project_path(config["search_output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"TQT 搜索目录非空，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    search = config["threshold_search"]
    thresholds = generate_thresholds(float(search["start"]), float(search["stop"]), float(search["step"]))
    reference = load_reference_metrics(resolve_project_path(config["reference_validation_metrics"]))
    constraints = {name: float(value) for name, value in config["constraints"].items()}
    validation_path = resolve_project_path(config["validation_manifest"])
    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for steps in config["tqt_steps"]:
        candidate_name = f"tqt_{steps}_steps"
        candidate_dir = output_dir / candidate_name
        candidate_dir.mkdir(parents=True, exist_ok=False)
        try:
            graph, device, checkpoint = build_tqt_candidate(config, base_config, candidate_dir, int(steps))
            prediction_cache = collect_prediction_cache(graph, validation_path, device)
            boundary_width = int(checkpoint["config"]["boundary_width_pixels"])
            for threshold in thresholds:
                metrics, _ = evaluate_cached_predictions(prediction_cache, threshold, boundary_width)
                candidates.append(
                    {
                        "candidate": candidate_name,
                        "tqt_steps": int(steps),
                        "threshold": threshold,
                        **metrics,
                    }
                )
        except Exception as error:  # noqa: BLE001
            failures.append(
                {
                    "candidate": candidate_name,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    if failures:
        write_json(output_dir / "failures.json", failures)
    if not candidates:
        raise RuntimeError("全部 TQT 候选均失败。")
    selected = select_candidate(candidates, reference, constraints)
    candidate_rows = [
        {"eligible": candidate_is_eligible(item, reference, constraints), **item} for item in candidates
    ]
    write_csv(output_dir / "validation_candidates.csv", candidate_rows)
    selection = {
        "name": config["name"],
        "reference_metrics": reference,
        "constraints": constraints,
        "selected": selected,
        "selected_minus_reference": {
            name: selected[name] - reference[name] for name in METRIC_NAMES
        },
        "boundary_improved": selected["boundary_iou"] > reference["boundary_iou"],
        "successful_tqt_configs": len(candidates) // len(thresholds),
        "failed_tqt_configs": len(failures),
        "evaluated_validation_candidates": len(candidates),
    }
    write_json(output_dir / "selection.json", selection)
    print(json.dumps(selection, ensure_ascii=False, indent=2))


def run_test(config: dict[str, Any], base_config: dict[str, Any]) -> None:
    """重建验证集锁定的 TQT 配置并执行一次最终测试。"""
    selection_path = resolve_project_path(config["search_output_dir"]) / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("缺少 TQT 验证选择，禁止直接读取测试集。")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection["selected"]
    output_dir = resolve_project_path(config["test_output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"TQT 测试目录非空，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    graph, device, checkpoint = build_tqt_candidate(
        config, base_config, output_dir, int(selected["tqt_steps"])
    )
    candidate_onnx = output_dir / "light_unet_64_boundary_guided_esp32s3.onnx"
    graph_comparison = compare_graph_structure(resolve_project_path(config["stage2_reference_onnx"]), candidate_onnx)
    write_json(output_dir / "graph_comparison.json", graph_comparison)
    if not graph_comparison["equal"]:
        raise RuntimeError("TQT 候选与阶段二基线图结构不一致。")

    prediction_cache = collect_prediction_cache(graph, resolve_project_path(config["test_manifest"]), device)
    boundary_width = int(checkpoint["config"]["boundary_width_pixels"])
    selected_metrics, sample_metrics = evaluate_cached_predictions(
        prediction_cache, float(selected["threshold"]), boundary_width
    )
    write_csv(output_dir / "test_sample_metrics.csv", sample_metrics)
    reference = load_reference_metrics(resolve_project_path(config["reference_test_metrics"]))
    result = {
        "name": config["name"],
        "selected_configuration": {
            "tqt_steps": selected["tqt_steps"],
            "threshold": selected["threshold"],
        },
        "reference_metrics": reference,
        "selected_metrics": selected_metrics,
        "selected_minus_reference": {
            name: selected_metrics[name] - reference[name] for name in METRIC_NAMES
        },
        "graph_comparison": graph_comparison,
    }
    write_json(output_dir / "metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    """按模式执行 TQT 验证搜索或最终测试。"""
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    base_config = json.loads(resolve_project_path(config["base_ptq_config"]).read_text(encoding="utf-8"))
    if args.mode == "search":
        run_search(config, base_config)
    else:
        run_test(config, base_config)


if __name__ == "__main__":
    main()
