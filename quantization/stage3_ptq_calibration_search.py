"""搜索 ESP32-S3 PTQ 校准集、激活截断算法和统一后处理阈值。"""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage3_ptq_calibration_search.json"


def parse_args() -> argparse.Namespace:
    """读取搜索配置和严格分离的验证搜索或最终测试模式。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("search", "test"), help="search 只读验证集；test 只评价锁定配置。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="PTQ 校准策略搜索配置。")
    return parser.parse_args()


def candidate_is_eligible(
    metrics: dict[str, float], reference: dict[str, float], constraints: dict[str, float]
) -> bool:
    """执行中心误差和空预测率均不得增加的硬约束。"""
    tolerance = 1e-12
    return (
        metrics["center_mae_pixels"]
        <= reference["center_mae_pixels"] + constraints["max_center_mae_increase_pixels"] + tolerance
        and metrics["empty_prediction_rate"]
        <= reference["empty_prediction_rate"]
        + constraints["max_empty_prediction_rate_increase"]
        + tolerance
    )


def select_candidate(
    candidates: list[dict[str, Any]], reference: dict[str, float], constraints: dict[str, float]
) -> dict[str, Any]:
    """在硬约束内最大化 Boundary IoU，并以 Dice 和几何稳定性依次破除并列。"""
    eligible = [item for item in candidates if candidate_is_eligible(item, reference, constraints)]
    if not eligible:
        raise RuntimeError("没有 PTQ 候选满足中心误差与空预测率硬约束。")
    return max(
        eligible,
        key=lambda item: (
            item["boundary_iou"],
            item["dice"],
            -item["center_mae_pixels"],
            -item["empty_prediction_rate"],
            -abs(item["threshold"] - 0.5),
        ),
    )


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    """按首条记录字段顺序写入完整候选或逐样本指标。"""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def load_reference_metrics(path: Path) -> dict[str, float]:
    """读取 PTQ、通用评估或已选候选指标作为不可变选择基线。"""
    document = json.loads(path.read_text(encoding="utf-8"))
    for key in ("ptq_metrics", "metrics", "selected_metrics"):
        if key in document:
            return {name: float(document[key][name]) for name in METRIC_NAMES}
    raise KeyError(f"参考指标文件缺少受支持的指标字段：{path}")


def run_search(config: dict[str, Any], base_config: dict[str, Any]) -> None:
    """运行全部 PTQ 组合并仅依据验证集锁定最终配置。"""
    output_dir = resolve_project_path(config["search_output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"PTQ 搜索输出目录非空，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_path = resolve_project_path(config["validation_manifest"])
    calibration_dir = resolve_project_path(config["calibration_manifest_dir"])
    search = config["threshold_search"]
    thresholds = generate_thresholds(float(search["start"]), float(search["stop"]), float(search["step"]))
    reference = load_reference_metrics(resolve_project_path(config["reference_validation_metrics"]))
    constraints = {name: float(value) for name, value in config["constraints"].items()}
    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for sample_count in config["calibration_sample_counts"]:
        calibration_path = calibration_dir / f"calibration_lpw_{sample_count}.csv"
        if not calibration_path.is_file():
            raise FileNotFoundError(f"缺少固定校准清单：{calibration_path}")
        for algorithm in config["activation_algorithms"]:
            candidate_name = f"{algorithm}_{sample_count}"
            candidate_dir = output_dir / candidate_name
            candidate_dir.mkdir(parents=True, exist_ok=False)
            try:
                graph, device, checkpoint = build_quantized_graph(
                    base_config,
                    candidate_dir,
                    calibration_path_override=calibration_path,
                    activation_algorithm=str(algorithm),
                )
                prediction_cache = collect_prediction_cache(graph, validation_path, device)
                boundary_width = int(checkpoint["config"]["boundary_width_pixels"])
                for threshold in thresholds:
                    metrics, _ = evaluate_cached_predictions(prediction_cache, threshold, boundary_width)
                    candidates.append(
                        {
                            "candidate": candidate_name,
                            "activation_algorithm": algorithm,
                            "calibration_sample_count": sample_count,
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
        raise RuntimeError("全部 PTQ 候选均失败。")
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
        "successful_quantization_configs": len(candidates) // len(thresholds),
        "failed_quantization_configs": len(failures),
        "evaluated_validation_candidates": len(candidates),
    }
    write_json(output_dir / "selection.json", selection)
    print(json.dumps(selection, ensure_ascii=False, indent=2))


def run_test(config: dict[str, Any], base_config: dict[str, Any]) -> None:
    """重建锁定 PTQ 配置并对测试集执行一次最终评价。"""
    search_dir = resolve_project_path(config["search_output_dir"])
    selection_path = search_dir / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("缺少验证集 PTQ 选择结果，禁止直接读取测试集。")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection["selected"]
    output_dir = resolve_project_path(config["test_output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"PTQ 最终测试目录非空，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_dir = resolve_project_path(config["calibration_manifest_dir"])
    calibration_path = calibration_dir / f"calibration_lpw_{selected['calibration_sample_count']}.csv"
    graph, device, checkpoint = build_quantized_graph(
        base_config,
        output_dir,
        calibration_path_override=calibration_path,
        activation_algorithm=str(selected["activation_algorithm"]),
    )
    candidate_onnx = output_dir / "light_unet_64_boundary_guided_esp32s3.onnx"
    graph_comparison = compare_graph_structure(resolve_project_path(config["stage2_reference_onnx"]), candidate_onnx)
    write_json(output_dir / "graph_comparison.json", graph_comparison)
    if not graph_comparison["equal"]:
        raise RuntimeError("最终 PTQ 候选与阶段二基线图结构不一致。")

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
            "activation_algorithm": selected["activation_algorithm"],
            "calibration_sample_count": selected["calibration_sample_count"],
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
    """按模式执行 PTQ 验证搜索或锁定后的最终测试。"""
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    base_config = json.loads(resolve_project_path(config["base_ptq_config"]).read_text(encoding="utf-8"))
    if args.mode == "search":
        run_search(config, base_config)
    else:
        run_test(config, base_config)


if __name__ == "__main__":
    main()
