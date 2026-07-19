"""在固定 ESP32-S3 PTQ 图上执行边界指标驱动的统一阈值选择与最终评价。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from torch.utils.data import DataLoader

from datasets.pupil_segmentation import PupilSegmentationDataset
from evaluation.segmentation_metrics import evaluate_sample
from models.unet import LightUNet
from quantization.quantize_espdl_ptq import (
    CalibrationImageDataset,
    collate_to_device,
    resolve_project_path,
    set_reproducible_seed,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage3_boundary_guided_threshold.json"
METRIC_NAMES = (
    "dice",
    "boundary_iou",
    "boundary_dice",
    "interior_dice",
    "center_mae_pixels",
    "empty_prediction_rate",
)


def parse_args() -> argparse.Namespace:
    """读取阶段三配置和严格分离的搜索或测试模式。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("search", "test"), help="search 只读验证集；test 读取已固化选择后评价测试集。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="阶段三阈值优化配置。")
    return parser.parse_args()


def generate_thresholds(start: float, stop: float, step: float) -> list[float]:
    """生成包含端点且消除浮点累计误差的阈值序列。"""
    if not 0.0 < start <= stop < 1.0 or step <= 0.0:
        raise ValueError("阈值范围必须位于 (0, 1)，且步长为正。")
    count = int(round((stop - start) / step))
    thresholds = [round(start + index * step, 10) for index in range(count + 1)]
    if not np.isclose(thresholds[-1], stop):
        raise ValueError("阈值范围不能被步长整除。")
    return thresholds


def build_quantized_graph(
    base_config: dict[str, Any],
    output_dir: Path,
    calibration_path_override: Path | None = None,
    activation_algorithm: str | None = None,
    setting_override: Any | None = None,
) -> tuple[Any, torch.device, dict[str, Any]]:
    """按阶段二固定配置加载权重、校准真实样本并导出量化图。"""
    from esp_ppq.api import QuantizationSettingFactory, espdl_quantize_torch

    seed = int(base_config["seed"])
    set_reproducible_seed(seed)
    device = torch.device(base_config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("量化配置要求 CUDA，但当前环境不可用。")

    checkpoint_path = resolve_project_path(base_config["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint["config"]
    model = LightUNet(upsample_mode=str(checkpoint_config.get("upsample_mode", "bilinear"))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    manifest_dir = resolve_project_path(base_config["manifest_dir"])
    calibration_path = (
        calibration_path_override.resolve()
        if calibration_path_override is not None
        else manifest_dir / base_config["calibration_manifest"]
    )
    calibration_dataset = CalibrationImageDataset(calibration_path)
    expected_calibration_steps = int(base_config["calibration_steps"])
    if calibration_path_override is None and len(calibration_dataset) != expected_calibration_steps:
        raise ValueError("阶段三必须复用完整的阶段二固定校准清单。")
    calibration_steps = len(calibration_dataset)
    calibration_loader = DataLoader(calibration_dataset, batch_size=1, shuffle=False, num_workers=0)
    collate_fn = partial(collate_to_device, device=device)
    test_input = calibration_dataset[0].unsqueeze(0).to(device)
    espdl_path = output_dir / "light_unet_64_boundary_guided_esp32s3.espdl"
    setting = setting_override
    if setting is None and activation_algorithm is not None:
        setting = QuantizationSettingFactory.espdl_setting()
        setting.quantize_activation_setting.calib_algorithm = activation_algorithm
    graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=str(espdl_path),
        calib_dataloader=calibration_loader,
        calib_steps=calibration_steps,
        input_shape=base_config["input_shape"],
        inputs=[test_input],
        target=base_config["target"],
        num_of_bits=int(base_config["num_of_bits"]),
        setting=setting,
        collate_fn=collate_fn,
        device=str(device),
        error_report=False,
        export_config=True,
        export_test_values=True,
        verbose=0,
        opset_version=int(base_config["opset_version"]),
    )
    return graph, device, checkpoint


def collect_prediction_cache(graph: Any, manifest_path: Path, device: torch.device) -> list[dict[str, Any]]:
    """对固定清单运行一次量化推理并缓存概率图、真值与追溯字段。"""
    from esp_ppq import TorchExecutor

    dataset = PupilSegmentationDataset(manifest_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    executor = TorchExecutor(graph=graph, device=str(device))
    records: list[dict[str, Any]] = []
    for batch in loader:
        probability = executor.forward(batch["image"].to(device))[0].detach().cpu().numpy()[0, 0]
        records.append(
            {
                "sample_id": batch["sample_id"][0],
                "source": batch["source"][0],
                "subject": batch["subject"][0],
                "probability": probability,
                "target": batch["mask"].numpy()[0, 0],
            }
        )
    return records


def evaluate_cached_predictions(
    records: list[dict[str, Any]], threshold: float, boundary_width: int
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """以指定统一阈值评价缓存预测，并返回汇总与逐样本指标。"""
    metric_sums: defaultdict[str, float] = defaultdict(float)
    sample_metrics: list[dict[str, object]] = []
    for record in records:
        metrics, _ = evaluate_sample(record["probability"], record["target"], threshold, boundary_width)
        for name in METRIC_NAMES:
            metric_sums[name] += metrics[name]
        sample_metrics.append(
            {
                "sample_id": record["sample_id"],
                "source": record["source"],
                "subject": record["subject"],
                "threshold": threshold,
                **metrics,
            }
        )
    sample_count = len(records)
    summary = {name: metric_sums[name] / sample_count for name in METRIC_NAMES}
    summary["sample_count"] = float(sample_count)
    return summary, sample_metrics


def candidate_is_eligible(
    metrics: dict[str, float], standard: dict[str, float], constraints: dict[str, float]
) -> bool:
    """判断候选是否满足预先固定的 Dice、中心误差和空预测约束。"""
    return (
        metrics["dice"] >= standard["dice"] - constraints["max_dice_drop"]
        and metrics["center_mae_pixels"]
        <= standard["center_mae_pixels"] + constraints["max_center_mae_increase_pixels"]
        and metrics["empty_prediction_rate"]
        <= standard["empty_prediction_rate"] + constraints["max_empty_prediction_rate_increase"]
    )


def select_candidate(
    candidates: list[dict[str, float]], standard_threshold: float, constraints: dict[str, float]
) -> dict[str, float]:
    """在合格候选中最大化 Boundary IoU，无真实改善时保留标准阈值。"""
    standard = next(item for item in candidates if np.isclose(item["threshold"], standard_threshold))
    eligible = [item for item in candidates if candidate_is_eligible(item, standard, constraints)]
    best = max(
        eligible,
        key=lambda item: (
            item["boundary_iou"],
            item["dice"],
            -item["center_mae_pixels"],
            -abs(item["threshold"] - standard_threshold),
        ),
    )
    return best if best["boundary_iou"] > standard["boundary_iou"] else standard


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    """按首条记录字段顺序写入候选或逐样本 CSV。"""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def onnx_graph_signature(path: Path) -> dict[str, object]:
    """提取不含权重数值的 ONNX 节点、连接与张量形状签名。"""
    model = onnx.load(path)
    graph = model.graph
    return {
        "nodes": [
            {
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
            }
            for node in graph.node
        ],
        "initializers": [
            {"name": item.name, "data_type": item.data_type, "dims": list(item.dims)}
            for item in graph.initializer
        ],
        "inputs": [item.name for item in graph.input],
        "outputs": [item.name for item in graph.output],
    }


def compare_graph_structure(reference_path: Path, candidate_path: Path) -> dict[str, object]:
    """比较阶段二与阶段三 ONNX 图结构并返回可追溯摘要。"""
    reference = onnx_graph_signature(reference_path)
    candidate = onnx_graph_signature(candidate_path)
    reference_text = json.dumps(reference, ensure_ascii=False, sort_keys=True)
    candidate_text = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
    return {
        "equal": reference == candidate,
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "reference_sha256": hashlib.sha256(reference_text.encode("utf-8")).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate_text.encode("utf-8")).hexdigest(),
        "node_count": len(candidate["nodes"]),
    }


def run_search(config: dict[str, Any], base_config: dict[str, Any]) -> None:
    """仅使用验证集搜索候选阈值并固化最终选择。"""
    output_dir = resolve_project_path(config["search_output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"搜索输出目录非空，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    graph, device, checkpoint = build_quantized_graph(base_config, output_dir)
    candidate_onnx = output_dir / "light_unet_64_boundary_guided_esp32s3.onnx"
    graph_comparison = compare_graph_structure(resolve_project_path(config["stage2_reference_onnx"]), candidate_onnx)
    write_json(output_dir / "graph_comparison.json", graph_comparison)
    if not graph_comparison["equal"]:
        raise RuntimeError("阶段三导出图与阶段二基线结构不一致。")

    records = collect_prediction_cache(graph, resolve_project_path(config["validation_manifest"]), device)
    search = config["threshold_search"]
    thresholds = generate_thresholds(float(search["start"]), float(search["stop"]), float(search["step"]))
    boundary_width = int(checkpoint["config"]["boundary_width_pixels"])
    candidates: list[dict[str, float]] = []
    for threshold in thresholds:
        metrics, _ = evaluate_cached_predictions(records, threshold, boundary_width)
        candidates.append({"threshold": threshold, **metrics})

    standard_threshold = float(config["standard_threshold"])
    constraints = {name: float(value) for name, value in config["constraints"].items()}
    standard = next(item for item in candidates if np.isclose(item["threshold"], standard_threshold))
    selected = select_candidate(candidates, standard_threshold, constraints)
    candidate_rows = [
        {"eligible": candidate_is_eligible(item, standard, constraints), **item} for item in candidates
    ]
    write_csv(output_dir / "validation_candidates.csv", candidate_rows)
    _, selected_sample_metrics = evaluate_cached_predictions(records, selected["threshold"], boundary_width)
    write_csv(output_dir / "validation_selected_sample_metrics.csv", selected_sample_metrics)
    selection = {
        "name": config["name"],
        "selection_dataset": str(resolve_project_path(config["validation_manifest"])),
        "standard_threshold": standard_threshold,
        "constraints": constraints,
        "standard_metrics": standard,
        "selected_threshold": selected["threshold"],
        "selected_metrics": selected,
        "selected_minus_standard": {
            name: selected[name] - standard[name] for name in METRIC_NAMES
        },
        "boundary_improved": selected["boundary_iou"] > standard["boundary_iou"],
        "candidate_count": len(candidates),
        "graph_comparison": graph_comparison,
    }
    write_json(output_dir / "selection.json", selection)
    print(json.dumps(selection, ensure_ascii=False, indent=2))


def run_test(config: dict[str, Any], base_config: dict[str, Any]) -> None:
    """读取已固化验证选择，对测试集执行一次标准与候选阈值评价。"""
    search_dir = resolve_project_path(config["search_output_dir"])
    selection_path = search_dir / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("缺少验证集选择结果，禁止直接评价测试集。")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    output_dir = resolve_project_path(config["test_output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"测试输出目录非空，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    graph, device, checkpoint = build_quantized_graph(base_config, output_dir)
    candidate_onnx = output_dir / "light_unet_64_boundary_guided_esp32s3.onnx"
    graph_comparison = compare_graph_structure(resolve_project_path(config["stage2_reference_onnx"]), candidate_onnx)
    write_json(output_dir / "graph_comparison.json", graph_comparison)
    if not graph_comparison["equal"]:
        raise RuntimeError("最终候选图与阶段二基线结构不一致，终止测试评价。")

    records = collect_prediction_cache(graph, resolve_project_path(config["test_manifest"]), device)
    boundary_width = int(checkpoint["config"]["boundary_width_pixels"])
    standard_threshold = float(selection["standard_threshold"])
    selected_threshold = float(selection["selected_threshold"])
    standard_metrics, _ = evaluate_cached_predictions(records, standard_threshold, boundary_width)
    selected_metrics, selected_sample_metrics = evaluate_cached_predictions(records, selected_threshold, boundary_width)
    write_csv(output_dir / "test_selected_sample_metrics.csv", selected_sample_metrics)
    result = {
        "name": config["name"],
        "test_dataset": str(resolve_project_path(config["test_manifest"])),
        "standard_threshold": standard_threshold,
        "selected_threshold": selected_threshold,
        "standard_metrics": standard_metrics,
        "selected_metrics": selected_metrics,
        "selected_minus_standard": {
            name: selected_metrics[name] - standard_metrics[name] for name in METRIC_NAMES
        },
        "graph_comparison": graph_comparison,
    }
    write_json(output_dir / "metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    """按模式执行验证集选择或一次性测试集评价。"""
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    base_config = json.loads(resolve_project_path(config["base_ptq_config"]).read_text(encoding="utf-8"))
    if args.mode == "search":
        run_search(config, base_config)
    else:
        run_test(config, base_config)


if __name__ == "__main__":
    main()
