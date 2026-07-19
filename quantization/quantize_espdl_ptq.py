"""使用真实 LPW 校准样本生成 ESP32-S3 PTQ 模型并执行同口径评价。"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.pupil_segmentation import PupilSegmentationDataset
from evaluation.segmentation_metrics import evaluate_sample
from models.unet import LightUNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage2_ptq_64.json"
METRIC_NAMES = (
    "dice",
    "boundary_iou",
    "boundary_dice",
    "interior_dice",
    "center_mae_pixels",
    "empty_prediction_rate",
)


class CalibrationImageDataset(Dataset[torch.Tensor]):
    """只暴露校准图像张量，避免标签进入 ESP-PPQ 输入。"""

    def __init__(self, manifest_path: Path) -> None:
        """加载固定校准清单。"""
        self.dataset = PupilSegmentationDataset(manifest_path)

    def __len__(self) -> int:
        """返回固定校准样本数量。"""
        return len(self.dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        """返回单张归一化灰度图。"""
        return self.dataset[index]["image"]


def parse_args() -> argparse.Namespace:
    """读取量化配置及可选评价清单覆盖项。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="阶段二 PTQ 配置。")
    parser.add_argument("--evaluation-manifest", type=Path, default=None, help="覆盖待评价清单。")
    parser.add_argument("--fp32-metrics", type=Path, default=None, help="覆盖对应 FP32 指标文件。")
    parser.add_argument("--output", type=Path, default=None, help="覆盖独立输出目录。")
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    """将配置中的项目相对路径解析为绝对路径。"""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def set_reproducible_seed(seed: int) -> None:
    """固定 Python、NumPy 与 PyTorch 随机状态。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def collate_to_device(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    """把校准批次转换为 ESP-PPQ 执行设备上的 FP32 张量。"""
    return batch.to(device=device, dtype=torch.float32)


def to_jsonable(value: Any) -> Any:
    """递归转换量化报告中的张量、元组和 NumPy 标量。"""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, content: object) -> None:
    """以 UTF-8 和稳定缩进写入 JSON。"""
    path.write_text(json.dumps(to_jsonable(content), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate_quantized_graph(
    graph: Any,
    manifest_path: Path,
    device: torch.device,
    threshold: float,
    boundary_width: int,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """使用 PPQ 量化图评价固定清单，并返回汇总与逐样本指标。"""
    from esp_ppq import TorchExecutor

    dataset = PupilSegmentationDataset(manifest_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    executor = TorchExecutor(graph=graph, device=str(device))
    metric_sums: defaultdict[str, float] = defaultdict(float)
    records: list[dict[str, object]] = []
    for batch in loader:
        probability = executor.forward(batch["image"].to(device))[0].detach().cpu().numpy()[0, 0]
        target = batch["mask"].numpy()[0, 0]
        metrics, _ = evaluate_sample(probability, target, threshold, boundary_width)
        for name in METRIC_NAMES:
            metric_sums[name] += metrics[name]
        records.append(
            {
                "sample_id": batch["sample_id"][0],
                "source": batch["source"][0],
                "subject": batch["subject"][0],
                **metrics,
            }
        )
    count = len(records)
    return {name: metric_sums[name] / count for name in METRIC_NAMES} | {"sample_count": float(count)}, records


def write_sample_metrics(path: Path, records: list[dict[str, object]]) -> None:
    """写入量化图逐样本指标。"""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def calculate_metric_deltas(ptq_metrics: dict[str, float], fp32_metrics: dict[str, float]) -> dict[str, float]:
    """计算 PTQ 相对同清单 FP32 指标的有符号变化。"""
    return {name: ptq_metrics[name] - fp32_metrics[name] for name in METRIC_NAMES}


def main() -> None:
    """加载最佳 FP32 权重、执行 PTQ 导出、误差分析与同口径评价。"""
    from esp_ppq import graphwise_error_analyse, layerwise_error_analyse
    from esp_ppq.api import espdl_quantize_torch

    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    output_dir = args.output.resolve() if args.output else resolve_project_path(config["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    set_reproducible_seed(seed)
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PTQ 配置要求 CUDA，但当前环境不可用。")

    checkpoint_path = resolve_project_path(config["checkpoint"])
    manifest_dir = resolve_project_path(config["manifest_dir"])
    calibration_path = manifest_dir / config["calibration_manifest"]
    evaluation_path = args.evaluation_manifest.resolve() if args.evaluation_manifest else manifest_dir / config["evaluation_manifest"]
    fp32_metrics_path = args.fp32_metrics.resolve() if args.fp32_metrics else resolve_project_path(config["fp32_metrics"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint["config"]
    model = LightUNet(upsample_mode=str(checkpoint_config.get("upsample_mode", "bilinear"))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    calibration_dataset = CalibrationImageDataset(calibration_path)
    calibration_loader = DataLoader(calibration_dataset, batch_size=1, shuffle=False, num_workers=0)
    calibration_steps = int(config["calibration_steps"])
    if calibration_steps != len(calibration_dataset):
        raise ValueError("正式 PTQ 必须使用完整固定校准清单。")
    collate_fn = partial(collate_to_device, device=device)
    test_input = calibration_dataset[0].unsqueeze(0).to(device)
    espdl_path = output_dir / "light_unet_64_ptq_esp32s3.espdl"

    graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=str(espdl_path),
        calib_dataloader=calibration_loader,
        calib_steps=calibration_steps,
        input_shape=config["input_shape"],
        inputs=[test_input],
        target=config["target"],
        num_of_bits=int(config["num_of_bits"]),
        collate_fn=collate_fn,
        device=str(device),
        error_report=False,
        export_config=True,
        export_test_values=True,
        verbose=1,
        opset_version=int(config["opset_version"]),
    )

    analysis_steps = int(config["analysis_steps"])
    graphwise = graphwise_error_analyse(
        graph=graph,
        running_device=str(device),
        dataloader=calibration_loader,
        collate_fn=collate_fn,
        steps=analysis_steps,
        verbose=False,
    )
    layerwise = layerwise_error_analyse(
        graph=graph,
        running_device=str(device),
        dataloader=calibration_loader,
        collate_fn=collate_fn,
        steps=analysis_steps,
        verbose=False,
    )
    write_json(output_dir / "graphwise_error.json", graphwise)
    write_json(output_dir / "layerwise_error.json", layerwise)

    ptq_metrics, sample_records = evaluate_quantized_graph(
        graph,
        evaluation_path,
        device,
        float(checkpoint["config"]["prediction_threshold"]),
        int(checkpoint["config"]["boundary_width_pixels"]),
    )
    fp32_document = json.loads(fp32_metrics_path.read_text(encoding="utf-8"))
    fp32_metrics = fp32_document.get("metrics", fp32_document)
    result = {
        "config": config,
        "checkpoint": str(checkpoint_path),
        "calibration_manifest": str(calibration_path),
        "evaluation_manifest": str(evaluation_path),
        "ptq_metrics": ptq_metrics,
        "fp32_metrics": {name: fp32_metrics[name] for name in METRIC_NAMES},
        "ptq_minus_fp32": calculate_metric_deltas(ptq_metrics, fp32_metrics),
    }
    write_json(output_dir / "metrics.json", result)
    write_sample_metrics(output_dir / "sample_metrics.csv", sample_records)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
