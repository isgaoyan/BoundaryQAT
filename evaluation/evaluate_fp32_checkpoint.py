"""独立评价阶段一最佳 FP32 检查点并生成指标、逐样本结果与预测预览。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets.pupil_segmentation import PupilSegmentationDataset
from evaluation.segmentation_metrics import evaluate_sample, inner_boundary
from models.unet import LightUNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "results" / "runs" / "stage1_fp32_lpw_only_v2" / "best_model.pt"
DEFAULT_TEST_MANIFEST = (
    PROJECT_ROOT / "data" / "processed" / "pupil_segmentation_128" / "manifests" / "stage1" / "test.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "runs" / "stage1_fp32_lpw_only_v2" / "test_evaluation"
METRIC_NAMES = (
    "dice",
    "boundary_iou",
    "boundary_dice",
    "interior_dice",
    "center_mae_pixels",
    "empty_prediction_rate",
)


def parse_args() -> argparse.Namespace:
    """读取检查点、清单、输出目录和运行设备。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="最佳 FP32 检查点。")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_TEST_MANIFEST, help="待评价固定清单。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="独立评价输出目录。")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="推理设备。")
    parser.add_argument("--batch-size", type=int, default=32, help="推理批次大小。")
    parser.add_argument("--num-workers", type=int, default=0, help="数据读取进程数。")
    parser.add_argument("--preview-count", type=int, default=12, help="按清单顺序保存的预览数量。")
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    """解析独立评价设备，并拒绝不可用的显式 CUDA 请求。"""
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("已要求使用 CUDA，但当前环境不可用。")
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def save_preview(
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
    boundary_width: int,
) -> None:
    """保存原图、绿色真值边界和红色预测边界三联预览。"""
    gray = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    base = np.repeat(gray[:, :, None], 3, axis=2)
    target_panel = base.copy()
    prediction_panel = base.copy()
    target_panel[inner_boundary(target, boundary_width)] = (0, 255, 0)
    prediction_panel[inner_boundary(prediction, boundary_width)] = (255, 0, 0)
    canvas = np.concatenate([base, target_panel, prediction_panel], axis=1)
    Image.fromarray(canvas, mode="RGB").save(output_path)


def write_sample_metrics(records: list[dict[str, object]], output_path: Path) -> None:
    """写入逐样本指标，便于后续定位最差案例和绘制误差分布。"""
    if not records:
        raise RuntimeError("没有可写入的逐样本指标。")
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    """加载最佳检查点，对指定固定清单执行一次独立评价。"""
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    manifest_path = args.manifest.resolve()
    output_dir = args.output.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"评价输出目录非空，拒绝覆盖：{output_dir}")
    if args.batch_size <= 0 or args.num_workers < 0 or args.preview_count < 0:
        raise ValueError("batch_size 必须为正，num_workers 与 preview_count 不能为负。")
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = LightUNet(upsample_mode=str(config.get("upsample_mode", "bilinear"))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = PupilSegmentationDataset(manifest_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    metric_sums: defaultdict[str, float] = defaultdict(float)
    sample_records: list[dict[str, object]] = []
    preview_index = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            probabilities = model(images).cpu().numpy()[:, 0]
            targets = batch["mask"].numpy()[:, 0]
            image_arrays = batch["image"].numpy()[:, 0]
            for index, (probability, target, image_array) in enumerate(
                zip(probabilities, targets, image_arrays, strict=True)
            ):
                metrics, prediction = evaluate_sample(
                    probability,
                    target,
                    float(config["prediction_threshold"]),
                    int(config["boundary_width_pixels"]),
                )
                record: dict[str, object] = {
                    "sample_id": batch["sample_id"][index],
                    "source": batch["source"][index],
                    "subject": batch["subject"][index],
                    **metrics,
                }
                sample_records.append(record)
                for metric_name in METRIC_NAMES:
                    metric_sums[metric_name] += metrics[metric_name]
                if preview_index < args.preview_count:
                    save_preview(
                        image_array,
                        target >= 0.5,
                        prediction,
                        preview_dir / f"{preview_index:02d}_{record['sample_id']}.png",
                        int(config["boundary_width_pixels"]),
                    )
                    preview_index += 1

    sample_count = len(sample_records)
    if sample_count == 0:
        raise RuntimeError("独立评价没有读取任何样本。")
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "manifest": str(manifest_path),
        "sample_count": sample_count,
        "prediction_threshold": config["prediction_threshold"],
        "boundary_width_pixels": config["boundary_width_pixels"],
        "metrics": {metric_name: metric_sums[metric_name] / sample_count for metric_name in METRIC_NAMES},
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_sample_metrics(sample_records, output_dir / "sample_metrics.csv")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
