"""训练阶段一 LightUNet FP32 基线，并按验证集 Dice 保存最佳检查点。"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.pupil_segmentation import PupilSegmentationDataset
from evaluation.segmentation_metrics import evaluate_batch
from models.unet import LightUNet
from training.losses import BCEDiceLoss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage1_baseline.json"
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "data" / "processed" / "pupil_segmentation_128" / "manifests" / "stage1"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "runs" / "stage1_fp32_baseline_v1"
METRIC_NAMES = (
    "dice",
    "boundary_iou",
    "boundary_dice",
    "interior_dice",
    "center_mae_pixels",
    "empty_prediction_rate",
)


def parse_args() -> argparse.Namespace:
    """读取训练参数；可选覆盖项主要用于冒烟验证。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="阶段一基线配置。")
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR, help="固定切分清单目录。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="本次运行的独立输出目录。")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="训练设备。")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的训练轮数。")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖配置中的批次大小。")
    parser.add_argument("--num-workers", type=int, default=None, help="覆盖配置中的数据进程数。")
    parser.add_argument("--max-train-batches", type=int, default=None, help="每轮最多训练批次数，仅用于冒烟验证。")
    parser.add_argument("--max-validation-batches", type=int, default=None, help="每轮最多验证批次数，仅用于冒烟验证。")
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    """读取训练配置并检查本训练器依赖的固定口径。"""
    with config_path.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = json.load(handle)
    if config.get("loss") != "bce_plus_dice":
        raise ValueError("FP32 基线训练器只接受 bce_plus_dice。")
    if config.get("checkpoint_metric") != "validation_dice":
        raise ValueError("FP32 基线训练器只按 validation_dice 选择检查点。")
    return config


def resolve_device(requested_device: str) -> torch.device:
    """解析训练设备，并在用户明确要求 CUDA 但不可用时失败。"""
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("已要求使用 CUDA，但当前环境不可用。")
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def set_reproducible_seed(seed: int) -> None:
    """固定 Python、NumPy 与 PyTorch 随机状态，并关闭 CuDNN 自动调优。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """基于 PyTorch 工作进程种子固定 NumPy 与 Python 随机状态。"""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_loader(
    manifest_path: Path,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader[dict[str, object]]:
    """为一份固定清单创建具有可复现顺序的数据加载器。"""
    dataset = PupilSegmentationDataset(manifest_path)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def validate_stage1_sample(loader: DataLoader[dict[str, object]], expected_size: int) -> None:
    """在训练前确认真实清单输出符合阶段一尺寸和二值契约。"""
    sample = loader.dataset[0]
    image = sample["image"]
    mask = sample["mask"]
    expected_shape = (1, expected_size, expected_size)
    if not isinstance(image, torch.Tensor) or tuple(image.shape) != expected_shape:
        raise ValueError(f"图像形状不符合配置：{getattr(image, 'shape', None)} != {expected_shape}")
    if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != expected_shape:
        raise ValueError(f"掩码形状不符合配置：{getattr(mask, 'shape', None)} != {expected_shape}")
    if not set(torch.unique(mask).tolist()).issubset({0.0, 1.0}):
        raise ValueError("数据加载器输出的掩码不是 0/1 二值张量。")


def train_one_epoch(
    model: LightUNet,
    loader: DataLoader[dict[str, object]],
    optimizer: torch.optim.Optimizer,
    loss_function: BCEDiceLoss,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int | None,
) -> float:
    """执行一轮 FP32 参数更新并返回按样本加权的平均训练损失。"""
    model.train()
    loss_sum = 0.0
    sample_count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            probabilities = model(images)
        # 模型已内置 Sigmoid；概率版 BCE 必须离开自动混合精度区域，以 FP32 计算。
        loss = loss_function(probabilities.float(), masks.float())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"训练损失出现非有限值：{loss.item()}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        current_batch_size = images.shape[0]
        loss_sum += loss.item() * current_batch_size
        sample_count += current_batch_size
    if sample_count == 0:
        raise RuntimeError("训练循环没有读取任何样本。")
    return loss_sum / sample_count


def validate_one_epoch(
    model: LightUNet,
    loader: DataLoader[dict[str, object]],
    loss_function: BCEDiceLoss,
    device: torch.device,
    amp_enabled: bool,
    threshold: float,
    boundary_width: int,
    max_batches: int | None,
) -> dict[str, float]:
    """执行一轮验证并返回损失、主指标与诊断指标。"""
    model.eval()
    loss_sum = 0.0
    metric_sums = {name: 0.0 for name in METRIC_NAMES}
    sample_count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                probabilities = model(images)
            # 与训练保持相同口径，避免概率版 BCE 在自动混合精度区域触发运行时错误。
            loss = loss_function(probabilities.float(), masks.float())
            batch_metrics = evaluate_batch(probabilities, masks, threshold, boundary_width)
            current_batch_size = images.shape[0]
            loss_sum += loss.item() * current_batch_size
            for metric_name in METRIC_NAMES:
                metric_sums[metric_name] += batch_metrics[metric_name] * current_batch_size
            sample_count += current_batch_size
    if sample_count == 0:
        raise RuntimeError("验证循环没有读取任何样本。")
    return {
        "loss": loss_sum / sample_count,
        **{metric_name: metric_sum / sample_count for metric_name, metric_sum in metric_sums.items()},
        "sample_count": float(sample_count),
    }


def is_better_checkpoint(metrics: dict[str, float], best_metrics: dict[str, float] | None) -> bool:
    """按验证 Dice 优先、Boundary IoU 次优的固定规则比较检查点。"""
    if best_metrics is None:
        return True
    if metrics["dice"] > best_metrics["dice"]:
        return True
    return math.isclose(metrics["dice"], best_metrics["dice"], abs_tol=1e-12) and (
        metrics["boundary_iou"] > best_metrics["boundary_iou"]
    )


def write_json(path: Path, content: object) -> None:
    """以 UTF-8 和稳定缩进写入可追溯 JSON。"""
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """执行可复现 FP32 训练、验证、早停与最佳检查点保存。"""
    args = parse_args()
    config_path = args.config.resolve()
    manifest_dir = args.manifest_dir.resolve()
    output_dir = args.output.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖已有运行：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    training_config = config["training"]
    epochs = args.epochs if args.epochs is not None else int(training_config["epochs"])
    batch_size = args.batch_size if args.batch_size is not None else int(training_config["batch_size"])
    num_workers = args.num_workers if args.num_workers is not None else int(training_config["num_workers"])
    if epochs <= 0 or batch_size <= 0 or num_workers < 0:
        raise ValueError("epochs、batch_size 必须为正，num_workers 不能为负。")

    seed = int(config["seed"])
    set_reproducible_seed(seed)
    device = resolve_device(args.device)
    amp_enabled = bool(training_config["amp"]) and device.type == "cuda"
    train_loader = create_loader(
        manifest_dir / str(config["training_manifest"]),
        batch_size,
        num_workers,
        True,
        seed,
        device.type == "cuda",
    )
    validation_loader = create_loader(
        manifest_dir / "validation.csv", batch_size, num_workers, False, seed, device.type == "cuda"
    )
    validate_stage1_sample(train_loader, int(config["image_size"]))
    validate_stage1_sample(validation_loader, int(config["image_size"]))

    model = LightUNet(upsample_mode=str(config.get("upsample_mode", "bilinear"))).to(device)
    loss_function = BCEDiceLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    history: list[dict[str, object]] = []
    best_metrics: dict[str, float] | None = None
    epochs_without_improvement = 0
    patience = int(training_config["early_stopping_patience"])

    run_metadata = {
        "config": config,
        "config_path": str(config_path),
        "manifest_dir": str(manifest_dir),
        "training_manifest": str(config["training_manifest"]),
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "amp_enabled": amp_enabled,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "max_train_batches": args.max_train_batches,
        "max_validation_batches": args.max_validation_batches,
    }
    write_json(output_dir / "run_metadata.json", run_metadata)

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            scaler,
            device,
            amp_enabled,
            args.max_train_batches,
        )
        validation_metrics = validate_one_epoch(
            model,
            validation_loader,
            loss_function,
            device,
            amp_enabled,
            float(config["prediction_threshold"]),
            int(config["boundary_width_pixels"]),
            args.max_validation_batches,
        )
        epoch_record: dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": validation_metrics,
            "duration_seconds": time.perf_counter() - epoch_start,
        }
        history.append(epoch_record)
        write_json(output_dir / "history.json", history)

        if is_better_checkpoint(validation_metrics, best_metrics):
            best_metrics = dict(validation_metrics)
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_metrics": validation_metrics,
                    "config": config,
                },
                output_dir / "best_model.pt",
            )
            write_json(output_dir / "best_validation_metrics.json", {"epoch": epoch, **validation_metrics})
        else:
            epochs_without_improvement += 1

        print(
            f"轮次 {epoch:03d}/{epochs:03d} "
            f"训练损失={train_loss:.6f} 验证Dice={validation_metrics['dice']:.6f} "
            f"边界IoU={validation_metrics['boundary_iou']:.6f} "
            f"中心误差={validation_metrics['center_mae_pixels']:.3f}px"
        )
        if epochs_without_improvement >= patience:
            print(f"验证 Dice 连续 {patience} 轮未改善，提前停止。")
            break

    write_json(
        output_dir / "completed.json",
        {"completed_epochs": len(history), "best_validation": best_metrics},
    )
    print(f"训练完成：{output_dir}")


if __name__ == "__main__":
    main()
