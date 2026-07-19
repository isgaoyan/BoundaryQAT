"""从阶段一固定清单读取统一瞳孔分割图像与掩码。"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PupilSegmentationDataset(Dataset[dict[str, object]]):
    """读取一份固定切分清单，并返回可直接训练的单通道张量。"""

    def __init__(self, manifest_path: Path, dataset_root: Path | None = None) -> None:
        """加载清单并确定统一数据根目录，不在初始化时读取全部图像。"""
        self.manifest_path = Path(manifest_path).resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"切分清单不存在：{self.manifest_path}")
        self.dataset_root = (
            Path(dataset_root).resolve()
            if dataset_root is not None
            else self._infer_dataset_root()
        )
        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            self.records = list(csv.DictReader(handle))
        if not self.records:
            raise ValueError(f"切分清单为空：{self.manifest_path}")

    def _infer_dataset_root(self) -> Path:
        """从清单上级目录中查找同时包含 images 与 masks 的数据根目录。"""
        for parent in self.manifest_path.parents:
            if (parent / "images").is_dir() and (parent / "masks").is_dir():
                return parent
        raise ValueError("无法从清单位置推断数据根目录，请显式传入 dataset_root。")

    def __len__(self) -> int:
        """返回清单中的有效样本数量。"""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        """读取并验证单个样本，返回图像、掩码及追溯字段。"""
        record = self.records[index]
        image_path = self.dataset_root / record["image_path"]
        mask_path = self.dataset_root / record["mask_path"]
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(
                f"样本文件缺失：image={image_path.is_file()} mask={mask_path.is_file()} "
                f"sample_id={record.get('sample_id', '')}"
            )

        with Image.open(image_path) as raw_image:
            image_array = np.array(raw_image.convert("L"), dtype=np.float32, copy=True)
        with Image.open(mask_path) as raw_mask:
            mask_array = np.array(raw_mask.convert("L"), dtype=np.uint8, copy=True)

        if image_array.shape != mask_array.shape:
            raise ValueError(
                f"图像与掩码尺寸不一致：{image_array.shape} != {mask_array.shape} "
                f"sample_id={record.get('sample_id', '')}"
            )
        mask_values = set(np.unique(mask_array).tolist())
        if not mask_values.issubset({0, 255}):
            raise ValueError(
                f"掩码含非法像素值：{sorted(mask_values)} "
                f"sample_id={record.get('sample_id', '')}"
            )

        image_tensor = torch.from_numpy(image_array).unsqueeze(0).div_(255.0)
        mask_tensor = torch.from_numpy((mask_array > 0).astype(np.float32)).unsqueeze(0)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "sample_id": record["sample_id"],
            "source": record["source"],
            "subject": record["subject"],
        }
