"""阶段一固定切分与训练数据读取的单元测试。"""

import csv
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from datasets.pupil_segmentation import PupilSegmentationDataset
from scripts.create_stage1_splits import build_subject_to_split, split_records, write_outputs


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_stage1_config() -> dict[str, object]:
    """读取版本化阶段一配置，供切分测试使用。"""
    config_path = PROJECT_ROOT / "configs" / "stage1_baseline.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def write_gray_image(path: Path, size: tuple[int, int], value: int) -> None:
    """创建测试用单通道 PNG，并自动建立父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=value).save(path)


def write_manifest(path: Path, records: list[dict[str, str]]) -> None:
    """将测试记录写为数据集可读取的最小清单。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "source", "subject", "image_path", "mask_path"],
        )
        writer.writeheader()
        writer.writerows(records)


def test_fixed_split_keeps_subjects_isolated_and_hmep_train_only(tmp_path: Path) -> None:
    """验证固定 LPW 受试者互斥，且 HMEPS 只进入训练集。"""
    config = load_stage1_config()
    configured_subjects = config["lpw_subjects"]
    records: list[dict[str, str]] = []

    for split_subjects in configured_subjects.values():
        for subject in split_subjects:
            sample_id = f"lpw__{subject}"
            mask_relative = f"masks/lpw/{sample_id}.png"
            write_gray_image(tmp_path / mask_relative, (4, 4), 255)
            records.append(
                {
                    "sample_id": sample_id,
                    "source": "lpw",
                    "subject": subject,
                    "image_path": f"images/lpw/{sample_id}.png",
                    "mask_path": mask_relative,
                }
            )

    hmep_record = {
        "sample_id": "hmep__sample",
        "source": "hmep",
        "subject": "53",
        "image_path": "images/hmep/hmep__sample.png",
        "mask_path": "masks/hmep/hmep__sample.png",
    }
    write_gray_image(tmp_path / hmep_record["mask_path"], (4, 4), 255)
    records.append(hmep_record)

    subject_to_split = build_subject_to_split(config, records)
    splits, excluded = split_records(records, tmp_path, subject_to_split)

    assert excluded == []
    assert {record["subject"] for record in splits["train"] if record["source"] == "lpw"} == set(
        configured_subjects["train"]
    )
    assert {record["subject"] for record in splits["validation"]} == set(configured_subjects["validation"])
    assert {record["subject"] for record in splits["test"]} == set(configured_subjects["test"])
    assert [record["sample_id"] for record in splits["train"] if record["source"] == "hmep"] == [
        "hmep__sample"
    ]


def test_fixed_split_records_empty_mask_exclusion(tmp_path: Path) -> None:
    """验证空掩码不会进入训练、验证或测试清单，并保留排除原因。"""
    config = load_stage1_config()
    records = []
    for split_subjects in config["lpw_subjects"].values():
        for subject in split_subjects:
            sample_id = f"lpw__{subject}"
            mask_relative = f"masks/lpw/{sample_id}.png"
            write_gray_image(tmp_path / mask_relative, (4, 4), 0 if subject == "05" else 255)
            records.append(
                {
                    "sample_id": sample_id,
                    "source": "lpw",
                    "subject": subject,
                    "image_path": f"images/lpw/{sample_id}.png",
                    "mask_path": mask_relative,
                }
            )

    mapping = build_subject_to_split(config, records)
    splits, excluded = split_records(records, tmp_path, mapping)

    assert sum(len(items) for items in splits.values()) == 21
    assert len(excluded) == 1
    assert excluded[0]["sample_id"] == "lpw__05"
    assert excluded[0]["split"] == "train"
    assert excluded[0]["exclusion_reason"] == "empty_mask"


def test_split_outputs_separate_lpw_primary_training_manifest(tmp_path: Path) -> None:
    """验证主训练清单只含 LPW，而辅助训练清单可包含 HMEPS。"""
    config = load_stage1_config()
    original_fields = ["sample_id", "source", "subject", "image_path", "mask_path"]
    lpw_record = {
        "sample_id": "lpw__01",
        "source": "lpw",
        "subject": "01",
        "image_path": "images/lpw/lpw__01.png",
        "mask_path": "masks/lpw/lpw__01.png",
        "split": "train",
    }
    hmep_record = {
        "sample_id": "hmep__01",
        "source": "hmep",
        "subject": "53",
        "image_path": "images/hmep/hmep__01.png",
        "mask_path": "masks/hmep/hmep__01.png",
        "split": "train",
    }
    splits = {"train": [lpw_record, hmep_record], "validation": [], "test": []}

    write_outputs(splits, [], tmp_path, config, original_fields)

    primary_rows = list(csv.DictReader((tmp_path / "train_lpw.csv").open(encoding="utf-8")))
    auxiliary_rows = list(csv.DictReader((tmp_path / "train.csv").open(encoding="utf-8")))
    assert [record["source"] for record in primary_rows] == ["lpw"]
    assert {record["source"] for record in auxiliary_rows} == {"lpw", "hmep"}


def test_dataset_returns_normalized_128_tensors(tmp_path: Path) -> None:
    """验证数据集返回形状正确、范围固定的 128×128 FP32 张量。"""
    image_relative = "images/lpw/lpw__01.png"
    mask_relative = "masks/lpw/lpw__01.png"
    write_gray_image(tmp_path / image_relative, (128, 128), 128)
    write_gray_image(tmp_path / mask_relative, (128, 128), 0)
    with Image.open(tmp_path / mask_relative) as raw_mask:
        mask = raw_mask.copy()
    mask.paste(255, (32, 32, 96, 96))
    mask.save(tmp_path / mask_relative)

    manifest_path = tmp_path / "manifests" / "stage1" / "train.csv"
    write_manifest(
        manifest_path,
        [
            {
                "sample_id": "lpw__01",
                "source": "lpw",
                "subject": "01",
                "image_path": image_relative,
                "mask_path": mask_relative,
            }
        ],
    )

    dataset = PupilSegmentationDataset(manifest_path)
    sample = dataset[0]

    assert sample["image"].shape == (1, 128, 128)
    assert sample["mask"].shape == (1, 128, 128)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].dtype == torch.float32
    assert torch.allclose(sample["image"], torch.full((1, 128, 128), 128 / 255))
    assert set(torch.unique(sample["mask"]).tolist()) == {0.0, 1.0}


def test_dataset_rejects_non_binary_mask(tmp_path: Path) -> None:
    """验证非法掩码像素不会被静默转换为前景。"""
    image_relative = "images/lpw/lpw__01.png"
    mask_relative = "masks/lpw/lpw__01.png"
    write_gray_image(tmp_path / image_relative, (128, 128), 128)
    write_gray_image(tmp_path / mask_relative, (128, 128), 127)
    manifest_path = tmp_path / "manifests" / "train.csv"
    write_manifest(
        manifest_path,
        [
            {
                "sample_id": "lpw__01",
                "source": "lpw",
                "subject": "01",
                "image_path": image_relative,
                "mask_path": mask_relative,
            }
        ],
    )

    dataset = PupilSegmentationDataset(manifest_path, dataset_root=tmp_path)

    with pytest.raises(ValueError, match="非法像素值"):
        _ = dataset[0]
