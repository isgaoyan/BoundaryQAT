"""按固定受试者配置生成阶段一训练、验证、测试与排除清单。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "pupil_segmentation_128" / "manifests" / "samples.csv"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage1_baseline.json"


def parse_args() -> argparse.Namespace:
    """读取切分程序参数并提供项目内的默认位置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="统一数据的 samples.csv。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="阶段一固定切分配置。")
    parser.add_argument("--output", type=Path, default=None, help="输出目录，默认位于清单旁的 stage1。")
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, object]:
    """读取并验证阶段一切分配置的基本字段。"""
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("hmep_usage") != "train_only":
        raise ValueError("阶段一首版只支持 HMEPS 进入训练集。")
    if config.get("empty_mask_policy") != "exclude":
        raise ValueError("阶段一首版只支持显式排除空掩码。")
    return config


def read_records(manifest_path: Path) -> list[dict[str, str]]:
    """读取统一样本清单并确认必需字段存在。"""
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "source", "subject", "image_path", "mask_path"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"统一清单缺少字段：{sorted(missing)}")
        records = list(reader)
    if not records:
        raise ValueError("统一样本清单为空。")
    return records


def build_subject_to_split(config: dict[str, object], records: list[dict[str, str]]) -> dict[str, str]:
    """验证 LPW 受试者集合完整且互斥，并建立受试者到集合的映射。"""
    subject_groups = config.get("lpw_subjects")
    if not isinstance(subject_groups, dict):
        raise ValueError("配置缺少 lpw_subjects。")

    mapping: dict[str, str] = {}
    for config_name, output_name in (("train", "train"), ("validation", "validation"), ("test", "test")):
        subjects = subject_groups.get(config_name)
        if not isinstance(subjects, list) or not all(isinstance(item, str) for item in subjects):
            raise ValueError(f"配置中的 {config_name} 受试者列表无效。")
        for subject in subjects:
            if subject in mapping:
                raise ValueError(f"LPW 受试者重复出现在多个集合：{subject}")
            mapping[subject] = output_name

    actual_subjects = {record["subject"] for record in records if record["source"] == "lpw"}
    configured_subjects = set(mapping)
    if actual_subjects != configured_subjects:
        raise ValueError(
            "LPW 受试者配置与清单不一致："
            f"缺少配置={sorted(actual_subjects - configured_subjects)}，"
            f"多余配置={sorted(configured_subjects - actual_subjects)}"
        )
    return mapping


def mask_is_empty(mask_path: Path) -> bool:
    """检查统一二值掩码是否完全没有瞳孔前景。"""
    with Image.open(mask_path) as mask:
        gray_mask = mask.convert("L")
        values = set(gray_mask.get_flattened_data())
        if not values.issubset({0, 255}):
            raise ValueError(f"掩码含非法像素值：{mask_path}")
        return gray_mask.getbbox() is None


def split_records(
    records: list[dict[str, str]],
    dataset_root: Path,
    subject_to_split: dict[str, str],
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    """按来源和固定受试者分配样本，并单独保留空掩码排除记录。"""
    splits = {"train": [], "validation": [], "test": []}
    excluded: list[dict[str, str]] = []
    for record in records:
        mask_path = dataset_root / record["mask_path"]
        if not mask_path.is_file():
            raise FileNotFoundError(f"清单引用的掩码不存在：{mask_path}")

        if record["source"] == "lpw":
            split_name = subject_to_split[record["subject"]]
        elif record["source"] == "hmep":
            split_name = "train"
        else:
            raise ValueError(f"未知数据来源：{record['source']}")

        output_record = {**record, "split": split_name}
        if mask_is_empty(mask_path):
            excluded.append({**output_record, "exclusion_reason": "empty_mask"})
        else:
            splits[split_name].append(output_record)
    return splits, excluded


def write_csv(records: list[dict[str, str]], output_path: Path, fieldnames: list[str]) -> None:
    """以固定字段顺序写入一份可复核的 CSV 清单。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_outputs(
    splits: dict[str, list[dict[str, str]]],
    excluded: list[dict[str, str]],
    output_dir: Path,
    config: dict[str, object],
    original_fields: list[str],
) -> None:
    """写入全部切分清单和包含来源计数、受试者名单的摘要。"""
    split_fields = [*original_fields, "split"]
    for split_name, records in splits.items():
        write_csv(records, output_dir / f"{split_name}.csv", split_fields)
    lpw_train_records = [record for record in splits["train"] if record["source"] == "lpw"]
    write_csv(lpw_train_records, output_dir / "train_lpw.csv", split_fields)
    write_csv(excluded, output_dir / "excluded.csv", [*split_fields, "exclusion_reason"])

    summary = {
        "config_name": config["name"],
        "seed": config["seed"],
        "counts": {name: len(records) for name, records in splits.items()},
        "source_counts": {
            name: dict(sorted(Counter(record["source"] for record in records).items()))
            for name, records in splits.items()
        },
        "primary_training_manifest": "train_lpw.csv",
        "primary_training_count": len(lpw_train_records),
        "auxiliary_training_manifest": "train.csv",
        "lpw_subjects": config["lpw_subjects"],
        "excluded_count": len(excluded),
        "excluded_by_source": dict(sorted(Counter(record["source"] for record in excluded).items())),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """执行固定切分并打印可人工核对的统计结果。"""
    args = parse_args()
    manifest_path = args.manifest.resolve()
    config_path = args.config.resolve()
    output_dir = args.output.resolve() if args.output else manifest_path.parent / "stage1"
    config = load_config(config_path)
    records = read_records(manifest_path)
    subject_to_split = build_subject_to_split(config, records)
    splits, excluded = split_records(records, manifest_path.parents[1], subject_to_split)
    write_outputs(splits, excluded, output_dir, config, list(records[0].keys()))

    print(f"阶段一切分已生成：{output_dir}")
    for split_name, split_items in splits.items():
        print(f"{split_name}: {len(split_items)}")
    print(f"excluded: {len(excluded)}")


if __name__ == "__main__":
    main()
