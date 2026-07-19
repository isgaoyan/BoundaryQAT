"""从固定训练清单生成受试者均衡、可复现的 PTQ 校准清单。"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage2_64_fp32_baseline.json"
DEFAULT_MANIFEST_DIR = (
    PROJECT_ROOT / "data" / "processed" / "pupil_segmentation_64" / "manifests" / "stage2"
)


def parse_args() -> argparse.Namespace:
    """读取配置和清单目录参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="包含校准规则的阶段二配置。")
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR, help="固定切分清单目录。")
    parser.add_argument("--sample-count", type=int, default=None, help="覆盖配置中的校准样本数量。")
    parser.add_argument("--output-name", type=str, default=None, help="覆盖配置中的校准清单文件名。")
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """读取 CSV 并返回字段顺序与全部记录。"""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        records = list(reader)
    if not records or "subject" not in fieldnames or "source" not in fieldnames:
        raise ValueError(f"校准来源清单为空或缺少字段：{path}")
    return fieldnames, records


def select_subject_balanced_records(
    records: list[dict[str, str]], sample_count: int, seed: int
) -> list[dict[str, str]]:
    """按受试者轮转抽取固定数量样本，使校准集覆盖训练受试者。"""
    if sample_count <= 0 or sample_count > len(records):
        raise ValueError("校准样本数必须为正且不能超过来源清单数量。")
    if any(record["source"] != "lpw" for record in records):
        raise ValueError("正式校准来源必须只包含 LPW 训练样本。")

    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[record["subject"]].append(record)
    generator = random.Random(seed)
    for subject_records in grouped.values():
        subject_records.sort(key=lambda item: item["sample_id"])
        generator.shuffle(subject_records)

    selected: list[dict[str, str]] = []
    subjects = sorted(grouped)
    while len(selected) < sample_count:
        progress = False
        for subject in subjects:
            if grouped[subject] and len(selected) < sample_count:
                selected.append(grouped[subject].pop())
                progress = True
        if not progress:
            raise RuntimeError("校准候选样本不足。")
    return selected


def write_csv(path: Path, fieldnames: list[str], records: list[dict[str, str]]) -> None:
    """写入固定字段顺序的校准清单。"""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    """根据阶段二配置生成并汇报固定校准清单。"""
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    calibration = config["calibration"]
    manifest_dir = args.manifest_dir.resolve()
    source_path = manifest_dir / calibration["source_manifest"]
    sample_count = args.sample_count if args.sample_count is not None else int(calibration["sample_count"])
    output_name = args.output_name if args.output_name is not None else str(calibration["output_manifest"])
    output_path = manifest_dir / output_name
    if output_path.exists():
        raise FileExistsError(f"校准清单已存在，拒绝覆盖：{output_path}")
    if calibration["strategy"] != "subject_balanced_fixed_seed":
        raise ValueError("当前只支持受试者均衡的固定随机抽样。")

    fieldnames, records = read_csv(source_path)
    selected = select_subject_balanced_records(records, sample_count, int(config["seed"]))
    write_csv(output_path, fieldnames, selected)
    print(f"校准清单已生成：{output_path}")
    print(f"样本数：{len(selected)}；受试者数：{len({item['subject'] for item in selected})}")


if __name__ == "__main__":
    main()
