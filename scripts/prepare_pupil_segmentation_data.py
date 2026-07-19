"""将 LPW 与 HMEPS 人眼数据统一为可直接训练的瞳孔分割数据。"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


# 项目根目录用于固定输入、输出的相对位置，避免依赖当前工作目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 阶段一离线基线的统一边长；128 与 HMEPS 原始人眼裁剪尺寸一致，并保留 LPW 的宽高比。
DEFAULT_IMAGE_SIZE = 128
# LPW GT 文件名由受试者、视频与帧号组成。
LPW_FILE_PATTERN = re.compile(r"(?P<subject>\d+)_(?P<sequence>\d+)_(?P<frame>\d+)")


@dataclass(frozen=True)
class Sample:
    """描述一条已通过来源筛选、但尚未变换的图像与掩码配对。"""

    sample_id: str
    source: str
    image_path: Path
    mask_path: Path
    subject: str
    sequence: str
    frame: str


def parse_args() -> argparse.Namespace:
    """读取命令行参数，并提供版本化的默认输出目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"统一后的正方形边长，默认 {DEFAULT_IMAGE_SIZE}。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / f"pupil_segmentation_{DEFAULT_IMAGE_SIZE}",
        help="处理结果目录；目录必须尚不存在或为空。",
    )
    return parser.parse_args()


def validate_paths(size: int, output_dir: Path) -> None:
    """验证目标尺寸、原始目录与输出目录，防止覆盖下载的原始数据。"""
    if size <= 0:
        raise ValueError("--size 必须为正整数。")

    required_dirs = [
        PROJECT_ROOT / "data" / "LPW" / "gt" / "imgs",
        PROJECT_ROOT / "data" / "LPW" / "gt" / "masks",
        PROJECT_ROOT / "data" / "NN_human_mouse_eyes" / "fullFrames",
        PROJECT_ROOT / "data" / "NN_human_mouse_eyes" / "annotation" / "png",
    ]
    required_files = [
        PROJECT_ROOT / "data" / "NN_human_mouse_eyes" / "annotation" / "annotations.csv",
    ]
    missing = [str(path) for path in [*required_dirs, *required_files] if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少输入数据：\n" + "\n".join(missing))

    output_dir = output_dir.resolve()
    for source_dir in required_dirs:
        if output_dir == source_dir.resolve() or source_dir.resolve() in output_dir.parents:
            raise ValueError("输出目录不能位于原始数据目录中。")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{output_dir}")


def iter_lpw_samples() -> Iterable[Sample]:
    """枚举 LPW GT 的同名 PNG 图像与掩码，并解析受试者、视频和帧号。"""
    image_dir = PROJECT_ROOT / "data" / "LPW" / "gt" / "imgs"
    mask_dir = PROJECT_ROOT / "data" / "LPW" / "gt" / "masks"

    for image_path in sorted(image_dir.glob("*.png")):
        match = LPW_FILE_PATTERN.fullmatch(image_path.stem)
        if match is None:
            raise ValueError(f"无法解析 LPW 文件名：{image_path.name}")
        mask_path = mask_dir / image_path.name
        if not mask_path.is_file():
            raise FileNotFoundError(f"LPW 缺少同名掩码：{mask_path}")
        fields = match.groupdict()
        yield Sample(
            sample_id=f"lpw__{image_path.stem}",
            source="lpw",
            image_path=image_path,
            mask_path=mask_path,
            subject=fields["subject"],
            sequence=fields["sequence"],
            frame=fields["frame"],
        )


def iter_hmep_samples() -> Iterable[Sample]:
    """枚举 HMEPS 中无眨眼、有人眼的人类样本及其红通道掩码。"""
    dataset_dir = PROJECT_ROOT / "data" / "NN_human_mouse_eyes"
    image_dir = dataset_dir / "fullFrames"
    mask_dir = dataset_dir / "annotation" / "png"
    annotation_path = dataset_dir / "annotation" / "annotations.csv"

    with annotation_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not (row["exp"] == "h" and row["eye"] == "1" and row["blink"] == "0"):
                continue
            image_path = image_dir / row["filename"]
            mask_path = mask_dir / f"{Path(row['filename']).stem}.png"
            if not image_path.is_file():
                raise FileNotFoundError(f"HMEPS 缺少原图：{image_path}")
            if not mask_path.is_file():
                raise FileNotFoundError(f"HMEPS 缺少掩码：{mask_path}")
            yield Sample(
                sample_id=f"hmep__{image_path.stem}",
                source="hmep",
                image_path=image_path,
                mask_path=mask_path,
                subject=row["sub"],
                sequence="",
                frame="",
            )


def load_binary_mask(mask_path: Path, source: str) -> Image.Image:
    """按来源读取掩码并返回像素值仅为 0 或 255 的单通道图像。"""
    with Image.open(mask_path) as raw_mask:
        if source == "lpw":
            mask = raw_mask.convert("L")
        elif source == "hmep":
            mask = raw_mask.convert("RGB").getchannel("R")
        else:
            raise ValueError(f"未知数据来源：{source}")
        return mask.point(lambda value: 255 if value > 0 else 0)


def letterbox_pair(image: Image.Image, mask: Image.Image, size: int) -> tuple[Image.Image, Image.Image, float, int, int]:
    """等比例缩放图像与掩码后居中补边，保持瞳孔几何形状不被拉伸。"""
    if image.size != mask.size:
        raise ValueError(f"图像和掩码尺寸不一致：{image.size} != {mask.size}")

    width, height = image.size
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    pad_left = (size - resized_width) // 2
    pad_top = (size - resized_height) // 2

    resized_image = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    resized_mask = mask.resize((resized_width, resized_height), Image.Resampling.NEAREST)
    output_image = Image.new("L", (size, size), color=0)
    output_mask = Image.new("L", (size, size), color=0)
    output_image.paste(resized_image, (pad_left, pad_top))
    output_mask.paste(resized_mask, (pad_left, pad_top))
    return output_image, output_mask, scale, pad_left, pad_top


def save_preview(image: Image.Image, mask: Image.Image, output_path: Path) -> None:
    """保存图像和掩码并排的灰度预览图，供人工抽检。"""
    preview = Image.new("L", (image.width * 2, image.height), color=0)
    preview.paste(image, (0, 0))
    preview.paste(mask, (image.width, 0))
    preview.save(output_path)


def relative_to_project(path: Path) -> str:
    """将绝对路径转换为相对项目根目录的跨平台清单路径。"""
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def process_samples(samples: Iterable[Sample], output_dir: Path, size: int) -> list[dict[str, object]]:
    """处理样本并写入统一 PNG、预览图与可追溯的清单记录。"""
    records: list[dict[str, object]] = []
    preview_written: set[str] = set()

    for index, sample in enumerate(samples, start=1):
        with Image.open(sample.image_path) as raw_image:
            image = raw_image.convert("L")
        mask = load_binary_mask(sample.mask_path, sample.source)
        original_width, original_height = image.size
        output_image, output_mask, scale, pad_left, pad_top = letterbox_pair(image, mask, size)

        image_relative = Path("images") / sample.source / f"{sample.sample_id}.png"
        mask_relative = Path("masks") / sample.source / f"{sample.sample_id}.png"
        image_output = output_dir / image_relative
        mask_output = output_dir / mask_relative
        image_output.parent.mkdir(parents=True, exist_ok=True)
        mask_output.parent.mkdir(parents=True, exist_ok=True)
        output_image.save(image_output, optimize=True)
        output_mask.save(mask_output, optimize=True)

        if sample.source not in preview_written:
            preview_dir = output_dir / "previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            save_preview(output_image, output_mask, preview_dir / f"{sample.source}_first_pair.png")
            preview_written.add(sample.source)

        records.append(
            {
                "sample_id": sample.sample_id,
                "source": sample.source,
                "subject": sample.subject,
                "sequence": sample.sequence,
                "frame": sample.frame,
                "image_path": image_relative.as_posix(),
                "mask_path": mask_relative.as_posix(),
                "source_image_path": relative_to_project(sample.image_path),
                "source_mask_path": relative_to_project(sample.mask_path),
                "original_width": original_width,
                "original_height": original_height,
                "processed_width": size,
                "processed_height": size,
                "scale": f"{scale:.8f}",
                "pad_left": pad_left,
                "pad_top": pad_top,
            }
        )

        if index % 500 == 0:
            print(f"已处理 {index} 张样本。")
    return records


def write_manifest(records: list[dict[str, object]], output_dir: Path) -> None:
    """写入全部样本的 CSV 清单，供后续受试者切分和训练读取。"""
    if not records:
        raise RuntimeError("没有可写入清单的样本。")
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "samples.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def write_metadata(records: list[dict[str, object]], output_dir: Path, size: int) -> None:
    """写入数据集说明与机器可读统计，记录变换规则和各来源数量。"""
    source_counts: dict[str, int] = {}
    for record in records:
        source = str(record["source"])
        source_counts[source] = source_counts.get(source, 0) + 1

    summary = {
        "target_size": [size, size],
        "image_mode": "L",
        "mask_mode": "L",
        "mask_values": [0, 255],
        "resize_policy": "等比例缩放后居中补边；图像使用双线性插值，掩码使用最近邻插值。",
        "sources": source_counts,
        "total_samples": len(records),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# 统一瞳孔分割数据\n\n"
        "- 图像：`images/<source>/<sample_id>.png`，8-bit 单通道、固定正方形尺寸。\n"
        "- 掩码：`masks/<source>/<sample_id>.png`，8-bit 单通道，像素值仅为 0（背景）或 255（瞳孔）。\n"
        "- 清单：`manifests/samples.csv`，保留来源、受试者、原始路径和几何变换参数。\n"
        "- 训练时应以 LPW 的受试者字段切分；HMEPS 只作为训练辅助数据。\n",
        encoding="utf-8",
    )


def main() -> None:
    """执行预处理，并打印可复核的输出位置和样本数量。"""
    args = parse_args()
    output_dir = args.output.resolve()
    validate_paths(args.size, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lpw_records = process_samples(iter_lpw_samples(), output_dir, args.size)
    hmep_records = process_samples(iter_hmep_samples(), output_dir, args.size)
    records = [*lpw_records, *hmep_records]
    write_manifest(records, output_dir)
    write_metadata(records, output_dir, args.size)

    print(f"处理完成：{output_dir}")
    print(f"LPW：{len(lpw_records)} 张")
    print(f"HMEPS 人眼：{len(hmep_records)} 张")
    print(f"合计：{len(records)} 张")


if __name__ == "__main__":
    main()
