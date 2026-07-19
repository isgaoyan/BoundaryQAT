"""统一瞳孔分割数据预处理的单元测试。"""

from pathlib import Path
import sys

from PIL import Image


# 将项目根目录加入模块搜索路径，便于直接导入批处理脚本。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_pupil_segmentation_data import letterbox_pair, load_binary_mask  # noqa: E402


def test_letterbox_pair_preserves_lpw_aspect_ratio() -> None:
    """验证 640×480 的 LPW 样本缩放到 128 后保持 4:3 内容区域与二值掩码。"""
    image = Image.new("L", (640, 480), color=64)
    mask = Image.new("L", (640, 480), color=0)
    mask.paste(255, (320, 240, 400, 320))

    output_image, output_mask, scale, pad_left, pad_top = letterbox_pair(image, mask, 128)

    assert output_image.size == (128, 128)
    assert output_mask.size == (128, 128)
    assert scale == 0.2
    assert pad_left == 0
    assert pad_top == 16
    assert set(output_mask.get_flattened_data()) == {0, 255}
    assert output_mask.getbbox() == (64, 64, 80, 80)


def test_load_binary_mask_extracts_hmep_red_channel(tmp_path: Path) -> None:
    """验证 HMEPS 掩码只将红色通道的瞳孔区域转换为前景。"""
    mask_path = tmp_path / "hmep_mask.png"
    raw_mask = Image.new("RGB", (2, 1), color=(0, 0, 255))
    raw_mask.putpixel((1, 0), (255, 0, 0))
    raw_mask.save(mask_path)

    binary_mask = load_binary_mask(mask_path, "hmep")

    assert list(binary_mask.get_flattened_data()) == [0, 255]
