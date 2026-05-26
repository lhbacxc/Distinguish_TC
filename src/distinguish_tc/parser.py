from __future__ import annotations

import re
from pathlib import Path

from .models import ImageSample

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
GROUP_TYPES = {"blank", "single", "binary", "ternary"}

FILENAME_PATTERN = re.compile(
    r"^(?P<buffer>[A-Za-z0-9]+)_(?P<group>blank|single|binary|ternary)"
    r"(?:_(?P<concs>\d+,\d+,\d+))?"
    r"(?:_(?P<repeat>\d+))?$"
)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def parse_image_file(root_dir: Path, image_path: Path) -> ImageSample:
    stem = image_path.stem
    match = FILENAME_PATTERN.match(stem)
    if not match:
        raise ValueError(f"文件名不符合预期规则: {image_path.name}")

    group_type = match.group("group")
    conc_text = match.group("concs")
    if group_type == "blank":
        tc_conc_uM, otc_conc_uM, ctc_conc_uM = 0, 0, 0
    else:
        if not conc_text:
            raise ValueError(f"文件名缺少浓度信息: {image_path.name}")
        tc_text, otc_text, ctc_text = conc_text.split(",")
        tc_conc_uM, otc_conc_uM, ctc_conc_uM = int(tc_text), int(otc_text), int(ctc_text)

    repeat_suffix = match.group("repeat")
    replicate_id = "R1" if repeat_suffix is None else f"R{int(repeat_suffix) + 1}"
    sample_name = f"{group_type}_TC{tc_conc_uM}_OTC{otc_conc_uM}_CTC{ctc_conc_uM}"

    return ImageSample(
        image_path=image_path,
        relative_path=image_path.relative_to(root_dir),
        buffer_name=match.group("buffer"),
        group_type=group_type,
        tc_conc_uM=tc_conc_uM,
        otc_conc_uM=otc_conc_uM,
        ctc_conc_uM=ctc_conc_uM,
        replicate_id=replicate_id,
        sample_name=sample_name,
    )


def scan_images(root_dir: Path) -> list[ImageSample]:
    if not root_dir.exists():
        raise FileNotFoundError(f"目录不存在: {root_dir}")

    samples: list[ImageSample] = []
    for image_path in sorted(root_dir.rglob("*")):
        if image_path.is_file() and is_image_file(image_path):
            samples.append(parse_image_file(root_dir, image_path))

    if not samples:
        raise ValueError(f"未在目录中找到图片: {root_dir}")
    return samples
