from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h


@dataclass(slots=True)
class ImageSample:
    image_path: Path
    relative_path: Path
    buffer_name: str
    group_type: str
    tc_conc_uM: int
    otc_conc_uM: int
    ctc_conc_uM: int
    replicate_id: str
    sample_name: str


@dataclass(slots=True)
class CropRecord:
    image_relative_path: str
    auto_rect: Rect | None
    final_rect: Rect | None
    crop_relative_path: str
    auto_flagged: bool
    manual_flagged: bool
    flag_reason: str
    review_status: str


@dataclass(slots=True)
class RoiProfile:
    profile_name: str
    source_crop_path: str
    left_ratio: float
    top_ratio: float
    right_ratio: float
    bottom_ratio: float
    trim_percent: float
    created_at: str


@dataclass(slots=True)
class ProcessingResult:
    image_path: str
    crop_path: str
    buffer_name: str
    group_type: str
    sample_name: str
    replicate_id: str
    tc_conc_uM: int
    otc_conc_uM: int
    ctc_conc_uM: int
    roi_mean_r: float | None
    roi_mean_g: float | None
    roi_mean_b: float | None
    roi_std_r: float | None
    roi_std_g: float | None
    roi_std_b: float | None
    hsv_mean_h: float | None
    hsv_mean_s: float | None
    hsv_mean_v: float | None
    hsv_std_h: float | None
    hsv_std_s: float | None
    hsv_std_v: float | None
    lab_mean_l: float | None
    lab_mean_a: float | None
    lab_mean_b: float | None
    lab_std_l: float | None
    lab_std_a: float | None
    lab_std_b: float | None
    pixel_count: int
    kept_pixel_count: int
    status: str
    warning: str
    roi_profile_name: str
    preview_path: str
    error_message: str


@dataclass(slots=True)
class ProjectState:
    version: int
    source_dir: str
    project_dir: str
    created_at: str
    updated_at: str
    samples: list[ImageSample] = field(default_factory=list)
    crops: list[CropRecord] = field(default_factory=list)
    roi_profile: RoiProfile | None = None
