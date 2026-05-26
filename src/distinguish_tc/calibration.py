from __future__ import annotations

from datetime import datetime

from .models import RoiProfile, Rect


def build_roi_profile(
    profile_name: str,
    source_crop_path: str,
    roi_rect: Rect,
    crop_width: int,
    crop_height: int,
    trim_percent: float,
) -> RoiProfile:
    return RoiProfile(
        profile_name=profile_name,
        source_crop_path=source_crop_path,
        left_ratio=roi_rect.x / crop_width,
        top_ratio=roi_rect.y / crop_height,
        right_ratio=(roi_rect.x + roi_rect.w) / crop_width,
        bottom_ratio=(roi_rect.y + roi_rect.h) / crop_height,
        trim_percent=trim_percent,
        created_at=datetime.now().isoformat(timespec="seconds"),
    )


def roi_rect_from_profile(profile: RoiProfile, crop_width: int, crop_height: int) -> Rect:
    left = int(crop_width * profile.left_ratio)
    top = int(crop_height * profile.top_ratio)
    right = int(crop_width * profile.right_ratio)
    bottom = int(crop_height * profile.bottom_ratio)
    return Rect(left, top, max(1, right - left), max(1, bottom - top))
