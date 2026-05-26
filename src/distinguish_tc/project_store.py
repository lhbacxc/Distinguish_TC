from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .models import CropRecord, ImageSample, ProjectState, Rect, RoiProfile


def _rect_from_dict(data: dict | None) -> Rect | None:
    if data is None:
        return None
    return Rect(**data)


def _sample_to_dict(sample: ImageSample) -> dict:
    return {
        "image_path": str(sample.image_path),
        "relative_path": sample.relative_path.as_posix(),
        "buffer_name": sample.buffer_name,
        "group_type": sample.group_type,
        "tc_conc_uM": sample.tc_conc_uM,
        "otc_conc_uM": sample.otc_conc_uM,
        "ctc_conc_uM": sample.ctc_conc_uM,
        "replicate_id": sample.replicate_id,
        "sample_name": sample.sample_name,
    }


def _sample_from_dict(data: dict) -> ImageSample:
    return ImageSample(
        image_path=Path(data["image_path"]),
        relative_path=Path(data["relative_path"]),
        buffer_name=data["buffer_name"],
        group_type=data["group_type"],
        tc_conc_uM=int(data["tc_conc_uM"]),
        otc_conc_uM=int(data["otc_conc_uM"]),
        ctc_conc_uM=int(data["ctc_conc_uM"]),
        replicate_id=data["replicate_id"],
        sample_name=data["sample_name"],
    )


def _crop_to_dict(crop: CropRecord) -> dict:
    return {
        "image_relative_path": crop.image_relative_path,
        "auto_rect": asdict(crop.auto_rect) if crop.auto_rect else None,
        "final_rect": asdict(crop.final_rect) if crop.final_rect else None,
        "crop_relative_path": crop.crop_relative_path,
        "auto_flagged": crop.auto_flagged,
        "manual_flagged": crop.manual_flagged,
        "flag_reason": crop.flag_reason,
        "review_status": crop.review_status,
    }


def _crop_from_dict(data: dict) -> CropRecord:
    return CropRecord(
        image_relative_path=data["image_relative_path"],
        auto_rect=_rect_from_dict(data.get("auto_rect")),
        final_rect=_rect_from_dict(data.get("final_rect")),
        crop_relative_path=data["crop_relative_path"],
        auto_flagged=bool(data.get("auto_flagged", False)),
        manual_flagged=bool(data.get("manual_flagged", False)),
        flag_reason=data.get("flag_reason", ""),
        review_status=data.get("review_status", "pending"),
    )


def _roi_to_dict(profile: RoiProfile | None) -> dict | None:
    return asdict(profile) if profile else None


def _roi_from_dict(data: dict | None) -> RoiProfile | None:
    if data is None:
        return None
    return RoiProfile(**data)


def save_project_state(state: ProjectState, project_file: Path) -> None:
    project_file.parent.mkdir(parents=True, exist_ok=True)
    state.updated_at = datetime.now().isoformat(timespec="seconds")
    payload = {
        "version": state.version,
        "source_dir": state.source_dir,
        "project_dir": state.project_dir,
        "created_at": state.created_at,
        "updated_at": state.updated_at,
        "samples": [_sample_to_dict(sample) for sample in state.samples],
        "crops": [_crop_to_dict(crop) for crop in state.crops],
        "roi_profile": _roi_to_dict(state.roi_profile),
    }
    project_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_project_state(project_file: Path) -> ProjectState:
    payload = json.loads(project_file.read_text(encoding="utf-8"))
    return ProjectState(
        version=int(payload["version"]),
        source_dir=payload["source_dir"],
        project_dir=payload["project_dir"],
        created_at=payload["created_at"],
        updated_at=payload["updated_at"],
        samples=[_sample_from_dict(item) for item in payload.get("samples", [])],
        crops=[_crop_from_dict(item) for item in payload.get("crops", [])],
        roi_profile=_roi_from_dict(payload.get("roi_profile")),
    )
