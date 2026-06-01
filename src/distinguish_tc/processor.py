from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .calibration import roi_rect_from_profile
from .detector import detect_cuvette_rect
from .image_io import read_image, write_image
from .models import CropRecord, ImageSample, ProcessingResult, ProjectState, Rect, RoiProfile


def clip_rect(rect: Rect, width: int, height: int) -> Rect:
    x = max(0, min(rect.x, width - 1))
    y = max(0, min(rect.y, height - 1))
    w = max(1, min(rect.w, width - x))
    h = max(1, min(rect.h, height - y))
    return Rect(x, y, w, h)


def crop_with_rect(image_bgr: np.ndarray, rect: Rect) -> np.ndarray:
    clipped = clip_rect(rect, image_bgr.shape[1], image_bgr.shape[0])
    return image_bgr[clipped.y:clipped.bottom, clipped.x:clipped.right].copy()


def detect_crops_for_samples(
    samples: list[ImageSample],
    project_dir: Path,
    progress_callback=None,
) -> list[CropRecord]:
    detections: list[tuple[ImageSample, CropRecord, float, tuple[int, int]]] = []
    total = len(samples)

    for index, sample in enumerate(samples, start=1):
        try:
            image_bgr = read_image(sample.image_path)
            detection = detect_cuvette_rect(image_bgr)
            crop_relative_path = (
                Path("cropped_cuvettes")
                / sample.buffer_name
                / sample.group_type
                / f"{sample.relative_path.stem}.png"
            ).as_posix()

            if detection.rect is None:
                record = CropRecord(
                    image_relative_path=sample.relative_path.as_posix(),
                    auto_rect=None,
                    final_rect=None,
                    crop_relative_path=crop_relative_path,
                    auto_flagged=True,
                    manual_flagged=False,
                    flag_reason=detection.reason or "自动检测失败",
                    review_status="pending",
                )
                score = 0.0
            else:
                crop_image = crop_with_rect(image_bgr, detection.rect)
                write_image(project_dir / crop_relative_path, crop_image)
                record = CropRecord(
                    image_relative_path=sample.relative_path.as_posix(),
                    auto_rect=detection.rect,
                    final_rect=detection.rect,
                    crop_relative_path=crop_relative_path,
                    auto_flagged=False,
                    manual_flagged=False,
                    flag_reason="",
                    review_status="auto_ok",
                )
                score = detection.score

            detections.append((sample, record, score, (image_bgr.shape[1], image_bgr.shape[0])))
            if progress_callback is not None:
                progress_callback(index, total, sample, record)
        except Exception as exc:
            record = CropRecord(
                image_relative_path=sample.relative_path.as_posix(),
                auto_rect=None,
                final_rect=None,
                crop_relative_path=(
                    Path("cropped_cuvettes")
                    / sample.buffer_name
                    / sample.group_type
                    / f"{sample.relative_path.stem}.png"
                ).as_posix(),
                auto_flagged=True,
                manual_flagged=False,
                flag_reason=str(exc),
                review_status="pending",
            )
            detections.append((sample, record, 0.0, (0, 0)))
            if progress_callback is not None:
                progress_callback(index, total, sample, record)

    apply_auto_flags(detections)
    return [record for _, record, _, _ in detections]


def apply_auto_flags(detections: list[tuple[ImageSample, CropRecord, float, tuple[int, int]]]) -> None:
    valid_rects = [record.final_rect for _, record, _, _ in detections if record.final_rect is not None]
    if not valid_rects:
        return

    widths = np.array([rect.w for rect in valid_rects], dtype=np.float32)
    heights = np.array([rect.h for rect in valid_rects], dtype=np.float32)
    width_median = float(np.median(widths))
    height_median = float(np.median(heights))

    for _, record, score, image_size in detections:
        rect = record.final_rect
        if rect is None:
            continue

        reasons: list[str] = []
        if width_median > 0 and abs(rect.w - width_median) / width_median > 0.18:
            reasons.append("宽度异常")
        if height_median > 0 and abs(rect.h - height_median) / height_median > 0.18:
            reasons.append("高度异常")

        _, image_height = image_size
        if rect.y < image_height * 0.25 or rect.bottom > image_height * 0.95:
            reasons.append("过于贴近上下边界")
        if score < 0.95:
            reasons.append("检测置信偏低")

        if reasons:
            record.auto_flagged = True
            record.flag_reason = "；".join(reasons)
            record.review_status = "pending"


def refresh_single_crop(sample: ImageSample, record: CropRecord, project_dir: Path) -> None:
    if record.final_rect is None:
        return
    image_bgr = read_image(sample.image_path)
    crop_image = crop_with_rect(image_bgr, record.final_rect)
    write_image(project_dir / record.crop_relative_path, crop_image)


def roi_rect_on_crop(profile: RoiProfile, crop_image: np.ndarray) -> Rect:
    return roi_rect_from_profile(profile, crop_image.shape[1], crop_image.shape[0])


def trim_pixels(rgb_pixels: np.ndarray, trim_percent: float) -> np.ndarray:
    if rgb_pixels.size == 0 or trim_percent <= 0:
        return rgb_pixels
    intensities = rgb_pixels.mean(axis=1)
    low = np.quantile(intensities, trim_percent)
    high = np.quantile(intensities, 1 - trim_percent)
    kept = rgb_pixels[(intensities >= low) & (intensities <= high)]
    return kept if kept.size else rgb_pixels


def make_warning(pixel_count: int, kept_count: int, std_rgb: np.ndarray) -> str:
    warnings: list[str] = []
    if pixel_count == 0:
        warnings.append("取色区无像素")
    elif kept_count / pixel_count < 0.65:
        warnings.append("有效像素保留比例偏低")
    if float(np.max(std_rgb)) > 25:
        warnings.append("RGB标准差偏高")
    return "；".join(warnings)


def compute_color_statistics(kept_rgb_pixels: np.ndarray) -> dict[str, np.ndarray]:
    rgb_pixels = kept_rgb_pixels.astype(np.float32)
    rgb_uint8 = np.clip(np.rint(rgb_pixels), 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    hsv_pixels = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    lab_pixels = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    return {
        "mean_rgb": rgb_pixels.mean(axis=0),
        "std_rgb": rgb_pixels.std(axis=0),
        "mean_hsv": hsv_pixels.mean(axis=0),
        "std_hsv": hsv_pixels.std(axis=0),
        "mean_lab": lab_pixels.mean(axis=0),
        "std_lab": lab_pixels.std(axis=0),
    }


def draw_crop_preview(
    crop_bgr: np.ndarray,
    roi_rect: Rect,
    label: str,
    output_path: Path,
) -> None:
    preview = crop_bgr.copy()
    cv2.rectangle(preview, (roi_rect.x, roi_rect.y), (roi_rect.right, roi_rect.bottom), (0, 165, 255), 3)
    cv2.putText(preview, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    write_image(output_path, preview)


def process_rgb_results(
    state: ProjectState,
    progress_callback=None,
) -> list[ProcessingResult]:
    if state.roi_profile is None:
        raise ValueError("请先保存 ROI 配置")

    project_dir = Path(state.project_dir)
    sample_map = {sample.relative_path.as_posix(): sample for sample in state.samples}
    results: list[ProcessingResult] = []
    total = len(state.crops)

    for index, crop_record in enumerate(state.crops, start=1):
        sample = sample_map[crop_record.image_relative_path]
        try:
            crop_path = project_dir / crop_record.crop_relative_path
            crop_bgr = read_image(crop_path)
            roi_rect = roi_rect_on_crop(state.roi_profile, crop_bgr)
            roi_rect = clip_rect(roi_rect, crop_bgr.shape[1], crop_bgr.shape[0])
            roi_bgr = crop_bgr[roi_rect.y:roi_rect.bottom, roi_rect.x:roi_rect.right]
            roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            pixel_count = int(roi_rgb.shape[0] * roi_rgb.shape[1])
            rgb_pixels = roi_rgb.reshape(-1, 3).astype(np.float32)
            kept_pixels = trim_pixels(rgb_pixels, state.roi_profile.trim_percent)

            color_stats = compute_color_statistics(kept_pixels)
            mean_rgb = color_stats["mean_rgb"]
            std_rgb = color_stats["std_rgb"]
            mean_hsv = color_stats["mean_hsv"]
            std_hsv = color_stats["std_hsv"]
            mean_lab = color_stats["mean_lab"]
            std_lab = color_stats["std_lab"]
            kept_count = int(kept_pixels.shape[0])
            warning = make_warning(pixel_count, kept_count, std_rgb)

            preview_relative = (
                Path("previews") / sample.buffer_name / sample.group_type / f"{Path(crop_record.crop_relative_path).stem}.png"
            ).as_posix()
            draw_crop_preview(
                crop_bgr,
                roi_rect,
                (
                    f"{sample.sample_name} "
                    f"RGB={mean_rgb[0]:.0f},{mean_rgb[1]:.0f},{mean_rgb[2]:.0f} "
                    f"HSV={mean_hsv[0]:.0f},{mean_hsv[1]:.0f},{mean_hsv[2]:.0f}"
                ),
                project_dir / preview_relative,
            )

            result = ProcessingResult(
                image_path=sample.relative_path.as_posix(),
                crop_path=crop_record.crop_relative_path,
                buffer_name=sample.buffer_name,
                group_type=sample.group_type,
                sample_name=sample.sample_name,
                replicate_id=sample.replicate_id,
                tc_conc_uM=sample.tc_conc_uM,
                otc_conc_uM=sample.otc_conc_uM,
                ctc_conc_uM=sample.ctc_conc_uM,
                roi_mean_r=float(mean_rgb[0]),
                roi_mean_g=float(mean_rgb[1]),
                roi_mean_b=float(mean_rgb[2]),
                roi_std_r=float(std_rgb[0]),
                roi_std_g=float(std_rgb[1]),
                roi_std_b=float(std_rgb[2]),
                hsv_mean_h=float(mean_hsv[0]),
                hsv_mean_s=float(mean_hsv[1]),
                hsv_mean_v=float(mean_hsv[2]),
                hsv_std_h=float(std_hsv[0]),
                hsv_std_s=float(std_hsv[1]),
                hsv_std_v=float(std_hsv[2]),
                lab_mean_l=float(mean_lab[0]),
                lab_mean_a=float(mean_lab[1]),
                lab_mean_b=float(mean_lab[2]),
                lab_std_l=float(std_lab[0]),
                lab_std_a=float(std_lab[1]),
                lab_std_b=float(std_lab[2]),
                pixel_count=pixel_count,
                kept_pixel_count=kept_count,
                status="ok",
                warning=warning,
                roi_profile_name=state.roi_profile.profile_name,
                preview_path=preview_relative,
                error_message="",
            )
        except Exception as exc:
            result = ProcessingResult(
                image_path=sample.relative_path.as_posix(),
                crop_path=crop_record.crop_relative_path,
                buffer_name=sample.buffer_name,
                group_type=sample.group_type,
                sample_name=sample.sample_name,
                replicate_id=sample.replicate_id,
                tc_conc_uM=sample.tc_conc_uM,
                otc_conc_uM=sample.otc_conc_uM,
                ctc_conc_uM=sample.ctc_conc_uM,
                roi_mean_r=None,
                roi_mean_g=None,
                roi_mean_b=None,
                roi_std_r=None,
                roi_std_g=None,
                roi_std_b=None,
                hsv_mean_h=None,
                hsv_mean_s=None,
                hsv_mean_v=None,
                hsv_std_h=None,
                hsv_std_s=None,
                hsv_std_v=None,
                lab_mean_l=None,
                lab_mean_a=None,
                lab_mean_b=None,
                lab_std_l=None,
                lab_std_a=None,
                lab_std_b=None,
                pixel_count=0,
                kept_pixel_count=0,
                status="failed",
                warning="",
                roi_profile_name=state.roi_profile.profile_name,
                preview_path="",
                error_message=str(exc),
            )

        results.append(result)
        if progress_callback is not None:
            progress_callback(index, total, sample, result)

    return results


def export_results(results: list[ProcessingResult], project_dir: Path) -> Path:
    output_dir = project_dir / "rgb_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rgb_results.csv"
    frame = pd.DataFrame(asdict(result) for result in results)
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


def build_project_state(source_dir: Path, project_dir: Path, samples: list[ImageSample]) -> ProjectState:
    now = datetime.now().isoformat(timespec="seconds")
    return ProjectState(
        version=2,
        source_dir=str(source_dir),
        project_dir=str(project_dir),
        created_at=now,
        updated_at=now,
        samples=samples,
        crops=[],
        roi_profile=None,
    )
