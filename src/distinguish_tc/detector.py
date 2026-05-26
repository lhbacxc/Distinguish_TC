from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import Rect


@dataclass(slots=True)
class DetectionResult:
    rect: Rect | None
    score: float
    reason: str


def detect_cuvette_rect(image_bgr: np.ndarray) -> DetectionResult:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), dtype=np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape[:2]
    candidates: list[tuple[float, Rect]] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 0.01 * width * height:
            continue
        aspect = h / max(w, 1)
        if aspect < 1.2:
            continue
        center_x = x + w / 2
        center_y = y + h / 2
        center_score = 1.0 - min(abs(center_x - width / 2) / (width / 2), 1.0)
        vertical_score = 1.0 - min(abs(center_y - height * 0.58) / (height * 0.58), 1.0)
        area_score = min(area / (width * height * 0.06), 1.0)
        score = aspect * 0.35 + center_score * 0.35 + vertical_score * 0.2 + area_score * 0.1
        candidates.append((score, Rect(x, y, w, h)))

    if not candidates:
        return DetectionResult(None, 0.0, "未找到符合条件的比色皿轮廓")

    best_score, best_rect = max(candidates, key=lambda item: item[0])
    return DetectionResult(best_rect, float(best_score), "")
