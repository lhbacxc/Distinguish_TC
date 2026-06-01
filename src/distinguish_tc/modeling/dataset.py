from __future__ import annotations

from itertools import combinations
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

LONG_COLUMNS = [
    "buffer_name",
    "feature_prefix",
    "sample_name",
    "replicate_id",
    "group_type",
    "tc_conc_uM",
    "otc_conc_uM",
    "ctc_conc_uM",
    "roi_mean_r",
    "roi_mean_g",
    "roi_mean_b",
    "roi_std_r",
    "roi_std_g",
    "roi_std_b",
    "hsv_mean_h",
    "hsv_mean_s",
    "hsv_mean_v",
    "hsv_std_h",
    "hsv_std_s",
    "hsv_std_v",
    "lab_mean_l",
    "lab_mean_a",
    "lab_mean_b",
    "lab_std_l",
    "lab_std_a",
    "lab_std_b",
    "pixel_count",
    "kept_pixel_count",
    "status",
    "warning",
    "source_csv",
    "source_image_path",
    "source_crop_path",
]

FEATURE_SUFFIX_MAP = {
    "roi_mean_r": "mean_r",
    "roi_mean_g": "mean_g",
    "roi_mean_b": "mean_b",
    "roi_std_r": "std_r",
    "roi_std_g": "std_g",
    "roi_std_b": "std_b",
    "hsv_mean_h": "hsv_mean_h",
    "hsv_mean_s": "hsv_mean_s",
    "hsv_mean_v": "hsv_mean_v",
    "hsv_std_h": "hsv_std_h",
    "hsv_std_s": "hsv_std_s",
    "hsv_std_v": "hsv_std_v",
    "lab_mean_l": "lab_mean_l",
    "lab_mean_a": "lab_mean_a",
    "lab_mean_b": "lab_mean_b",
    "lab_std_l": "lab_std_l",
    "lab_std_a": "lab_std_a",
    "lab_std_b": "lab_std_b",
}


@dataclass(slots=True)
class BufferSource:
    buffer_name: str
    feature_prefix: str
    csv_path: Path


@dataclass(slots=True)
class ModelingConfig:
    min_presence_uM: int
    test_size: float
    random_seed: int
    buffers: list[BufferSource]


def sanitize_feature_prefix(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", value.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "buffer"
    if cleaned[0].isdigit():
        cleaned = f"b_{cleaned}"
    return cleaned


def load_modeling_config(config_path: Path) -> ModelingConfig:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    buffer_sources: list[BufferSource] = []
    for item in payload.get("buffers", []):
        csv_path = (config_path.parent.parent / item["csv_path"]).resolve()
        feature_prefix = item.get("feature_prefix") or sanitize_feature_prefix(item["buffer_name"])
        buffer_sources.append(
            BufferSource(
                buffer_name=item["buffer_name"],
                feature_prefix=feature_prefix,
                csv_path=csv_path,
            )
        )
    if not buffer_sources:
        raise ValueError("配置中未包含任何缓冲结果表")
    return ModelingConfig(
        min_presence_uM=int(payload.get("min_presence_uM", 6)),
        test_size=float(payload.get("test_size", 0.2)),
        random_seed=int(payload.get("random_seed", 42)),
        buffers=buffer_sources,
    )


def _read_single_buffer_csv(source: BufferSource) -> pd.DataFrame:
    frame = pd.read_csv(source.csv_path)
    required_columns = {
        "buffer_name",
        "sample_name",
        "replicate_id",
        "group_type",
        "tc_conc_uM",
        "otc_conc_uM",
        "ctc_conc_uM",
        "roi_mean_r",
        "roi_mean_g",
        "roi_mean_b",
        "roi_std_r",
        "roi_std_g",
        "roi_std_b",
        "hsv_mean_h",
        "hsv_mean_s",
        "hsv_mean_v",
        "hsv_std_h",
        "hsv_std_s",
        "hsv_std_v",
        "lab_mean_l",
        "lab_mean_a",
        "lab_mean_b",
        "lab_std_l",
        "lab_std_a",
        "lab_std_b",
        "pixel_count",
        "kept_pixel_count",
        "status",
        "warning",
        "image_path",
        "crop_path",
    }
    missing = required_columns - set(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{source.csv_path} 缺少字段: {missing_text}")

    frame = frame.copy()
    frame["buffer_name"] = source.buffer_name
    frame["feature_prefix"] = source.feature_prefix
    frame["source_csv"] = source.csv_path.as_posix()
    frame["source_image_path"] = frame["image_path"]
    frame["source_crop_path"] = frame["crop_path"]
    return frame[LONG_COLUMNS].copy()


def build_long_table(config: ModelingConfig) -> pd.DataFrame:
    frames = [_read_single_buffer_csv(source) for source in config.buffers]
    long_frame = pd.concat(frames, ignore_index=True)
    long_frame = long_frame.sort_values(
        by=["sample_name", "replicate_id", "buffer_name", "source_image_path"],
        kind="stable",
    ).reset_index(drop=True)
    return long_frame


def build_qc_summary(long_frame: pd.DataFrame, expected_buffers: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    expected_set = set(expected_buffers)

    for (sample_name, replicate_id), group in long_frame.groupby(["sample_name", "replicate_id"], sort=True):
        group = group.reset_index(drop=True)
        first = group.iloc[0]

        available_buffers = sorted(set(group["buffer_name"]))
        ok_buffers = sorted(set(group.loc[group["status"] == "ok", "buffer_name"]))
        failed_buffers = sorted(set(group.loc[group["status"] != "ok", "buffer_name"]))
        warning_buffers = sorted(
            set(group.loc[(group["status"] == "ok") & group["warning"].fillna("").astype(str).str.strip().ne(""), "buffer_name"])
        )

        counts = group.groupby("buffer_name").size()
        duplicate_buffers = sorted(counts[counts > 1].index.tolist())
        missing_buffers = sorted(expected_set - set(ok_buffers))

        if duplicate_buffers:
            qc_status = "duplicate_buffer"
        elif failed_buffers:
            qc_status = "failed_buffer"
        elif missing_buffers:
            qc_status = "missing_buffer"
        else:
            qc_status = "ready"

        records.append(
            {
                "sample_name": sample_name,
                "replicate_id": replicate_id,
                "group_type": first["group_type"],
                "tc_conc_uM": int(first["tc_conc_uM"]),
                "otc_conc_uM": int(first["otc_conc_uM"]),
                "ctc_conc_uM": int(first["ctc_conc_uM"]),
                "expected_buffers": ",".join(expected_buffers),
                "available_buffers": ",".join(available_buffers),
                "ok_buffers": ",".join(ok_buffers),
                "missing_buffers": ",".join(missing_buffers),
                "failed_buffers": ",".join(failed_buffers),
                "warning_buffers": ",".join(warning_buffers),
                "duplicate_buffers": ",".join(duplicate_buffers),
                "qc_status": qc_status,
            }
        )

    return pd.DataFrame(records)


def build_training_table(long_frame: pd.DataFrame, qc_frame: pd.DataFrame, min_presence_uM: int) -> pd.DataFrame:
    ready_keys = set(
        zip(
            qc_frame.loc[qc_frame["qc_status"] == "ready", "sample_name"],
            qc_frame.loc[qc_frame["qc_status"] == "ready", "replicate_id"],
        )
    )

    filtered = long_frame[
        long_frame.apply(lambda row: (row["sample_name"], row["replicate_id"]) in ready_keys, axis=1)
    ].copy()

    if filtered.empty:
        raise ValueError("没有满足训练条件的完整多缓冲样本")

    index_columns = [
        "sample_name",
        "replicate_id",
        "group_type",
        "tc_conc_uM",
        "otc_conc_uM",
        "ctc_conc_uM",
    ]
    wide_base = filtered[index_columns].drop_duplicates().set_index(["sample_name", "replicate_id"])

    feature_frames: list[pd.DataFrame] = []
    for raw_column, suffix in FEATURE_SUFFIX_MAP.items():
        pivot = filtered.pivot(
            index=["sample_name", "replicate_id"],
            columns="feature_prefix",
            values=raw_column,
        )
        pivot = pivot.rename(columns={column: f"{column}_{suffix}" for column in pivot.columns})
        feature_frames.append(pivot)

    wide = pd.concat([wide_base] + feature_frames, axis=1).reset_index()
    wide = add_derived_color_features(wide)
    wide["tc_present"] = (wide["tc_conc_uM"] >= min_presence_uM).astype(int)
    wide["otc_present"] = (wide["otc_conc_uM"] >= min_presence_uM).astype(int)
    wide["ctc_present"] = (wide["ctc_conc_uM"] >= min_presence_uM).astype(int)

    public_columns = [
        "sample_name",
        "replicate_id",
        "group_type",
        "tc_conc_uM",
        "otc_conc_uM",
        "ctc_conc_uM",
        "tc_present",
        "otc_present",
        "ctc_present",
    ]
    feature_columns = sorted([column for column in wide.columns if column not in public_columns])
    return wide[public_columns + feature_columns].sort_values(by=["sample_name", "replicate_id"], kind="stable").reset_index(drop=True)


def _get_buffer_prefixes(training_frame: pd.DataFrame) -> list[str]:
    prefixes: list[str] = []
    for column in training_frame.columns:
        if column.endswith("_mean_r"):
            prefix = column[: -len("_mean_r")]
            if prefix.startswith(("hsv", "lab")):
                continue
            if prefix not in prefixes:
                prefixes.append(prefix)
    return prefixes


def add_derived_color_features(training_frame: pd.DataFrame) -> pd.DataFrame:
    frame = training_frame.copy()
    prefixes = _get_buffer_prefixes(frame)
    for prefix in prefixes:
        mean_r = f"{prefix}_mean_r"
        mean_g = f"{prefix}_mean_g"
        mean_b = f"{prefix}_mean_b"
        if {mean_r, mean_g, mean_b}.issubset(frame.columns):
            total = frame[mean_r] + frame[mean_g] + frame[mean_b]
            frame[f"{prefix}_norm_r"] = frame[mean_r] / total
            frame[f"{prefix}_norm_g"] = frame[mean_g] / total
            frame[f"{prefix}_norm_b"] = frame[mean_b] / total

    for right, left in combinations(prefixes, 2):
        for channel in ("r", "g", "b"):
            left_column = f"{left}_mean_{channel}"
            right_column = f"{right}_mean_{channel}"
            if {left_column, right_column}.issubset(frame.columns):
                frame[f"{left}_minus_{right}_mean_{channel}"] = frame[left_column] - frame[right_column]
    return frame


def get_feature_columns(training_frame: pd.DataFrame) -> list[str]:
    excluded = {
        "sample_name",
        "replicate_id",
        "group_type",
        "tc_conc_uM",
        "otc_conc_uM",
        "ctc_conc_uM",
        "tc_present",
        "otc_present",
        "ctc_present",
    }
    return [column for column in training_frame.columns if column not in excluded]


def get_default_feature_columns(training_frame: pd.DataFrame) -> list[str]:
    feature_columns = get_feature_columns(training_frame)
    selected: list[str] = []
    for column in feature_columns:
        if "_lab_" in column:
            continue
        if "_minus_" in column and "_mean_" not in column:
            continue
        if "_minus_" in column and not column.endswith(("_mean_r", "_mean_g", "_mean_b")):
            continue
        if "_hsv_" in column:
            selected.append(column)
            continue
        if column.endswith(("_mean_r", "_mean_g", "_mean_b", "_std_r", "_std_g", "_std_b")):
            selected.append(column)
            continue
        if "_norm_" in column:
            selected.append(column)
            continue
    return sorted(dict.fromkeys(selected))
