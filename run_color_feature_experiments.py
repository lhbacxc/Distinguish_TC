from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from distinguish_tc.modeling.baseline import train_baseline_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较不同颜色特征组合的建模效果。")
    parser.add_argument("--training-path", default="data_processed/training_features_wide.csv", help="训练宽表路径")
    parser.add_argument("--runs-dir", default="model_runs", help="实验结果输出目录")
    parser.add_argument("--seed-count", type=int, default=30, help="重复随机切分次数")
    return parser.parse_args()


def build_experiment_frame(frame: pd.DataFrame) -> pd.DataFrame:
    aug = frame.copy()
    prefixes = ["water", "tris8", "tris5"]
    pairs = [("tris5", "water"), ("tris8", "water"), ("tris5", "tris8")]

    for prefix in prefixes:
        r = f"{prefix}_mean_r"
        g = f"{prefix}_mean_g"
        b = f"{prefix}_mean_b"
        total = aug[r] + aug[g] + aug[b]
        aug[f"{prefix}_norm_r"] = aug[r] / total
        aug[f"{prefix}_norm_g"] = aug[g] / total
        aug[f"{prefix}_norm_b"] = aug[b] / total

    for left, right in pairs:
        for channel in ["r", "g", "b"]:
            aug[f"{left}_minus_{right}_mean_{channel}"] = aug[f"{left}_mean_{channel}"] - aug[f"{right}_mean_{channel}"]
        for suffix in ["h", "s", "v"]:
            aug[f"{left}_minus_{right}_hsv_mean_{suffix}"] = (
                aug[f"{left}_hsv_mean_{suffix}"] - aug[f"{right}_hsv_mean_{suffix}"]
            )
        for suffix in ["l", "a", "b"]:
            aug[f"{left}_minus_{right}_lab_mean_{suffix}"] = (
                aug[f"{left}_lab_mean_{suffix}"] - aug[f"{right}_lab_mean_{suffix}"]
            )
    return aug


def build_feature_cases(frame: pd.DataFrame) -> list[tuple[str, list[str]]]:
    all_columns = list(frame.columns)
    rgb_mean = sorted(
        column
        for column in all_columns
        if column.endswith(("mean_r", "mean_g", "mean_b"))
        and "hsv_" not in column
        and "lab_" not in column
        and any(column.startswith(prefix) for prefix in ("water_", "tris8_", "tris5_"))
    )
    rgb_std = sorted(
        column
        for column in all_columns
        if column.endswith(("std_r", "std_g", "std_b"))
        and "hsv_" not in column
        and "lab_" not in column
        and any(column.startswith(prefix) for prefix in ("water_", "tris8_", "tris5_"))
    )
    hsv_columns = sorted(
        column
        for column in all_columns
        if any(column.startswith(f"{prefix}_hsv_") for prefix in ("water", "tris8", "tris5")) and "_minus_" not in column
    )
    lab_columns = sorted(
        column
        for column in all_columns
        if any(column.startswith(f"{prefix}_lab_") for prefix in ("water", "tris8", "tris5")) and "_minus_" not in column
    )

    rgb_norm = sorted(column for column in all_columns if "_norm_" in column)
    rgb_delta = sorted(
        column
        for column in all_columns
        if "_minus_" in column and column.endswith(("mean_r", "mean_g", "mean_b")) and "hsv_" not in column and "lab_" not in column
    )
    hsv_delta = sorted(column for column in all_columns if "_minus_" in column and "_hsv_mean_" in column)
    lab_delta = sorted(column for column in all_columns if "_minus_" in column and "_lab_mean_" in column)

    rgb_base = sorted(rgb_mean + rgb_std)
    old_rgb_best = sorted(rgb_base + rgb_norm + rgb_delta)

    cases = [
        ("old_rgb_best", old_rgb_best),
        ("rgb_plus_hsv_raw", sorted(old_rgb_best + hsv_columns)),
        ("rgb_plus_lab_raw", sorted(old_rgb_best + lab_columns)),
        ("rgb_plus_hsv_lab_raw", sorted(old_rgb_best + hsv_columns + lab_columns)),
        ("rgb_plus_lab_raw_delta", sorted(old_rgb_best + lab_columns + lab_delta)),
        ("rgb_plus_hsv_lab_raw_delta", sorted(old_rgb_best + hsv_columns + lab_columns + hsv_delta + lab_delta)),
        ("all_raw_only", sorted(rgb_base + hsv_columns + lab_columns)),
    ]
    return [(name, sorted(dict.fromkeys(columns))) for name, columns in cases]


def evaluate_case(frame: pd.DataFrame, feature_columns: list[str], seed_count: int) -> dict[str, object]:
    macro_f1_values: list[float] = []
    exact_values: list[float] = []
    hamming_values: list[float] = []
    tc_values: list[float] = []
    otc_values: list[float] = []
    ctc_values: list[float] = []

    for seed in range(seed_count):
        _, metrics = train_baseline_model(
            training_frame=frame,
            feature_columns=feature_columns,
            test_size=0.2,
            random_seed=seed,
        )
        macro_f1_values.append(metrics["macro_f1"])
        exact_values.append(metrics["exact_match_ratio"])
        hamming_values.append(metrics["hamming_loss"])
        tc_values.append(metrics["label_metrics"]["tc_present"]["f1"])
        otc_values.append(metrics["label_metrics"]["otc_present"]["f1"])
        ctc_values.append(metrics["label_metrics"]["ctc_present"]["f1"])

    return {
        "feature_count": len(feature_columns),
        "macro_f1_mean": sum(macro_f1_values) / len(macro_f1_values),
        "macro_f1_std": statistics.pstdev(macro_f1_values),
        "exact_match_mean": sum(exact_values) / len(exact_values),
        "hamming_loss_mean": sum(hamming_values) / len(hamming_values),
        "tc_f1_mean": sum(tc_values) / len(tc_values),
        "otc_f1_mean": sum(otc_values) / len(otc_values),
        "ctc_f1_mean": sum(ctc_values) / len(ctc_values),
    }


def main() -> None:
    args = parse_args()
    training_path = (ROOT_DIR / args.training_path).resolve()
    runs_dir = (ROOT_DIR / args.runs_dir).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)

    base_frame = pd.read_csv(training_path)
    experiment_frame = build_experiment_frame(base_frame)
    cases = build_feature_cases(experiment_frame)

    summaries: list[dict[str, object]] = []
    for name, feature_columns in cases:
        summary = evaluate_case(experiment_frame, feature_columns, seed_count=args.seed_count)
        summary["name"] = name
        summary["feature_columns"] = feature_columns
        summaries.append(summary)
    summaries.sort(key=lambda item: item["macro_f1_mean"], reverse=True)

    run_dir = runs_dir / f"color_feature_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    json_path = run_dir / "summary.json"
    csv_path = run_dir / "summary.csv"

    json_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {key: value for key, value in item.items() if key != "feature_columns"}
            for item in summaries
        ]
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"training_path={training_path.as_posix()}")
    print(f"run_dir={run_dir.as_posix()}")
    print(f"summary_json={json_path.as_posix()}")
    print(f"summary_csv={csv_path.as_posix()}")


if __name__ == "__main__":
    main()
