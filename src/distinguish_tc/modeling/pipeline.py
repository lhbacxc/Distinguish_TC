from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import shutil

from .baseline import evaluate_repeated_group_splits, train_baseline_model
from .dataset import (
    build_long_table,
    build_qc_summary,
    build_training_table,
    get_default_feature_columns,
    load_modeling_config,
)


def run_modeling_pipeline(
    config_path: Path,
    processed_dir: Path,
    runs_dir: Path,
    overwrite_run_dir: Path | None = None,
    grouped_eval_seed_count: int = 30,
) -> dict[str, str]:
    config = load_modeling_config(config_path)
    processed_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    long_frame = build_long_table(config)
    qc_frame = build_qc_summary(long_frame, [item.buffer_name for item in config.buffers])
    training_frame = build_training_table(long_frame, qc_frame, min_presence_uM=config.min_presence_uM)
    feature_columns = get_default_feature_columns(training_frame)

    long_path = processed_dir / "buffer_rgb_long.csv"
    qc_path = processed_dir / "training_qc_summary.csv"
    training_path = processed_dir / "training_features_wide.csv"
    long_frame.to_csv(long_path, index=False, encoding="utf-8-sig")
    qc_frame.to_csv(qc_path, index=False, encoding="utf-8-sig")
    training_frame.to_csv(training_path, index=False, encoding="utf-8-sig")

    predictions_frame, metrics = train_baseline_model(
        training_frame=training_frame,
        feature_columns=feature_columns,
        test_size=config.test_size,
        random_seed=config.random_seed,
    )
    grouped_eval_frame, grouped_eval_summary = evaluate_repeated_group_splits(
        training_frame=training_frame,
        feature_columns=feature_columns,
        test_size=config.test_size,
        seed_count=grouped_eval_seed_count,
        random_seed_start=0,
    )

    metrics["buffer_names"] = [item.buffer_name for item in config.buffers]
    metrics["feature_prefixes"] = [item.feature_prefix for item in config.buffers]
    metrics["default_feature_set"] = "rgb_best_plus_hsv_raw"
    metrics["input_csv_paths"] = [item.csv_path.as_posix() for item in config.buffers]
    metrics["processed_outputs"] = {
        "buffer_rgb_long": long_path.as_posix(),
        "training_qc_summary": qc_path.as_posix(),
        "training_features_wide": training_path.as_posix(),
    }
    metrics["grouped_evaluation"] = {
        "evaluation_name": "repeated_grouped_holdout",
        "seed_count": grouped_eval_seed_count,
        "summary_file": "grouped_cv_summary.json",
        "details_file": "grouped_cv_details.csv",
    }

    if overwrite_run_dir is not None:
        run_dir = overwrite_run_dir
        if run_dir.exists():
            for child in run_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = runs_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = run_dir / "model_metrics.json"
    predictions_path = run_dir / "predictions.csv"
    features_path = run_dir / "feature_columns.txt"
    grouped_eval_json_path = run_dir / "grouped_cv_summary.json"
    grouped_eval_csv_path = run_dir / "grouped_cv_details.csv"

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions_frame.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    features_path.write_text("\n".join(feature_columns) + "\n", encoding="utf-8")
    grouped_eval_json_path.write_text(json.dumps(grouped_eval_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    grouped_eval_frame.to_csv(grouped_eval_csv_path, index=False, encoding="utf-8-sig")

    return {
        "long_path": long_path.as_posix(),
        "qc_path": qc_path.as_posix(),
        "training_path": training_path.as_posix(),
        "run_dir": run_dir.as_posix(),
        "metrics_path": metrics_path.as_posix(),
        "predictions_path": predictions_path.as_posix(),
        "features_path": features_path.as_posix(),
        "grouped_eval_json_path": grouped_eval_json_path.as_posix(),
        "grouped_eval_csv_path": grouped_eval_csv_path.as_posix(),
    }
