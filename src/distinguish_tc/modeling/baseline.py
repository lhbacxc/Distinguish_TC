from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import numpy as np
import pandas as pd


LABEL_COLUMNS = ["tc_present", "otc_present", "ctc_present"]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(slots=True)
class BinaryLogisticRegression:
    learning_rate: float = 0.1
    epochs: int = 4000
    l2: float = 1e-3
    weights: np.ndarray | None = None
    bias: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        samples, features = x.shape
        self.weights = np.zeros(features, dtype=np.float64)
        self.bias = 0.0

        for _ in range(self.epochs):
            linear = x @ self.weights + self.bias
            probs = _sigmoid(linear)
            error = probs - y
            grad_w = (x.T @ error) / samples + self.l2 * self.weights
            grad_b = float(np.mean(error))
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("模型尚未训练")
        return _sigmoid(x @ self.weights + self.bias)


def split_groups(sample_names: np.ndarray, test_size: float, random_seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(sample_names)
    if unique_groups.size < 2:
        raise ValueError("可用于训练的 sample_name 分组不足 2 个")

    rng = np.random.default_rng(random_seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)

    test_group_count = max(1, int(math.ceil(unique_groups.size * test_size)))
    test_group_count = min(test_group_count, unique_groups.size - 1)
    test_groups = set(shuffled[:test_group_count].tolist())

    test_mask = np.array([name in test_groups for name in sample_names], dtype=bool)
    train_mask = ~test_mask
    return train_mask, test_mask


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(1, y_true.size)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def train_baseline_model(
    training_frame: pd.DataFrame,
    feature_columns: list[str],
    test_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    sample_names = training_frame["sample_name"].to_numpy()
    train_mask, test_mask = split_groups(sample_names, test_size=test_size, random_seed=random_seed)

    x = training_frame[feature_columns].to_numpy(dtype=np.float64)
    y = training_frame[LABEL_COLUMNS].to_numpy(dtype=np.float64)

    x_train, x_test = x[train_mask], x[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    x_train_scaled, x_test_scaled, mean, std = standardize_train_test(x_train, x_test)

    probability_columns: list[np.ndarray] = []
    prediction_columns: list[np.ndarray] = []
    label_metrics: dict[str, dict[str, float]] = {}

    for label_index, label_name in enumerate(LABEL_COLUMNS):
        model = BinaryLogisticRegression()
        model.fit(x_train_scaled, y_train[:, label_index])
        probabilities = model.predict_proba(x_test_scaled)
        predictions = (probabilities >= 0.5).astype(int)
        probability_columns.append(probabilities)
        prediction_columns.append(predictions)
        label_metrics[label_name] = _binary_metrics(y_test[:, label_index].astype(int), predictions.astype(int))

    probabilities_matrix = np.column_stack(probability_columns)
    predictions_matrix = np.column_stack(prediction_columns).astype(int)
    y_test_int = y_test.astype(int)

    exact_match_ratio = float(np.mean(np.all(predictions_matrix == y_test_int, axis=1)))
    hamming_loss = float(np.mean(predictions_matrix != y_test_int))
    macro_f1 = float(np.mean([metrics["f1"] for metrics in label_metrics.values()]))

    test_frame = training_frame.loc[test_mask, ["sample_name", "replicate_id", "group_type"] + LABEL_COLUMNS].copy().reset_index(drop=True)
    for label_index, label_name in enumerate(LABEL_COLUMNS):
        base_name = label_name.replace("_present", "")
        test_frame[f"pred_{label_name}"] = predictions_matrix[:, label_index]
        test_frame[f"prob_{base_name}_present"] = probabilities_matrix[:, label_index]

    metrics = {
        "model_name": "numpy_logistic_regression_ovr",
        "train_rows": int(np.sum(train_mask)),
        "test_rows": int(np.sum(test_mask)),
        "train_groups": int(len(np.unique(sample_names[train_mask]))),
        "test_groups": int(len(np.unique(sample_names[test_mask]))),
        "test_size": test_size,
        "random_seed": random_seed,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "feature_scaler_mean": mean.tolist(),
        "feature_scaler_std": std.tolist(),
        "label_metrics": label_metrics,
        "exact_match_ratio": exact_match_ratio,
        "hamming_loss": hamming_loss,
        "macro_f1": macro_f1,
    }
    return test_frame, metrics


def evaluate_repeated_group_splits(
    training_frame: pd.DataFrame,
    feature_columns: list[str],
    test_size: float,
    seed_count: int,
    random_seed_start: int = 0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if seed_count < 1:
        raise ValueError("重复分组评估次数必须至少为 1")

    records: list[dict[str, float | int]] = []
    macro_f1_values: list[float] = []
    exact_match_values: list[float] = []
    hamming_loss_values: list[float] = []

    per_label_f1_values: dict[str, list[float]] = {label: [] for label in LABEL_COLUMNS}
    per_label_precision_values: dict[str, list[float]] = {label: [] for label in LABEL_COLUMNS}
    per_label_recall_values: dict[str, list[float]] = {label: [] for label in LABEL_COLUMNS}

    for seed in range(random_seed_start, random_seed_start + seed_count):
        _, metrics = train_baseline_model(
            training_frame=training_frame,
            feature_columns=feature_columns,
            test_size=test_size,
            random_seed=seed,
        )

        row: dict[str, float | int] = {
            "random_seed": seed,
            "train_rows": int(metrics["train_rows"]),
            "test_rows": int(metrics["test_rows"]),
            "train_groups": int(metrics["train_groups"]),
            "test_groups": int(metrics["test_groups"]),
            "macro_f1": float(metrics["macro_f1"]),
            "exact_match_ratio": float(metrics["exact_match_ratio"]),
            "hamming_loss": float(metrics["hamming_loss"]),
        }
        macro_f1_values.append(float(metrics["macro_f1"]))
        exact_match_values.append(float(metrics["exact_match_ratio"]))
        hamming_loss_values.append(float(metrics["hamming_loss"]))

        for label_name in LABEL_COLUMNS:
            label_metrics = metrics["label_metrics"][label_name]
            precision = float(label_metrics["precision"])
            recall = float(label_metrics["recall"])
            f1 = float(label_metrics["f1"])
            base_name = label_name.replace("_present", "")
            row[f"{base_name}_precision"] = precision
            row[f"{base_name}_recall"] = recall
            row[f"{base_name}_f1"] = f1
            per_label_precision_values[label_name].append(precision)
            per_label_recall_values[label_name].append(recall)
            per_label_f1_values[label_name].append(f1)

        records.append(row)

    details_frame = pd.DataFrame(records).sort_values(by="random_seed", kind="stable").reset_index(drop=True)

    label_summary: dict[str, dict[str, float]] = {}
    for label_name in LABEL_COLUMNS:
        label_summary[label_name] = {
            "precision_mean": float(sum(per_label_precision_values[label_name]) / seed_count),
            "precision_std": float(statistics.pstdev(per_label_precision_values[label_name])),
            "recall_mean": float(sum(per_label_recall_values[label_name]) / seed_count),
            "recall_std": float(statistics.pstdev(per_label_recall_values[label_name])),
            "f1_mean": float(sum(per_label_f1_values[label_name]) / seed_count),
            "f1_std": float(statistics.pstdev(per_label_f1_values[label_name])),
        }

    summary = {
        "evaluation_name": "repeated_grouped_holdout",
        "seed_count": seed_count,
        "random_seed_start": random_seed_start,
        "test_size": test_size,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "macro_f1_mean": float(sum(macro_f1_values) / seed_count),
        "macro_f1_std": float(statistics.pstdev(macro_f1_values)),
        "exact_match_ratio_mean": float(sum(exact_match_values) / seed_count),
        "exact_match_ratio_std": float(statistics.pstdev(exact_match_values)),
        "hamming_loss_mean": float(sum(hamming_loss_values) / seed_count),
        "hamming_loss_std": float(statistics.pstdev(hamming_loss_values)),
        "label_metrics_summary": label_summary,
    }
    return details_frame, summary
