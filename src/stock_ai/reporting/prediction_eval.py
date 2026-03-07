"""Utilities for evaluating historical prediction outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from stock_ai.utils import ConfigError
from stock_ai.utils.io import ensure_directory, project_path, timestamp_for_filename, write_json_file


@dataclass(frozen=True)
class PredictionEvaluationResult:
    """Result of evaluating a past prediction file."""

    report_output_path: Path
    details_output_path: Path
    metadata_path: Path
    prediction_input_path: Path
    dataset_input_path: Path
    row_count: int


def _latest_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern))
    if not files:
        raise ConfigError(f"No files found in {directory} matching {pattern}")
    return files[-1]


def _resolve_input(input_path: str | None, default_dir: str, pattern: str) -> Path:
    if input_path:
        path = Path(input_path)
        if not path.is_absolute():
            path = project_path(path)
        if not path.exists():
            raise ConfigError(f"Input file not found: {path}")
        return path
    return _latest_file(project_path(default_dir), pattern)


def _prediction_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    evaluated = frame.dropna(subset=["label", "future_return_60bd"]).copy()
    if evaluated.empty:
        return {
            "row_count": 0,
            "accuracy": float("nan"),
            "predicted_positive_count": 0,
            "realized_positive_count": 0,
            "avg_realized_future_return": float("nan"),
            "avg_realized_future_return_top5": float("nan"),
            "avg_realized_future_return_top10": float("nan"),
            "positive_rate_top5": float("nan"),
            "positive_rate_top10": float("nan"),
        }

    scored = evaluated.sort_values("probability", ascending=False).copy()
    top5 = scored.head(5)
    top10 = scored.head(10)
    return {
        "row_count": int(len(evaluated)),
        "accuracy": float((evaluated["prediction"].astype(int) == evaluated["label"].astype(int)).mean()),
        "predicted_positive_count": int(evaluated["prediction"].astype(int).sum()),
        "realized_positive_count": int(evaluated["label"].astype(int).sum()),
        "avg_realized_future_return": float(evaluated["future_return_60bd"].mean()),
        "avg_realized_future_return_top5": float(top5["future_return_60bd"].mean()) if not top5.empty else float("nan"),
        "avg_realized_future_return_top10": float(top10["future_return_60bd"].mean()) if not top10.empty else float("nan"),
        "positive_rate_top5": float(top5["label"].mean()) if not top5.empty else float("nan"),
        "positive_rate_top10": float(top10["label"].mean()) if not top10.empty else float("nan"),
    }


def evaluate_prediction(
    prediction_input_path: str | None = None,
    dataset_input_path: str | None = None,
) -> PredictionEvaluationResult:
    """Evaluate a saved prediction file against dataset labels and future returns."""
    resolved_prediction_path = _resolve_input(
        prediction_input_path,
        "reports/tables/predictions",
        "predict_default_inference_*.csv",
    )
    resolved_dataset_path = _resolve_input(dataset_input_path, "data/processed/datasets", "dataset_*.csv")

    predictions = pd.read_csv(resolved_prediction_path)
    dataset = pd.read_csv(resolved_dataset_path)

    required_prediction_columns = {"date", "ticker", "probability", "prediction"}
    missing_prediction = sorted(required_prediction_columns - set(predictions.columns))
    if missing_prediction:
        raise ConfigError(f"Prediction file is missing columns: {missing_prediction}")

    required_dataset_columns = {"date", "ticker", "label", "future_return_60bd"}
    missing_dataset = sorted(required_dataset_columns - set(dataset.columns))
    if missing_dataset:
        raise ConfigError(f"Dataset file is missing columns: {missing_dataset}")

    merged = predictions.merge(
        dataset[["date", "ticker", "label", "future_return_60bd", "adjusted_close_t_plus_60bd"]],
        on=["date", "ticker"],
        how="left",
        suffixes=("", "_dataset"),
    )
    merged["hit"] = (pd.to_numeric(merged["prediction"], errors="coerce") == pd.to_numeric(merged["label"], errors="coerce"))
    merged.sort_values("probability", ascending=False, inplace=True)

    metrics = _prediction_metrics(merged)
    timestamp = timestamp_for_filename()
    report_output_path = project_path(f"reports/tables/prediction_evaluation_{timestamp}.json")
    details_output_path = project_path(f"reports/tables/prediction_evaluation_{timestamp}.csv")
    metadata_path = project_path(f"data/metadata/prediction_evaluation_{timestamp}.json")

    ensure_directory(report_output_path.parent)
    merged.to_csv(details_output_path, index=False)

    payload = {
        "command": "report evaluate-prediction",
        "timestamp_utc": timestamp,
        "prediction_input_path": str(resolved_prediction_path),
        "dataset_input_path": str(resolved_dataset_path),
        "prediction_date": str(merged["date"].iloc[0]) if not merged.empty else None,
        "summary": metrics,
        "details_output_path": str(details_output_path),
    }
    write_json_file(report_output_path, payload)
    write_json_file(metadata_path, payload)
    return PredictionEvaluationResult(
        report_output_path=report_output_path,
        details_output_path=details_output_path,
        metadata_path=metadata_path,
        prediction_input_path=resolved_prediction_path,
        dataset_input_path=resolved_dataset_path,
        row_count=int(metrics["row_count"]),
    )
