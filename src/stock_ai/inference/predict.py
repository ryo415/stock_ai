"""Inference utilities for trained models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from stock_ai.modeling.train import extract_feature_importance, prepare_feature_matrix
from stock_ai.utils import ConfigError, load_config
from stock_ai.utils.io import ensure_directory, project_path, timestamp_for_filename, write_json_file


@dataclass(frozen=True)
class PredictResult:
    """Result of an inference run."""

    output_path: Path
    metadata_path: Path
    prediction_row_count: int
    prediction_date: str
    dataset_input_path: Path
    model_input_path: Path


def _latest_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern))
    if not files:
        raise ConfigError(f"No files found in {directory} matching {pattern}")
    return files[-1]


def _resolve_optional_input(input_path: str | None, default_dir: str, pattern: str) -> Path:
    if input_path:
        path = Path(input_path)
        if not path.is_absolute():
            path = project_path(path)
        if not path.exists():
            raise ConfigError(f"Input file not found: {path}")
        return path
    return _latest_file(project_path(default_dir), pattern)


def _resolve_prediction_date(frame: pd.DataFrame, configured_date: str | None, cli_date: str | None) -> str:
    requested = cli_date or configured_date
    if requested:
        return str(pd.to_datetime(requested, errors="raise").date())
    latest = pd.to_datetime(frame["date"], errors="coerce").max()
    if pd.isna(latest):
        raise ConfigError("Dataset does not contain any valid dates for inference.")
    return str(latest.date())


def _save_predictions(frame: pd.DataFrame, output_dir: Path, run_name: str, output_format: str, timestamp: str) -> Path:
    ensure_directory(output_dir)
    normalized_format = output_format.lower()
    if normalized_format == "parquet":
        try:
            output_path = output_dir / f"predict_{run_name}_{timestamp}.parquet"
            frame.to_parquet(output_path, index=False)
            return output_path
        except Exception:
            normalized_format = "csv"
    output_path = output_dir / f"predict_{run_name}_{timestamp}.csv"
    frame.to_csv(output_path, index=False)
    return output_path


def predict(
    inference_config_name: str,
    train_config_name: str,
    dataset_input_path: str | None = None,
    model_input_path: str | None = None,
    prediction_date: str | None = None,
) -> PredictResult:
    """Run inference for a specific prediction date using a trained model."""
    inference_config = load_config("inference", inference_config_name)
    _ = load_config("train", train_config_name)

    resolved_dataset_path = _resolve_optional_input(
        dataset_input_path, "data/processed/datasets", "dataset_*.csv"
    )
    resolved_model_path = _resolve_optional_input(model_input_path, "models", "*.joblib")

    artifact = joblib.load(resolved_model_path)
    pipeline = artifact.get("pipeline")
    feature_columns = artifact.get("feature_columns")
    if pipeline is None or not feature_columns:
        raise ConfigError(f"Model artifact is missing required fields: {resolved_model_path}")

    dataset = pd.read_csv(resolved_dataset_path)
    if "date" not in dataset.columns or "ticker" not in dataset.columns:
        raise ConfigError("Dataset must contain date and ticker columns for inference.")

    resolved_prediction_date = _resolve_prediction_date(
        dataset,
        configured_date=inference_config["inference"].get("prediction_date"),
        cli_date=prediction_date,
    )
    candidates = dataset[dataset["date"] == resolved_prediction_date].copy()
    if candidates.empty:
        raise ConfigError(f"No rows found for prediction date: {resolved_prediction_date}")

    missing_features = [column for column in feature_columns if column not in candidates.columns]
    if missing_features:
        raise ConfigError(f"Dataset is missing model feature columns: {missing_features}")

    x = prepare_feature_matrix(candidates, feature_columns)
    probabilities = pipeline.predict_proba(x)[:, 1]
    candidates["probability"] = probabilities
    candidates["prediction"] = (candidates["probability"] >= 0.5).astype(int)

    selection = inference_config["selection"]
    sort_by = selection.get("sort_by", "probability")
    descending = bool(selection.get("descending", True))
    top_n = int(selection.get("return_top_n", len(candidates)))
    candidates.sort_values(sort_by, ascending=not descending, inplace=True)
    candidates = candidates.head(top_n).copy()

    run_name = inference_config["inference"]["run_name"]
    output_dir = project_path(inference_config["inference"]["output_dir"])
    timestamp = timestamp_for_filename()
    output_format = inference_config["inference"].get("output_format", "csv")
    output_path = _save_predictions(candidates, output_dir, run_name, output_format, timestamp)

    metadata_path = project_path(f"data/metadata/predict_{run_name}_{timestamp}.json")
    metadata_payload: dict[str, Any] = {
        "command": "inference predict",
        "timestamp_utc": timestamp,
        "inference_config_name": inference_config_name,
        "train_config_name": train_config_name,
        "dataset_input_path": str(resolved_dataset_path),
        "model_input_path": str(resolved_model_path),
        "prediction_date": resolved_prediction_date,
        "prediction_row_count": int(len(candidates)),
        "output_path": str(output_path),
        "feature_columns": feature_columns,
    }

    if inference_config.get("explainability", {}).get("save_feature_importance"):
        feature_importance = extract_feature_importance(pipeline, feature_columns)
        if feature_importance:
            metadata_payload["feature_importance"] = feature_importance

    write_json_file(metadata_path, metadata_payload)
    return PredictResult(
        output_path=output_path,
        metadata_path=metadata_path,
        prediction_row_count=int(len(candidates)),
        prediction_date=resolved_prediction_date,
        dataset_input_path=resolved_dataset_path,
        model_input_path=resolved_model_path,
    )
