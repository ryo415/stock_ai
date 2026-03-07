"""Training utilities for baseline models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stock_ai.utils import ConfigError, load_config
from stock_ai.utils.io import ensure_directory, project_path, timestamp_for_filename, write_json_file

EXCLUDED_FEATURE_COLUMNS = {
    "date",
    "ticker",
    "market",
    "source",
    "label",
    "future_return_60bd",
    "adjusted_close_t_plus_60bd",
}


@dataclass(frozen=True)
class TrainResult:
    """Result of a training run."""

    model_output_path: Path
    report_output_path: Path
    metadata_path: Path
    train_row_count: int
    validation_row_count: int
    test_row_count: int
    dataset_input_path: Path


def _latest_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern))
    if not files:
        raise ConfigError(f"No files found in {directory} matching {pattern}")
    return files[-1]


def _resolve_dataset_path(input_path: str | None) -> Path:
    if input_path:
        path = Path(input_path)
        if not path.is_absolute():
            path = project_path(path)
        if not path.exists():
            raise ConfigError(f"Dataset file not found: {path}")
        return path
    return _latest_file(project_path("data/processed/datasets"), "dataset_*.csv")


def _split_dataset(frame: pd.DataFrame, dataset_config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    train_start = pd.to_datetime(dataset_config["train_start_date"])
    validation_start = pd.to_datetime(dataset_config["validation_start_date"])
    test_start = pd.to_datetime(dataset_config["test_start_date"])

    train = frame[(frame["date"] >= train_start) & (frame["date"] < validation_start)].copy()
    validation = frame[(frame["date"] >= validation_start) & (frame["date"] < test_start)].copy()
    test = frame[frame["date"] >= test_start].copy()

    if train.empty:
        raise ConfigError("Training split is empty. Expand dataset history or adjust train config dates.")
    if validation.empty:
        raise ConfigError("Validation split is empty. Expand dataset history or adjust train config dates.")
    if test.empty:
        raise ConfigError("Test split is empty. Expand dataset history or adjust train config dates.")
    return train, validation, test


def _metric_dict(y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series) -> dict[str, float]:
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def _top_k_analysis(scored: pd.DataFrame, top_k_values: list[int]) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    sorted_frame = scored.sort_values("probability", ascending=False)
    for top_k in top_k_values:
        top = sorted_frame.head(top_k)
        if top.empty:
            results[str(top_k)] = {"avg_future_return": float("nan"), "positive_rate": float("nan")}
            continue
        results[str(top_k)] = {
            "avg_future_return": float(top["future_return_60bd"].mean()),
            "positive_rate": float(top["label"].mean()),
        }
    return results


def get_numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return usable numeric feature columns from a dataset frame."""
    candidate_features = [column for column in frame.columns if column not in EXCLUDED_FEATURE_COLUMNS]
    numeric_features: list[str] = []
    for column in candidate_features:
        numeric = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if numeric.notna().any():
            numeric_features.append(column)
    return numeric_features


def prepare_feature_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Coerce model input features to numeric values and replace infinities."""
    prepared = frame[feature_columns].copy()
    for column in feature_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return prepared


def build_training_pipeline(train_config: dict[str, Any]) -> Pipeline:
    """Create the sklearn pipeline for the configured model."""
    model_type = train_config["model"]["type"]
    params = train_config["model"]["params"]
    random_seed = int(train_config["experiment"]["random_seed"])

    if model_type == "logistic_regression":
        logistic_kwargs = {
            "C": float(params.get("c", 1.0)),
            "class_weight": params.get("class_weight"),
            "max_iter": int(params.get("max_iter", 1000)),
            "random_state": random_seed,
        }
        penalty = params.get("penalty", "l2")
        if penalty not in {None, "l2"}:
            logistic_kwargs["penalty"] = penalty

        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(**logistic_kwargs)),
            ]
        )
        pipeline.set_output(transform="pandas")
        return pipeline

    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ConfigError(
                "lightgbm is not installed. Run `pip install -e .` after adding project dependencies."
            ) from exc

        lightgbm_kwargs = {
            "objective": params.get("objective", "binary"),
            "learning_rate": float(params.get("learning_rate", 0.05)),
            "num_leaves": int(params.get("num_leaves", 31)),
            "n_estimators": int(params.get("n_estimators", 300)),
            "subsample": float(params.get("subsample", 1.0)),
            "colsample_bytree": float(params.get("colsample_bytree", 1.0)),
            "class_weight": params.get("class_weight"),
            "random_state": random_seed,
            "verbosity": int(params.get("verbosity", -1)),
        }
        max_depth = params.get("max_depth")
        if max_depth is not None:
            lightgbm_kwargs["max_depth"] = int(max_depth)
        min_child_samples = params.get("min_child_samples")
        if min_child_samples is not None:
            lightgbm_kwargs["min_child_samples"] = int(min_child_samples)
        reg_alpha = params.get("reg_alpha")
        if reg_alpha is not None:
            lightgbm_kwargs["reg_alpha"] = float(reg_alpha)
        reg_lambda = params.get("reg_lambda")
        if reg_lambda is not None:
            lightgbm_kwargs["reg_lambda"] = float(reg_lambda)

        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LGBMClassifier(**lightgbm_kwargs)),
            ]
        )
        pipeline.set_output(transform="pandas")
        return pipeline

    raise ConfigError(f"Unsupported model type: {model_type}")


def extract_feature_importance(pipeline: Pipeline, feature_columns: list[str]) -> list[dict[str, float]]:
    """Extract model-specific feature importance in a common shape."""
    model = pipeline.named_steps.get("model")
    if model is None:
        return []

    if hasattr(model, "coef_"):
        coefficients = model.coef_[0]
        return sorted(
            (
                {"feature": feature, "importance": float(coef)}
                for feature, coef in zip(feature_columns, coefficients, strict=False)
            ),
            key=lambda item: abs(item["importance"]),
            reverse=True,
        )

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return sorted(
            (
                {"feature": feature, "importance": float(value)}
                for feature, value in zip(feature_columns, importances, strict=False)
            ),
            key=lambda item: item["importance"],
            reverse=True,
        )

    return []


def train_model(train_config_name: str, dataset_input_path: str | None = None) -> TrainResult:
    """Train a configured baseline model and save outputs."""
    train_config = load_config("train", train_config_name)
    model_type = train_config["model"]["type"]

    dataset_path = _resolve_dataset_path(dataset_input_path)
    frame = pd.read_csv(dataset_path)
    train_split, validation_split, test_split = _split_dataset(frame, train_config["dataset"])

    feature_columns = [
        column for column in get_numeric_feature_columns(train_split) if pd.to_numeric(train_split[column], errors="coerce").notna().any()
    ]
    if not feature_columns:
        raise ConfigError("No usable numeric feature columns found for training.")

    pipeline = build_training_pipeline(train_config)

    x_train = prepare_feature_matrix(train_split, feature_columns)
    y_train = train_split["label"].astype(int)
    x_validation = prepare_feature_matrix(validation_split, feature_columns)
    y_validation = validation_split["label"].astype(int)
    x_test = prepare_feature_matrix(test_split, feature_columns)
    y_test = test_split["label"].astype(int)

    pipeline.fit(x_train, y_train)

    validation_proba = pd.Series(pipeline.predict_proba(x_validation)[:, 1], index=validation_split.index)
    validation_pred = (validation_proba >= 0.5).astype(int)
    test_proba = pd.Series(pipeline.predict_proba(x_test)[:, 1], index=test_split.index)
    test_pred = (test_proba >= 0.5).astype(int)

    validation_metrics = _metric_dict(y_validation, validation_pred, validation_proba)
    test_metrics = _metric_dict(y_test, test_pred, test_proba)

    top_k_values = train_config["evaluation"]["top_k_analysis"]["top_k_values"]
    scored_test = test_split[["date", "ticker", "label", "future_return_60bd"]].copy()
    scored_test["probability"] = test_proba.values
    top_k_results = _top_k_analysis(scored_test, top_k_values)

    timestamp = timestamp_for_filename()
    experiment_name = train_config["experiment"]["name"]
    model_output_path = project_path(f"models/{experiment_name}_{timestamp}.joblib")
    report_output_path = project_path(f"reports/tables/train_{experiment_name}_{timestamp}.json")
    metadata_path = project_path(f"data/metadata/train_{experiment_name}_{timestamp}.json")

    ensure_directory(model_output_path.parent)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": feature_columns,
            "train_config_name": train_config_name,
            "dataset_path": str(dataset_path),
        },
        model_output_path,
    )

    report_payload = {
        "command": "train run",
        "timestamp_utc": timestamp,
        "train_config_name": train_config_name,
        "experiment_name": experiment_name,
        "model_type": model_type,
        "dataset_input_path": str(dataset_path),
        "feature_columns": feature_columns,
        "train_row_count": int(len(train_split)),
        "validation_row_count": int(len(validation_split)),
        "test_row_count": int(len(test_split)),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "top_k_test_analysis": top_k_results,
        "model_output_path": str(model_output_path),
        "feature_importance": extract_feature_importance(pipeline, feature_columns),
    }
    write_json_file(report_output_path, report_payload)
    write_json_file(metadata_path, report_payload)

    return TrainResult(
        model_output_path=model_output_path,
        report_output_path=report_output_path,
        metadata_path=metadata_path,
        train_row_count=int(len(train_split)),
        validation_row_count=int(len(validation_split)),
        test_row_count=int(len(test_split)),
        dataset_input_path=dataset_path,
    )
