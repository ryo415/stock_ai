"""Inference command handlers."""

from __future__ import annotations

from typing import Any

from stock_ai.utils import ConfigError


def run_predict_command(
    inference_config_name: str,
    train_config_name: str,
    dataset_input_path: str | None = None,
    model_input_path: str | None = None,
    prediction_date: str | None = None,
) -> dict[str, Any]:
    """Run real inference using a trained model and processed dataset."""
    try:
        from stock_ai.inference.predict import predict
    except ImportError as exc:
        raise ConfigError("Inference dependencies are missing. Install project dependencies first.") from exc

    result = predict(
        inference_config_name=inference_config_name,
        train_config_name=train_config_name,
        dataset_input_path=dataset_input_path,
        model_input_path=model_input_path,
        prediction_date=prediction_date,
    )
    return {
        "command": "inference predict",
        "prediction_date": result.prediction_date,
        "dataset_input_path": str(result.dataset_input_path),
        "model_input_path": str(result.model_input_path),
        "output_path": str(result.output_path),
        "metadata_path": str(result.metadata_path),
        "prediction_row_count": result.prediction_row_count,
    }
