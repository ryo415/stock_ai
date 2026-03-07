"""Model training command handlers."""

from __future__ import annotations

from typing import Any

from stock_ai.utils import ConfigError


def run_train_command(train_config_name: str, dataset_input_path: str | None = None) -> dict[str, Any]:
    """Train a baseline model and persist outputs."""
    try:
        from stock_ai.modeling.train import train_model
    except ImportError as exc:
        raise ConfigError("Training dependencies are missing. Install project dependencies first.") from exc

    result = train_model(train_config_name=train_config_name, dataset_input_path=dataset_input_path)
    return {
        "command": "train run",
        "train_config_name": train_config_name,
        "dataset_input_path": str(result.dataset_input_path),
        "model_output_path": str(result.model_output_path),
        "report_output_path": str(result.report_output_path),
        "metadata_path": str(result.metadata_path),
        "train_row_count": result.train_row_count,
        "validation_row_count": result.validation_row_count,
        "test_row_count": result.test_row_count,
    }
