"""Feature command handlers."""

from __future__ import annotations

from typing import Any

from stock_ai.utils import ConfigError


def run_build_labels_command(
    input_path: str | None = None,
    label_config_name: str = "labels",
) -> dict[str, Any]:
    """Build future-return labels from normalized price data."""
    try:
        from stock_ai.features.labels import build_labels
    except ImportError as exc:
        raise ConfigError(
            "Label generation dependencies are missing. Install project dependencies first."
        ) from exc

    result = build_labels(input_path=input_path, label_config_name=label_config_name)
    return {
        "command": "features build-labels",
        "input_path": str(result.source_path),
        "output_path": str(result.output_path),
        "metadata_path": str(result.metadata_path),
        "row_count": result.row_count,
        "labeled_row_count": result.labeled_row_count,
    }


def run_build_dataset_command(
    feature_config_name: str = "feature_set_baseline",
    label_config_name: str = "labels",
    price_input_path: str | None = None,
    label_input_path: str | None = None,
    macro_input_path: str | None = None,
    fundamentals_input_path: str | None = None,
) -> dict[str, Any]:
    """Build a minimal training dataset from labels and price features."""
    try:
        from stock_ai.features.dataset import build_dataset
    except ImportError as exc:
        raise ConfigError(
            "Dataset build dependencies are missing. Install project dependencies first."
        ) from exc

    result = build_dataset(
        feature_config_name=feature_config_name,
        label_config_name=label_config_name,
        price_input_path=price_input_path,
        label_input_path=label_input_path,
        macro_input_path=macro_input_path,
        fundamentals_input_path=fundamentals_input_path,
    )
    return {
        "command": "features build-dataset",
        "price_input_path": str(result.price_input_path),
        "label_input_path": str(result.label_input_path),
        "macro_input_path": None if result.macro_input_path is None else str(result.macro_input_path),
        "fundamentals_input_path": None
        if result.fundamentals_input_path is None
        else str(result.fundamentals_input_path),
        "output_path": str(result.output_path),
        "metadata_path": str(result.metadata_path),
        "row_count": result.row_count,
    }
