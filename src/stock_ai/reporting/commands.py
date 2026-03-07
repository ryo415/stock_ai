"""Reporting command handlers."""

from __future__ import annotations

from typing import Any

from stock_ai.utils import ConfigError


def run_compare_models_command(
    left_train_report_path: str | None = None,
    right_train_report_path: str | None = None,
    left_walk_forward_report_path: str | None = None,
    right_walk_forward_report_path: str | None = None,
    left_name: str = "baseline_logreg",
    right_name: str = "baseline_lightgbm",
) -> dict[str, Any]:
    """Generate a model comparison report."""
    try:
        from stock_ai.reporting.compare import compare_models
    except ImportError as exc:
        raise ConfigError("Reporting dependencies are missing. Install project dependencies first.") from exc

    result = compare_models(
        left_train_report_path=left_train_report_path,
        right_train_report_path=right_train_report_path,
        left_walk_forward_report_path=left_walk_forward_report_path,
        right_walk_forward_report_path=right_walk_forward_report_path,
        left_name=left_name,
        right_name=right_name,
    )
    return {
        "command": "report compare-models",
        "json_output_path": str(result.json_output_path),
        "markdown_output_path": str(result.markdown_output_path),
        "metadata_path": str(result.metadata_path),
    }
