"""Comparison reporting utilities for model evaluation outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stock_ai.utils import ConfigError
from stock_ai.utils.io import ensure_directory, project_path, timestamp_for_filename, write_json_file


@dataclass(frozen=True)
class CompareModelsResult:
    """Result of a model comparison report generation run."""

    json_output_path: Path
    markdown_output_path: Path
    metadata_path: Path


def _latest_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern))
    if not files:
        raise ConfigError(f"No files found in {directory} matching {pattern}")
    return files[-1]


def _resolve_optional_path(input_path: str | None, default_dir: str, pattern: str) -> Path:
    if input_path:
        path = Path(input_path)
        if not path.is_absolute():
            path = project_path(path)
        if not path.exists():
            raise ConfigError(f"Input file not found: {path}")
        return path
    return _latest_file(project_path(default_dir), pattern)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_json_by_field(
    directory: str,
    pattern: str,
    field_name: str,
    expected_value: str,
) -> Path:
    candidates = sorted(project_path(directory).glob(pattern), reverse=True)
    for path in candidates:
        payload = _load_json(path)
        if payload.get(field_name) == expected_value:
            return path
    raise ConfigError(
        f"No JSON files found in {project_path(directory)} matching {pattern} with {field_name}={expected_value}"
    )


def _safe_get(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _model_summary(train_report: dict[str, Any], walk_forward_report: dict[str, Any]) -> dict[str, Any]:
    top_k = train_report.get("top_k_test_analysis", {})
    walk_summary = walk_forward_report.get("summary", {})
    return {
        "model_type": train_report.get("model_type"),
        "train_config_name": train_report.get("train_config_name"),
        "dataset_input_path": train_report.get("dataset_input_path"),
        "feature_count": len(train_report.get("feature_columns", [])),
        "validation_metrics": train_report.get("validation_metrics", {}),
        "test_metrics": train_report.get("test_metrics", {}),
        "top_k_test_analysis": top_k,
        "walk_forward_summary": walk_summary,
        "top_feature_importance": train_report.get("feature_importance", [])[:10],
    }


def _diff(left: float | int | None, right: float | int | None) -> float | None:
    if left is None or right is None:
        return None
    return float(right) - float(left)


def _format_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_markdown(comparison: dict[str, Any]) -> str:
    left_name = comparison["models"][0]["name"]
    right_name = comparison["models"][1]["name"]
    left = comparison["models"][0]["summary"]
    right = comparison["models"][1]["summary"]

    validation_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    walk_metrics = [
        "total_return",
        "benchmark_total_return",
        "avg_portfolio_return",
        "win_rate",
        "max_drawdown",
        "rebalance_count",
        "trade_count",
        "trained_window_count",
    ]

    lines = [
        "# Model Comparison Report",
        "",
        f"- Left model: `{left_name}`",
        f"- Right model: `{right_name}`",
        f"- Generated at: `{comparison['timestamp_utc']}`",
        "",
        "## Test Metrics",
        "",
        "| Metric | " + left_name + " | " + right_name + " | Delta (right-left) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in validation_metrics:
        left_value = _safe_get(left, "test_metrics", metric)
        right_value = _safe_get(right, "test_metrics", metric)
        lines.append(
            f"| {metric} | {_format_metric(left_value)} | {_format_metric(right_value)} | {_format_metric(_diff(left_value, right_value))} |"
        )

    lines.extend(
        [
            "",
            "## Walk-Forward Summary",
            "",
            "| Metric | " + left_name + " | " + right_name + " | Delta (right-left) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for metric in walk_metrics:
        left_value = _safe_get(left, "walk_forward_summary", metric)
        right_value = _safe_get(right, "walk_forward_summary", metric)
        lines.append(
            f"| {metric} | {_format_metric(left_value)} | {_format_metric(right_value)} | {_format_metric(_diff(left_value, right_value))} |"
        )

    lines.extend(
        [
            "",
            "## Top-K Test Analysis",
            "",
            "| Metric | " + left_name + " top10 | " + right_name + " top10 | " + left_name + " top30 | " + right_name + " top30 | " + left_name + " top50 | " + right_name + " top50 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for metric in ["avg_future_return", "positive_rate"]:
        lines.append(
            "| "
            + metric
            + " | "
            + " | ".join(
                _format_metric(_safe_get(summary, "top_k_test_analysis", top_k, metric))
                for top_k in ["10", "30", "50"]
                for summary in [left, right]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Walk-forward result can match across models if the investable universe is very small and both models select the same names.",
            "- Compare both classification metrics and top-k return metrics; they measure different behavior.",
        ]
    )
    return "\n".join(lines) + "\n"


def compare_models(
    left_train_report_path: str | None = None,
    right_train_report_path: str | None = None,
    left_walk_forward_report_path: str | None = None,
    right_walk_forward_report_path: str | None = None,
    left_name: str = "baseline_logreg",
    right_name: str = "baseline_lightgbm",
) -> CompareModelsResult:
    """Generate a comparison report for two model variants."""
    left_train_path = _resolve_optional_path(
        left_train_report_path,
        "reports/tables",
        "train_baseline_logreg*.json",
    )
    right_train_path = _resolve_optional_path(
        right_train_report_path,
        "reports/tables",
        "train_baseline_lightgbm*.json",
    )
    left_walk_path = None
    right_walk_path = None
    if left_walk_forward_report_path is not None:
        left_walk_path = _resolve_optional_path(
            left_walk_forward_report_path,
            "reports/tables",
            "backtest_*walk_forward*.json",
        )
    if right_walk_forward_report_path is not None:
        right_walk_path = _resolve_optional_path(
            right_walk_forward_report_path,
            "reports/tables",
            "backtest_*walk_forward*.json",
        )

    left_train = _load_json(left_train_path)
    right_train = _load_json(right_train_path)
    if left_walk_path is None:
        left_walk_path = _resolve_json_by_field(
            "reports/tables",
            "backtest_*walk_forward*.json",
            "train_config_name",
            str(left_train.get("train_config_name")),
        )
    if right_walk_path is None:
        right_walk_path = _resolve_json_by_field(
            "reports/tables",
            "backtest_*walk_forward*.json",
            "train_config_name",
            str(right_train.get("train_config_name")),
        )

    left_walk = _load_json(left_walk_path)
    right_walk = _load_json(right_walk_path)

    if left_train.get("train_config_name") == right_train.get("train_config_name"):
        raise ConfigError("Comparison inputs must point to different train reports.")

    comparison = {
        "command": "report compare-models",
        "timestamp_utc": timestamp_for_filename(),
        "models": [
            {
                "name": left_name,
                "train_report_path": str(left_train_path),
                "walk_forward_report_path": str(left_walk_path),
                "summary": _model_summary(left_train, left_walk),
            },
            {
                "name": right_name,
                "train_report_path": str(right_train_path),
                "walk_forward_report_path": str(right_walk_path),
                "summary": _model_summary(right_train, right_walk),
            },
        ],
    }

    comparison["comparison"] = {
        "test_metrics_delta_right_minus_left": {
            metric: _diff(
                _safe_get(comparison["models"][0]["summary"], "test_metrics", metric),
                _safe_get(comparison["models"][1]["summary"], "test_metrics", metric),
            )
            for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        },
        "walk_forward_delta_right_minus_left": {
            metric: _diff(
                _safe_get(comparison["models"][0]["summary"], "walk_forward_summary", metric),
                _safe_get(comparison["models"][1]["summary"], "walk_forward_summary", metric),
            )
            for metric in [
                "total_return",
                "benchmark_total_return",
                "avg_portfolio_return",
                "win_rate",
                "max_drawdown",
                "rebalance_count",
                "trade_count",
                "trained_window_count",
            ]
        },
    }

    timestamp = comparison["timestamp_utc"]
    json_output_path = project_path(f"reports/tables/model_comparison_{timestamp}.json")
    markdown_output_path = project_path(f"reports/tables/model_comparison_{timestamp}.md")
    metadata_path = project_path(f"data/metadata/model_comparison_{timestamp}.json")

    ensure_directory(json_output_path.parent)
    write_json_file(json_output_path, comparison)
    markdown_output_path.write_text(_render_markdown(comparison), encoding="utf-8")
    write_json_file(
        metadata_path,
        {
            "command": "report compare-models",
            "json_output_path": str(json_output_path),
            "markdown_output_path": str(markdown_output_path),
            "left_train_report_path": str(left_train_path),
            "right_train_report_path": str(right_train_path),
            "left_walk_forward_report_path": str(left_walk_path),
            "right_walk_forward_report_path": str(right_walk_path),
        },
    )
    return CompareModelsResult(
        json_output_path=json_output_path,
        markdown_output_path=markdown_output_path,
        metadata_path=metadata_path,
    )
