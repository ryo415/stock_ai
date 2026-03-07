"""Label generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stock_ai.utils import ConfigError, load_config
from stock_ai.utils.io import ensure_directory, project_path, timestamp_for_filename, write_json_file


@dataclass(frozen=True)
class LabelBuildResult:
    """Result of a label generation run."""

    output_path: Path
    metadata_path: Path
    row_count: int
    labeled_row_count: int
    source_path: Path


def _latest_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern))
    if not files:
        raise ConfigError(f"No files found in {directory} matching {pattern}")
    return files[-1]


def _resolve_input_path(input_path: str | None) -> Path:
    if input_path:
        path = Path(input_path)
        if not path.is_absolute():
            path = project_path(path)
        if not path.exists():
            raise ConfigError(f"Input file not found: {path}")
        return path
    return _latest_file(project_path("data/interim/prices"), "prices_normalized_*.csv")


def _apply_threshold(series: pd.Series, operator: str, value: float) -> pd.Series:
    if operator == ">=":
        return (series >= value).astype("Int64")
    if operator == ">":
        return (series > value).astype("Int64")
    if operator == "<=":
        return (series <= value).astype("Int64")
    if operator == "<":
        return (series < value).astype("Int64")
    raise ConfigError(f"Unsupported label threshold operator: {operator}")


def build_labels(input_path: str | None = None, label_config_name: str = "labels") -> LabelBuildResult:
    """Build future return and binary labels from normalized price data."""
    source_path = _resolve_input_path(input_path)
    config = load_config("features", label_config_name)

    label_config = config["label"]
    price_reference = config["price_reference"]
    sample_definition = config["sample_definition"]

    horizon = int(label_config["horizon_business_days"])
    return_column = str(label_config["return_column"])
    operator = str(label_config["threshold"]["operator"])
    threshold_value = float(label_config["threshold"]["value"])
    base_price_column = str(price_reference["base_price_column"])
    future_price_column = str(price_reference["future_price_column"])
    drop_missing_future = bool(sample_definition.get("drop_rows_with_missing_future_price", True))

    frame = pd.read_csv(source_path)
    required_columns = {"date", "ticker", base_price_column}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ConfigError(f"Normalized price file is missing columns: {missing_columns}")

    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working["ticker"] = working["ticker"].astype(str).str.strip().str.upper()
    working[base_price_column] = pd.to_numeric(working[base_price_column], errors="coerce")
    working.sort_values(["ticker", "date"], inplace=True)

    working[future_price_column] = working.groupby("ticker")[base_price_column].shift(-horizon)
    working[return_column] = (working[future_price_column] - working[base_price_column]) / working[
        base_price_column
    ]
    working["label"] = _apply_threshold(working[return_column], operator, threshold_value)

    if drop_missing_future:
        working = working[working[future_price_column].notna()].copy()

    working["date"] = working["date"].dt.strftime("%Y-%m-%d")

    ordered_columns = [
        "date",
        "ticker",
        "market",
        "source",
        base_price_column,
        future_price_column,
        return_column,
        "label",
    ]
    optional_columns = ["close", "adjusted_close", "volume"]
    for column in optional_columns:
        if column in working.columns and column not in ordered_columns:
            ordered_columns.append(column)
    for column in ordered_columns:
        if column not in working.columns:
            working[column] = None
    output_frame = working[ordered_columns].copy()

    label_name = str(label_config["name"])
    timestamp = timestamp_for_filename()
    output_dir = ensure_directory(project_path("data/processed/labels"))
    output_path = output_dir / f"labels_{label_name}_{timestamp}.csv"
    output_frame.to_csv(output_path, index=False)

    labeled_row_count = int(output_frame["label"].notna().sum())
    metadata_path = project_path(f"data/metadata/build_labels_{timestamp}.json")
    metadata = {
        "command": "features build-labels",
        "timestamp_utc": timestamp,
        "input_path": str(source_path),
        "output_path": str(output_path),
        "label_config_name": label_config_name,
        "label_name": label_name,
        "horizon_business_days": horizon,
        "base_price_column": base_price_column,
        "future_price_column": future_price_column,
        "return_column": return_column,
        "threshold_operator": operator,
        "threshold_value": threshold_value,
        "row_count": int(len(output_frame)),
        "labeled_row_count": labeled_row_count,
    }
    write_json_file(metadata_path, metadata)

    return LabelBuildResult(
        output_path=output_path,
        metadata_path=metadata_path,
        row_count=int(len(output_frame)),
        labeled_row_count=labeled_row_count,
        source_path=source_path,
    )
