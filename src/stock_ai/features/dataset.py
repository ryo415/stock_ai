"""Dataset building utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_ai.utils import ConfigError, load_config
from stock_ai.utils.io import ensure_directory, project_path, timestamp_for_filename, write_json_file


@dataclass(frozen=True)
class DatasetBuildResult:
    """Result of dataset generation."""

    output_path: Path
    metadata_path: Path
    row_count: int
    price_input_path: Path
    label_input_path: Path
    macro_input_path: Path | None
    fundamentals_input_path: Path | None


def _latest_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern))
    if not files:
        raise ConfigError(f"No files found in {directory} matching {pattern}")
    return files[-1]


def _resolve_input_path(input_path: str | None, default_dir: str, pattern: str) -> Path:
    if input_path:
        path = Path(input_path)
        if not path.is_absolute():
            path = project_path(path)
        if not path.exists():
            raise ConfigError(f"Input file not found: {path}")
        return path
    return _latest_file(project_path(default_dir), pattern)


def _build_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build minimal price-based features per ticker."""
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame.sort_values(["ticker", "date"], inplace=True)

    grouped_close = frame.groupby("ticker")["adjusted_close"]
    grouped_volume = frame.groupby("ticker")["volume"]

    frame["return_1d"] = grouped_close.pct_change(1)
    frame["return_5d"] = grouped_close.pct_change(5)
    frame["return_20d"] = grouped_close.pct_change(20)

    daily_return = grouped_close.pct_change(1)
    frame["volatility_20d"] = daily_return.groupby(frame["ticker"]).rolling(20).std().reset_index(
        level=0, drop=True
    )

    frame["volume_change_5d"] = grouped_volume.pct_change(5)

    ma_5 = grouped_close.transform(lambda series: series.rolling(5).mean())
    ma_20 = grouped_close.transform(lambda series: series.rolling(20).mean())
    ma_60 = grouped_close.transform(lambda series: series.rolling(60).mean())
    frame["ma_gap_5d"] = (frame["adjusted_close"] / ma_5) - 1.0
    frame["ma_gap_20d"] = (frame["adjusted_close"] / ma_20) - 1.0
    frame["ma_gap_60d"] = (frame["adjusted_close"] / ma_60) - 1.0

    return frame


def _build_macro_features(market: pd.DataFrame) -> pd.DataFrame:
    """Build daily benchmark and macro features."""
    frame = market.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["series_name"] = frame["series_name"].astype(str).str.strip().str.lower()
    frame.sort_values(["series_name", "date"], inplace=True)
    frame["return_20d"] = frame.groupby("series_name")["value"].pct_change(20)

    pivot = frame.pivot(index="date", columns="series_name", values="value").sort_index()
    returns = frame.pivot(index="date", columns="series_name", values="return_20d").sort_index()

    output = pd.DataFrame(index=pivot.index)
    if "usd_jpy" in pivot.columns:
        output["usd_jpy"] = pivot["usd_jpy"]
    if "nikkei225" in returns.columns:
        output["nikkei225_return_20d"] = returns["nikkei225"]
    if "topix" in returns.columns:
        output["topix_return_20d"] = returns["topix"]

    output["policy_rate"] = pd.NA
    output["cpi_yoy"] = pd.NA
    output = output.reset_index().rename(columns={"index": "date"})
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    return output


def build_dataset(
    feature_config_name: str = "feature_set_baseline",
    label_config_name: str = "labels",
    price_input_path: str | None = None,
    label_input_path: str | None = None,
    macro_input_path: str | None = None,
    fundamentals_input_path: str | None = None,
) -> DatasetBuildResult:
    """Build a minimal training dataset from normalized prices and generated labels."""
    feature_config = load_config("features", feature_config_name)
    label_config = load_config("features", label_config_name)

    resolved_price_input = _resolve_input_path(
        price_input_path,
        "data/interim/prices",
        "prices_normalized_*.csv",
    )
    resolved_label_input = _resolve_input_path(
        label_input_path,
        "data/processed/labels",
        "labels_*.csv",
    )
    resolved_macro_input: Path | None = None
    resolved_fundamentals_input: Path | None = None
    if feature_config["groups"].get("macro", {}).get("enabled") or feature_config["groups"].get(
        "relative_strength", {}
    ).get("enabled"):
        try:
            resolved_macro_input = _resolve_input_path(
                macro_input_path,
                "data/interim/market",
                "market_normalized_*.csv",
            )
        except ConfigError:
            resolved_macro_input = None
    if feature_config["groups"].get("fundamentals", {}).get("enabled"):
        try:
            resolved_fundamentals_input = _resolve_input_path(
                fundamentals_input_path,
                "data/interim/fundamentals",
                "fundamentals_features_*.csv",
            )
        except ConfigError:
            resolved_fundamentals_input = None

    prices = pd.read_csv(resolved_price_input)
    labels = pd.read_csv(resolved_label_input)
    macro_features = None
    if resolved_macro_input:
        macro = pd.read_csv(resolved_macro_input)
        macro_features = _build_macro_features(macro)
    fundamentals_features = None
    if resolved_fundamentals_input:
        fundamentals_features = pd.read_csv(resolved_fundamentals_input)
        fundamentals_features["date"] = pd.to_datetime(fundamentals_features["date"], errors="coerce")
        fundamentals_features["ticker"] = (
            fundamentals_features["ticker"].astype(str).str.strip().str.upper()
        )
        fundamentals_features.sort_values(["ticker", "date"], inplace=True)

    required_price_columns = {"date", "ticker", "adjusted_close", "volume"}
    missing_price_columns = sorted(required_price_columns - set(prices.columns))
    if missing_price_columns:
        raise ConfigError(f"Normalized price file is missing columns: {missing_price_columns}")

    required_label_columns = {"date", "ticker", "future_return_60bd", "label"}
    missing_label_columns = sorted(required_label_columns - set(labels.columns))
    if missing_label_columns:
        raise ConfigError(f"Label file is missing columns: {missing_label_columns}")

    price_features = _build_price_features(prices)
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce")
    labels["ticker"] = labels["ticker"].astype(str).str.strip().str.upper()
    dataset = labels.merge(
        price_features,
        on=["date", "ticker"],
        how="left",
        suffixes=("", "_price"),
    )
    if macro_features is not None:
        dataset = dataset.merge(macro_features, on="date", how="left")
        if "return_20d" in dataset.columns and "topix_return_20d" in dataset.columns:
            dataset["excess_return_vs_topix_20d"] = dataset["return_20d"] - dataset["topix_return_20d"]
        if "return_20d" in dataset.columns and "nikkei225_return_20d" in dataset.columns:
            dataset["excess_return_vs_nikkei225_20d"] = (
                dataset["return_20d"] - dataset["nikkei225_return_20d"]
            )
    if fundamentals_features is not None:
        dataset = dataset.sort_values(["date", "ticker"]).copy()
        dataset = pd.merge_asof(
            dataset,
            fundamentals_features.sort_values(["date", "ticker"]),
            on="date",
            by="ticker",
            direction="backward",
            suffixes=("", "_fundamentals"),
        )
        if "shares_outstanding" in dataset.columns:
            dataset["market_cap"] = dataset["adjusted_close"] * dataset["shares_outstanding"]
        if "eps" in dataset.columns:
            dataset["per"] = dataset["adjusted_close"] / dataset["eps"].replace(0, np.nan)
        if "book_value" in dataset.columns:
            dataset["pbr"] = dataset["market_cap"] / dataset["book_value"].replace(0, np.nan)

    feature_groups = feature_config["groups"]
    enabled_groups = [
        group_name for group_name, group_config in feature_groups.items() if group_config.get("enabled")
    ]

    # Price features are implemented now. Other groups remain reserved for later integration.
    missing_feature_columns: list[str] = []
    for group_name in ("relative_strength", "fundamentals", "macro"):
        group = feature_groups.get(group_name, {})
        if group.get("enabled"):
            for feature_name in group.get("features", []):
                if feature_name not in dataset.columns:
                    dataset[feature_name] = pd.NA
                    missing_feature_columns.append(feature_name)

    ordered_columns = [
        "date",
        "ticker",
        "market",
        "source",
        "label",
        "future_return_60bd",
        "adjusted_close",
        "adjusted_close_t_plus_60bd",
        "close",
        "volume",
    ]
    for group_name in enabled_groups:
        for feature_name in feature_groups[group_name].get("features", []):
            if feature_name not in ordered_columns:
                ordered_columns.append(feature_name)

    for column in ordered_columns:
        if column not in dataset.columns:
            dataset[column] = pd.NA
    dataset = dataset[ordered_columns].copy()
    dataset.sort_values(["ticker", "date"], inplace=True)

    threshold_ratio = float(feature_config["missing_values"]["drop_if_missing_ratio_over"])
    implemented_feature_columns = [
        "future_return_60bd",
        "adjusted_close",
        "adjusted_close_t_plus_60bd",
        "close",
        "volume",
        "return_1d",
        "return_5d",
        "return_20d",
        "volatility_20d",
        "volume_change_5d",
        "ma_gap_5d",
        "ma_gap_20d",
        "ma_gap_60d",
        "excess_return_vs_topix_20d",
        "excess_return_vs_nikkei225_20d",
        "usd_jpy",
        "nikkei225_return_20d",
        "topix_return_20d",
        "market_cap",
        "per",
        "pbr",
        "roe",
        "revenue_growth_yoy",
        "operating_margin",
    ]
    feature_columns = [column for column in implemented_feature_columns if column in dataset.columns]
    missing_ratio = dataset[feature_columns].isna().mean(axis=1)
    dataset = dataset[missing_ratio <= threshold_ratio].copy()
    dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    feature_set_name = feature_config["feature_set"]["name"]
    timestamp = timestamp_for_filename()
    output_dir = ensure_directory(project_path("data/processed/datasets"))
    output_path = output_dir / f"dataset_{feature_set_name}_{timestamp}.csv"
    dataset.to_csv(output_path, index=False)

    metadata_path = project_path(f"data/metadata/build_dataset_{timestamp}.json")
    metadata = {
        "command": "features build-dataset",
        "timestamp_utc": timestamp,
        "feature_config_name": feature_config_name,
        "label_config_name": label_config_name,
        "feature_set_name": feature_set_name,
        "price_input_path": str(resolved_price_input),
        "label_input_path": str(resolved_label_input),
        "macro_input_path": None if resolved_macro_input is None else str(resolved_macro_input),
        "fundamentals_input_path": None
        if resolved_fundamentals_input is None
        else str(resolved_fundamentals_input),
        "row_count": int(len(dataset)),
        "enabled_groups": enabled_groups,
        "missing_feature_columns": missing_feature_columns,
        "output_path": str(output_path),
    }
    write_json_file(metadata_path, metadata)

    return DatasetBuildResult(
        output_path=output_path,
        metadata_path=metadata_path,
        row_count=int(len(dataset)),
        price_input_path=resolved_price_input,
        label_input_path=resolved_label_input,
        macro_input_path=resolved_macro_input,
        fundamentals_input_path=resolved_fundamentals_input,
    )
