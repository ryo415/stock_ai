"""Universe construction utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from stock_ai.utils import ConfigError, load_config
from stock_ai.utils.io import ensure_directory, project_path, timestamp_for_filename, write_json_file


@dataclass(frozen=True)
class UniverseBuildResult:
    """Result of a liquidity-filtered universe build."""

    output_path: Path
    metadata_path: Path
    source_path: Path
    selected_ticker_count: int


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

    interim_dir = project_path("data/interim/prices")
    raw_dir = project_path("data/raw/prices")
    if interim_dir.exists():
        try:
            return _latest_file(interim_dir, "prices_normalized_*.csv")
        except ConfigError:
            pass
    return _latest_file(raw_dir, "prices_yfinance_*.csv")


def _normalize_price_input(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ConfigError(f"Universe build input is missing columns: {missing}")

    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["ticker"] = normalized["ticker"].astype(str).str.strip().str.upper()
    normalized["close"] = pd.to_numeric(normalized["close"], errors="coerce")
    normalized["volume"] = pd.to_numeric(normalized["volume"], errors="coerce")
    normalized = normalized.dropna(subset=["date", "ticker", "close", "volume"])
    normalized["daily_value_jpy"] = normalized["close"] * normalized["volume"]
    return normalized


def build_liquidity_filtered_universe(input_path: str | None = None) -> UniverseBuildResult:
    """Build a universe file by filtering tickers on average daily traded value."""
    universe_config = load_config("data", "universe")
    liquidity_config = universe_config["universe"].get("liquidity_filter", {})
    ticker_selection = universe_config["universe"].get("ticker_selection", {})

    if not liquidity_config.get("enabled", False):
        raise ConfigError("universe.liquidity_filter.enabled is false. Enable it before building a universe.")

    source_path = _resolve_input_path(input_path)
    frame = pd.read_csv(source_path)
    normalized = _normalize_price_input(frame)
    normalized.sort_values(["ticker", "date"], inplace=True)

    lookback_days = int(liquidity_config.get("lookback_days", 20))
    min_average_daily_value_jpy = liquidity_config.get("min_average_daily_value_jpy")
    if min_average_daily_value_jpy is None:
        raise ConfigError("Set universe.liquidity_filter.min_average_daily_value_jpy before building a universe.")
    min_average_daily_value_jpy = float(min_average_daily_value_jpy)
    min_observation_days = int(liquidity_config.get("min_observation_days", lookback_days))
    max_tickers = liquidity_config.get("max_tickers")
    mode = str(ticker_selection.get("mode", "explicit_list"))

    if mode not in {"explicit_list", "liquidity_filtered"}:
        raise ConfigError(f"Unsupported ticker_selection.mode for universe build: {mode}")

    if mode == "liquidity_filtered":
        candidate_tickers = ticker_selection.get("candidate_tickers", [])
    else:
        candidate_tickers = ticker_selection.get("tickers", [])
    if candidate_tickers:
        normalized = normalized[normalized["ticker"].isin(candidate_tickers)].copy()

    latest_dates = normalized.groupby("ticker")["date"].max().rename("latest_date")
    normalized = normalized.merge(latest_dates, on="ticker", how="left")
    lookback_cutoff = normalized["latest_date"] - pd.to_timedelta(max(lookback_days * 3, lookback_days), unit="D")
    recent = normalized[normalized["date"] >= lookback_cutoff].copy()
    recent["rank_desc"] = recent.groupby("ticker")["date"].rank(method="first", ascending=False)
    recent = recent[recent["rank_desc"] <= lookback_days].copy()

    summary = (
        recent.groupby("ticker")
        .agg(
            latest_date=("date", "max"),
            observation_days=("date", "nunique"),
            average_daily_value_jpy=("daily_value_jpy", "mean"),
            median_daily_value_jpy=("daily_value_jpy", "median"),
            average_close=("close", "mean"),
            average_volume=("volume", "mean"),
        )
        .reset_index()
    )
    summary = summary[summary["observation_days"] >= min_observation_days].copy()
    summary = summary[summary["average_daily_value_jpy"] >= min_average_daily_value_jpy].copy()
    summary.sort_values(["average_daily_value_jpy", "ticker"], ascending=[False, True], inplace=True)
    if max_tickers is not None:
        summary = summary.head(int(max_tickers)).copy()

    if summary.empty:
        raise ConfigError("Liquidity filter selected zero tickers. Lower the threshold or expand candidate tickers.")

    selected_tickers = summary["ticker"].tolist()
    summary["latest_date"] = pd.to_datetime(summary["latest_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    timestamp = timestamp_for_filename()
    output_path = project_path(f"data/processed/universe/liquidity_universe_{timestamp}.json")
    latest_output_path = project_path("data/processed/universe/latest.json")
    metadata_path = project_path(f"data/metadata/build_universe_{timestamp}.json")

    payload = {
        "command": "data build-universe",
        "timestamp_utc": timestamp,
        "mode": "liquidity_filtered",
        "source_path": str(source_path),
        "selected_tickers": selected_tickers,
        "selected_ticker_count": len(selected_tickers),
        "criteria": {
            "lookback_days": lookback_days,
            "min_average_daily_value_jpy": min_average_daily_value_jpy,
            "min_observation_days": min_observation_days,
            "max_tickers": max_tickers,
        },
        "summary": summary.to_dict(orient="records"),
    }

    ensure_directory(output_path.parent)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    output_path.write_text(serialized, encoding="utf-8")
    latest_output_path.write_text(serialized, encoding="utf-8")
    write_json_file(metadata_path, payload)
    return UniverseBuildResult(
        output_path=output_path,
        metadata_path=metadata_path,
        source_path=source_path,
        selected_ticker_count=len(selected_tickers),
    )
