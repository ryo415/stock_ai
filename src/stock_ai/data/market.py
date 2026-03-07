"""Market and macro series fetching utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from stock_ai.utils import ConfigError, load_config
from stock_ai.utils.io import ensure_directory, project_path, timestamp_for_filename, write_json_file


@dataclass(frozen=True)
class MarketFetchResult:
    """Result of a market series fetch operation."""

    output_path: Path
    metadata_path: Path
    row_count: int
    series_count: int
    series_names: list[str]


def _resolve_end_date(raw_end_date: str | None) -> str:
    if raw_end_date:
        return raw_end_date
    return datetime.now(UTC).strftime("%Y-%m-%d")


def fetch_market_data(
    start_date: str | None = None,
    end_date: str | None = None,
) -> MarketFetchResult:
    """Fetch benchmark and FX series via yfinance."""
    sources_config = load_config("data", "sources")
    universe_config = load_config("data", "universe")
    macro_config = sources_config["sources"]["macro"]
    provider = macro_config.get("provider")
    if provider != "yfinance":
        raise ConfigError(f"Unsupported macro provider for current implementation: {provider}")

    ticker_map: dict[str, str] = macro_config.get("ticker_map", {})
    if not ticker_map:
        raise ConfigError("No macro ticker_map configured.")

    resolved_start_date = start_date or universe_config["date_range"]["start_date"]
    resolved_end_date = _resolve_end_date(end_date or universe_config["date_range"]["end_date"])

    frames: list[pd.DataFrame] = []
    failed_series: list[str] = []
    for series_name, ticker in ticker_map.items():
        frame = yf.download(
            ticker,
            start=resolved_start_date,
            end=resolved_end_date,
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
        )
        if frame.empty:
            failed_series.append(series_name)
            continue
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        normalized = frame.reset_index().copy()
        normalized.columns = [str(column).lower().replace(" ", "_") for column in normalized.columns]
        normalized["date"] = pd.to_datetime(normalized["date"]).dt.strftime("%Y-%m-%d")
        normalized["series_name"] = series_name
        normalized["ticker"] = ticker
        normalized["value"] = pd.to_numeric(normalized.get("adj_close", normalized.get("close")), errors="coerce")
        normalized["close"] = pd.to_numeric(normalized.get("close"), errors="coerce")
        normalized["open"] = pd.to_numeric(normalized.get("open"), errors="coerce")
        normalized["high"] = pd.to_numeric(normalized.get("high"), errors="coerce")
        normalized["low"] = pd.to_numeric(normalized.get("low"), errors="coerce")
        normalized["volume"] = pd.to_numeric(normalized.get("volume"), errors="coerce")
        frames.append(
            normalized[
                ["date", "series_name", "ticker", "value", "close", "open", "high", "low", "volume"]
            ]
        )

    if not frames:
        raise ConfigError("No market series fetched. Check provider availability and ticker_map settings.")

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["series_name", "date"], inplace=True)

    timestamp = timestamp_for_filename()
    output_dir = ensure_directory(project_path("data/raw/market"))
    output_path = output_dir / f"market_yfinance_{timestamp}.csv"
    combined.to_csv(output_path, index=False)

    metadata_path = project_path(f"data/metadata/fetch_macro_{timestamp}.json")
    metadata = {
        "command": "data fetch-macro",
        "timestamp_utc": timestamp,
        "provider": provider,
        "start_date": resolved_start_date,
        "end_date": resolved_end_date,
        "series_names": sorted(combined["series_name"].dropna().unique().tolist()),
        "failed_series": failed_series,
        "row_count": int(len(combined)),
        "output_path": str(output_path),
    }
    write_json_file(metadata_path, metadata)

    return MarketFetchResult(
        output_path=output_path,
        metadata_path=metadata_path,
        row_count=int(len(combined)),
        series_count=int(combined["series_name"].nunique()),
        series_names=sorted(combined["series_name"].dropna().unique().tolist()),
    )
