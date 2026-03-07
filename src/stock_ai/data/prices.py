"""Price data fetching utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from stock_ai.utils import ConfigError, load_config
from stock_ai.utils.io import ensure_directory, project_path, timestamp_for_filename, write_json_file


@dataclass(frozen=True)
class PriceFetchResult:
    """Result of a price fetch operation."""

    output_path: Path
    metadata_path: Path
    row_count: int
    ticker_count: int
    provider: str
    tickers: list[str]


def _resolve_end_date(raw_end_date: str | None) -> str:
    if raw_end_date:
        return raw_end_date
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _resolve_price_tickers(
    universe_config: dict[str, Any],
    tickers_override: list[str] | None = None,
) -> list[str]:
    if tickers_override:
        return tickers_override

    ticker_selection = universe_config["universe"].get("ticker_selection", {})
    mode = str(ticker_selection.get("mode", "explicit_list"))
    if mode == "explicit_list":
        tickers = ticker_selection.get("tickers", [])
        if not tickers:
            raise ConfigError("No price tickers configured. Set universe.ticker_selection.tickers.")
        return tickers

    if mode == "liquidity_filtered":
        generated_path = ticker_selection.get("generated_universe_path")
        if generated_path:
            path = Path(generated_path)
            if not path.is_absolute():
                path = project_path(path)
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                selected_tickers = payload.get("selected_tickers")
                if selected_tickers:
                    return list(selected_tickers)
        generated_dir = project_path("data/processed/universe")
        generated_files = sorted(generated_dir.glob("liquidity_universe_*.json")) if generated_dir.exists() else []
        if generated_files:
            payload = json.loads(generated_files[-1].read_text(encoding="utf-8"))
            selected_tickers = payload.get("selected_tickers")
            if selected_tickers:
                return list(selected_tickers)

        candidate_tickers = ticker_selection.get("candidate_tickers", [])
        if candidate_tickers:
            return candidate_tickers
        raise ConfigError(
            "No liquidity-filtered universe available. Run `python -m stock_ai data build-universe` first or set candidate_tickers."
        )

    raise ConfigError(f"Unsupported ticker_selection.mode: {mode}")


def _normalize_price_frame(ticker: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output into a flat daily price table."""
    if frame.empty:
        return pd.DataFrame()

    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)

    normalized = normalized.reset_index()
    normalized.columns = [str(column).lower().replace(" ", "_") for column in normalized.columns]
    normalized.rename(
        columns={
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "adj_close": "adjusted_close",
            "volume": "volume",
            "dividends": "dividends",
            "stock_splits": "stock_splits",
        },
        inplace=True,
    )
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.strftime("%Y-%m-%d")
    normalized["ticker"] = ticker
    ordered_columns = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "dividends",
        "stock_splits",
    ]
    for column in ordered_columns:
        if column not in normalized.columns:
            normalized[column] = None
    return normalized[ordered_columns]


def fetch_price_data(
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> PriceFetchResult:
    """Fetch daily price data and persist it to data/raw/prices."""
    sources_config = load_config("data", "sources")
    universe_config = load_config("data", "universe")
    provider = sources_config["sources"]["prices"].get("provider")

    if provider != "yfinance":
        raise ConfigError(f"Unsupported price provider for current implementation: {provider}")

    resolved_tickers = _resolve_price_tickers(universe_config, tickers_override=tickers)
    resolved_start_date = start_date or universe_config["date_range"]["start_date"]
    resolved_end_date = _resolve_end_date(end_date or universe_config["date_range"]["end_date"])

    frames: list[pd.DataFrame] = []
    failed_tickers: list[str] = []
    for ticker in resolved_tickers:
        frame = yf.download(
            ticker,
            start=resolved_start_date,
            end=resolved_end_date,
            auto_adjust=False,
            actions=True,
            progress=False,
            threads=False,
        )
        normalized = _normalize_price_frame(ticker, frame)
        if normalized.empty:
            failed_tickers.append(ticker)
            continue
        frames.append(normalized)

    if not frames:
        raise ConfigError("No price data fetched. Check provider availability and ticker settings.")

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["ticker", "date"], inplace=True)

    timestamp = timestamp_for_filename()
    output_dir = ensure_directory(project_path("data/raw/prices"))
    output_path = output_dir / f"prices_yfinance_{timestamp}.csv"
    combined.to_csv(output_path, index=False)

    metadata_path = project_path(f"data/metadata/fetch_prices_{timestamp}.json")
    metadata = {
        "command": "data fetch-prices",
        "timestamp_utc": timestamp,
        "provider": provider,
        "start_date": resolved_start_date,
        "end_date": resolved_end_date,
        "requested_tickers": resolved_tickers,
        "fetched_tickers": sorted(combined["ticker"].dropna().unique().tolist()),
        "failed_tickers": failed_tickers,
        "row_count": int(len(combined)),
        "output_path": str(output_path),
    }
    write_json_file(metadata_path, metadata)

    return PriceFetchResult(
        output_path=output_path,
        metadata_path=metadata_path,
        row_count=int(len(combined)),
        ticker_count=int(combined["ticker"].nunique()),
        provider=provider,
        tickers=sorted(combined["ticker"].dropna().unique().tolist()),
    )
