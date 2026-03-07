"""Data command handlers."""

from __future__ import annotations

from typing import Any

from stock_ai.utils import ConfigError, load_config
from stock_ai.utils.io import project_path, timestamp_for_filename, write_json_file


def run_normalize_prices_command(input_path: str | None = None) -> dict[str, Any]:
    """Normalize raw price data into an interim table."""
    try:
        from stock_ai.data.normalize import normalize_prices
    except ImportError as exc:
        raise ConfigError(
            "Price normalization dependencies are missing. Install project dependencies first."
        ) from exc

    result = normalize_prices(input_path=input_path)
    return {
        "command": "data normalize-prices",
        "input_path": str(result.source_path),
        "output_path": str(result.output_path),
        "metadata_path": str(result.metadata_path),
        "row_count": result.row_count,
    }


def run_normalize_fundamentals_command(input_path: str | None = None) -> dict[str, Any]:
    """Normalize raw fundamentals data into an interim table."""
    try:
        from stock_ai.data.normalize import normalize_fundamentals
    except ImportError as exc:
        raise ConfigError(
            "Fundamentals normalization dependencies are missing. Install project dependencies first."
        ) from exc

    result = normalize_fundamentals(input_path=input_path)
    return {
        "command": "data normalize-fundamentals",
        "input_path": str(result.source_path),
        "output_path": str(result.output_path),
        "metadata_path": str(result.metadata_path),
        "row_count": result.row_count,
    }


def run_normalize_macro_command(input_path: str | None = None) -> dict[str, Any]:
    """Normalize raw market data into an interim table."""
    try:
        from stock_ai.data.normalize import normalize_macro
    except ImportError as exc:
        raise ConfigError(
            "Macro normalization dependencies are missing. Install project dependencies first."
        ) from exc

    result = normalize_macro(input_path=input_path)
    return {
        "command": "data normalize-macro",
        "input_path": str(result.source_path),
        "output_path": str(result.output_path),
        "metadata_path": str(result.metadata_path),
        "row_count": result.row_count,
    }


def run_fetch_fundamentals_command(
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Fetch and persist raw fundamentals source documents."""
    try:
        from stock_ai.data.fundamentals import fetch_fundamentals_data
    except ImportError as exc:
        raise ConfigError(
            "Fundamentals fetching dependencies are missing. Install project dependencies first."
        ) from exc

    result = fetch_fundamentals_data(tickers=tickers, start_date=start_date, end_date=end_date)
    return {
        "command": "data fetch-fundamentals",
        "document_count": result.document_count,
        "downloaded_count": result.downloaded_count,
        "skipped_count": result.skipped_count,
        "failed_count": result.failed_count,
        "requested_tickers": result.requested_tickers,
        "summary_path": str(result.summary_path),
        "metadata_path": str(result.metadata_path),
    }


def run_fetch_prices_command(
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Fetch and persist actual daily price data."""
    try:
        from stock_ai.data.prices import fetch_price_data
    except ImportError as exc:
        raise ConfigError(
            "Price fetching dependencies are missing. Install project dependencies first."
        ) from exc

    result = fetch_price_data(tickers=tickers, start_date=start_date, end_date=end_date)
    return {
        "command": "data fetch-prices",
        "provider": result.provider,
        "row_count": result.row_count,
        "ticker_count": result.ticker_count,
        "tickers": result.tickers,
        "output_path": str(result.output_path),
        "metadata_path": str(result.metadata_path),
    }


def run_fetch_macro_command(
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Fetch and persist raw market/macro benchmark series."""
    try:
        from stock_ai.data.market import fetch_market_data
    except ImportError as exc:
        raise ConfigError(
            "Macro fetching dependencies are missing. Install project dependencies first."
        ) from exc

    result = fetch_market_data(start_date=start_date, end_date=end_date)
    return {
        "command": "data fetch-macro",
        "series_count": result.series_count,
        "series_names": result.series_names,
        "row_count": result.row_count,
        "output_path": str(result.output_path),
        "metadata_path": str(result.metadata_path),
    }


def run_build_universe_command(input_path: str | None = None) -> dict[str, Any]:
    """Build a liquidity-filtered universe file from price history."""
    try:
        from stock_ai.data.universe import build_liquidity_filtered_universe
    except ImportError as exc:
        raise ConfigError("Universe build dependencies are missing. Install project dependencies first.") from exc

    result = build_liquidity_filtered_universe(input_path=input_path)
    return {
        "command": "data build-universe",
        "input_path": str(result.source_path),
        "output_path": str(result.output_path),
        "metadata_path": str(result.metadata_path),
        "selected_ticker_count": result.selected_ticker_count,
    }


def run_fetch_command(dataset_kind: str) -> dict[str, Any]:
    """Create a fetch manifest for a data source kind."""
    sources_config = load_config("data", "sources")
    universe_config = load_config("data", "universe")
    timestamp = timestamp_for_filename()

    provider_config = sources_config["sources"][dataset_kind]
    output_path = project_path(f"data/metadata/fetch_{dataset_kind}_{timestamp}.json")

    payload = {
        "command": f"data fetch-{dataset_kind}",
        "timestamp_utc": timestamp,
        "dataset_kind": dataset_kind,
        "provider": provider_config.get("provider"),
        "priority": provider_config.get("priority"),
        "candidates": provider_config.get("candidates", []),
        "project_market": universe_config["project"]["market"],
        "date_range": universe_config["date_range"],
        "notes": "External fetch implementation is pending. This manifest records the planned fetch run.",
    }
    write_json_file(output_path, payload)
    return payload | {"output_path": str(output_path)}
