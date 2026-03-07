"""Fundamentals data fetching via EDINET API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

from stock_ai.utils import ConfigError, load_config
from stock_ai.utils.io import ensure_directory, project_path, timestamp_for_filename, write_json_file

EDINET_BASE_URL = "https://api.edinet-fsa.go.jp/api/v2"


@dataclass(frozen=True)
class FundamentalsFetchResult:
    """Result of a fundamentals fetch operation."""

    summary_path: Path
    metadata_path: Path
    document_count: int
    downloaded_count: int
    skipped_count: int
    failed_count: int
    requested_tickers: list[str]


def _resolve_edinet_api_key(fundamentals_config: dict[str, Any]) -> str:
    env_var = fundamentals_config.get("api_key_env_var", "EDINET_API_KEY")
    api_key = os.getenv(env_var)
    if not api_key:
        raise ConfigError(f"EDINET API key is missing. Set environment variable: {env_var}")
    return api_key


def _resolve_fundamentals_tickers(
    universe_config: dict[str, Any],
    tickers_override: list[str] | None = None,
) -> list[str]:
    if tickers_override:
        return tickers_override

    tickers = universe_config["universe"].get("ticker_selection", {}).get("tickers", [])
    if not tickers:
        raise ConfigError("No tickers configured for fundamentals fetch.")
    return tickers


def _ticker_to_sec_code(ticker: str) -> str:
    base = ticker.split(".")[0]
    if not base.isdigit():
        raise ConfigError(f"Unsupported ticker format for EDINET mapping: {ticker}")
    return f"{base}0"


def _daterange(start: date, end: date) -> list[date]:
    days = (end - start).days
    return [start + timedelta(days=offset) for offset in range(days + 1)]


def _request_json(session: requests.Session, url: str, api_key: str, params: dict[str, Any]) -> dict[str, Any]:
    response = session.get(url, params=params, headers={"Subscription-Key": api_key}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if str(payload.get("metadata", {}).get("status")) != "200":
        raise ConfigError(f"EDINET API returned non-200 metadata status: {payload}")
    return payload


def _download_document(
    session: requests.Session,
    api_key: str,
    doc_id: str,
    download_type: int,
    output_path: Path,
) -> None:
    response = session.get(
        f"{EDINET_BASE_URL}/documents/{doc_id}",
        params={"type": download_type},
        headers={"Subscription-Key": api_key},
        timeout=120,
    )
    response.raise_for_status()
    ensure_directory(output_path.parent)
    output_path.write_bytes(response.content)


def fetch_fundamentals_data(
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> FundamentalsFetchResult:
    """Fetch fundamentals source documents from EDINET and persist raw files."""
    sources_config = load_config("data", "sources")
    universe_config = load_config("data", "universe")
    fundamentals_config = sources_config["sources"]["fundamentals"]
    provider = fundamentals_config.get("provider")
    if provider != "edinet":
        raise ConfigError(f"Unsupported fundamentals provider for current implementation: {provider}")

    api_key = _resolve_edinet_api_key(fundamentals_config)
    resolved_tickers = _resolve_fundamentals_tickers(universe_config, tickers_override=tickers)
    sec_codes = {_ticker_to_sec_code(ticker): ticker for ticker in resolved_tickers}

    resolved_start = date.fromisoformat(start_date or universe_config["date_range"]["start_date"])
    resolved_end = date.fromisoformat(end_date or datetime.utcnow().strftime("%Y-%m-%d"))
    if resolved_start > resolved_end:
        raise ConfigError("Fundamentals start date must be earlier than or equal to end date.")

    target_form_codes = set(fundamentals_config.get("target_form_codes", []))
    document_list_type = int(fundamentals_config.get("document_list_type", 2))
    preferred_download = fundamentals_config.get("download_preference", {}).get("primary", "csv")

    session = requests.Session()
    matched_documents: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()

    for day in _daterange(resolved_start, resolved_end):
        payload = _request_json(
            session,
            f"{EDINET_BASE_URL}/documents.json",
            api_key,
            params={"date": day.isoformat(), "type": document_list_type},
        )
        results = payload.get("results", [])
        for item in results:
            sec_code = item.get("secCode")
            doc_id = item.get("docID")
            if not sec_code or not doc_id or doc_id in seen_doc_ids:
                continue
            if sec_code not in sec_codes:
                continue
            if target_form_codes and item.get("formCode") not in target_form_codes:
                continue
            seen_doc_ids.add(doc_id)
            matched_documents.append(item)

    timestamp = timestamp_for_filename()
    raw_base_dir = ensure_directory(project_path(f"data/raw/fundamentals/edinet_{timestamp}"))

    downloaded_count = 0
    skipped_count = 0
    failed_documents: list[dict[str, Any]] = []
    downloaded_documents: list[dict[str, Any]] = []

    for item in matched_documents:
        doc_id = item["docID"]
        sec_code = item.get("secCode")
        ticker = sec_codes.get(sec_code, sec_code)
        csv_flag = item.get("csvFlag") == "1"

        if preferred_download == "csv" and csv_flag:
            download_type = 5
            suffix = ".zip"
            download_kind = "csv"
        else:
            download_type = 1
            suffix = ".zip"
            download_kind = "xbrl_zip"

        output_path = raw_base_dir / ticker / f"{doc_id}_{download_kind}{suffix}"
        try:
            _download_document(session, api_key, doc_id, download_type, output_path)
            downloaded_count += 1
            downloaded_documents.append(
                {
                    "doc_id": doc_id,
                    "ticker": ticker,
                    "sec_code": sec_code,
                    "form_code": item.get("formCode"),
                    "doc_description": item.get("docDescription"),
                    "submit_date_time": item.get("submitDateTime"),
                    "download_kind": download_kind,
                    "output_path": str(output_path),
                }
            )
        except Exception as exc:
            failed_documents.append(
                {
                    "doc_id": doc_id,
                    "ticker": ticker,
                    "error": str(exc),
                }
            )

    skipped_count = len(matched_documents) - downloaded_count - len(failed_documents)

    summary_path = project_path(f"data/raw/fundamentals/fundamentals_summary_{timestamp}.json")
    summary_payload = {
        "provider": provider,
        "start_date": resolved_start.isoformat(),
        "end_date": resolved_end.isoformat(),
        "requested_tickers": resolved_tickers,
        "document_count": len(matched_documents),
        "downloaded_count": downloaded_count,
        "skipped_count": skipped_count,
        "failed_count": len(failed_documents),
        "downloaded_documents": downloaded_documents,
        "failed_documents": failed_documents,
    }
    write_json_file(summary_path, summary_payload)

    metadata_path = project_path(f"data/metadata/fetch_fundamentals_{timestamp}.json")
    metadata_payload = {
        "command": "data fetch-fundamentals",
        "timestamp_utc": timestamp,
        "provider": provider,
        "api_key_env_var": fundamentals_config.get("api_key_env_var", "EDINET_API_KEY"),
        "start_date": resolved_start.isoformat(),
        "end_date": resolved_end.isoformat(),
        "requested_tickers": resolved_tickers,
        "matched_document_count": len(matched_documents),
        "downloaded_count": downloaded_count,
        "failed_count": len(failed_documents),
        "summary_path": str(summary_path),
    }
    write_json_file(metadata_path, metadata_payload)

    return FundamentalsFetchResult(
        summary_path=summary_path,
        metadata_path=metadata_path,
        document_count=len(matched_documents),
        downloaded_count=downloaded_count,
        skipped_count=skipped_count,
        failed_count=len(failed_documents),
        requested_tickers=resolved_tickers,
    )
