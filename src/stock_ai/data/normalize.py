"""Normalization utilities for raw datasets."""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_ai.utils import ConfigError, load_config
from stock_ai.utils.io import ensure_directory, project_path, timestamp_for_filename, write_json_file


@dataclass(frozen=True)
class NormalizeResult:
    """Result of a normalization run."""

    output_path: Path
    metadata_path: Path
    row_count: int
    source_path: Path


FUNDAMENTALS_FIELD_PATTERNS: dict[str, list[str]] = {
    "revenue": [
        r"sales",
        r"revenue",
        r"netsales",
        r"売上高",
        r"営業収益",
    ],
    "operating_income": [
        r"operatingincome",
        r"営業利益",
        r"operatingprofit",
    ],
    "net_income": [
        r"netincome",
        r"profitattributabletoownersofparent",
        r"親会社株主に帰属する当期純利益",
        r"当期純利益",
        r"profitlossattributabletoownersofparent",
    ],
    "eps": [
        r"basic.*earnings.*per.*share",
        r"basicearningspershare",
        r"earningspershare",
        r"１株当たり当期純利益",
        r"一株当たり当期純利益",
    ],
    "book_value": [
        r"netassets",
        r"equityattributabletoownersofparent",
        r"純資産",
        r"資本合計",
        r"equity",
    ],
    "shares_outstanding": [
        r"numberofissuedandoutstandingshares",
        r"issuedandoutstandingshares",
        r"発行済株式総数",
        r"期末発行済株式総数",
    ],
}


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


def _read_zipped_csv_frames(zip_path: Path) -> list[pd.DataFrame]:
    """Read all CSV files in a ZIP archive with a few tolerant fallbacks."""
    if not zip_path.exists():
        return []

    candidate_encodings = ["utf-16", "utf-16le", "utf-8-sig", "cp932", "utf-8"]
    candidate_separators = ["\t", ","]
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        for csv_name in csv_names:
            raw_bytes = zf.read(csv_name)
            parsed = None
            for encoding in candidate_encodings:
                for separator in candidate_separators:
                    try:
                        parsed = pd.read_csv(
                            BytesIO(raw_bytes),
                            encoding=encoding,
                            sep=separator,
                            dtype=str,
                            on_bad_lines="skip",
                        )
                    except Exception:
                        continue
                    if parsed is not None and parsed.shape[1] >= 2:
                        break
                if parsed is not None and parsed.shape[1] >= 2:
                    break
            if parsed is not None and not parsed.empty:
                parsed.columns = [str(column).strip() for column in parsed.columns]
                parsed["_source_csv_name"] = csv_name
                frames.append(parsed)
    return frames


def _find_column(columns: list[str], patterns: list[str]) -> str | None:
    normalized = {column: re.sub(r"\s+", "", column).lower() for column in columns}
    for column, key in normalized.items():
        if any(re.search(pattern, key) for pattern in patterns):
            return column
    return None


def _parse_numeric_value(raw_value: Any) -> float | None:
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return None
    text = str(raw_value).strip()
    if not text or text.lower() in {"nan", "none", "null", "-"}:
        return None
    text = text.replace(",", "")
    if text.startswith("(") and text.endswith(")"):
        text = f"-{text[1:-1]}"
    try:
        return float(text)
    except ValueError:
        return None


def _context_score(context_text: str, field_name: str, form_code: str | None) -> int:
    key = context_text.lower()
    score = 0
    if "consolidated" in key or "連結" in context_text:
        score += 4
    if field_name in {"revenue", "operating_income", "net_income"}:
        if "currentyearduration" in key or "currentquarterduration" in key:
            score += 4
    if field_name in {"book_value", "shares_outstanding"}:
        if "currentyearinstant" in key or "currentquarterinstant" in key:
            score += 4
    if field_name == "eps" and ("currentyearduration" in key or "currentquarterduration" in key):
        score += 4
    if form_code and form_code.startswith("043") and "quarter" in key:
        score += 2
    if form_code == "030000" and "year" in key:
        score += 2
    return score


def _extract_document_fundamentals(
    zip_path: Path,
    ticker: str,
    published_at: str | None,
    form_code: str | None,
) -> dict[str, Any]:
    """Extract a small set of fundamentals from an EDINET CSV ZIP."""
    frames = _read_zipped_csv_frames(zip_path)
    extracted: dict[str, Any] = {
        "ticker": ticker,
        "published_at": published_at,
        "form_code": form_code,
        "source_path": str(zip_path),
    }
    if not frames:
        return extracted

    candidates: dict[str, tuple[int, float]] = {}
    for frame in frames:
        columns = list(frame.columns)
        concept_column = _find_column(columns, [r"element", r"要素"])
        label_column = _find_column(columns, [r"項目名", r"label", r"labelja", r"科目"])
        value_column = _find_column(columns, [r"値", r"value"])
        context_column = _find_column(columns, [r"context", r"コンテキスト"])
        if value_column is None or (concept_column is None and label_column is None):
            continue

        for _, row in frame.iterrows():
            value = _parse_numeric_value(row.get(value_column))
            if value is None:
                continue
            concept_text = str(row.get(concept_column, "")).strip()
            label_text = str(row.get(label_column, "")).strip()
            search_text = f"{concept_text} {label_text}".lower()
            context_text = str(row.get(context_column, "")).strip()
            for field_name, patterns in FUNDAMENTALS_FIELD_PATTERNS.items():
                if field_name == "net_income" and (
                    "１株当たり" in search_text or "一株当たり" in search_text
                ):
                    continue
                if any(re.search(pattern, search_text) for pattern in patterns):
                    score = _context_score(context_text, field_name, form_code)
                    current = candidates.get(field_name)
                    if current is None or score > current[0]:
                        candidates[field_name] = (score, value)
                    break

    for field_name in FUNDAMENTALS_FIELD_PATTERNS:
        if field_name in candidates:
            extracted[field_name] = candidates[field_name][1]

    return extracted


def normalize_prices(input_path: str | None = None) -> NormalizeResult:
    """Normalize raw price CSV into an interim dataset."""
    source_path = _resolve_input_path(input_path, "data/raw/prices", "prices_yfinance_*.csv")
    universe_config = load_config("data", "universe")

    frame = pd.read_csv(source_path)
    required_columns = {
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ConfigError(f"Raw price file is missing columns: {missing_columns}")

    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    normalized["ticker"] = normalized["ticker"].astype(str).str.strip().str.upper()
    normalized["market"] = universe_config["project"]["market"]
    normalized["source"] = "yfinance"

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "dividends",
        "stock_splits",
    ]
    for column in numeric_columns:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized.dropna(subset=["date", "ticker", "close"], inplace=True)
    normalized.drop_duplicates(subset=["date", "ticker"], keep="last", inplace=True)
    normalized.sort_values(["ticker", "date"], inplace=True)

    ordered_columns = [
        "date",
        "ticker",
        "market",
        "source",
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
    normalized = normalized[ordered_columns]

    timestamp = timestamp_for_filename()
    output_dir = ensure_directory(project_path("data/interim/prices"))
    output_path = output_dir / f"prices_normalized_{timestamp}.csv"
    normalized.to_csv(output_path, index=False)

    metadata_path = project_path(f"data/metadata/normalize_prices_{timestamp}.json")
    metadata = {
        "command": "data normalize-prices",
        "timestamp_utc": timestamp,
        "input_path": str(source_path),
        "output_path": str(output_path),
        "row_count": int(len(normalized)),
        "ticker_count": int(normalized["ticker"].nunique()),
        "date_min": None if normalized.empty else normalized["date"].min(),
        "date_max": None if normalized.empty else normalized["date"].max(),
    }
    write_json_file(metadata_path, metadata)

    return NormalizeResult(
        output_path=output_path,
        metadata_path=metadata_path,
        row_count=int(len(normalized)),
        source_path=source_path,
    )


def normalize_fundamentals(input_path: str | None = None) -> NormalizeResult:
    """Normalize raw fundamentals summary JSON into document and feature tables."""
    source_path = _resolve_input_path(
        input_path,
        "data/raw/fundamentals",
        "fundamentals_summary_*.json",
    )
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    downloaded_documents = payload.get("downloaded_documents", [])

    frame = pd.DataFrame(downloaded_documents)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "doc_id",
                "ticker",
                "sec_code",
                "form_code",
                "doc_description",
                "submit_date_time",
                "download_kind",
                "output_path",
            ]
        )

    if "submit_date_time" in frame.columns:
        published_at = pd.to_datetime(frame["submit_date_time"], errors="coerce")
        frame["published_at"] = published_at.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        frame["date"] = published_at.dt.strftime("%Y-%m-%d")
    else:
        frame["published_at"] = None
        frame["date"] = None

    if "ticker" in frame.columns:
        frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    else:
        frame["ticker"] = None

    frame["market"] = "jp_equities"
    frame["source"] = payload.get("provider", "edinet")
    frame["as_of_date"] = frame["date"]

    ordered_columns = [
        "date",
        "ticker",
        "market",
        "source",
        "doc_id",
        "sec_code",
        "form_code",
        "doc_description",
        "published_at",
        "as_of_date",
        "download_kind",
        "output_path",
    ]
    for column in ordered_columns:
        if column not in frame.columns:
            frame[column] = None
    normalized = frame[ordered_columns].sort_values(["ticker", "date", "doc_id"], na_position="last")

    timestamp = timestamp_for_filename()
    output_dir = ensure_directory(project_path("data/interim/fundamentals"))
    documents_output_path = output_dir / f"fundamentals_documents_{timestamp}.csv"
    normalized.to_csv(documents_output_path, index=False)

    feature_rows: list[dict[str, Any]] = []
    for item in downloaded_documents:
        if item.get("download_kind") != "csv":
            continue
        raw_zip_path = Path(str(item.get("output_path", "")))
        if not raw_zip_path.exists():
            continue
        feature_rows.append(
            _extract_document_fundamentals(
                zip_path=raw_zip_path,
                ticker=str(item.get("ticker", "")).strip().upper(),
                published_at=item.get("submit_date_time"),
                form_code=item.get("form_code"),
            )
        )

    features_frame = pd.DataFrame(feature_rows)
    if features_frame.empty:
        features_frame = pd.DataFrame(
            columns=[
                "ticker",
                "published_at",
                "form_code",
                "revenue",
                "operating_income",
                "net_income",
                "eps",
                "book_value",
                "shares_outstanding",
                "revenue_growth_yoy",
                "operating_margin",
                "roe",
                "source_path",
            ]
        )
    else:
        features_frame["published_at"] = pd.to_datetime(
            features_frame["published_at"], errors="coerce"
        )
        features_frame["date"] = features_frame["published_at"].dt.strftime("%Y-%m-%d")
        features_frame["published_at"] = features_frame["published_at"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        features_frame.sort_values(["ticker", "published_at"], inplace=True)
        for column in [
            "revenue",
            "operating_income",
            "net_income",
            "eps",
            "book_value",
            "shares_outstanding",
        ]:
            if column in features_frame.columns:
                features_frame[column] = pd.to_numeric(features_frame[column], errors="coerce")

        features_frame["operating_margin"] = features_frame["operating_income"] / features_frame["revenue"]
        features_frame["roe"] = features_frame["net_income"] / features_frame["book_value"]
        features_frame["report_type"] = features_frame["form_code"].astype(str).str[:3]
        features_frame["revenue_growth_yoy"] = features_frame.groupby(
            ["ticker", "report_type"]
        )["revenue"].pct_change()
        features_frame["market"] = "jp_equities"
        features_frame["source"] = payload.get("provider", "edinet")
        ordered_feature_columns = [
            "date",
            "published_at",
            "ticker",
            "market",
            "source",
            "form_code",
            "revenue",
            "operating_income",
            "net_income",
            "eps",
            "book_value",
            "shares_outstanding",
            "revenue_growth_yoy",
            "operating_margin",
            "roe",
            "source_path",
        ]
        for column in ordered_feature_columns:
            if column not in features_frame.columns:
                features_frame[column] = None
        features_frame = features_frame[ordered_feature_columns]

    features_output_path = output_dir / f"fundamentals_features_{timestamp}.csv"
    features_frame.to_csv(features_output_path, index=False)

    metadata_path = project_path(f"data/metadata/normalize_fundamentals_{timestamp}.json")
    metadata = {
        "command": "data normalize-fundamentals",
        "timestamp_utc": timestamp,
        "input_path": str(source_path),
        "documents_output_path": str(documents_output_path),
        "features_output_path": str(features_output_path),
        "documents_row_count": int(len(normalized)),
        "features_row_count": int(len(features_frame)),
        "ticker_count": int(normalized["ticker"].dropna().nunique()) if "ticker" in normalized else 0,
    }
    write_json_file(metadata_path, metadata)

    return NormalizeResult(
        output_path=features_output_path,
        metadata_path=metadata_path,
        row_count=int(len(features_frame)),
        source_path=source_path,
    )


def normalize_macro(input_path: str | None = None) -> NormalizeResult:
    """Normalize raw market/macro CSV into an interim daily table."""
    source_path = _resolve_input_path(input_path, "data/raw/market", "market_yfinance_*.csv")
    frame = pd.read_csv(source_path)
    required_columns = {"date", "series_name", "value"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ConfigError(f"Raw market file is missing columns: {missing_columns}")

    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    normalized["series_name"] = normalized["series_name"].astype(str).str.strip().str.lower()
    for column in ["value", "close", "open", "high", "low", "volume"]:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized.dropna(subset=["date", "series_name", "value"], inplace=True)
    normalized.drop_duplicates(subset=["date", "series_name"], keep="last", inplace=True)
    normalized.sort_values(["series_name", "date"], inplace=True)

    timestamp = timestamp_for_filename()
    output_dir = ensure_directory(project_path("data/interim/market"))
    output_path = output_dir / f"market_normalized_{timestamp}.csv"
    normalized.to_csv(output_path, index=False)

    metadata_path = project_path(f"data/metadata/normalize_macro_{timestamp}.json")
    metadata = {
        "command": "data normalize-macro",
        "timestamp_utc": timestamp,
        "input_path": str(source_path),
        "output_path": str(output_path),
        "row_count": int(len(normalized)),
        "series_count": int(normalized["series_name"].nunique()),
    }
    write_json_file(metadata_path, metadata)

    return NormalizeResult(
        output_path=output_path,
        metadata_path=metadata_path,
        row_count=int(len(normalized)),
        source_path=source_path,
    )
