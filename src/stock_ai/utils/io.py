"""File output helpers for CLI commands."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stock_ai.utils.config import get_project_root


def timestamp_for_filename() -> str:
    """Return a UTC timestamp formatted for filenames."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def project_path(relative_path: str | Path) -> Path:
    """Resolve a project-relative path."""
    return get_project_root() / relative_path


def write_json_file(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON payload with stable formatting."""
    file_path = Path(path)
    ensure_directory(file_path.parent)
    file_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return file_path
