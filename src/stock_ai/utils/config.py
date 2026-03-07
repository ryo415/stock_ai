"""Utilities for loading project YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ConfigError(Exception):
    """Raised when a config file cannot be resolved or parsed."""


def get_project_root() -> Path:
    """Return the repository root based on the installed source layout."""
    return Path(__file__).resolve().parents[3]


def get_configs_root() -> Path:
    """Return the configs directory."""
    return get_project_root() / "configs"


def resolve_config_path(config_path: str | Path) -> Path:
    """Resolve a config path from either an absolute path or a project-relative path."""
    path = Path(config_path)
    if path.is_absolute():
        return path
    return get_project_root() / path


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return its contents as a dictionary."""
    resolved_path = resolve_config_path(config_path)
    if not resolved_path.exists():
        raise ConfigError(f"Config file not found: {resolved_path}")

    if resolved_path.suffix not in {".yaml", ".yml"}:
        raise ConfigError(f"Unsupported config file type: {resolved_path.suffix}")

    try:
        import yaml
    except ImportError as exc:
        raise ConfigError(
            "PyYAML is required to load YAML config files. Install project dependencies first."
        ) from exc

    with resolved_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping: {resolved_path}")

    return data


def load_config(section: str, name: str) -> dict[str, Any]:
    """Load a config from configs/<section>/<name>.yaml."""
    relative_path = Path("configs") / section / f"{name}.yaml"
    return load_yaml_config(relative_path)


def list_config_files(section: str) -> list[Path]:
    """List YAML config files in a config section."""
    section_dir = get_configs_root() / section
    if not section_dir.exists():
        raise ConfigError(f"Config section not found: {section_dir}")

    return sorted(
        path for path in section_dir.iterdir() if path.is_file() and path.suffix in {".yaml", ".yml"}
    )
