"""Configuration loading.

Precedence (highest wins): environment variables  >  YAML files  >  schema defaults.

Each YAML file under ``config/`` maps 1:1 to a top-level key of
:class:`~afe.config.schemas.RootConfig`. Environment variables use the
prefix ``AFE__<SECTION>__<FIELD>`` with ``__`` as the nesting delimiter,
e.g. ``AFE__MODEL__PLANNER__PROVIDER=anthropic``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from afe.config.schemas import RootConfig
from afe.exceptions.errors import ConfigurationError

_SECTION_FILES = {
    "app": "app.yaml",
    "model": "model.yaml",
    "context": "context.yaml",
    "feature_engineering": "feature_engineering.yaml",
    "storage": "storage.yaml",
    "logging": "logging.yaml",
    "security": "security.yaml",
    "cli": "cli.yaml",
    "agent": "agent.yaml",
}

_ENV_PREFIX = "AFE__"


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"Config file {path} must contain a mapping at the top level")
    return data


def _coerce_env_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    result = {k: (dict(v) if isinstance(v, dict) else v) for k, v in config.items()}
    for env_key, raw_value in os.environ.items():
        if not env_key.startswith(_ENV_PREFIX):
            continue
        path_parts = env_key[len(_ENV_PREFIX) :].lower().split("__")
        if not path_parts:
            continue
        cursor = result
        for part in path_parts[:-1]:
            if part not in cursor or not isinstance(cursor.get(part), dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[path_parts[-1]] = _coerce_env_value(raw_value)
    return result


def load_config(config_dir: str | Path = "config", dotenv_path: str | Path | None = None) -> RootConfig:
    """Load, merge, and validate the full application configuration.

    Raises:
        ConfigurationError: if required sections are missing or invalid.
    """
    load_dotenv(dotenv_path=dotenv_path, override=False)

    config_dir = Path(config_dir)
    merged: dict[str, Any] = {}
    for section, filename in _SECTION_FILES.items():
        section_data = _load_yaml_file(config_dir / filename)
        merged[section] = section_data

    merged = _apply_env_overrides(merged)

    try:
        return RootConfig.model_validate(merged)
    except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
        raise ConfigurationError(f"Failed to load configuration from {config_dir}: {exc}") from exc


_cached_config: RootConfig | None = None


def get_config(config_dir: str | Path = "config", force_reload: bool = False) -> RootConfig:
    """Return a process-wide cached configuration, loading it on first use."""
    global _cached_config
    if _cached_config is None or force_reload:
        _cached_config = load_config(config_dir)
    return _cached_config
