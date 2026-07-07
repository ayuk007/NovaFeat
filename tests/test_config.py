from __future__ import annotations

import os

import pytest

from afe.config.loader import load_config
from afe.exceptions.errors import ConfigurationError


def test_load_config_from_repo_config_dir():
    config = load_config("config")
    assert config.model.planner.provider == "openai"
    assert config.security.pii_confidence_threshold == pytest.approx(0.6)
    assert config.feature_engineering.preview_row_count == 5


def test_env_override_beats_yaml(monkeypatch):
    monkeypatch.setenv("AFE__MODEL__PLANNER__PROVIDER", "anthropic")
    config = load_config("config")
    assert config.model.planner.provider == "anthropic"


def test_missing_config_dir_raises():
    with pytest.raises(ConfigurationError):
        load_config("nonexistent_config_dir_xyz")
