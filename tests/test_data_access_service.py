from __future__ import annotations

import pytest

from afe.exceptions.errors import DataAccessError, PolicyViolationError
from afe.security.pii.aggregator import PIIAggregator
from afe.security.pii.regex_detector import RegexPIIDetector
from afe.services.data_access_service import PandasDataAccessService


def _make_service(sample_df, blocked_columns=None):
    aggregator = PIIAggregator(detectors=[RegexPIIDetector()], confidence_threshold=0.6)
    return PandasDataAccessService(
        df=sample_df, pii_aggregator=aggregator, blocked_columns=blocked_columns or [], preview_row_cap=3
    )


def test_pii_columns_are_marked_and_hidden(sample_df):
    service = _make_service(sample_df)
    schema = service.get_schema()
    pii_columns = {c.name for c in schema.columns if c.is_pii}
    assert "email" in pii_columns
    assert "customer_id" in pii_columns
    visible = {c.name for c in schema.visible_columns()}
    assert "email" not in visible
    assert "age" in visible


def test_get_column_profile_blocks_pii_column(sample_df):
    service = _make_service(sample_df)
    with pytest.raises(PolicyViolationError):
        service.get_column_profile("email")


def test_get_column_profile_allows_visible_column(sample_df):
    service = _make_service(sample_df)
    profile = service.get_column_profile("age")
    assert profile.name == "age"


def test_blocked_columns_config_hides_column(sample_df):
    service = _make_service(sample_df, blocked_columns=["income"])
    with pytest.raises(PolicyViolationError):
        service.get_column_profile("income")


def test_apply_transformation_creates_new_version(sample_df):
    service = _make_service(sample_df)
    before = service.get_metadata().version
    after_meta = service.apply_transformation("log_transform", {"column": "income"})
    assert after_meta.version == before + 1
    assert "income_log" in service.get_all_column_profiles()


def test_rollback_restores_previous_version(sample_df):
    service = _make_service(sample_df)
    service.apply_transformation("log_transform", {"column": "income"})
    rolled_back = service.rollback(0)
    assert rolled_back.version == 0
    assert "income_log" not in service.get_all_column_profiles()


def test_request_preview_is_row_capped(sample_df):
    service = _make_service(sample_df)  # preview_row_cap=3
    preview = service.request_preview(["age", "income"], max_rows=100)
    assert preview.row_count == 3


def test_request_preview_blocks_pii_column(sample_df):
    service = _make_service(sample_df)
    with pytest.raises(PolicyViolationError):
        service.request_preview(["email"])


def test_unknown_transformation_raises(sample_df):
    service = _make_service(sample_df)
    with pytest.raises(DataAccessError):
        service.apply_transformation("not_a_real_transformation", {})
