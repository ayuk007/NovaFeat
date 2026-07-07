from __future__ import annotations

from afe.models.pii import PIICategory
from afe.security.pii.aggregator import PIIAggregator
from afe.security.pii.regex_detector import RegexPIIDetector


def test_regex_detector_flags_email_by_value():
    detector = RegexPIIDetector()
    result = detector.assess("contact", ["a@example.com", "b@example.com", "c@example.com"])
    assert result.category == PIICategory.EMAIL
    assert result.confidence > 0.6


def test_regex_detector_flags_by_column_name():
    detector = RegexPIIDetector()
    result = detector.assess("customer_email", [])
    assert result.category == PIICategory.EMAIL


def test_regex_detector_returns_none_for_clean_column():
    detector = RegexPIIDetector()
    result = detector.assess("age", ["25", "30", "40"])
    assert result.category == PIICategory.NONE


def test_aggregator_respects_confidence_threshold():
    aggregator = PIIAggregator(detectors=[RegexPIIDetector()], confidence_threshold=0.99)
    result = aggregator.assess_column("email", ["a@example.com"])
    # Regex-only confidence tops out below 0.99, so a very high threshold should suppress it.
    assert result.category == PIICategory.NONE


def test_aggregator_flags_sensitive_column_with_default_threshold():
    aggregator = PIIAggregator(detectors=[RegexPIIDetector()], confidence_threshold=0.6)
    result = aggregator.assess_column("email", ["a@example.com", "b@example.com"])
    assert result.is_sensitive
    assert result.votes[0].stage_name == "regex"
