"""Regex-tier PII detection.

This is stage one of the PII pipeline (Regex -> NER -> LLM classification
-> confidence aggregation, per spec). It operates only on column *names*
and a small bounded sample of values supplied by the caller — it never
requests or receives the full column.
"""

from __future__ import annotations

import re

from afe.interfaces.pii_detector import PIIDetector
from afe.models.pii import ColumnPIIAssessment, PIIAction, PIICategory

_NAME_HINTS: dict[PIICategory, tuple[str, ...]] = {
    PIICategory.EMAIL: ("email", "e_mail", "mail_address"),
    PIICategory.PHONE: ("phone", "mobile", "contact_number", "tel"),
    PIICategory.SSN: ("ssn", "social_security"),
    PIICategory.AADHAAR: ("aadhaar", "aadhar"),
    PIICategory.PAN: ("pan_number", "pan_no"),
    PIICategory.PASSPORT: ("passport",),
    PIICategory.CREDIT_CARD: ("credit_card", "card_number", "ccn"),
    PIICategory.NAME: ("first_name", "last_name", "full_name", "customer_name", "given_name", "surname"),
    PIICategory.ADDRESS: ("address", "street", "zipcode", "postal_code"),
    PIICategory.IP_ADDRESS: ("ip_address", "ip_addr"),
    PIICategory.UUID: ("uuid", "guid"),
    PIICategory.DEVICE_ID: ("device_id", "imei", "mac_address"),
    PIICategory.USER_ID: ("user_id", "customer_id", "account_id"),
    PIICategory.SESSION_ID: ("session_id", "session_token"),
}

_VALUE_PATTERNS: dict[PIICategory, re.Pattern[str]] = {
    PIICategory.EMAIL: re.compile(r"^[\w.\-]+@[\w\-]+\.[a-zA-Z]{2,}$"),
    PIICategory.PHONE: re.compile(r"^\+?\d{7,15}$"),
    PIICategory.SSN: re.compile(r"^\d{3}-\d{2}-\d{4}$"),
    PIICategory.CREDIT_CARD: re.compile(r"^(\d{4}[- ]?){3}\d{1,4}$"),
    PIICategory.IP_ADDRESS: re.compile(r"^(\d{1,3}\.){3}\d{1,3}$"),
    PIICategory.UUID: re.compile(
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    ),
    PIICategory.HASH: re.compile(r"^[a-fA-F0-9]{32,64}$"),
}

_ACTION_MAP: dict[PIICategory, PIIAction] = {
    PIICategory.EMAIL: PIIAction.MASK,
    PIICategory.PHONE: PIIAction.MASK,
    PIICategory.SSN: PIIAction.REMOVE,
    PIICategory.AADHAAR: PIIAction.REMOVE,
    PIICategory.PAN: PIIAction.REMOVE,
    PIICategory.PASSPORT: PIIAction.REMOVE,
    PIICategory.CREDIT_CARD: PIIAction.REMOVE,
    PIICategory.NAME: PIIAction.MASK,
    PIICategory.ADDRESS: PIIAction.MASK,
    PIICategory.IP_ADDRESS: PIIAction.SUMMARIZE_ONLY,
    PIICategory.UUID: PIIAction.SUMMARIZE_ONLY,
    PIICategory.HASH: PIIAction.SUMMARIZE_ONLY,
    PIICategory.DEVICE_ID: PIIAction.SUMMARIZE_ONLY,
    PIICategory.USER_ID: PIIAction.SUMMARIZE_ONLY,
    PIICategory.SESSION_ID: PIIAction.SUMMARIZE_ONLY,
}


class RegexPIIDetector(PIIDetector):
    """Fast, deterministic first-pass PII detector using name heuristics
    and value pattern matching over a bounded sample."""

    @property
    def stage_name(self) -> str:
        return "regex"

    def assess(self, column_name: str, sample_values: list[str]) -> ColumnPIIAssessment:
        name_category, name_confidence = self._assess_name(column_name)
        value_category, value_confidence = self._assess_values(sample_values)

        category: PIICategory
        confidence: float
        basis: str
        if value_confidence >= name_confidence and value_category != PIICategory.NONE:
            category, confidence, basis = value_category, value_confidence, "value pattern match"
        elif name_category != PIICategory.NONE:
            category, confidence, basis = name_category, name_confidence, "column name heuristic"
        else:
            category, confidence, basis = PIICategory.NONE, 0.0, "no signal"

        explanation = (
            f"Regex stage classified column '{column_name}' as {category.value} "
            f"based on {basis} (confidence={confidence:.2f})."
        )

        return ColumnPIIAssessment(
            column_name=column_name,
            category=category,
            confidence=confidence,
            explanation=explanation,
            votes=[],
            recommended_action=_ACTION_MAP.get(category, PIIAction.ALLOW),
        )

    def _assess_name(self, column_name: str) -> tuple[PIICategory, float]:
        lowered = column_name.lower()
        for category, hints in _NAME_HINTS.items():
            matched_category: PIICategory = category
            if any(hint in lowered for hint in hints):
                return matched_category, 0.7
        return PIICategory.NONE, 0.0

    def _assess_values(self, sample_values: list[str]) -> tuple[PIICategory, float]:
        if not sample_values:
            return PIICategory.NONE, 0.0

        best_category = PIICategory.NONE
        best_ratio = 0.0
        for category, pattern in _VALUE_PATTERNS.items():
            matches = sum(1 for v in sample_values if isinstance(v, str) and pattern.match(v.strip()))
            ratio = matches / len(sample_values)
            if ratio > best_ratio:
                best_ratio = ratio
                best_category = category

        if best_ratio == 0.0:
            return PIICategory.NONE, 0.0
        # Scale ratio into a confidence score, capped below 1.0 for a single stage.
        confidence = min(0.95, 0.5 + best_ratio * 0.45)
        return best_category, confidence
