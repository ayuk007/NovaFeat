"""PII detection result models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PIICategory(str, Enum):
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    AADHAAR = "aadhaar"
    PAN = "pan"
    PASSPORT = "passport"
    CREDIT_CARD = "credit_card"
    NAME = "name"
    ADDRESS = "address"
    IP_ADDRESS = "ip_address"
    UUID = "uuid"
    HASH = "hash"
    DEVICE_ID = "device_id"
    USER_ID = "user_id"
    SESSION_ID = "session_id"
    NONE = "none"


class PIIAction(str, Enum):
    MASK = "mask"
    REMOVE = "remove"
    SUMMARIZE_ONLY = "summarize_only"
    ALLOW = "allow"


class DetectorVote(BaseModel):
    stage_name: str
    category: PIICategory
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str


class ColumnPIIAssessment(BaseModel):
    column_name: str
    category: PIICategory
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
    votes: list[DetectorVote] = Field(default_factory=list)
    recommended_action: PIIAction = PIIAction.ALLOW

    @property
    def is_sensitive(self) -> bool:
        return self.category != PIICategory.NONE
