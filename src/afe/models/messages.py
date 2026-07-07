"""Typed inter-agent message objects.

Nodes in the LangGraph graph communicate through these structured
messages rather than raw strings, so downstream nodes can rely on a
schema instead of parsing free text.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _new_id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MessagePriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class BaseMessage(BaseModel):
    id: str = Field(default_factory=_new_id)
    timestamp: datetime = Field(default_factory=_now)
    source_agent: str
    destination_agent: str
    correlation_id: str = Field(default_factory=_new_id)
    priority: MessagePriority = MessagePriority.NORMAL
    metadata: dict[str, Any] = Field(default_factory=dict)
    version: str = "1.0"


class TaskMessage(BaseMessage):
    task_description: str
    business_goal: str | None = None
    constraints: list[str] = Field(default_factory=list)


class FeatureProposal(BaseModel):
    name: str
    operation: str
    source_columns: list[str]
    rationale: str
    risk_level: str = "low"
    params: dict[str, Any] = Field(default_factory=dict)


class PlanMessage(BaseMessage):
    plan_id: str = Field(default_factory=_new_id)
    proposals: list[FeatureProposal]
    summary: str


class ExecutionMessage(BaseMessage):
    plan_id: str
    executed_proposals: list[str]
    dataset_version_before: int
    dataset_version_after: int
    execution_log: list[str] = Field(default_factory=list)


class FeedbackMessage(BaseMessage):
    raw_feedback: str
    parsed_constraints: list[str] = Field(default_factory=list)
    affected_columns: list[str] = Field(default_factory=list)


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskMessage(BaseMessage):
    subject: str
    risk_level: RiskLevel
    reasons: list[str] = Field(default_factory=list)


class ApprovalDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    SKIP = "skip"
    ROLLBACK = "rollback"
    CANCEL = "cancel"


class ApprovalMessage(BaseMessage):
    plan_id: str
    decision: ApprovalDecision
    modified_proposals: list[FeatureProposal] | None = None
    comment: str | None = None


class ValidationResult(BaseMessage):
    target: str
    passed: bool
    violations: list[str] = Field(default_factory=list)


class ToolResultMessage(BaseMessage):
    tool_name: str
    success: bool
    payload: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
