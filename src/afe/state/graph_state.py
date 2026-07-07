"""The strongly typed state that flows through the LangGraph graph.

Nothing here holds a raw DataFrame — only metadata, summaries, and typed
messages. This is the object LangGraph checkpoints, so it must be fully
serializable (pure Pydantic, no engine handles beyond an opaque id).
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field

from afe.models.dataset_metadata import ColumnProfile, CorrelationSummary, DatasetMetadata, DatasetSchema
from afe.models.messages import (
    ApprovalDecision,
    ExecutionMessage,
    FeedbackMessage,
    PlanMessage,
    RiskMessage,
)


class GraphPhase(str, Enum):
    INIT = "init"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    SUMMARIZING = "summarizing"
    AWAITING_FEEDBACK = "awaiting_feedback"
    REPLANNING = "replanning"
    DONE = "done"
    CANCELLED = "cancelled"


class TokenUsageRecord(BaseModel):
    role: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    model_name: str = ""
    provider: str = ""


def _append(existing: list[Any], new: list[Any]) -> list[Any]:
    """Reducer: append new items to existing list (LangGraph-style)."""
    return [*existing, *new]


class HarnessState(BaseModel):
    """The full session state persisted/checkpointed by LangGraph."""

    session_id: str
    phase: GraphPhase = GraphPhase.INIT

    # Dataset
    dataset_metadata: DatasetMetadata | None = None
    dataset_schema: DatasetSchema | None = None
    column_profiles: dict[str, ColumnProfile] = Field(default_factory=dict)
    correlation_summary: CorrelationSummary | None = None

    # Planning / execution
    current_plan: PlanMessage | None = None
    pending_feature_names: list[str] = Field(default_factory=list)
    approved_feature_names: list[str] = Field(default_factory=list)
    executed_transformations: Annotated[list[ExecutionMessage], _append] = Field(default_factory=list)
    execution_history: Annotated[list[str], _append] = Field(default_factory=list)

    # Human interaction
    last_approval_decision: ApprovalDecision | None = None
    human_feedback: Annotated[list[FeedbackMessage], _append] = Field(default_factory=list)
    active_constraints: list[str] = Field(default_factory=list)

    # Risk / replanning
    risk_assessments: Annotated[list[RiskMessage], _append] = Field(default_factory=list)
    replan_count: int = 0

    # Context engineering
    conversation_summary: str = ""
    context_token_budget_remaining: int = 0
    token_usage: Annotated[list[TokenUsageRecord], _append] = Field(default_factory=list)

    # Errors / rollback
    errors: Annotated[list[str], _append] = Field(default_factory=list)
    rollback_target_version: int | None = None

    model_config = {"arbitrary_types_allowed": True}

    def total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self.token_usage)
