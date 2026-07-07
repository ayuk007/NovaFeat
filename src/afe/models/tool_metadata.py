"""Tool registry metadata models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class ToolRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolSource(str, Enum):
    PREBUILT = "prebuilt"
    GENERATED = "generated"
    PLUGIN = "plugin"


class ToolCapability(BaseModel):
    name: str
    description: str
    input_schema: dict = Field(default_factory=dict)
    output_schema: dict = Field(default_factory=dict)
    risk_level: ToolRiskLevel = ToolRiskLevel.LOW
    requires_human_approval: bool = False
    source: ToolSource = ToolSource.PREBUILT
    version: str = "1.0.0"
    dependencies: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    output: dict = Field(default_factory=dict)
    error_message: str | None = None
    duration_seconds: float = 0.0
