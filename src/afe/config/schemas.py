"""Strongly typed configuration schemas.

Every configurable subsystem in the harness has a corresponding Pydantic
model here. Nothing in the application should hardcode a setting that
belongs in one of these models — if a new tunable is needed, it is added
to the relevant schema and to the matching YAML file under ``config/``.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    PLAIN = "plain"
    JSON = "json"


class ModelRoleConfig(BaseModel):
    """Configuration for a single named model role (planner, code_gen, ...).

    Kept independent per-role so a multi-model architecture (different
    provider/model for planning vs. summarisation vs. PII detection) is a
    configuration change, never a code change.
    """

    provider: str = Field(
        ..., description="init_chat_model provider id, e.g. 'openai', 'anthropic', 'ollama'"
    )
    model: str = Field(..., description="Provider-specific model name")
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, gt=0)
    timeout_seconds: float = Field(60.0, gt=0)
    max_retries: int = Field(3, ge=0)
    extra: dict = Field(default_factory=dict, description="Provider-specific passthrough kwargs")


class ModelConfig(BaseModel):
    """The full multi-model architecture: one entry per responsibility."""

    default_provider: str = "openai"
    planner: ModelRoleConfig
    reasoning: ModelRoleConfig
    code_generation: ModelRoleConfig
    pii_detection: ModelRoleConfig
    summarization: ModelRoleConfig
    critique: ModelRoleConfig
    validation: ModelRoleConfig
    context_compression: ModelRoleConfig
    feedback: ModelRoleConfig
    tool_selection: ModelRoleConfig


class ContextEngineeringConfig(BaseModel):
    max_context_tokens: int = Field(8000, gt=0)
    summarization_trigger_tokens: int = Field(6000, gt=0)
    rolling_memory_turns: int = Field(20, ge=1)
    reserve_tokens_for_response: int = Field(1024, ge=0)

    @field_validator("summarization_trigger_tokens")
    @classmethod
    def _trigger_below_max(cls, v: int, info) -> int:
        max_tokens = info.data.get("max_context_tokens")
        if max_tokens is not None and v > max_tokens:
            raise ValueError("summarization_trigger_tokens must be <= max_context_tokens")
        return v


class ApprovalPolicy(str, Enum):
    ALWAYS = "always"
    RISKY_ONLY = "risky_only"
    NEVER = "never"


class FeatureEngineeringConfig(BaseModel):
    max_rows_in_memory: int = Field(2_000_000, gt=0)
    preview_row_count: int = Field(5, ge=0, le=50)
    approval_policy: ApprovalPolicy = ApprovalPolicy.ALWAYS
    cache_enabled: bool = True
    cache_ttl_seconds: int = Field(3600, ge=0)
    large_dataset_threshold_gb: float = Field(5.0, gt=0)
    medium_dataset_threshold_gb: float = Field(0.5, gt=0)


class StorageConfig(BaseModel):
    temp_directory: Path = Path(".afe_tmp")
    artifact_directory: Path = Path(".afe_artifacts")
    checkpoint_directory: Path = Path(".afe_checkpoints")

    @field_validator("temp_directory", "artifact_directory", "checkpoint_directory")
    @classmethod
    def _resolve(cls, v: Path) -> Path:
        return Path(v).expanduser()


class LoggingConfig(BaseModel):
    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.PLAIN
    log_to_file: bool = True
    log_file_path: Path = Path(".afe_tmp/logs/afe.log")
    log_to_json: bool = False


class SecurityConfig(BaseModel):
    pii_detection_enabled: bool = True
    pii_confidence_threshold: float = Field(0.6, ge=0.0, le=1.0)
    allowed_columns: list[str] = Field(default_factory=list)
    blocked_columns: list[str] = Field(default_factory=list)
    sandbox_backend: str = "subprocess"
    sandbox_cpu_seconds: int = Field(10, gt=0)
    sandbox_memory_mb: int = Field(512, gt=0)
    sandbox_timeout_seconds: int = Field(15, gt=0)
    sandbox_network_enabled: bool = False
    require_human_approval_for_generated_code: bool = True


class CLIConfig(BaseModel):
    colors_enabled: bool = True
    verbosity: str = "normal"
    require_confirmations: bool = True
    history_file: Path = Path(".afe_tmp/cli_history")


class AppConfig(BaseModel):
    app_name: str = "agentic-feature-engineering-harness"
    environment: str = "development"


class AgentConfig(BaseModel):
    max_planning_iterations: int = Field(5, ge=1)
    max_replans: int = Field(3, ge=0)
    require_human_approval: bool = True
    dynamic_tool_generation_enabled: bool = True
    dynamic_tool_requires_approval: bool = True


class RootConfig(BaseModel):
    """Top-level composed configuration, assembled by the loader from the
    individual YAML files plus environment variable overrides."""

    app: AppConfig
    model: ModelConfig
    context: ContextEngineeringConfig
    feature_engineering: FeatureEngineeringConfig
    storage: StorageConfig
    logging: LoggingConfig
    security: SecurityConfig
    cli: CLIConfig
    agent: AgentConfig
