"""Pydantic models describing dataset state *without* holding raw data.

These are the only objects that ever cross into the LLM-visible context.
None of them can hold a DataFrame — that constraint is enforced simply by
never giving these models a field capable of it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class EngineType(str, Enum):
    PANDAS = "pandas"
    POLARS_LAZY = "polars_lazy"
    DUCKDB = "duckdb"
    DASK = "dask"
    SPARK = "spark"


class ColumnDType(str, Enum):
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    STRING = "string"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    UNKNOWN = "unknown"


class ColumnSchema(BaseModel):
    name: str
    dtype: ColumnDType
    nullable: bool = True
    is_pii: bool = False
    is_blocked: bool = False


class DatasetSchema(BaseModel):
    columns: list[ColumnSchema]
    target_column: str | None = None

    def visible_columns(self) -> list[ColumnSchema]:
        """Columns an agent is allowed to see (excludes blocked/PII)."""
        return [c for c in self.columns if not c.is_blocked and not c.is_pii]


class ColumnProfile(BaseModel):
    """Full statistical summary of one column. Never contains raw values."""

    name: str
    dtype: ColumnDType
    null_count: int = 0
    null_percentage: float = 0.0
    unique_count: int = 0
    cardinality_ratio: float = 0.0
    mean: float | None = None
    median: float | None = None
    mode: str | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    quantiles: dict[str, float] = Field(default_factory=dict)
    skewness: float | None = None
    kurtosis: float | None = None
    is_constant: bool = False
    is_duplicate_of: str | None = None
    histogram_bins: list[float] = Field(default_factory=list)
    histogram_counts: list[int] = Field(default_factory=list)


class CorrelationSummary(BaseModel):
    method: str
    columns: list[str]
    matrix: list[list[float]]
    high_correlation_pairs: list[tuple[str, str, float]] = Field(default_factory=list)


class PreviewResult(BaseModel):
    """A bounded, human-approved preview. Row count is always capped by
    ``feature_engineering.preview_row_count`` in configuration."""

    columns: list[str]
    row_count: int
    rows: list[dict[str, str]]
    approved_by: str
    approved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DatasetMetadata(BaseModel):
    dataset_id: str
    engine: EngineType
    row_count: int
    column_count: int
    size_bytes: int
    is_lazy: bool = False
    version: int = 0
    temp_artifact_paths: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024**3)
