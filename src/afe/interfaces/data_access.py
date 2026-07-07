"""The Data Access Service interface.

This is the single abstraction through which any agent, tool, or graph
node may learn about a dataset. It is the enforcement point for the hard
rule that the LLM never sees raw rows: every method here returns a
summary/statistical object, never a DataFrame.

Concrete engines (pandas today; Polars/DuckDB/Dask later) implement this
interface, so swapping the underlying engine never requires touching
agent or graph code (Dependency Inversion + Open/Closed).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from afe.models.dataset_metadata import (
    ColumnProfile,
    CorrelationSummary,
    DatasetMetadata,
    DatasetSchema,
    PreviewResult,
)


class DataAccessService(ABC):
    """Controlled read/write gateway to a dataset.

    Every method here must uphold the data-privacy contract: no method
    may return complete raw column values to a caller unless it is an
    explicitly human-approved preview request (:meth:`request_preview`),
    and even then the row count is capped by configuration.
    """

    @abstractmethod
    def get_metadata(self) -> DatasetMetadata:
        """Return dataset-level metadata (row/col counts, size, engine, version)."""

    @abstractmethod
    def get_schema(self) -> DatasetSchema:
        """Return column names, dtypes, and nullability — no values."""

    @abstractmethod
    def get_column_profile(self, column: str) -> ColumnProfile:
        """Return the full statistical profile for a single column."""

    @abstractmethod
    def get_all_column_profiles(self) -> dict[str, ColumnProfile]:
        """Return statistical profiles for every non-blocked column."""

    @abstractmethod
    def get_correlation_summary(self, method: str = "pearson") -> CorrelationSummary:
        """Return a correlation matrix summary over numeric columns."""

    @abstractmethod
    def request_preview(self, columns: list[str], max_rows: int | None = None) -> PreviewResult:
        """Return a small, human-approved preview. Callers must have
        approval; implementations must enforce the configured row cap and
        must refuse blocked/PII columns."""

    @abstractmethod
    def apply_transformation(self, name: str, params: dict[str, Any]) -> DatasetMetadata:
        """Apply a named, registered transformation to the dataset and
        return updated metadata. Must create a new dataset version."""

    @abstractmethod
    def rollback(self, to_version: int) -> DatasetMetadata:
        """Roll the dataset handle back to a previous version."""

    @abstractmethod
    def list_versions(self) -> list[int]:
        """Return all available dataset versions for this session."""
