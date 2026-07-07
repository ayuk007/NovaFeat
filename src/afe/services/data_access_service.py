"""Pandas-backed implementation of the Data Access Service.

This is the only object in the small-dataset path that holds the actual
DataFrame. Every public method returns a summary/metadata object; no
public method returns a DataFrame or raw values, except
:meth:`request_preview`, which is bounded and gated by an explicit
``approved`` flag supplied by the caller (the human-approval graph node).
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd

from afe.analysis.profiler import compute_correlation_summary, infer_dtype, profile_dataframe
from afe.exceptions.errors import DataAccessError, PolicyViolationError
from afe.interfaces.data_access import DataAccessService
from afe.models.dataset_metadata import (
    ColumnProfile,
    ColumnSchema,
    CorrelationSummary,
    DatasetMetadata,
    DatasetSchema,
    EngineType,
    PreviewResult,
)
from afe.security.pii.aggregator import PIIAggregator


class PandasDataAccessService(DataAccessService):
    """Small-dataset (in-memory pandas) implementation."""

    def __init__(
        self,
        df: pd.DataFrame,
        pii_aggregator: PIIAggregator,
        blocked_columns: list[str] | None = None,
        preview_row_cap: int = 5,
        temp_dir: str | Path = ".afe_tmp",
    ) -> None:
        self._df = df
        self._pii_aggregator = pii_aggregator
        self._blocked_columns = set(blocked_columns or [])
        self._preview_row_cap = preview_row_cap
        self._temp_dir = Path(temp_dir)
        self._temp_dir.mkdir(parents=True, exist_ok=True)

        self._dataset_id = str(uuid.uuid4())
        self._version = 0
        self._version_history: dict[int, pd.DataFrame] = {0: df.copy()}
        self._schema = self._build_schema()
        self._profiles: dict[str, ColumnProfile] = {}
        self._refresh_profiles()

    # -- construction helpers -------------------------------------------------

    def _build_schema(self) -> DatasetSchema:
        df = self._current_df()
        columns = []
        for col in df.columns:
            sample = df[col].dropna().astype(str).head(20).tolist()
            assessment = self._pii_aggregator.assess_column(col, sample)
            columns.append(
                ColumnSchema(
                    name=col,
                    dtype=infer_dtype(df[col]),
                    nullable=bool(df[col].isna().any()),
                    is_pii=assessment.is_sensitive,
                    is_blocked=col in self._blocked_columns,
                )
            )
        return DatasetSchema(columns=columns)

    def _refresh_profiles(self) -> None:
        self._profiles = profile_dataframe(self._current_df())

    def _current_df(self) -> pd.DataFrame:
        return self._version_history[self._version]

    def _assert_visible(self, column: str) -> None:
        schema_col = next((c for c in self._schema.columns if c.name == column), None)
        if schema_col is None:
            raise DataAccessError(f"Unknown column '{column}'")
        if schema_col.is_blocked or schema_col.is_pii:
            raise PolicyViolationError(
                f"Column '{column}' is blocked or PII-classified and cannot be accessed by an agent"
            )

    # -- DataAccessService interface -----------------------------------------

    def get_metadata(self) -> DatasetMetadata:
        df = self._current_df()
        return DatasetMetadata(
            dataset_id=self._dataset_id,
            engine=EngineType.PANDAS,
            row_count=len(df),
            column_count=df.shape[1],
            size_bytes=int(df.memory_usage(deep=True).sum()),
            is_lazy=False,
            version=self._version,
            temp_artifact_paths=[],
        )

    def get_schema(self) -> DatasetSchema:
        return self._schema

    def get_column_profile(self, column: str) -> ColumnProfile:
        self._assert_visible(column)
        if column not in self._profiles:
            raise DataAccessError(f"No profile available for column '{column}'")
        return self._profiles[column]

    def get_all_column_profiles(self) -> dict[str, ColumnProfile]:
        visible = {c.name for c in self._schema.visible_columns()}
        return {name: profile for name, profile in self._profiles.items() if name in visible}

    def get_correlation_summary(self, method: str = "pearson") -> CorrelationSummary:
        visible = [c.name for c in self._schema.visible_columns()]
        return compute_correlation_summary(self._current_df()[visible], method=method)

    def request_preview(self, columns: list[str], max_rows: int | None = None) -> PreviewResult:
        for col in columns:
            self._assert_visible(col)
        row_count = min(max_rows or self._preview_row_cap, self._preview_row_cap)
        subset = self._current_df()[columns].head(row_count)
        rows = subset.where(subset.notna(), "").astype(str).to_dict(orient="records")
        return PreviewResult(
            columns=columns,
            row_count=len(rows),
            rows=rows,
            approved_by="human",
        )

    def apply_transformation(self, name: str, params: dict) -> DatasetMetadata:
        from afe.services.transformations import get_transformation

        transformation = get_transformation(name)
        new_df = transformation(self._current_df(), params)

        self._version += 1
        self._version_history[self._version] = new_df
        self._schema = self._build_schema()
        self._refresh_profiles()
        return self.get_metadata()

    def rollback(self, to_version: int) -> DatasetMetadata:
        if to_version not in self._version_history:
            raise DataAccessError(f"No such dataset version: {to_version}")
        self._version = to_version
        self._schema = self._build_schema()
        self._refresh_profiles()
        return self.get_metadata()

    def list_versions(self) -> list[int]:
        return sorted(self._version_history.keys())
