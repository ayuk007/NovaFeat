"""Statistical profiling of a pandas DataFrame.

This module is one of the very few places in the codebase permitted to
touch raw data directly. Its job is to compress a DataFrame into
:class:`ColumnProfile` / :class:`CorrelationSummary` objects that are safe
to hand to an LLM. It must never return a raw value, row, or sample
outside of the explicit, bounded preview path.
"""

from __future__ import annotations

import pandas as pd

from afe.models.dataset_metadata import ColumnDType, ColumnProfile, CorrelationSummary

_DTYPE_MAP = {
    "int64": ColumnDType.INTEGER,
    "int32": ColumnDType.INTEGER,
    "float64": ColumnDType.FLOAT,
    "float32": ColumnDType.FLOAT,
    "bool": ColumnDType.BOOLEAN,
    "object": ColumnDType.STRING,
    "str": ColumnDType.STRING,
    "string": ColumnDType.STRING,
    "category": ColumnDType.CATEGORICAL,
    "datetime64[ns]": ColumnDType.DATETIME,
}


def infer_dtype(series: pd.Series) -> ColumnDType:
    return _DTYPE_MAP.get(str(series.dtype), ColumnDType.UNKNOWN)


def profile_column(series: pd.Series, all_columns: dict[str, pd.Series] | None = None) -> ColumnProfile:
    """Compute a full statistical profile for one column."""
    dtype = infer_dtype(series)
    n = len(series)
    null_count = int(series.isna().sum())
    unique_count = int(series.nunique(dropna=True))

    profile = ColumnProfile(
        name=str(series.name),
        dtype=dtype,
        null_count=null_count,
        null_percentage=round((null_count / n) * 100, 4) if n else 0.0,
        unique_count=unique_count,
        cardinality_ratio=round(unique_count / n, 6) if n else 0.0,
        is_constant=unique_count <= 1,
    )

    if dtype in (ColumnDType.INTEGER, ColumnDType.FLOAT):
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if not numeric.empty:
            quantiles = numeric.quantile([0.25, 0.5, 0.75]).to_dict()
            hist_counts, hist_bins = _safe_histogram(numeric)
            profile.mean = float(numeric.mean())
            profile.median = float(numeric.median())
            profile.std = float(numeric.std()) if len(numeric) > 1 else 0.0
            profile.min = float(numeric.min())
            profile.max = float(numeric.max())
            profile.quantiles = {f"p{int(k * 100)}": float(v) for k, v in quantiles.items()}
            profile.skewness = float(numeric.skew()) if len(numeric) > 2 else 0.0
            profile.kurtosis = float(numeric.kurt()) if len(numeric) > 3 else 0.0
            profile.histogram_bins = hist_bins
            profile.histogram_counts = hist_counts
    else:
        mode_vals = series.mode(dropna=True)
        profile.mode = str(mode_vals.iloc[0]) if not mode_vals.empty else None

    if all_columns:
        profile.is_duplicate_of = _find_duplicate_column(series, all_columns)

    return profile


def _safe_histogram(numeric: pd.Series, bins: int = 10) -> tuple[list[int], list[float]]:
    try:
        counts, edges = pd.cut(numeric, bins=bins, retbins=True, duplicates="drop")
        value_counts = counts.value_counts(sort=False)
        return [int(v) for v in value_counts.tolist()], [float(e) for e in edges.tolist()]
    except Exception:  # noqa: BLE001 - histogramming is best-effort
        return [], []


def _find_duplicate_column(series: pd.Series, all_columns: dict[str, pd.Series]) -> str | None:
    for other_name, other_series in all_columns.items():
        if other_name == series.name:
            continue
        if len(other_series) == len(series) and other_series.equals(series):
            return other_name
    return None


def profile_dataframe(df: pd.DataFrame) -> dict[str, ColumnProfile]:
    columns = {col: df[col] for col in df.columns}
    return {col: profile_column(series, columns) for col, series in columns.items()}


def compute_correlation_summary(
    df: pd.DataFrame, method: str = "pearson", high_corr_threshold: float = 0.85
) -> CorrelationSummary:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return CorrelationSummary(method=method, columns=list(numeric_df.columns), matrix=[])

    corr = numeric_df.corr(method=method).fillna(0.0)
    columns = list(corr.columns)
    matrix = [[round(float(v), 6) for v in row] for row in corr.values]

    high_pairs: list[tuple[str, str, float]] = []
    for i, col_a in enumerate(columns):
        for j, col_b in enumerate(columns):
            if j <= i:
                continue
            value = matrix[i][j]
            if abs(value) >= high_corr_threshold:
                high_pairs.append((col_a, col_b, value))

    return CorrelationSummary(
        method=method, columns=columns, matrix=matrix, high_correlation_pairs=high_pairs
    )
