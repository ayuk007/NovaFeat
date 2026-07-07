from __future__ import annotations

from afe.analysis.profiler import compute_correlation_summary, infer_dtype, profile_column, profile_dataframe
from afe.models.dataset_metadata import ColumnDType


def test_infer_dtype_numeric(sample_df):
    assert infer_dtype(sample_df["age"]) == ColumnDType.FLOAT
    assert infer_dtype(sample_df["income"]) == ColumnDType.INTEGER


def test_profile_column_numeric_stats(sample_df):
    profile = profile_column(sample_df["income"])
    assert profile.null_count == 0
    assert profile.mean == sample_df["income"].mean()
    assert profile.min == sample_df["income"].min()
    assert profile.max == sample_df["income"].max()
    assert profile.is_constant is False


def test_profile_column_null_percentage(sample_df):
    profile = profile_column(sample_df["age"])
    assert profile.null_count == 1
    assert profile.null_percentage == 10.0


def test_profile_dataframe_covers_all_columns(sample_df):
    profiles = profile_dataframe(sample_df)
    assert set(profiles.keys()) == set(sample_df.columns)


def test_correlation_summary_detects_high_correlation(sample_df):
    summary = compute_correlation_summary(sample_df[["age", "income"]], high_corr_threshold=0.5)
    assert summary.columns == ["age", "income"]
    assert len(summary.high_correlation_pairs) == 1


def test_correlation_summary_handles_single_numeric_column():
    import pandas as pd

    df = pd.DataFrame({"only_numeric": [1, 2, 3]})
    summary = compute_correlation_summary(df)
    assert summary.matrix == []
