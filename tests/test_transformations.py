from __future__ import annotations

import pandas as pd
import pytest

from afe.exceptions.errors import DataAccessError
from afe.services.transformations import get_transformation, list_transformations


def test_list_transformations_includes_core_set():
    names = list_transformations()
    for expected in ("fill_missing", "log_transform", "ratio", "standard_scale", "one_hot_encode", "binning"):
        assert expected in names


def test_fill_missing_median():
    df = pd.DataFrame({"x": [1.0, None, 3.0]})
    fn = get_transformation("fill_missing")
    out = fn(df, {"column": "x", "strategy": "median"})
    assert out["x"].isna().sum() == 0
    assert out["x"].iloc[1] == 2.0


def test_log_transform_creates_new_column():
    df = pd.DataFrame({"x": [0, 1, 10]})
    fn = get_transformation("log_transform")
    out = fn(df, {"column": "x"})
    assert "x_log" in out.columns
    assert out["x_log"].iloc[0] == 0.0


def test_ratio_transform():
    df = pd.DataFrame({"a": [10, 20], "b": [2, 4]})
    fn = get_transformation("ratio")
    out = fn(df, {"numerator": "a", "denominator": "b"})
    assert out["a_over_b"].tolist() == [5.0, 5.0]


def test_get_unknown_transformation_raises():
    with pytest.raises(DataAccessError):
        get_transformation("does_not_exist")


def test_one_hot_encode_adds_dummy_columns():
    df = pd.DataFrame({"cat": ["a", "b", "a"]})
    fn = get_transformation("one_hot_encode")
    out = fn(df, {"column": "cat"})
    assert "cat_a" in out.columns and "cat_b" in out.columns
