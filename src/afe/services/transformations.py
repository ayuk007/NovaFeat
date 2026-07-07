"""Registered feature-engineering transformations.

Each transformation is a pure function ``(df, params) -> new_df``,
registered by name. New transformations are added by writing a function
and registering it here — no existing code needs modification
(Open/Closed via a simple registry + Strategy pattern).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from afe.exceptions.errors import DataAccessError

Transformation = Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]

_REGISTRY: dict[str, Transformation] = {}


def register(name: str) -> Callable[[Transformation], Transformation]:
    def decorator(fn: Transformation) -> Transformation:
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_transformation(name: str) -> Transformation:
    if name not in _REGISTRY:
        raise DataAccessError(f"Unknown transformation '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_transformations() -> list[str]:
    return sorted(_REGISTRY)


@register("fill_missing")
def fill_missing(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    column = params["column"]
    strategy = params.get("strategy", "median")
    out = df.copy()
    if strategy == "median":
        out[column] = out[column].fillna(out[column].median())
    elif strategy == "mean":
        out[column] = out[column].fillna(out[column].mean())
    elif strategy == "mode":
        mode = out[column].mode(dropna=True)
        out[column] = out[column].fillna(mode.iloc[0] if not mode.empty else 0)
    elif strategy == "constant":
        out[column] = out[column].fillna(params.get("value", 0))
    else:
        raise DataAccessError(f"Unknown fill strategy '{strategy}'")
    return out


@register("log_transform")
def log_transform(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    column = params["column"]
    out = df.copy()
    new_col = params.get("new_column", f"{column}_log")
    out[new_col] = np.log1p(out[column].clip(lower=0))
    return out


@register("ratio")
def ratio(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    numerator, denominator = params["numerator"], params["denominator"]
    out = df.copy()
    new_col = params.get("new_column", f"{numerator}_over_{denominator}")
    out[new_col] = out[numerator] / out[denominator].replace(0, np.nan)
    return out


@register("standard_scale")
def standard_scale(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    column = params["column"]
    out = df.copy()
    std = out[column].std()
    out[f"{column}_scaled"] = (out[column] - out[column].mean()) / (std if std else 1.0)
    return out


@register("one_hot_encode")
def one_hot_encode(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    column = params["column"]
    out = df.copy()
    dummies = pd.get_dummies(out[column], prefix=column)
    return pd.concat([out, dummies], axis=1)


@register("binning")
def binning(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    column = params["column"]
    bins = params.get("bins", 5)
    out = df.copy()
    out[f"{column}_bin"] = pd.cut(out[column], bins=bins, labels=False, duplicates="drop")
    return out


@register("date_features")
def date_features(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    column = params["column"]
    out = df.copy()
    parsed = pd.to_datetime(out[column], errors="coerce")
    out[f"{column}_year"] = parsed.dt.year
    out[f"{column}_month"] = parsed.dt.month
    out[f"{column}_dayofweek"] = parsed.dt.dayofweek
    return out
