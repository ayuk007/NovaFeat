from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, None, 50, 22, 60, 45, 33],
            "income": [50000, 60000, 55000, 80000, 62000, 90000, 48000, 100000, 75000, 58000],
            "email": [f"user{i}@example.com" for i in range(10)],
            "customer_id": [f"CUST{i:04d}" for i in range(10)],
            "category": ["a", "b", "a", "c", "b", "a", "c", "b", "a", "c"],
        }
    )
