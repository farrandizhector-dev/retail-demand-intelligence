"""Unit tests for baseline forecasting models.

Uses synthetic sales series with known properties to verify:
- Output format (columns, dtypes, row count)
- Forecasts are non-negative
- All requested model names appear in the result
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from src.models.baselines import (
    HORIZON,
    forecast_croston,
    forecast_moving_average,
    forecast_seasonal_naive,
    forecast_tsb,
    run_baselines,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CUTOFF = date(2020, 3, 1)
EXPECTED_COLS = {"date", "store_id", "item_id", "forecast_p50", "model_name"}


def _make_series(
    values: list[int],
    item: str = "ITEM_1",
    store: str = "CA_1",
) -> pl.DataFrame:
    """Build a minimal long-format sales DataFrame."""
    n = len(values)
    start = date(2020, 1, 1)
    return pl.DataFrame({
        "id": [f"{item}_{store}"] * n,
        "item_id": [item] * n,
        "store_id": [store] * n,
        "date": [start + timedelta(days=i) for i in range(n)],
        "sales": values,
    }).with_columns(pl.col("date").cast(pl.Date))


def _make_multi(specs: list[tuple[list[int], str, str]]) -> pl.DataFrame:
    """Concatenate multiple series."""
    return pl.concat([_make_series(v, item, store) for v, item, store in specs])


def _assert_forecast_format(result: pl.DataFrame, n_series: int, model_name: str) -> None:
    """Assert canonical output format."""
    assert EXPECTED_COLS.issubset(set(result.columns)), f"Missing cols in {result.columns}"
    assert len(result) == n_series * HORIZON, (
        f"Expected {n_series * HORIZON} rows, got {len(result)}"
    )
    assert (result["forecast_p50"] >= 0).all(), "Negative forecasts detected"
    assert set(result["model_name"].to_list()) == {model_name}


# ---------------------------------------------------------------------------
# Per-model format tests
# ---------------------------------------------------------------------------

# 61 days of data ending at CUTOFF = 2020-03-01 (starting 2020-01-01)
N_DAYS = (CUTOFF - date(2020, 1, 1)).days + 1  # 61 days


def test_seasonal_naive_format():
    df = _make_series([max(0, 2 + (i % 7)) for i in range(N_DAYS)])
    result = forecast_seasonal_naive(df, CUTOFF)
    _assert_forecast_format(result, n_series=1, model_name="SeasonalNaive")


def test_moving_average_format():
    df = _make_series([3] * N_DAYS)
    result = forecast_moving_average(df, CUTOFF)
    _assert_forecast_format(result, n_series=1, model_name="MovingAverage28")


def test_croston_format():
    # Intermittent pattern
    sales = [0] * N_DAYS
    for i in range(0, N_DAYS, 5):
        sales[i] = 2
    df = _make_series(sales)
    result = forecast_croston(df, CUTOFF)
    _assert_forecast_format(result, n_series=1, model_name="Croston")


def test_tsb_format():
    # Lumpy pattern
    sales = [0] * N_DAYS
    for i in range(0, N_DAYS, 7):
        sales[i] = i + 1  # varying demand size
    df = _make_series(sales)
    result = forecast_tsb(df, CUTOFF)
    _assert_forecast_format(result, n_series=1, model_name="TSB")


# ---------------------------------------------------------------------------
# Seasonal Naive correctness
# ---------------------------------------------------------------------------


def test_seasonal_naive_repeats_same_weekday():
    """SeasonalNaive on constant series → constant forecast."""
    df = _make_series([5] * N_DAYS)
    result = forecast_seasonal_naive(df, CUTOFF, horizon=7)
    vals = result["forecast_p50"].to_list()
    assert all(abs(v - 5.0) < 1e-6 for v in vals), f"Expected all 5.0, got {vals}"


def test_moving_average_constant_series():
    """MA on constant series → constant forecast."""
    df = _make_series([4] * N_DAYS)
    result = forecast_moving_average(df, CUTOFF)
    vals = result["forecast_p50"].to_list()
    assert all(abs(v - 4.0) < 1e-6 for v in vals), f"Expected all 4.0, got {vals}"


# ---------------------------------------------------------------------------
# Multi-series
# ---------------------------------------------------------------------------


def test_multi_series_forecast_row_count():
    """Two series → 2 × HORIZON rows per model."""
    df = _make_multi([
        ([3] * N_DAYS, "A", "CA_1"),
        ([2] * N_DAYS, "B", "CA_1"),
    ])
    result = forecast_seasonal_naive(df, CUTOFF)
    assert len(result) == 2 * HORIZON


def test_multi_series_has_both_items():
    df = _make_multi([
        ([3] * N_DAYS, "A", "CA_1"),
        ([2] * N_DAYS, "B", "CA_1"),
    ])
    result = forecast_seasonal_naive(df, CUTOFF)
    assert set(result["item_id"].to_list()) == {"A", "B"}


# ---------------------------------------------------------------------------
# run_baselines
# ---------------------------------------------------------------------------


def test_run_baselines_all_models():
    df = _make_series([2] * N_DAYS)
    result = run_baselines(df, CUTOFF)
    # 4 models × 28 horizon rows
    assert len(result) == 4 * HORIZON
    assert set(result["model_name"].to_list()) == {
        "SeasonalNaive", "MovingAverage28", "Croston", "TSB"
    }


def test_run_baselines_subset():
    df = _make_series([2] * N_DAYS)
    result = run_baselines(df, CUTOFF, models=["SeasonalNaive", "Croston"])
    assert set(result["model_name"].to_list()) == {"SeasonalNaive", "Croston"}
    assert len(result) == 2 * HORIZON


def test_run_baselines_unknown_model_raises():
    df = _make_series([1] * N_DAYS)
    with pytest.raises(ValueError, match="Unknown model"):
        run_baselines(df, CUTOFF, models=["FakeModel"])


def test_run_baselines_non_negative():
    # Series with many zeros
    sales = [0] * N_DAYS
    for i in range(0, N_DAYS, 7):
        sales[i] = 1
    df = _make_series(sales)
    result = run_baselines(df, CUTOFF)
    assert (result["forecast_p50"] >= 0).all()
