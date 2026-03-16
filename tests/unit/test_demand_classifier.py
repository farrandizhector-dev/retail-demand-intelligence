"""Unit tests for demand classification (ADI/CV²) and ABC/XYZ.

Uses synthetic series with known analytical properties so we can verify
the numeric outputs without any real M5 data.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from src.classification.abc_xyz import (
    ABC_A_THRESHOLD,
    XYZ_X_THRESHOLD,
    XYZ_Y_THRESHOLD,
    compute_abc,
    compute_xyz,
    enrich_with_abc_xyz,
)
from src.classification.demand_classifier import (
    ADI_THRESHOLD,
    CV2_THRESHOLD,
    classify_all_series,
    classify_demand,
    compute_adi,
    compute_cv2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_series(values: list[int], item: str = "ITM_1", store: str = "CA_1") -> pl.DataFrame:
    """Build a minimal long-format sales DataFrame from a list of daily values."""
    n = len(values)
    start = date(2011, 1, 29)
    rows = {
        "id": [f"{item}_{store}"] * n,
        "item_id": [item] * n,
        "dept_id": ["FOODS_1"] * n,
        "cat_id": ["FOODS"] * n,
        "store_id": [store] * n,
        "state_id": ["CA"] * n,
        "date": [start + timedelta(days=i) for i in range(n)],
        "sales": values,
    }
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def _multi_series(*series: tuple[list[int], str, str]) -> pl.DataFrame:
    """Concatenate multiple series DataFrames."""
    return pl.concat([_make_series(v, item, store) for v, item, store in series])


# ---------------------------------------------------------------------------
# compute_adi
# ---------------------------------------------------------------------------


def test_adi_all_nonzero():
    """If every period has demand, ADI = 1.0."""
    assert compute_adi([1, 2, 3, 4, 5]) == pytest.approx(1.0)


def test_adi_every_other():
    """Demand on alternating days → ADI = 2.0."""
    sales = [1, 0, 1, 0, 1, 0]  # 3 non-zero in 6 days
    assert compute_adi(sales) == pytest.approx(2.0)


def test_adi_all_zero():
    """All-zero series → ADI = inf."""
    assert compute_adi([0, 0, 0]) == math.inf


def test_adi_single_nonzero():
    """Single non-zero in 10 days → ADI = 10."""
    assert compute_adi([0, 0, 0, 0, 0, 0, 0, 0, 0, 5]) == pytest.approx(10.0)


def test_adi_accepts_polars_series():
    assert compute_adi(pl.Series([1, 0, 1, 0])) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# compute_cv2
# ---------------------------------------------------------------------------


def test_cv2_constant_nonzero():
    """Constant non-zero demand → CV² = 0."""
    assert compute_cv2([5, 5, 5, 5]) == pytest.approx(0.0)


def test_cv2_all_zero():
    """All-zero series → CV² = 0 (fewer than 2 non-zero values)."""
    assert compute_cv2([0, 0, 0]) == pytest.approx(0.0)


def test_cv2_single_nonzero():
    """Single non-zero → CV² = 0 (insufficient data)."""
    assert compute_cv2([0, 5, 0]) == pytest.approx(0.0)


def test_cv2_known_value():
    """Hand-compute: values [2, 4], mean=3, var=1, CV²=1/9."""
    # population std of [2,4] = 1.0, mean = 3.0 → CV² = (1/3)² = 1/9
    assert compute_cv2([2, 4]) == pytest.approx(1 / 9, rel=1e-6)


def test_cv2_accepts_polars_series():
    assert compute_cv2(pl.Series([2, 4])) == pytest.approx(1 / 9, rel=1e-6)


# ---------------------------------------------------------------------------
# classify_demand (scalar)
# ---------------------------------------------------------------------------


def test_classify_smooth():
    assert classify_demand(adi=1.0, cv2=0.0) == "smooth"


def test_classify_erratic():
    assert classify_demand(adi=1.0, cv2=0.5) == "erratic"


def test_classify_intermittent():
    assert classify_demand(adi=2.0, cv2=0.3) == "intermittent"


def test_classify_lumpy():
    assert classify_demand(adi=2.0, cv2=0.6) == "lumpy"


def test_classify_boundary_adi():
    """ADI exactly at threshold → high ADI."""
    assert classify_demand(adi=ADI_THRESHOLD, cv2=0.0) == "intermittent"


def test_classify_boundary_cv2():
    """CV² exactly at threshold → high CV²."""
    assert classify_demand(adi=1.0, cv2=CV2_THRESHOLD) == "erratic"


def test_classify_inf_adi():
    """All-zero series (ADI=inf) → lumpy (high ADI + low CV²=0)."""
    assert classify_demand(adi=math.inf, cv2=0.0) == "intermittent"


# ---------------------------------------------------------------------------
# classify_all_series (batch)
# ---------------------------------------------------------------------------


def test_classify_all_series_returns_one_row_per_id():
    df = _multi_series(
        ([1, 1, 1, 1], "A", "CA_1"),
        ([0, 0, 1, 0], "B", "CA_1"),
    )
    result = classify_all_series(df)
    assert len(result) == 2
    assert set(result["id"].to_list()) == {"A_CA_1", "B_CA_1"}


def test_classify_all_series_smooth_constant():
    """Constant non-zero demand → smooth."""
    df = _make_series([5] * 20)
    result = classify_all_series(df)
    assert result["demand_class"][0] == "smooth"


def test_classify_all_series_intermittent():
    """Sparse zeros with constant non-zero values → intermittent."""
    # 10 periods, 2 non-zero → ADI=5 (>1.32); CV²=0 → intermittent
    df = _make_series([0, 0, 0, 0, 5, 0, 0, 0, 0, 5])
    result = classify_all_series(df)
    assert result["demand_class"][0] == "intermittent"


def test_classify_all_series_all_zero():
    """All-zero series → lumpy (ADI=inf, CV²=0) → intermittent."""
    df = _make_series([0] * 20)
    result = classify_all_series(df)
    # ADI=inf (>=1.32), CV²=0 (<0.49) → intermittent
    assert result["demand_class"][0] == "intermittent"


def test_classify_all_series_contains_required_columns():
    df = _make_series([1, 2, 0, 1])
    result = classify_all_series(df)
    required = {"id", "adi", "cv_squared", "demand_class", "avg_daily_demand", "pct_zero_days"}
    assert required.issubset(set(result.columns))


def test_classify_all_series_pct_zero():
    """Series with 2 zeros out of 4 → pct_zero_days = 0.5."""
    df = _make_series([1, 0, 1, 0])
    result = classify_all_series(df)
    assert result["pct_zero_days"][0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_abc
# ---------------------------------------------------------------------------


def _make_sales_for_abc() -> pl.DataFrame:
    """3 items with known revenue ranks: A=high, B=medium, C=low."""
    start = date(2011, 1, 1)
    rows = []
    # A_CA_1: 100 units/day for 10 days → 1000 units
    for i in range(10):
        rows.append(("A", "CA_1", "CA", start + timedelta(i), 100))
    # B_CA_1: 10 units/day for 10 days → 100 units
    for i in range(10):
        rows.append(("B", "CA_1", "CA", start + timedelta(i), 10))
    # C_CA_1: 1 unit/day for 10 days → 10 units
    for i in range(10):
        rows.append(("C", "CA_1", "CA", start + timedelta(i), 1))
    return pl.DataFrame(
        {"item_id": [r[0] for r in rows], "store_id": [r[1] for r in rows],
         "state_id": [r[2] for r in rows], "date": [r[3] for r in rows],
         "sales": [r[4] for r in rows]}
    ).with_columns(pl.col("date").cast(pl.Date))


def test_abc_item_a_is_class_a():
    df = _make_sales_for_abc()
    result = compute_abc(df)
    row_a = result.filter(pl.col("item_id") == "A")
    assert row_a["abc_class"][0] == "A"


def test_abc_returns_expected_columns():
    df = _make_sales_for_abc()
    result = compute_abc(df)
    assert {"item_id", "total_revenue_proxy", "abc_class"}.issubset(set(result.columns))


def test_abc_classes_are_valid():
    df = _make_sales_for_abc()
    result = compute_abc(df)
    assert set(result["abc_class"].to_list()).issubset({"A", "B", "C"})


def test_abc_all_zero_sales():
    """All-zero sales → all items get class C."""
    df = _make_sales_for_abc().with_columns(pl.lit(0).alias("sales"))
    result = compute_abc(df)
    assert all(c == "C" for c in result["abc_class"].to_list())


# ---------------------------------------------------------------------------
# compute_xyz
# ---------------------------------------------------------------------------


def _make_sales_for_xyz() -> pl.DataFrame:
    """3 items with known CV values."""
    start = date(2011, 1, 1)
    rows = []
    # X: constant demand → CV = 0
    for i in range(30):
        rows.append(("X_item", "CA_1", "CA", start + timedelta(i), 5))
    # Y: moderate variability → CV ~0.5-0.9
    for i in range(30):
        rows.append(("Y_item", "CA_1", "CA", start + timedelta(i), 5 + (i % 5) * 2))
    # Z: high variability → CV > 1 (27 zeros + 3 large bursts → CV = 3.0)
    for i in range(30):
        rows.append(("Z_item", "CA_1", "CA", start + timedelta(i), 100 if i % 10 == 0 else 0))
    return pl.DataFrame(
        {"item_id": [r[0] for r in rows], "store_id": [r[1] for r in rows],
         "state_id": [r[2] for r in rows], "date": [r[3] for r in rows],
         "sales": [r[4] for r in rows]}
    ).with_columns(pl.col("date").cast(pl.Date))


def test_xyz_stable_demand_is_x():
    df = _make_sales_for_xyz()
    result = compute_xyz(df)
    row = result.filter(pl.col("item_id") == "X_item")
    assert row["xyz_class"][0] == "X"


def test_xyz_high_variability_is_z():
    df = _make_sales_for_xyz()
    result = compute_xyz(df)
    row = result.filter(pl.col("item_id") == "Z_item")
    assert row["xyz_class"][0] == "Z"


def test_xyz_returns_expected_columns():
    df = _make_sales_for_xyz()
    result = compute_xyz(df)
    assert {"item_id", "cv", "xyz_class"}.issubset(set(result.columns))


def test_xyz_classes_are_valid():
    df = _make_sales_for_xyz()
    result = compute_xyz(df)
    assert set(result["xyz_class"].to_list()).issubset({"X", "Y", "Z"})


# ---------------------------------------------------------------------------
# enrich_with_abc_xyz
# ---------------------------------------------------------------------------


def test_enrich_with_abc_xyz_adds_abc_xyz_columns():
    sales = _make_sales_for_abc()
    # Build a minimal classification df
    classification = classify_all_series(
        sales.with_columns(
            (pl.col("item_id") + "_" + pl.col("store_id")).alias("id"),
            pl.col("item_id").alias("dept_id"),
            pl.col("item_id").alias("cat_id"),
        )
    )
    result = enrich_with_abc_xyz(classification, sales)
    assert "abc_class" in result.columns
    assert "xyz_class" in result.columns
    assert result["abc_class"].null_count() == 0
    assert result["xyz_class"].null_count() == 0
