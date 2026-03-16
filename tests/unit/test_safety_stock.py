"""Unit tests for safety stock and reorder point calculations.

Uses analytically known values to verify numeric correctness.
"""

from __future__ import annotations

import math

import pytest

from src.inventory.reorder_point import (
    reorder_point,
    reorder_point_avg_demand,
    reorder_point_from_arrays,
)
from src.inventory.safety_stock import (
    safety_stock_classic,
    safety_stock_quantile,
    safety_stock_quantile_from_arrays,
    z_score,
)


# ---------------------------------------------------------------------------
# z_score
# ---------------------------------------------------------------------------


def test_z_score_95():
    """z(0.95) ≈ 1.645."""
    assert z_score(0.95) == pytest.approx(1.6449, abs=1e-3)


def test_z_score_98():
    """z(0.98) ≈ 2.054."""
    assert z_score(0.98) == pytest.approx(2.0537, abs=1e-3)


def test_z_score_90():
    """z(0.90) ≈ 1.282."""
    assert z_score(0.90) == pytest.approx(1.2816, abs=1e-3)


def test_z_score_invalid_zero():
    with pytest.raises(ValueError):
        z_score(0.0)


def test_z_score_invalid_one():
    with pytest.raises(ValueError):
        z_score(1.0)


def test_z_score_invalid_gt_one():
    with pytest.raises(ValueError):
        z_score(1.5)


# ---------------------------------------------------------------------------
# safety_stock_classic
# ---------------------------------------------------------------------------


def test_classic_known_value():
    """SS = z(0.95) × 2 × sqrt(9) = 1.645 × 2 × 3 = 9.869."""
    ss = safety_stock_classic(service_level=0.95, sigma_forecast=2.0, lead_time_mean=9.0)
    assert ss == pytest.approx(1.6449 * 2.0 * 3.0, rel=1e-3)


def test_classic_sl_90():
    """z(0.90) × 1 × sqrt(4) = 1.282 × 1 × 2 = 2.564."""
    ss = safety_stock_classic(0.90, 1.0, 4.0)
    assert ss == pytest.approx(1.2816 * 2.0, rel=1e-2)


def test_classic_zero_sigma():
    """Zero demand variability → SS = 0."""
    assert safety_stock_classic(0.95, 0.0, 7.0) == pytest.approx(0.0)


def test_classic_zero_lead_time():
    """Zero lead time → SS = 0."""
    assert safety_stock_classic(0.95, 2.0, 0.0) == pytest.approx(0.0)


def test_classic_non_negative():
    """SS must always be ≥ 0 even with weird inputs."""
    assert safety_stock_classic(0.01, 100.0, 7.0) >= 0.0


# ---------------------------------------------------------------------------
# safety_stock_quantile
# ---------------------------------------------------------------------------


def test_quantile_basic():
    """SS = max(0, p90_cum - p50_cum)."""
    assert safety_stock_quantile(100.0, 130.0) == pytest.approx(30.0)


def test_quantile_clipped_to_zero():
    """When p90 < p50 (shouldn't happen in practice), SS = 0."""
    assert safety_stock_quantile(130.0, 100.0) == pytest.approx(0.0)


def test_quantile_equal():
    """When p90 == p50 → SS = 0."""
    assert safety_stock_quantile(100.0, 100.0) == pytest.approx(0.0)


def test_quantile_from_arrays_truncated_to_lt():
    """Only first round(LT) days of forecast are used."""
    # LT=3: sum first 3: p50=[1,1,1,10,10] → sum=3; p90=[2,2,2,10,10] → sum=6; SS=3
    p50 = [1.0, 1.0, 1.0, 10.0, 10.0]
    p90 = [2.0, 2.0, 2.0, 10.0, 10.0]
    ss = safety_stock_quantile_from_arrays(p50, p90, lead_time_mean=3.0)
    assert ss == pytest.approx(3.0)


def test_quantile_from_arrays_short_forecast():
    """If forecast shorter than LT, uses all available."""
    ss = safety_stock_quantile_from_arrays([1.0, 2.0], [3.0, 4.0], lead_time_mean=10.0)
    # sum p50 = 3, sum p90 = 7 → SS = 4
    assert ss == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# reorder_point
# ---------------------------------------------------------------------------


def test_rop_basic():
    """ROP = expected_demand + SS."""
    assert reorder_point(50.0, 10.0) == pytest.approx(60.0)


def test_rop_zero_ss():
    assert reorder_point(50.0, 0.0) == pytest.approx(50.0)


def test_rop_clipped_to_zero():
    """Negative expected demand + SS below zero → ROP = 0."""
    assert reorder_point(-5.0, 0.0) == pytest.approx(0.0)


def test_rop_avg_demand():
    """ROP = avg × LT + SS = 2 × 7 + 5 = 19."""
    assert reorder_point_avg_demand(2.0, 7.0, 5.0) == pytest.approx(19.0)


def test_rop_avg_demand_zero_lt():
    """Zero lead time → expected demand = 0 → ROP = SS."""
    assert reorder_point_avg_demand(5.0, 0.0, 3.0) == pytest.approx(3.0)


def test_rop_from_arrays():
    """sum(p50[:LT]) + SS = (1+2+3) + 5 = 11."""
    p50 = [1.0, 2.0, 3.0, 99.0, 99.0]
    rop = reorder_point_from_arrays(p50, safety_stock=5.0, lead_time_mean=3.0)
    assert rop == pytest.approx(11.0)


# ---------------------------------------------------------------------------
# synthetic_generator integration
# ---------------------------------------------------------------------------


def test_synthetic_params_synthetic_tag():
    """All generated rows must have is_synthetic=True and synthetic_tag=SYNTHETIC."""
    from src.inventory.synthetic_generator import generate_synthetic_params
    import polars as pl

    classification = pl.DataFrame({
        "id": ["ITM_1_CA_1", "ITM_2_TX_1"],
        "item_id": ["ITM_1", "ITM_2"],
        "store_id": ["CA_1", "TX_1"],
        "cat_id": ["FOODS", "HOUSEHOLD"],
        "abc_class": ["A", "C"],
        "avg_daily_demand": [5.0, 2.0],
    })
    params = generate_synthetic_params(classification)
    assert params["is_synthetic"].to_list() == [True, True]
    assert all(v == "SYNTHETIC" for v in params["synthetic_tag"].to_list())


def test_synthetic_params_service_levels():
    """A=0.98, B=0.95, C=0.90."""
    from src.inventory.synthetic_generator import generate_synthetic_params
    import polars as pl

    classification = pl.DataFrame({
        "id": ["A_CA_1", "B_CA_1", "C_CA_1"],
        "item_id": ["A", "B", "C"],
        "store_id": ["CA_1"] * 3,
        "cat_id": ["FOODS"] * 3,
        "abc_class": ["A", "B", "C"],
        "avg_daily_demand": [10.0, 5.0, 1.0],
    })
    params = generate_synthetic_params(classification)
    sl = {row["id"]: row["service_level_target"] for row in params.iter_rows(named=True)}
    assert sl["A_CA_1"] == pytest.approx(0.98)
    assert sl["B_CA_1"] == pytest.approx(0.95)
    assert sl["C_CA_1"] == pytest.approx(0.90)


def test_synthetic_params_initial_stock():
    """initial_stock = round(avg_daily_demand × 30)."""
    from src.inventory.synthetic_generator import generate_synthetic_params
    import polars as pl

    classification = pl.DataFrame({
        "id": ["X_CA_1"],
        "item_id": ["X"],
        "store_id": ["CA_1"],
        "cat_id": ["FOODS"],
        "abc_class": ["B"],
        "avg_daily_demand": [4.0],
    })
    params = generate_synthetic_params(classification)
    assert params["initial_stock_on_hand"][0] == 120  # round(4 × 30)


def test_synthetic_params_lead_time_positive():
    from src.inventory.synthetic_generator import generate_synthetic_params
    import polars as pl

    classification = pl.DataFrame({
        "id": [f"ITM_{i}_CA_1" for i in range(10)],
        "item_id": [f"ITM_{i}" for i in range(10)],
        "store_id": ["CA_1"] * 10,
        "cat_id": ["HOBBIES"] * 10,
        "abc_class": ["C"] * 10,
        "avg_daily_demand": [1.0] * 10,
    })
    params = generate_synthetic_params(classification)
    assert (params["lead_time_days"] > 0).all()
    assert (params["lead_time_mean"] > 0).all()
