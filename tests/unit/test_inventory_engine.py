"""Unit tests for the inventory simulation engine.

Tests use synthetic scenarios with known, analytically predictable outcomes.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from src.inventory.engine import (
    SYNTHETIC_TAG,
    SimulationResult,
    run_inventory_simulation,
    simulate_series,
)


# ---------------------------------------------------------------------------
# simulate_series — per-series unit tests
# ---------------------------------------------------------------------------


def test_no_stockout_when_stock_exceeds_demand():
    """Ample stock → zero lost sales, fill_rate = 1."""
    demand = [2.0] * 10  # 2 units/day for 10 days = 20 total
    result = simulate_series(
        series_id="S1",
        demand_series=demand,
        initial_stock=100.0,     # way more than needed
        reorder_point=10.0,
        order_quantity=100.0,
        lead_time_days=3,
    )
    assert result.total_lost_sales == pytest.approx(0.0)
    assert result.fill_rate == pytest.approx(1.0)
    assert result.stockout_days == 0


def test_full_stockout_when_no_stock():
    """Zero stock, no replenishment possible before end → all demand lost."""
    demand = [5.0] * 5
    result = simulate_series(
        series_id="S2",
        demand_series=demand,
        initial_stock=0.0,
        reorder_point=-1.0,   # never trigger (stock never ≤ -1)
        order_quantity=0.0,
        lead_time_days=999,   # orders arrive after simulation
    )
    assert result.total_demand == pytest.approx(25.0)
    assert result.total_fulfilled == pytest.approx(0.0)
    assert result.total_lost_sales == pytest.approx(25.0)
    assert result.fill_rate == pytest.approx(0.0)
    assert result.stockout_days == 5


def test_partial_stockout():
    """Stock = 6 units, demand = 2/day → 3 days fulfilled, then stockout."""
    demand = [2.0] * 10
    result = simulate_series(
        series_id="S3",
        demand_series=demand,
        initial_stock=6.0,
        reorder_point=-1.0,  # no reorder
        order_quantity=0.0,
        lead_time_days=999,
    )
    assert result.total_fulfilled == pytest.approx(6.0)
    assert result.total_lost_sales == pytest.approx(14.0)
    assert result.stockout_days == 7


def test_replenishment_arrives_and_fulfills_demand():
    """Order placed day 0 arrives on day lead_time+0; stock replenished."""
    # 5 units stock, demand 3/day, ROP=5 (triggers immediately on day 0)
    # Lead time = 2 days → order arrives day 2
    # Day 0: stock=5, demand=3 → fulfilled=3, stock=2 ≤ ROP(5) → place order
    # Day 1: stock=2, demand=3 → fulfilled=2, lost=1
    # Day 2: order arrives (qty=5), stock=0+5=5; demand=3 → fulfilled=3, stock=2
    # Day 3+: similar
    demand = [3.0] * 10
    result = simulate_series(
        series_id="S4",
        demand_series=demand,
        initial_stock=5.0,
        reorder_point=5.0,
        order_quantity=5.0,
        lead_time_days=2,
    )
    assert result.fill_rate > 0.5  # at least partially fulfilled
    assert result.total_fulfilled + result.total_lost_sales == pytest.approx(
        sum(demand), abs=1e-6
    )
    assert result.total_orders_placed >= 1


def test_fill_rate_bounds():
    """fill_rate must always be in [0, 1]."""
    demand = [1.0] * 10
    result = simulate_series("S5", demand, 5.0, 2.0, 5.0, 3)
    assert 0.0 <= result.fill_rate <= 1.0


def test_synthetic_tags():
    """is_synthetic must be True and synthetic_tag must match constant."""
    result = simulate_series("S6", [1.0] * 5, 10.0, 2.0, 10.0, 3)
    assert result.is_synthetic is True
    assert result.synthetic_tag == SYNTHETIC_TAG


def test_zero_demand_series():
    """All-zero demand → fill_rate = 1 (nothing to fulfill), no stockouts."""
    result = simulate_series("S7", [0.0] * 10, 5.0, 2.0, 5.0, 3)
    assert result.fill_rate == pytest.approx(1.0)
    assert result.stockout_days == 0
    assert result.total_lost_sales == pytest.approx(0.0)


def test_avg_inventory_non_negative():
    result = simulate_series("S8", [2.0] * 30, 10.0, 3.0, 10.0, 5)
    assert result.avg_inventory >= 0.0


def test_days_of_supply_non_negative():
    result = simulate_series("S9", [2.0] * 30, 10.0, 3.0, 10.0, 5)
    assert result.days_of_supply >= 0.0


# ---------------------------------------------------------------------------
# run_inventory_simulation — batch tests
# ---------------------------------------------------------------------------


def _make_sales(values: list[int], series_id: str = "ITM_CA") -> pl.DataFrame:
    n = len(values)
    start = date(2015, 1, 1)
    return pl.DataFrame({
        "id": [series_id] * n,
        "date": [start + timedelta(days=i) for i in range(n)],
        "sales": values,
    }).with_columns(pl.col("date").cast(pl.Date))


def _make_params(
    series_id: str = "ITM_CA",
    initial_stock: int = 30,
    lead_time: float = 7.0,
    sl: float = 0.95,
    cat: str = "FOODS",
    abc: str = "B",
    avg_daily: float = 1.0,
) -> pl.DataFrame:
    return pl.DataFrame({
        "id": [series_id],
        "cat_id": [cat],
        "abc_class": [abc],
        "initial_stock_on_hand": [initial_stock],
        "lead_time_days": [lead_time],
        "lead_time_mean": [lead_time],
        "service_level_target": [sl],
        "avg_daily_demand": [avg_daily],
    })


def _make_ss(series_id: str = "ITM_CA", ss: float = 5.0) -> pl.DataFrame:
    return pl.DataFrame({"id": [series_id], "safety_stock": [ss]})


def _make_rop(series_id: str = "ITM_CA", rop: float = 10.0) -> pl.DataFrame:
    return pl.DataFrame({"id": [series_id], "reorder_point": [rop]})


def test_batch_returns_one_row_per_series():
    sales = _make_sales([2] * 30, "A")
    params = _make_params("A")
    ss = _make_ss("A")
    rop = _make_rop("A")
    result = run_inventory_simulation(sales, params, ss, rop)
    assert len(result) == 1


def test_batch_multi_series():
    sales = pl.concat([_make_sales([2] * 30, "A"), _make_sales([3] * 30, "B")])
    params = pl.concat([_make_params("A"), _make_params("B")])
    ss = pl.concat([_make_ss("A"), _make_ss("B")])
    rop = pl.concat([_make_rop("A"), _make_rop("B")])
    result = run_inventory_simulation(sales, params, ss, rop)
    assert len(result) == 2


def test_batch_synthetic_tag():
    sales = _make_sales([1] * 30)
    params = _make_params()
    ss = _make_ss()
    rop = _make_rop()
    result = run_inventory_simulation(sales, params, ss, rop)
    assert result["is_synthetic"][0] is True
    assert result["synthetic_tag"][0] == SYNTHETIC_TAG


def test_batch_fill_rate_in_range():
    sales = _make_sales([2] * 90)
    params = _make_params(initial_stock=200, avg_daily=2.0)
    ss = _make_ss(ss=10.0)
    rop = _make_rop(rop=20.0)
    result = run_inventory_simulation(sales, params, ss, rop)
    assert 0.0 <= result["fill_rate"][0] <= 1.0


def test_batch_required_columns():
    sales = _make_sales([1] * 30)
    params = _make_params()
    ss = _make_ss()
    rop = _make_rop()
    result = run_inventory_simulation(sales, params, ss, rop)
    required = {"id", "fill_rate", "stockout_days", "days_of_supply",
                "avg_inventory", "total_demand", "total_fulfilled",
                "total_lost_sales", "is_synthetic", "synthetic_tag"}
    assert required.issubset(set(result.columns))


def test_batch_high_stock_zero_stockouts():
    """With massive stock, no stockouts expected."""
    sales = _make_sales([3] * 90)
    params = _make_params(initial_stock=10000, avg_daily=3.0)
    ss = _make_ss(ss=100.0)
    rop = _make_rop(rop=200.0)
    result = run_inventory_simulation(sales, params, ss, rop)
    assert result["total_lost_sales"][0] == pytest.approx(0.0)
    assert result["fill_rate"][0] == pytest.approx(1.0)
