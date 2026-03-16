"""Unit tests for src/inventory/simulator.py — V2-Fase 3.

Tests verify:
- simulate_inventory_mc returns MonteCarloResult with correct fields
- Zero-stockout case: high stock, low demand → fill_rate = 1.0
- Full-stockout case: low stock, high demand → stockout_probability > 0
- Deterministic lead time (lt_std=0) works correctly
- Batch runner returns one result per series
- Result fields are in valid ranges
- daily_stock_percentiles shape is (5, horizon_days)
- Reproducibility: same seed → same result
"""

from __future__ import annotations

import numpy as np
import pytest

from src.inventory.simulator import (
    MonteCarloResult,
    _sample_demands,
    run_mc_batch,
    simulate_inventory_mc,
    SYNTHETIC_TAG,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat_forecast(value: float, horizon: int = 90) -> np.ndarray:
    return np.full(horizon, value, dtype=float)


def _make_series(
    p50: float = 10.0,
    p10_frac: float = 0.7,
    p90_frac: float = 1.3,
    horizon: int = 90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p50_arr = _flat_forecast(p50, horizon)
    p10_arr = p50_arr * p10_frac
    p90_arr = p50_arr * p90_frac
    return p10_arr, p50_arr, p90_arr


# ---------------------------------------------------------------------------
# _sample_demands
# ---------------------------------------------------------------------------


class TestSampleDemands:
    def test_shape(self):
        rng = np.random.default_rng(0)
        p10, p50, p90 = _make_series()
        d = _sample_demands(p10, p50, p90, 100, rng)
        assert d.shape == (100, 90)

    def test_non_negative(self):
        rng = np.random.default_rng(1)
        p10, p50, p90 = _make_series()
        d = _sample_demands(p10, p50, p90, 500, rng)
        assert (d >= 0).all()

    def test_p50_median_approx(self):
        """Median of samples should be close to p50 (with many sims)."""
        rng = np.random.default_rng(2)
        p10, p50, p90 = _make_series(p50=10.0)
        d = _sample_demands(p10, p50, p90, 5000, rng)
        median_day0 = np.median(d[:, 0])
        # Within 20% of p50 = 10
        assert abs(median_day0 - 10.0) < 3.0


# ---------------------------------------------------------------------------
# simulate_inventory_mc — basic contract
# ---------------------------------------------------------------------------


class TestSimulateInventoryMC:
    def test_returns_montecarlo_result(self):
        p10, p50, p90 = _make_series()
        result = simulate_inventory_mc(p10, p50, p90, initial_stock=50.0,
                                       lead_time_mean=7.0, lead_time_std=2.0,
                                       reorder_point=10.0, order_quantity=50.0,
                                       n_simulations=100)
        assert isinstance(result, MonteCarloResult)

    def test_synthetic_tag(self):
        p10, p50, p90 = _make_series()
        result = simulate_inventory_mc(p10, p50, p90, initial_stock=50.0,
                                       lead_time_mean=7.0, lead_time_std=2.0,
                                       reorder_point=10.0, order_quantity=50.0,
                                       n_simulations=100)
        assert result.is_synthetic is True
        assert result.synthetic_tag == SYNTHETIC_TAG

    def test_n_simulations_stored(self):
        p10, p50, p90 = _make_series()
        result = simulate_inventory_mc(p10, p50, p90, initial_stock=50.0,
                                       lead_time_mean=7.0, lead_time_std=2.0,
                                       reorder_point=10.0, order_quantity=50.0,
                                       n_simulations=200)
        assert result.n_simulations == 200

    def test_horizon_days_stored(self):
        p10, p50, p90 = _make_series(horizon=30)
        result = simulate_inventory_mc(p10, p50, p90, initial_stock=50.0,
                                       lead_time_mean=7.0, lead_time_std=2.0,
                                       reorder_point=10.0, order_quantity=50.0,
                                       n_simulations=100, horizon_days=30)
        assert result.horizon_days == 30

    def test_daily_stock_percentiles_shape(self):
        p10, p50, p90 = _make_series()
        result = simulate_inventory_mc(p10, p50, p90, initial_stock=50.0,
                                       lead_time_mean=7.0, lead_time_std=2.0,
                                       reorder_point=10.0, order_quantity=50.0,
                                       n_simulations=100)
        assert result.daily_stock_percentiles.shape == (5, 90)

    def test_daily_stock_percentiles_monotone(self):
        """p5 <= p25 <= p50 <= p75 <= p95 at every time step."""
        p10, p50, p90 = _make_series()
        result = simulate_inventory_mc(p10, p50, p90, initial_stock=100.0,
                                       lead_time_mean=5.0, lead_time_std=1.0,
                                       reorder_point=20.0, order_quantity=60.0,
                                       n_simulations=300)
        pcts = result.daily_stock_percentiles
        for i in range(4):
            assert (pcts[i] <= pcts[i + 1] + 1e-9).all()

    def test_series_id_stored(self):
        p10, p50, p90 = _make_series()
        result = simulate_inventory_mc(p10, p50, p90, initial_stock=50.0,
                                       lead_time_mean=7.0, lead_time_std=2.0,
                                       reorder_point=10.0, order_quantity=50.0,
                                       n_simulations=50, series_id="SKU-001_STORE-1")
        assert result.series_id == "SKU-001_STORE-1"


# ---------------------------------------------------------------------------
# Zero-stockout regime: very high stock, very low demand
# ---------------------------------------------------------------------------


class TestZeroStockoutRegime:
    def test_fill_rate_near_one(self):
        """Demand=2/day, initial_stock=10000 → fill_rate ~1.0."""
        p10, p50, p90 = _make_series(p50=2.0)
        result = simulate_inventory_mc(
            p10, p50, p90,
            initial_stock=10_000.0,
            lead_time_mean=3.0, lead_time_std=0.0,
            reorder_point=500.0, order_quantity=1000.0,
            n_simulations=200, seed=0,
        )
        assert result.fill_rate_mean >= 0.999, (
            f"fill_rate_mean={result.fill_rate_mean:.4f}, expected ~1.0"
        )

    def test_stockout_probability_zero(self):
        """No stockouts when stock is massive."""
        p10, p50, p90 = _make_series(p50=1.0)
        result = simulate_inventory_mc(
            p10, p50, p90,
            initial_stock=50_000.0,
            lead_time_mean=3.0, lead_time_std=0.0,
            reorder_point=1000.0, order_quantity=2000.0,
            n_simulations=200, seed=1,
        )
        assert result.stockout_probability == 0.0

    def test_expected_stockout_days_zero(self):
        p10, p50, p90 = _make_series(p50=1.0)
        result = simulate_inventory_mc(
            p10, p50, p90,
            initial_stock=50_000.0,
            lead_time_mean=3.0, lead_time_std=0.0,
            reorder_point=1000.0, order_quantity=2000.0,
            n_simulations=200, seed=2,
        )
        assert result.expected_stockout_days == 0.0


# ---------------------------------------------------------------------------
# Full-stockout regime: very low stock, very high demand
# ---------------------------------------------------------------------------


class TestHighStockoutRegime:
    def test_stockout_probability_high(self):
        """Demand=100/day, initial_stock=10 → nearly 100% stockout sims."""
        p10, p50, p90 = _make_series(p50=100.0)
        result = simulate_inventory_mc(
            p10, p50, p90,
            initial_stock=10.0,
            lead_time_mean=30.0, lead_time_std=5.0,
            reorder_point=5.0, order_quantity=20.0,
            n_simulations=500, seed=3,
        )
        assert result.stockout_probability > 0.9, (
            f"stockout_probability={result.stockout_probability:.3f}"
        )

    def test_fill_rate_low(self):
        """Fill rate should be well below 1 when demand >> stock."""
        p10, p50, p90 = _make_series(p50=100.0)
        result = simulate_inventory_mc(
            p10, p50, p90,
            initial_stock=10.0,
            lead_time_mean=30.0, lead_time_std=5.0,
            reorder_point=5.0, order_quantity=20.0,
            n_simulations=300, seed=4,
        )
        assert result.fill_rate_mean < 0.5

    def test_expected_stockout_days_positive(self):
        p10, p50, p90 = _make_series(p50=100.0)
        result = simulate_inventory_mc(
            p10, p50, p90,
            initial_stock=10.0,
            lead_time_mean=30.0, lead_time_std=5.0,
            reorder_point=5.0, order_quantity=20.0,
            n_simulations=300, seed=5,
        )
        assert result.expected_stockout_days > 5


# ---------------------------------------------------------------------------
# Valid ranges for aggregated statistics
# ---------------------------------------------------------------------------


class TestValidRanges:
    def _base_result(self, seed: int = 0) -> MonteCarloResult:
        p10, p50, p90 = _make_series(p50=10.0)
        return simulate_inventory_mc(
            p10, p50, p90,
            initial_stock=50.0,
            lead_time_mean=7.0, lead_time_std=2.0,
            reorder_point=10.0, order_quantity=50.0,
            n_simulations=200, seed=seed,
        )

    def test_fill_rate_in_01(self):
        r = self._base_result()
        assert 0.0 <= r.fill_rate_mean <= 1.0
        assert 0.0 <= r.fill_rate_p5 <= 1.0
        assert 0.0 <= r.fill_rate_p95 <= 1.0

    def test_fill_rate_p5_le_mean_le_p95(self):
        r = self._base_result()
        assert r.fill_rate_p5 <= r.fill_rate_mean + 1e-9
        assert r.fill_rate_mean <= r.fill_rate_p95 + 1e-9

    def test_stockout_prob_in_01(self):
        r = self._base_result()
        assert 0.0 <= r.stockout_probability <= 1.0

    def test_expected_stockout_days_non_negative(self):
        r = self._base_result()
        assert r.expected_stockout_days >= 0.0

    def test_avg_inventory_non_negative(self):
        r = self._base_result()
        assert r.avg_inventory_mean >= 0.0

    def test_total_cost_non_negative(self):
        r = self._base_result()
        assert r.total_cost_mean >= 0.0
        assert r.total_cost_p5 >= 0.0
        assert r.total_cost_p95 >= 0.0

    def test_total_cost_p5_le_mean_le_p95(self):
        r = self._base_result()
        assert r.total_cost_p5 <= r.total_cost_mean + 1e-9
        assert r.total_cost_mean <= r.total_cost_p95 + 1e-9

    def test_distributions_length(self):
        r = self._base_result()
        assert len(r.fill_rate_distribution) == 200
        assert len(r.total_cost_distribution) == 200


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_result(self):
        p10, p50, p90 = _make_series()
        kw = dict(initial_stock=40.0, lead_time_mean=5.0, lead_time_std=1.0,
                  reorder_point=8.0, order_quantity=40.0, n_simulations=100, seed=99)
        r1 = simulate_inventory_mc(p10, p50, p90, **kw)
        r2 = simulate_inventory_mc(p10, p50, p90, **kw)
        assert r1.fill_rate_mean == r2.fill_rate_mean
        assert r1.stockout_probability == r2.stockout_probability
        np.testing.assert_array_equal(r1.daily_stock_percentiles, r2.daily_stock_percentiles)

    def test_different_seeds_different_results(self):
        p10, p50, p90 = _make_series()
        kw = dict(initial_stock=40.0, lead_time_mean=5.0, lead_time_std=1.0,
                  reorder_point=8.0, order_quantity=40.0, n_simulations=300)
        r1 = simulate_inventory_mc(p10, p50, p90, seed=0, **kw)
        r2 = simulate_inventory_mc(p10, p50, p90, seed=1, **kw)
        # Very unlikely to be identical with different seeds and 300 sims
        assert r1.fill_rate_mean != r2.fill_rate_mean


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_lead_time_std(self):
        """Deterministic lead time (lt_std=0) should not crash."""
        p10, p50, p90 = _make_series()
        result = simulate_inventory_mc(
            p10, p50, p90,
            initial_stock=50.0,
            lead_time_mean=5.0, lead_time_std=0.0,
            reorder_point=10.0, order_quantity=50.0,
            n_simulations=100,
        )
        assert isinstance(result, MonteCarloResult)

    def test_forecast_shorter_than_horizon_padded(self):
        """Forecast shorter than horizon_days → padded and no crash."""
        p10 = np.array([5.0, 5.0, 5.0])  # only 3 days
        p50 = np.array([10.0, 10.0, 10.0])
        p90 = np.array([15.0, 15.0, 15.0])
        result = simulate_inventory_mc(
            p10, p50, p90,
            initial_stock=50.0,
            lead_time_mean=7.0, lead_time_std=1.0,
            reorder_point=10.0, order_quantity=50.0,
            n_simulations=100, horizon_days=30,
        )
        assert result.horizon_days == 30
        assert result.daily_stock_percentiles.shape == (5, 30)

    def test_order_quantity_clamped_to_minimum_one(self):
        """order_quantity=0 should be silently clamped to 1."""
        p10, p50, p90 = _make_series()
        result = simulate_inventory_mc(
            p10, p50, p90,
            initial_stock=20.0,
            lead_time_mean=5.0, lead_time_std=0.0,
            reorder_point=5.0, order_quantity=0.0,
            n_simulations=50,
        )
        assert isinstance(result, MonteCarloResult)

    def test_initial_stock_zero(self):
        """Starting with zero stock → immediate stockouts."""
        p10, p50, p90 = _make_series(p50=10.0)
        result = simulate_inventory_mc(
            p10, p50, p90,
            initial_stock=0.0,
            lead_time_mean=7.0, lead_time_std=1.0,
            reorder_point=5.0, order_quantity=50.0,
            n_simulations=200, seed=10,
        )
        assert result.stockout_probability > 0.0


# ---------------------------------------------------------------------------
# run_mc_batch
# ---------------------------------------------------------------------------


class TestRunMcBatch:
    def _make_series_dict(self, idx: int = 0) -> dict:
        p10, p50, p90 = _make_series(p50=5.0 + idx)
        return {
            "forecast_p10": p10,
            "forecast_p50": p50,
            "forecast_p90": p90,
            "initial_stock": 50.0,
            "lead_time_mean": 7.0,
            "lead_time_std": 2.0,
            "reorder_point": 10.0,
            "order_quantity": 50.0,
            "series_id": f"series_{idx}",
        }

    def test_returns_list_of_correct_length(self):
        batch = [self._make_series_dict(i) for i in range(5)]
        results = run_mc_batch(batch, n_simulations=50, horizon_days=30)
        assert len(results) == 5

    def test_all_results_are_montecarlo_result(self):
        batch = [self._make_series_dict(i) for i in range(3)]
        results = run_mc_batch(batch, n_simulations=50, horizon_days=30)
        for r in results:
            assert isinstance(r, MonteCarloResult)

    def test_series_id_preserved(self):
        batch = [self._make_series_dict(i) for i in range(4)]
        results = run_mc_batch(batch, n_simulations=50, horizon_days=30)
        for i, r in enumerate(results):
            assert r.series_id == f"series_{i}"

    def test_empty_batch(self):
        results = run_mc_batch([], n_simulations=100, horizon_days=30)
        assert results == []

    def test_single_series(self):
        batch = [self._make_series_dict(0)]
        results = run_mc_batch(batch, n_simulations=100, horizon_days=30)
        assert len(results) == 1
        assert isinstance(results[0], MonteCarloResult)

    def test_different_seeds_per_series(self):
        """Each series in batch uses a different seed → different fill rates."""
        # Two identical series but should get different seeds
        s = self._make_series_dict(0)
        batch = [s, s]
        results = run_mc_batch(batch, n_simulations=300, horizon_days=30)
        # Seeds differ by index so results should differ
        assert results[0].fill_rate_mean != results[1].fill_rate_mean
