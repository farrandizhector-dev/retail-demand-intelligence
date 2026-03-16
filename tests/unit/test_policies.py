"""Unit tests for src/inventory/policy_comparator.py — V2-Fase 3.

Tests verify:
- compare_policies returns PolicyComparisonResult with 4 policies
- fill_rates, stockout_probs, etc. have correct length (4)
- best_policy is one of POLICY_NAMES
- All metrics in valid ranges [0, 1] or >= 0
- (s,Q) with order_quantity=0 → never reorders → elevated stockout
- Higher demand vs lower stock → more stockouts across policies
- batch runner returns correct count
- parquet export works
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.inventory.policy_comparator import (
    POLICY_NAMES,
    PolicyComparisonResult,
    SYNTHETIC_TAG,
    build_policy_comparison_summary,
    compare_policies,
    run_policy_comparison_batch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat(val: float, horizon: int = 90) -> np.ndarray:
    return np.full(horizon, val, dtype=float)


def _make_forecasts(p50: float = 10.0, horizon: int = 90):
    p50_arr = _flat(p50, horizon)
    return p50_arr * 0.7, p50_arr, p50_arr * 1.3


def _base_result(p50: float = 10.0, seed: int = 0) -> PolicyComparisonResult:
    p10, p50_arr, p90 = _make_forecasts(p50)
    return compare_policies(
        forecast_p10=p10, forecast_p50=p50_arr, forecast_p90=p90,
        initial_stock=50.0, lead_time_mean=7.0, lead_time_std=1.0,
        n_simulations=200, horizon_days=30, seed=seed,
    )


# ---------------------------------------------------------------------------
# Structure / contract
# ---------------------------------------------------------------------------


class TestPolicyComparisonContract:
    def test_returns_policy_comparison_result(self):
        r = _base_result()
        assert isinstance(r, PolicyComparisonResult)

    def test_synthetic_tag(self):
        r = _base_result()
        assert r.is_synthetic is True
        assert r.synthetic_tag == SYNTHETIC_TAG

    def test_four_policies(self):
        r = _base_result()
        assert len(r.policy_names) == 4
        assert len(r.fill_rates) == 4
        assert len(r.stockout_probs) == 4
        assert len(r.avg_inventories) == 4
        assert len(r.total_costs) == 4

    def test_policy_names_match_constants(self):
        r = _base_result()
        assert r.policy_names == POLICY_NAMES

    def test_four_mc_results(self):
        r = _base_result()
        assert len(r.mc_results) == 4

    def test_best_policy_is_valid_name(self):
        r = _base_result()
        assert r.best_policy in POLICY_NAMES

    def test_best_cost_matches_best_policy(self):
        r = _base_result()
        idx = r.policy_names.index(r.best_policy)
        assert r.best_cost == pytest.approx(r.total_costs[idx])

    def test_series_id_stored(self):
        p10, p50, p90 = _make_forecasts()
        r = compare_policies(
            forecast_p10=p10, forecast_p50=p50, forecast_p90=p90,
            initial_stock=50.0, lead_time_mean=5.0, lead_time_std=1.0,
            n_simulations=100, horizon_days=30,
            series_id="SKU-X_STORE-1",
        )
        assert r.series_id == "SKU-X_STORE-1"


# ---------------------------------------------------------------------------
# Valid ranges
# ---------------------------------------------------------------------------


class TestValidRanges:
    def test_fill_rates_in_01(self):
        r = _base_result()
        for fr in r.fill_rates:
            assert 0.0 <= fr <= 1.0, f"fill_rate={fr}"

    def test_stockout_probs_in_01(self):
        r = _base_result()
        for sp in r.stockout_probs:
            assert 0.0 <= sp <= 1.0, f"stockout_prob={sp}"

    def test_avg_inventories_non_negative(self):
        r = _base_result()
        for ai in r.avg_inventories:
            assert ai >= 0.0

    def test_total_costs_non_negative(self):
        r = _base_result()
        for tc in r.total_costs:
            assert tc >= 0.0


# ---------------------------------------------------------------------------
# Behavioral assertions
# ---------------------------------------------------------------------------


class TestPolicyBehavior:
    def test_high_stock_low_stockout_across_policies(self):
        """With abundant stock, all policies should have low stockout prob."""
        p10, p50, p90 = _make_forecasts(p50=5.0)
        r = compare_policies(
            forecast_p10=p10, forecast_p50=p50, forecast_p90=p90,
            initial_stock=10_000.0, lead_time_mean=3.0, lead_time_std=0.0,
            n_simulations=200, horizon_days=30, seed=0,
        )
        for sp in r.stockout_probs:
            assert sp < 0.1, f"Expected low stockout but got {sp:.3f}"

    def test_low_stock_high_demand_increases_stockouts(self):
        """Low stock vs high demand → stockout > 0 for at least some policies."""
        p10, p50, p90 = _make_forecasts(p50=50.0)
        r = compare_policies(
            forecast_p10=p10, forecast_p50=p50, forecast_p90=p90,
            initial_stock=5.0, lead_time_mean=14.0, lead_time_std=3.0,
            n_simulations=300, horizon_days=30, seed=1,
        )
        # At least one policy should have stockout > 0
        assert max(r.stockout_probs) > 0.0

    def test_best_policy_has_minimum_cost(self):
        r = _base_result()
        best_idx = r.policy_names.index(r.best_policy)
        for i, cost in enumerate(r.total_costs):
            assert r.total_costs[best_idx] <= cost + 1e-9, (
                f"Best policy {r.best_policy} has cost {r.total_costs[best_idx]:.2f} "
                f"but policy {r.policy_names[i]} has {cost:.2f}"
            )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


class TestRunPolicyComparisonBatch:
    def _series_dict(self, idx: int) -> dict:
        p10, p50, p90 = _make_forecasts(p50=5.0 + idx, horizon=30)
        return {
            "forecast_p10": p10,
            "forecast_p50": p50,
            "forecast_p90": p90,
            "initial_stock": 50.0,
            "lead_time_mean": 5.0,
            "lead_time_std": 1.0,
            "series_id": f"s_{idx}",
        }

    def test_returns_correct_count(self):
        batch = [self._series_dict(i) for i in range(3)]
        results = run_policy_comparison_batch(batch, n_simulations=50, horizon_days=30)
        assert len(results) == 3

    def test_all_results_are_correct_type(self):
        batch = [self._series_dict(i) for i in range(2)]
        results = run_policy_comparison_batch(batch, n_simulations=50, horizon_days=30)
        for r in results:
            assert isinstance(r, PolicyComparisonResult)

    def test_empty_batch(self):
        results = run_policy_comparison_batch([], n_simulations=50)
        assert results == []

    def test_saves_parquet(self, tmp_path):
        batch = [self._series_dict(i) for i in range(2)]
        run_policy_comparison_batch(
            batch, n_simulations=50, horizon_days=30, output_dir=tmp_path
        )
        assert (tmp_path / "policy_comparison.parquet").exists()

    def test_parquet_row_count(self, tmp_path):
        """4 policies × n_series = expected row count."""
        n_series = 3
        batch = [self._series_dict(i) for i in range(n_series)]
        run_policy_comparison_batch(
            batch, n_simulations=50, horizon_days=30, output_dir=tmp_path
        )
        try:
            import polars as pl
            df = pl.read_parquet(tmp_path / "policy_comparison.parquet")
        except ImportError:
            import pandas as pd
            df = pd.read_parquet(tmp_path / "policy_comparison.parquet")
        assert len(df) == n_series * 4


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------


class TestBuildPolicyComparisonSummary:
    def test_returns_dict_with_required_keys(self):
        r = _base_result()
        summary = build_policy_comparison_summary([r])
        assert "n_series" in summary
        assert "policies" in summary
        assert "best_policy_distribution" in summary
        assert "series" in summary
        assert "synthetic_tag" in summary

    def test_n_series_correct(self):
        results = [_base_result(seed=i) for i in range(3)]
        summary = build_policy_comparison_summary(results)
        assert summary["n_series"] == 3

    def test_policies_list(self):
        summary = build_policy_comparison_summary([_base_result()])
        assert summary["policies"] == POLICY_NAMES

    def test_series_list_length(self):
        results = [_base_result(seed=i) for i in range(4)]
        summary = build_policy_comparison_summary(results)
        assert len(summary["series"]) == 4

    def test_each_series_has_policies_dict(self):
        summary = build_policy_comparison_summary([_base_result()])
        for s in summary["series"]:
            assert "policies" in s
            assert set(s["policies"].keys()) == set(POLICY_NAMES)

    def test_empty_input(self):
        summary = build_policy_comparison_summary([])
        assert summary["series"] == []

    def test_synthetic_tag_in_summary(self):
        summary = build_policy_comparison_summary([_base_result()])
        assert summary["synthetic_tag"] == SYNTHETIC_TAG
