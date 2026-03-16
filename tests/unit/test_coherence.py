"""Unit tests for src/reconciliation/evaluate_reconciliation.py.

Tests all 4 coherence checks with synthetic hierarchical data.
Each test verifies both the PASS and FAIL cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.reconciliation.evaluate_reconciliation import (
    check_bottom_up_coherence,
    check_daily_total_coherence,
    check_non_negativity,
    check_quantile_monotonicity,
    run_coherence_tests,
)
from src.reconciliation.hierarchy import build_hierarchy_matrix


# ---------------------------------------------------------------------------
# Fixtures — tiny coherent hierarchy
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_catalog() -> pd.DataFrame:
    """4 SKU-Store bottom series across 2 states × 2 categories."""
    rows = [
        ("FOODS_1_001", "FOODS_1", "FOODS", "CA_1", "CA"),
        ("FOODS_1_002", "FOODS_1", "FOODS", "TX_1", "TX"),
        ("HOBBIES_1_001", "HOBBIES_1", "HOBBIES", "CA_1", "CA"),
        ("HOBBIES_1_002", "HOBBIES_1", "HOBBIES", "TX_1", "TX"),
    ]
    return pd.DataFrame(
        rows, columns=["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    )


@pytest.fixture
def hm(tiny_catalog):
    return build_hierarchy_matrix(tiny_catalog)


def _make_coherent_forecasts(hm, n_days: int = 3) -> pd.DataFrame:
    """Build a fully coherent forecast DataFrame from bottom-level values.

    Each bottom series gets value 1.0 at every timestep.  Aggregates
    are computed by summing membership.
    """
    tags = hm.tags
    S_df = hm.S_df
    bottom_ids = list(tags["SKU-Store"])
    n_bottom = len(bottom_ids)

    # Bottom-level forecast: all 1.0
    rows = []
    for uid in bottom_ids:
        for ds in range(n_days):
            rows.append({"unique_id": uid, "ds": ds, "forecast_p50": 1.0})

    # Aggregates: sum of S matrix row × bottom values (all 1.0)
    data_cols = [c for c in S_df.columns if c != "unique_id"]
    for _, row in S_df.iterrows():
        uid = row["unique_id"]
        if uid in set(bottom_ids):
            continue
        agg_val = float(row[data_cols].astype(float).sum())
        for ds in range(n_days):
            rows.append({"unique_id": uid, "ds": ds, "forecast_p50": agg_val})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: Bottom-up coherence
# ---------------------------------------------------------------------------


class TestBottomUpCoherence:
    def test_coherent_forecasts_pass(self, hm):
        df = _make_coherent_forecasts(hm)
        result = check_bottom_up_coherence(df, hm.S_df, hm.tags)
        assert result["passed"] is True
        assert result["n_violations"] == 0

    def test_incoherent_forecasts_fail(self, hm):
        df = _make_coherent_forecasts(hm)
        # Corrupt Total series to 999.0
        df.loc[df["unique_id"] == "Total", "forecast_p50"] = 999.0
        result = check_bottom_up_coherence(df, hm.S_df, hm.tags)
        assert result["passed"] is False
        assert result["n_violations"] > 0
        assert result["max_abs_error"] > 0.01

    def test_max_error_reported(self, hm):
        df = _make_coherent_forecasts(hm)
        n_bottom = len(hm.tags["SKU-Store"])
        # Introduce error of 5.0 in Total
        df.loc[df["unique_id"] == "Total", "forecast_p50"] = n_bottom + 5.0
        result = check_bottom_up_coherence(df, hm.S_df, hm.tags)
        assert abs(result["max_abs_error"] - 5.0) < 0.01

    def test_atol_respected(self, hm):
        df = _make_coherent_forecasts(hm)
        n_bottom = len(hm.tags["SKU-Store"])
        # Introduce a small error (< 0.01) — should still pass with default atol
        df.loc[df["unique_id"] == "Total", "forecast_p50"] = n_bottom + 0.005
        result = check_bottom_up_coherence(df, hm.S_df, hm.tags, atol=0.01)
        assert result["passed"] is True


# ---------------------------------------------------------------------------
# Test 2: Non-negativity
# ---------------------------------------------------------------------------


class TestNonNegativity:
    def test_all_positive_passes(self, hm):
        df = _make_coherent_forecasts(hm)
        result = check_non_negativity(df)
        assert result["passed"] is True
        assert result["n_negatives"] == 0

    def test_negative_values_fail(self, hm):
        df = _make_coherent_forecasts(hm)
        df.loc[0, "forecast_p50"] = -1.0
        result = check_non_negativity(df)
        assert result["passed"] is False
        assert result["n_negatives"] >= 1
        assert result["min_value"] < 0

    def test_zero_is_not_negative(self, hm):
        df = _make_coherent_forecasts(hm)
        df.loc[0, "forecast_p50"] = 0.0
        result = check_non_negativity(df)
        assert result["passed"] is True

    def test_missing_column_returns_failed(self, hm):
        df = pd.DataFrame({"unique_id": ["A"], "ds": [0], "other": [1.0]})
        result = check_non_negativity(df, model_col="forecast_p50")
        assert result["passed"] is False
        assert "error" in result

    def test_reports_min_value(self, hm):
        df = _make_coherent_forecasts(hm)
        df.loc[0, "forecast_p50"] = -3.14
        result = check_non_negativity(df)
        assert result["min_value"] == pytest.approx(-3.14)


# ---------------------------------------------------------------------------
# Test 3: Quantile monotonicity
# ---------------------------------------------------------------------------


class TestQuantileMonotonicity:
    def _add_quantiles(self, df: pd.DataFrame, offset: float = 0.5) -> pd.DataFrame:
        df = df.copy()
        df["forecast_p10"] = (df["forecast_p50"] - offset).clip(lower=0)
        df["forecast_p90"] = df["forecast_p50"] + offset
        return df

    def test_monotone_quantiles_pass(self, hm):
        df = self._add_quantiles(_make_coherent_forecasts(hm))
        result = check_quantile_monotonicity(df)
        assert result["passed"] is True
        assert result["n_p10_violations"] == 0
        assert result["n_p90_violations"] == 0

    def test_p10_exceeds_p50_fails(self, hm):
        df = self._add_quantiles(_make_coherent_forecasts(hm))
        # Set p10 > p50 at index 0
        df.loc[0, "forecast_p10"] = df.loc[0, "forecast_p50"] + 2.0
        result = check_quantile_monotonicity(df)
        assert result["passed"] is False
        assert result["n_p10_violations"] >= 1

    def test_p90_below_p50_fails(self, hm):
        df = self._add_quantiles(_make_coherent_forecasts(hm))
        df.loc[0, "forecast_p90"] = df.loc[0, "forecast_p50"] - 2.0
        result = check_quantile_monotonicity(df)
        assert result["passed"] is False
        assert result["n_p90_violations"] >= 1

    def test_skipped_when_no_quantile_columns(self, hm):
        df = _make_coherent_forecasts(hm)  # only forecast_p50
        result = check_quantile_monotonicity(df)
        assert result["passed"] is True
        assert result.get("skipped") is True

    def test_partial_quantiles_allowed(self, hm):
        """Only p50+p90 present — only p90 check runs."""
        df = _make_coherent_forecasts(hm)
        df["forecast_p90"] = df["forecast_p50"] + 0.5
        result = check_quantile_monotonicity(df)
        assert result["passed"] is True


# ---------------------------------------------------------------------------
# Test 4: Daily total coherence
# ---------------------------------------------------------------------------


class TestDailyTotalCoherence:
    def test_coherent_passes(self, hm):
        df = _make_coherent_forecasts(hm)
        result = check_daily_total_coherence(df, hm.tags)
        assert result["passed"] is True
        assert result["n_violations"] == 0

    def test_incoherent_total_fails(self, hm):
        df = _make_coherent_forecasts(hm)
        # Corrupt Total for one day
        df.loc[(df["unique_id"] == "Total") & (df["ds"] == 0), "forecast_p50"] = 999.0
        result = check_daily_total_coherence(df, hm.tags)
        assert result["passed"] is False
        assert result["n_violations"] >= 1
        assert result["max_abs_error"] > 1.0

    def test_atol_tolerance(self, hm):
        df = _make_coherent_forecasts(hm)
        n_bottom = len(hm.tags["SKU-Store"])
        # Add tiny error (< 1.0) to Total
        df.loc[(df["unique_id"] == "Total"), "forecast_p50"] = n_bottom + 0.5
        result = check_daily_total_coherence(df, hm.tags, atol=1.0)
        assert result["passed"] is True

    def test_no_total_series_skipped(self, hm):
        df = _make_coherent_forecasts(hm)
        # Remove Total
        df = df[df["unique_id"] != "Total"]
        result = check_daily_total_coherence(df, hm.tags)
        assert result.get("skipped") is True
        assert result["passed"] is True

    def test_reports_dates_checked(self, hm):
        n_days = 5
        df = _make_coherent_forecasts(hm, n_days=n_days)
        result = check_daily_total_coherence(df, hm.tags)
        assert result["n_dates_checked"] == n_days


# ---------------------------------------------------------------------------
# Orchestrator: run_coherence_tests
# ---------------------------------------------------------------------------


class TestRunCoherenceTests:
    def test_all_pass_on_coherent_data(self, hm):
        df = _make_coherent_forecasts(hm)
        results = run_coherence_tests(df, hm.S_df, hm.tags)
        assert results["all_passed"] is True
        assert results["bottom_up_coherence"]["passed"] is True
        assert results["non_negativity"]["passed"] is True
        assert results["quantile_monotonicity"]["passed"] is True
        assert results["daily_total_coherence"]["passed"] is True

    def test_all_keys_present(self, hm):
        df = _make_coherent_forecasts(hm)
        results = run_coherence_tests(df, hm.S_df, hm.tags)
        expected_keys = {
            "bottom_up_coherence",
            "non_negativity",
            "quantile_monotonicity",
            "daily_total_coherence",
            "all_passed",
        }
        assert set(results.keys()) == expected_keys

    def test_partial_failure_detected(self, hm):
        df = _make_coherent_forecasts(hm)
        # Introduce one negative value
        df.loc[0, "forecast_p50"] = -0.5
        results = run_coherence_tests(df, hm.S_df, hm.tags)
        assert results["all_passed"] is False
        assert results["non_negativity"]["passed"] is False
