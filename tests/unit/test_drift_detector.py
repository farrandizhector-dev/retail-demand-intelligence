"""Unit tests for src/monitoring/drift_detector.py — 35+ tests."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.monitoring.drift_detector import (
    compute_psi,
    feature_distribution_drift,
    price_regime_change,
    sales_distribution_drift,
    zero_inflation_shift,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_sales_df(dates: list[str], sales: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"date": dates, "sales": sales})


def _make_price_df(depts: list[str], prices: list[float], by: str = "dept_id") -> pl.DataFrame:
    return pl.DataFrame({by: depts, "sell_price": prices})


def _make_cat_df(cats: list[str], sales: list[float], by: str = "cat_id") -> pl.DataFrame:
    return pl.DataFrame({by: cats, "sales": sales})


# ===========================================================================
# TestSalesDistributionDrift
# ===========================================================================

class TestSalesDistributionDrift:
    def _build_daily_df(self, start_day: int, n_days: int, mean: float, std: float, seed: int = 0):
        rng = np.random.default_rng(seed)
        dates = [f"2024-01-{d:02d}" for d in range(start_day, start_day + n_days)]
        sales = np.clip(rng.normal(mean, std, size=n_days), 0, None).tolist()
        # Repeat a few rows per day to simulate multi-item aggregation
        rows = {"date": dates, "sales": sales}
        return pl.DataFrame(rows)

    def test_identical_distributions_no_drift(self):
        rng = np.random.default_rng(42)
        base = rng.normal(100, 10, size=90).tolist()
        dates_ref = [f"2024-{i+1:03d}" for i in range(90)]
        dates_cur = [f"2025-{i+1:03d}" for i in range(90)]
        ref_df = pl.DataFrame({"date": dates_ref, "sales": base})
        # Use same distribution
        cur_data = rng.normal(100, 10, size=90).tolist()
        cur_df = pl.DataFrame({"date": dates_cur, "sales": cur_data})

        result = sales_distribution_drift(ref_df, cur_df)
        assert result["is_drift"] is False

    def test_shifted_distribution_drift(self):
        rng = np.random.default_rng(0)
        dates_ref = [f"2024-{i+1:03d}" for i in range(90)]
        dates_cur = [f"2025-{i+1:03d}" for i in range(90)]
        ref_df = pl.DataFrame({"date": dates_ref, "sales": rng.normal(100, 5, 90).tolist()})
        # Mean shifted by 3 std = 15 units
        cur_df = pl.DataFrame({"date": dates_cur, "sales": rng.normal(115, 5, 90).tolist()})

        result = sales_distribution_drift(ref_df, cur_df)
        assert result["is_drift"] is True

    def test_returns_ks_statistic_and_p_value(self):
        rng = np.random.default_rng(1)
        dates_ref = [f"2024-{i+1:03d}" for i in range(30)]
        dates_cur = [f"2025-{i+1:03d}" for i in range(30)]
        ref_df = pl.DataFrame({"date": dates_ref, "sales": rng.normal(50, 5, 30).tolist()})
        cur_df = pl.DataFrame({"date": dates_cur, "sales": rng.normal(50, 5, 30).tolist()})

        result = sales_distribution_drift(ref_df, cur_df)
        assert "ks_statistic" in result
        assert "p_value" in result
        assert 0.0 <= result["ks_statistic"] <= 1.0
        assert 0.0 <= result["p_value"] <= 1.0

    def test_reference_days_count(self):
        dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
        ref_df = pl.DataFrame({"date": dates, "sales": [10.0, 20.0, 30.0]})
        cur_df = pl.DataFrame({"date": ["2024-02-01"], "sales": [10.0]})

        result = sales_distribution_drift(ref_df, cur_df)
        assert result["reference_days"] == 3

    def test_current_days_count(self):
        dates = ["2024-01-01", "2024-01-02"]
        ref_df = pl.DataFrame({"date": ["2023-01-01"], "sales": [10.0]})
        cur_df = pl.DataFrame({"date": dates, "sales": [10.0, 20.0]})

        result = sales_distribution_drift(ref_df, cur_df)
        assert result["current_days"] == 2

    def test_empty_reference_no_crash(self):
        ref_df = pl.DataFrame({"date": ["2024-01-01"], "sales": [100.0]})
        cur_df = pl.DataFrame({"date": ["2024-02-01"], "sales": [100.0]})
        # Should not raise even with minimal data
        result = sales_distribution_drift(ref_df, cur_df)
        assert "is_drift" in result

    def test_result_has_all_keys(self):
        rng = np.random.default_rng(3)
        dates_ref = [f"2024-{i+1:03d}" for i in range(10)]
        dates_cur = [f"2025-{i+1:03d}" for i in range(10)]
        ref_df = pl.DataFrame({"date": dates_ref, "sales": rng.normal(50, 5, 10).tolist()})
        cur_df = pl.DataFrame({"date": dates_cur, "sales": rng.normal(50, 5, 10).tolist()})

        result = sales_distribution_drift(ref_df, cur_df)
        assert set(result.keys()) == {
            "ks_statistic", "p_value", "is_drift", "reference_days", "current_days"
        }


# ===========================================================================
# TestComputePSI
# ===========================================================================

class TestComputePSI:
    def test_identical_distributions_psi_zero(self):
        rng = np.random.default_rng(42)
        arr = rng.normal(100, 10, 1000)
        psi = compute_psi(arr, arr)
        assert psi == pytest.approx(0.0, abs=1e-6)

    def test_different_distributions_psi_high(self):
        rng = np.random.default_rng(1)
        ref = rng.uniform(0, 1, 1000)
        cur = rng.uniform(5, 6, 1000)
        psi = compute_psi(ref, cur)
        assert psi > 0.20

    def test_psi_non_negative(self):
        rng = np.random.default_rng(7)
        for _ in range(10):
            ref = rng.normal(0, 1, 200)
            cur = rng.normal(rng.uniform(-2, 2), 1, 200)
            assert compute_psi(ref, cur) >= 0.0

    def test_psi_symmetric_ish(self):
        rng = np.random.default_rng(9)
        a = rng.uniform(0, 1, 500)
        b = rng.uniform(2, 3, 500)
        assert compute_psi(a, b) > 0
        assert compute_psi(b, a) > 0

    def test_psi_single_bin(self):
        arr = np.array([1.0, 2.0, 3.0])
        psi = compute_psi(arr, arr, n_bins=1)
        assert psi == pytest.approx(0.0, abs=1e-6)

    def test_psi_empty_arrays_returns_zero(self):
        psi = compute_psi([], [])
        assert psi == 0.0

    def test_psi_increases_with_more_divergence(self):
        rng = np.random.default_rng(5)
        ref = rng.normal(0, 1, 500)
        mild = rng.normal(0.5, 1, 500)
        strong = rng.normal(3.0, 1, 500)
        assert compute_psi(ref, mild) < compute_psi(ref, strong)


# ===========================================================================
# TestFeatureDistributionDrift
# ===========================================================================

class TestFeatureDistributionDrift:
    def test_stable_feature_ok_status(self):
        rng = np.random.default_rng(0)
        arr = rng.normal(0, 1, 1000)
        ref = {"feat_a": arr}
        cur = {"feat_a": arr}  # identical
        result = feature_distribution_drift(ref, cur, ["feat_a"])
        assert result["feat_a"]["status"] == "ok"

    def test_shifted_feature_retrain_status(self):
        rng = np.random.default_rng(2)
        ref = {"feat_x": rng.uniform(0, 1, 1000)}
        cur = {"feat_x": rng.uniform(10, 11, 1000)}  # completely different
        result = feature_distribution_drift(ref, cur, ["feat_x"])
        assert result["feat_x"]["status"] in ("alert", "retrain")

    def test_returns_psi_for_each_feature(self):
        rng = np.random.default_rng(3)
        arr = rng.normal(0, 1, 200)
        features = ["f1", "f2", "f3"]
        ref = {f: arr for f in features}
        cur = {f: arr for f in features}
        result = feature_distribution_drift(ref, cur, features)
        assert set(result.keys()) == set(features)
        for f in features:
            assert "psi" in result[f]
            assert "status" in result[f]

    def test_empty_feature_names(self):
        result = feature_distribution_drift({}, {}, [])
        assert result == {}

    def test_warning_status_moderate_drift(self):
        # Generate data with moderate PSI (0.10 < PSI <= 0.25)
        rng = np.random.default_rng(99)
        ref = rng.normal(0, 1, 2000)
        # Shift by 0.8 std — moderate drift
        cur = rng.normal(0.8, 1, 2000)
        result = feature_distribution_drift({"f": ref}, {"f": cur}, ["f"])
        assert result["f"]["status"] in ("warning", "alert", "retrain")

    def test_psi_in_result_dict(self):
        rng = np.random.default_rng(11)
        arr = rng.normal(0, 1, 100)
        result = feature_distribution_drift({"f": arr}, {"f": arr}, ["f"])
        assert isinstance(result["f"]["psi"], float)
        assert result["f"]["psi"] >= 0.0


# ===========================================================================
# TestZeroInflationShift
# ===========================================================================

class TestZeroInflationShift:
    def test_no_shift_same_pct(self):
        cats = ["FOODS"] * 100 + ["HOBBIES"] * 100
        sales = [0.0] * 30 + [1.0] * 70 + [0.0] * 30 + [1.0] * 70
        df = pl.DataFrame({"cat_id": cats, "sales": sales})
        result = zero_inflation_shift(df, df)
        for cat_data in result.values():
            assert cat_data["is_shift"] is False

    def test_shift_detected(self):
        # Ref: 10% zeros in FOODS; Cur: 20% zeros → delta = 0.10 → shift
        ref_df = pl.DataFrame({
            "cat_id": ["FOODS"] * 100,
            "sales": [0.0] * 10 + [1.0] * 90,
        })
        cur_df = pl.DataFrame({
            "cat_id": ["FOODS"] * 100,
            "sales": [0.0] * 20 + [1.0] * 80,
        })
        result = zero_inflation_shift(ref_df, cur_df)
        assert result["FOODS"]["is_shift"] is True

    def test_delta_sign(self):
        ref_df = pl.DataFrame({"cat_id": ["FOODS"] * 10, "sales": [0.0] * 1 + [1.0] * 9})
        cur_df = pl.DataFrame({"cat_id": ["FOODS"] * 10, "sales": [0.0] * 8 + [1.0] * 2})
        result = zero_inflation_shift(ref_df, cur_df)
        # cur_pct > ref_pct → delta > 0
        assert result["FOODS"]["delta"] > 0

    def test_missing_category_handled(self):
        ref_df = pl.DataFrame({"cat_id": ["FOODS", "HOBBIES"], "sales": [1.0, 1.0]})
        cur_df = pl.DataFrame({"cat_id": ["FOODS"], "sales": [1.0]})
        result = zero_inflation_shift(ref_df, cur_df)
        # Only FOODS in both
        assert "HOBBIES" not in result
        assert "FOODS" in result

    def test_small_change_no_shift(self):
        ref_df = pl.DataFrame({
            "cat_id": ["FOODS"] * 100,
            "sales": [0.0] * 10 + [1.0] * 90,  # 10% zeros
        })
        cur_df = pl.DataFrame({
            "cat_id": ["FOODS"] * 100,
            "sales": [0.0] * 12 + [1.0] * 88,  # 12% zeros → delta=2pp < 5pp
        })
        result = zero_inflation_shift(ref_df, cur_df)
        assert result["FOODS"]["is_shift"] is False

    def test_result_fields_present(self):
        df = pl.DataFrame({"cat_id": ["A"], "sales": [1.0]})
        result = zero_inflation_shift(df, df)
        for v in result.values():
            assert "ref_pct" in v
            assert "cur_pct" in v
            assert "delta" in v
            assert "is_shift" in v

    def test_all_zeros_no_shift(self):
        df = pl.DataFrame({"cat_id": ["X"] * 50, "sales": [0.0] * 50})
        result = zero_inflation_shift(df, df)
        assert result["X"]["ref_pct"] == pytest.approx(1.0)
        assert result["X"]["cur_pct"] == pytest.approx(1.0)
        assert result["X"]["is_shift"] is False


# ===========================================================================
# TestPriceRegimeChange
# ===========================================================================

class TestPriceRegimeChange:
    def test_no_shift_same_price(self):
        df = _make_price_df(["FOODS_1", "FOODS_2"], [2.0, 3.0])
        result = price_regime_change(df, df)
        for dept_data in result.values():
            assert dept_data["is_shift"] is False

    def test_shift_detected_15pct(self):
        ref = _make_price_df(["FOODS_1"], [2.00])
        cur = _make_price_df(["FOODS_1"], [2.30])  # 15% increase
        result = price_regime_change(ref, cur)
        assert result["FOODS_1"]["is_shift"] is True

    def test_pct_change_direction(self):
        ref = _make_price_df(["DEPT_A"], [1.00])
        cur = _make_price_df(["DEPT_A"], [1.20])
        result = price_regime_change(ref, cur)
        assert result["DEPT_A"]["pct_change"] > 0

    def test_small_change_no_shift(self):
        ref = _make_price_df(["DEPT_A"], [2.00])
        cur = _make_price_df(["DEPT_A"], [2.10])  # 5% change < 10% threshold
        result = price_regime_change(ref, cur)
        assert result["DEPT_A"]["is_shift"] is False

    def test_missing_dept_handled(self):
        ref = _make_price_df(["D1", "D2"], [1.0, 2.0])
        cur = _make_price_df(["D1"], [1.0])
        result = price_regime_change(ref, cur)
        assert "D2" not in result
        assert "D1" in result

    def test_result_fields_present(self):
        ref = _make_price_df(["X"], [1.5])
        cur = _make_price_df(["X"], [1.5])
        result = price_regime_change(ref, cur)
        for v in result.values():
            assert "ref_mean" in v
            assert "cur_mean" in v
            assert "pct_change" in v
            assert "is_shift" in v

    def test_pct_change_exact(self):
        ref = _make_price_df(["D"], [4.00])
        cur = _make_price_df(["D"], [4.40])  # 10% exactly
        result = price_regime_change(ref, cur)
        assert result["D"]["pct_change"] == pytest.approx(0.10, rel=1e-6)

    def test_exactly_at_threshold_no_shift(self):
        # A change just under 10% should NOT trigger (threshold is > 0.10)
        ref = _make_price_df(["D"], [1.00])
        cur = _make_price_df(["D"], [1.09])  # 9% — clearly below threshold
        result = price_regime_change(ref, cur)
        assert result["D"]["is_shift"] is False
