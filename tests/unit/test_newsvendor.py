"""Unit tests for src/inventory/newsvendor.py — V2-Fase 3.

Tests verify:
- Cu == Co → Q* = median of distribution
- Cu >> Co → Q* tends toward high quantile
- Cu << Co → Q* tends toward low quantile
- EOQ formula correctness
- Safety stock: SL=0.5 → z=0 → SS=0
- ROP = demand_during_LT + SS
- Invalid inputs raise ValueError
- compare_by_abc_segment returns A/B/C keys
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.inventory.newsvendor import (
    ABC_SERVICE_LEVELS,
    NewsvendorResult,
    SYNTHETIC_TAG,
    compare_by_abc_segment,
    compute_critical_ratio,
    economic_order_quantity,
    optimal_newsvendor_quantity,
    reorder_point,
    run_newsvendor_analysis,
    safety_stock,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normal_samples(mu: float = 100.0, sigma: float = 20.0, n: int = 10_000) -> np.ndarray:
    rng = np.random.default_rng(42)
    return np.maximum(0.0, rng.normal(mu, sigma, n))


def _uniform_samples(lo: float = 0.0, hi: float = 100.0, n: int = 10_000) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(lo, hi, n)


# ---------------------------------------------------------------------------
# critical_ratio
# ---------------------------------------------------------------------------


class TestCriticalRatio:
    def test_equal_costs(self):
        cr = compute_critical_ratio(cu=1.0, co=1.0)
        assert cr == pytest.approx(0.5)

    def test_cu_dominates(self):
        cr = compute_critical_ratio(cu=9.0, co=1.0)
        assert cr == pytest.approx(0.9)

    def test_co_dominates(self):
        cr = compute_critical_ratio(cu=1.0, co=9.0)
        assert cr == pytest.approx(0.1)

    def test_invalid_zero_cu(self):
        with pytest.raises(ValueError):
            compute_critical_ratio(cu=0.0, co=1.0)

    def test_invalid_negative_co(self):
        with pytest.raises(ValueError):
            compute_critical_ratio(cu=1.0, co=-1.0)


# ---------------------------------------------------------------------------
# optimal_newsvendor_quantity
# ---------------------------------------------------------------------------


class TestOptimalNewsvendorQuantity:
    def test_cu_eq_co_gives_median(self):
        """Cu = Co → critical ratio = 0.5 → Q* = median."""
        samples = _normal_samples(mu=100.0, sigma=20.0)
        q_star = optimal_newsvendor_quantity(samples, cu=1.0, co=1.0)
        median = float(np.median(samples))
        # Within 5% of median
        assert abs(q_star - median) / median < 0.05, (
            f"Q*={q_star:.2f} vs median={median:.2f}"
        )

    def test_cu_much_greater_than_co_gives_high_q(self):
        """Cu >> Co → Q* should be near 90th percentile."""
        samples = _uniform_samples(0.0, 100.0)
        q_star = optimal_newsvendor_quantity(samples, cu=9.0, co=1.0)
        p90 = float(np.percentile(samples, 90))
        # Q* should be close to 90th pct
        assert abs(q_star - p90) / max(1, p90) < 0.1

    def test_cu_much_less_than_co_gives_low_q(self):
        """Cu << Co → Q* should be near 10th percentile."""
        samples = _uniform_samples(0.0, 100.0)
        q_star = optimal_newsvendor_quantity(samples, cu=1.0, co=9.0)
        p10 = float(np.percentile(samples, 10))
        assert abs(q_star - p10) / max(1, p10) < 0.1

    def test_non_negative_result(self):
        samples = _normal_samples(mu=5.0, sigma=10.0)  # some negatives clipped at 0
        q_star = optimal_newsvendor_quantity(samples, cu=1.0, co=1.0)
        assert q_star >= 0.0

    def test_invalid_cu_zero(self):
        with pytest.raises(ValueError):
            optimal_newsvendor_quantity(np.ones(10), cu=0.0, co=1.0)

    def test_invalid_co_negative(self):
        with pytest.raises(ValueError):
            optimal_newsvendor_quantity(np.ones(10), cu=1.0, co=-1.0)

    def test_monotone_in_critical_ratio(self):
        """Higher Cu/Co → higher Q* from same distribution."""
        samples = _normal_samples()
        q_low = optimal_newsvendor_quantity(samples, cu=1.0, co=9.0)  # CR=0.1
        q_mid = optimal_newsvendor_quantity(samples, cu=1.0, co=1.0)  # CR=0.5
        q_high = optimal_newsvendor_quantity(samples, cu=9.0, co=1.0) # CR=0.9
        assert q_low <= q_mid <= q_high


# ---------------------------------------------------------------------------
# economic_order_quantity
# ---------------------------------------------------------------------------


class TestEOQ:
    def test_formula_correctness(self):
        """EOQ = sqrt(2DS/H)."""
        D, S, H = 1000.0, 50.0, 5.0
        expected = np.sqrt(2 * D * S / H)
        result = economic_order_quantity(D, S, H)
        assert result == pytest.approx(expected)

    def test_zero_demand_returns_one(self):
        assert economic_order_quantity(0.0, 50.0, 5.0) == 1.0

    def test_zero_order_cost_returns_one(self):
        assert economic_order_quantity(1000.0, 0.0, 5.0) == 1.0

    def test_zero_holding_cost_returns_one(self):
        assert economic_order_quantity(1000.0, 50.0, 0.0) == 1.0

    def test_positive_result(self):
        result = economic_order_quantity(500.0, 25.0, 2.5)
        assert result > 0

    def test_higher_demand_higher_eoq(self):
        low = economic_order_quantity(100.0, 50.0, 5.0)
        high = economic_order_quantity(1000.0, 50.0, 5.0)
        assert high > low


# ---------------------------------------------------------------------------
# safety_stock
# ---------------------------------------------------------------------------


class TestSafetyStock:
    def test_sl_half_gives_zero_ss(self):
        """At SL=0.5, z=0 → SS = 0."""
        ss = safety_stock(demand_std=10.0, lead_time=7.0, service_level=0.5)
        assert ss == pytest.approx(0.0, abs=1e-6)

    def test_higher_sl_higher_ss(self):
        ss_90 = safety_stock(10.0, 7.0, 0.90)
        ss_95 = safety_stock(10.0, 7.0, 0.95)
        ss_99 = safety_stock(10.0, 7.0, 0.99)
        assert ss_90 < ss_95 < ss_99

    def test_zero_std_gives_zero_ss(self):
        ss = safety_stock(demand_std=0.0, lead_time=7.0, service_level=0.95)
        assert ss == pytest.approx(0.0)

    def test_zero_lead_time_gives_zero_ss(self):
        ss = safety_stock(demand_std=10.0, lead_time=0.0, service_level=0.95)
        assert ss == pytest.approx(0.0)

    def test_ss_formula(self):
        """SS = z × σ × √LT."""
        sl, std, lt = 0.95, 10.0, 9.0
        z = float(stats.norm.ppf(sl))
        expected = z * std * np.sqrt(lt)
        result = safety_stock(std, lt, sl)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_negative_std_raises(self):
        with pytest.raises(ValueError):
            safety_stock(demand_std=-1.0, lead_time=7.0)

    def test_negative_lead_time_raises(self):
        with pytest.raises(ValueError):
            safety_stock(demand_std=5.0, lead_time=-1.0)

    def test_non_negative_result(self):
        ss = safety_stock(10.0, 7.0, 0.99)
        assert ss >= 0.0


# ---------------------------------------------------------------------------
# reorder_point
# ---------------------------------------------------------------------------


class TestReorderPoint:
    def test_rop_equals_demand_during_lt_plus_ss(self):
        mu, lt, std, sl = 10.0, 7.0, 3.0, 0.95
        expected_demand = mu * lt
        expected_ss = safety_stock(std, lt, sl)
        expected_rop = expected_demand + expected_ss
        result = reorder_point(mu, lt, std, sl)
        assert result == pytest.approx(expected_rop, rel=1e-6)

    def test_zero_demand_zero_std_rop_is_zero(self):
        rop = reorder_point(0.0, 7.0, 0.0, 0.95)
        assert rop == pytest.approx(0.0)

    def test_rop_non_negative(self):
        rop = reorder_point(5.0, 5.0, 2.0, 0.90)
        assert rop >= 0.0

    def test_higher_sl_higher_rop(self):
        rop_90 = reorder_point(10.0, 7.0, 3.0, 0.90)
        rop_99 = reorder_point(10.0, 7.0, 3.0, 0.99)
        assert rop_99 > rop_90


# ---------------------------------------------------------------------------
# run_newsvendor_analysis
# ---------------------------------------------------------------------------


class TestRunNewsvendorAnalysis:
    def test_returns_newsvendor_result(self):
        samples = _normal_samples()
        result = run_newsvendor_analysis(samples, cu=2.0, co=1.0)
        assert isinstance(result, NewsvendorResult)

    def test_synthetic_tag(self):
        samples = _normal_samples()
        result = run_newsvendor_analysis(samples, cu=2.0, co=1.0)
        assert result.is_synthetic is True
        assert result.synthetic_tag == SYNTHETIC_TAG

    def test_series_id_stored(self):
        samples = _normal_samples()
        result = run_newsvendor_analysis(samples, cu=2.0, co=1.0, series_id="SKU-X")
        assert result.series_id == "SKU-X"

    def test_optimal_quantity_non_negative(self):
        samples = _normal_samples()
        result = run_newsvendor_analysis(samples, cu=2.0, co=1.0)
        assert result.optimal_quantity >= 0.0

    def test_critical_ratio_correct(self):
        samples = _normal_samples()
        result = run_newsvendor_analysis(samples, cu=3.0, co=1.0)
        assert result.critical_ratio == pytest.approx(0.75)

    def test_eoq_positive(self):
        samples = _normal_samples()
        result = run_newsvendor_analysis(samples, cu=2.0, co=1.0)
        assert result.eoq > 0

    def test_safety_stock_non_negative(self):
        samples = _normal_samples()
        result = run_newsvendor_analysis(samples, cu=2.0, co=1.0)
        assert result.safety_stock >= 0.0

    def test_reorder_point_non_negative(self):
        samples = _normal_samples()
        result = run_newsvendor_analysis(samples, cu=2.0, co=1.0)
        assert result.reorder_point >= 0.0

    def test_empty_samples_raises(self):
        with pytest.raises(ValueError):
            run_newsvendor_analysis(np.array([]), cu=1.0, co=1.0)


# ---------------------------------------------------------------------------
# compare_by_abc_segment
# ---------------------------------------------------------------------------


class TestCompareByAbcSegment:
    def test_returns_abc_keys(self):
        samples = _normal_samples()
        result = compare_by_abc_segment(samples, cu=2.0, co=1.0)
        assert set(result.keys()) == {"A", "B", "C"}

    def test_all_are_newsvendor_results(self):
        samples = _normal_samples()
        result = compare_by_abc_segment(samples, cu=2.0, co=1.0)
        for v in result.values():
            assert isinstance(v, NewsvendorResult)

    def test_higher_sl_for_a_segment(self):
        """A segment has highest SL → highest safety stock."""
        samples = _normal_samples(sigma=20.0)
        result = compare_by_abc_segment(samples, cu=2.0, co=1.0)
        assert result["A"].safety_stock >= result["B"].safety_stock
        assert result["B"].safety_stock >= result["C"].safety_stock

    def test_service_levels_match_abc_defaults(self):
        samples = _normal_samples()
        result = compare_by_abc_segment(samples, cu=2.0, co=1.0)
        for seg, expected_sl in ABC_SERVICE_LEVELS.items():
            assert result[seg].service_level == pytest.approx(expected_sl)
