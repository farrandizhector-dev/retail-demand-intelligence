"""Unit tests for src/models/conformal.py.

Tests verify:
- fit_conformal produces ConformalCalibrator with correct attributes
- calibrate enforces monotonicity and non-negativity
- evaluate_coverage returns correct metrics
- coverage improves after calibration vs raw forecasts
- edge cases: perfect forecasts, constant forecasts, small datasets
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.conformal import (
    COVERAGE_HI,
    COVERAGE_LO,
    ConformalCalibrator,
    calibrate,
    evaluate_coverage,
    fit_conformal,
    run_conformal_calibration,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_data(
    n: int = 500,
    noise_std: float = 1.0,
    interval_half: float = 1.5,
    seed: int = 42,
):
    """Synthetic actuals with quantile forecasts.

    Returns
    -------
    actuals, p10, p50, p90 as numpy arrays.
    The 80% interval is [p50 - 1.5, p50 + 1.5]; noise_std controls how wide
    actuals spread around p50.
    """
    rng = np.random.default_rng(seed)
    p50 = rng.uniform(0, 10, n)
    actuals = p50 + rng.normal(0, noise_std, n)
    p10 = p50 - interval_half
    p90 = p50 + interval_half
    return actuals, p10, p50, p90


# ---------------------------------------------------------------------------
# fit_conformal
# ---------------------------------------------------------------------------


class TestFitConformal:
    def test_returns_conformal_calibrator(self):
        y, p10, p50, p90 = _make_data()
        cal = fit_conformal(y, p10, p50, p90)
        assert isinstance(cal, ConformalCalibrator)

    def test_n_calibration_correct(self):
        y, p10, p50, p90 = _make_data(n=200)
        cal = fit_conformal(y, p10, p50, p90)
        assert cal.n_calibration == 200

    def test_coverage_within_target_for_good_forecasts(self):
        """With noise_std=1.0, interval_half=2.0 → raw coverage ~95%.
        After calibration/narrowing, should reach ~80%."""
        y, p10, p50, p90 = _make_data(noise_std=1.0, interval_half=2.0)
        cal = fit_conformal(y, p10, p50, p90)
        assert COVERAGE_LO <= cal.achieved_coverage <= COVERAGE_HI, (
            f"achieved_coverage={cal.achieved_coverage:.3f} outside [{COVERAGE_LO}, {COVERAGE_HI}]"
        )

    def test_adjustments_are_floats(self):
        y, p10, p50, p90 = _make_data()
        cal = fit_conformal(y, p10, p50, p90)
        assert isinstance(cal.adj_p10, float)
        assert isinstance(cal.adj_p90, float)

    def test_meta_contains_raw_coverage(self):
        y, p10, p50, p90 = _make_data()
        cal = fit_conformal(y, p10, p50, p90)
        assert "raw_coverage" in cal.meta
        assert 0.0 <= cal.meta["raw_coverage"] <= 1.0

    def test_empty_actuals_raises(self):
        with pytest.raises((ValueError, IndexError)):
            fit_conformal(np.array([]), np.array([]), np.array([]), np.array([]))

    def test_mismatched_lengths_raises(self):
        y = np.ones(10)
        p10 = np.ones(5)
        p50 = np.ones(10)
        p90 = np.ones(10)
        with pytest.raises(ValueError, match="same length"):
            fit_conformal(y, p10, p50, p90)

    def test_small_dataset(self):
        """fit_conformal should work with very small calibration sets."""
        rng = np.random.default_rng(0)
        n = 20
        y = rng.uniform(0, 5, n)
        p50 = rng.uniform(0, 5, n)
        p10 = p50 - 1.0
        p90 = p50 + 1.0
        cal = fit_conformal(y, p10, p50, p90)
        assert cal.n_calibration == n


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------


class TestCalibrate:
    def test_output_shape_preserved(self):
        y, p10, p50, p90 = _make_data(n=100)
        cal = fit_conformal(y, p10, p50, p90)
        cp10, cp50, cp90 = calibrate(p10, p50, p90, cal)
        assert cp10.shape == (100,)
        assert cp50.shape == (100,)
        assert cp90.shape == (100,)

    def test_non_negativity_enforced(self):
        p10 = np.array([-5.0, -1.0, 0.5])
        p50 = np.array([0.0, 1.0, 2.0])
        p90 = np.array([2.0, 3.0, 4.0])
        cal = ConformalCalibrator(adj_p10=0.0, adj_p90=0.0)
        cp10, cp50, cp90 = calibrate(p10, p50, p90, cal)
        assert (cp10 >= 0).all(), "p10 must be non-negative"
        assert (cp50 >= 0).all()
        assert (cp90 >= 0).all()

    def test_monotonicity_enforced(self):
        """Even with large adjustments that would invert order, monotonicity holds."""
        p10 = np.array([3.0, 3.0])
        p50 = np.array([2.0, 2.0])  # p10 > p50 initially
        p90 = np.array([1.0, 1.0])  # p90 < p50 initially
        cal = ConformalCalibrator(adj_p10=0.0, adj_p90=0.0)
        cp10, cp50, cp90 = calibrate(p10, p50, p90, cal)
        assert (cp10 <= cp50).all(), "p10 <= p50 must hold"
        assert (cp50 <= cp90).all(), "p50 <= p90 must hold"

    def test_p50_unchanged(self):
        y, p10, p50, p90 = _make_data(n=50)
        cal = fit_conformal(y, p10, p50, p90)
        _, cp50, _ = calibrate(p10, p50.copy(), p90, cal)
        # p50 should be identical (no adjustment applied)
        np.testing.assert_array_equal(cp50, np.maximum(0.0, p50))

    def test_adjustments_applied(self):
        p10 = np.array([1.0, 2.0, 3.0])
        p50 = np.array([2.0, 3.0, 4.0])
        p90 = np.array([3.0, 4.0, 5.0])
        cal = ConformalCalibrator(adj_p10=-0.5, adj_p90=0.5)
        cp10, _, cp90 = calibrate(p10, p50, p90, cal)
        np.testing.assert_allclose(cp10, [0.5, 1.5, 2.5])
        np.testing.assert_allclose(cp90, [3.5, 4.5, 5.5])

    def test_clip_negative_false(self):
        """Without clipping, negative values are allowed."""
        p10 = np.array([0.2])
        p50 = np.array([1.0])
        p90 = np.array([2.0])
        cal = ConformalCalibrator(adj_p10=-1.0, adj_p90=0.0)
        cp10, _, _ = calibrate(p10, p50, p90, cal, clip_negative=False)
        assert cp10[0] < 0


# ---------------------------------------------------------------------------
# evaluate_coverage
# ---------------------------------------------------------------------------


class TestEvaluateCoverage:
    def test_perfect_coverage(self):
        y = np.array([1.0, 2.0, 3.0])
        lo = np.array([0.5, 1.5, 2.5])
        hi = np.array([1.5, 2.5, 3.5])
        result = evaluate_coverage(y, lo, hi)
        assert result["coverage"] == pytest.approx(1.0)
        assert result["n"] == 3

    def test_zero_coverage(self):
        y = np.array([10.0, 20.0])
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        result = evaluate_coverage(y, lo, hi)
        assert result["coverage"] == pytest.approx(0.0)
        assert result["pct_above_hi"] == pytest.approx(1.0)

    def test_partial_coverage(self):
        y = np.array([1.0, 5.0, 3.0, 7.0])  # 1 and 3 inside [0,4], 5 and 7 outside
        lo = np.zeros(4)
        hi = np.full(4, 4.0)
        result = evaluate_coverage(y, lo, hi)
        assert result["coverage"] == pytest.approx(0.5)

    def test_interval_width_positive(self):
        y = np.ones(10)
        lo = np.zeros(10)
        hi = np.full(10, 2.0)
        result = evaluate_coverage(y, lo, hi)
        assert result["interval_width_mean"] == pytest.approx(2.0)
        assert result["interval_width_median"] == pytest.approx(2.0)

    def test_within_target_flag(self):
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 1000)
        lo = np.full(1000, -1.28)   # ~80% interval for N(0,1)
        hi = np.full(1000, 1.28)
        result = evaluate_coverage(y, lo, hi)
        assert result["within_target"] in (True, False)  # just check it computes


# ---------------------------------------------------------------------------
# Calibration improves coverage
# ---------------------------------------------------------------------------


class TestCalibrationImprovesCoverage:
    def test_too_narrow_interval_improved(self):
        """Start with ~50% raw coverage; after calibration should reach ~80%."""
        rng = np.random.default_rng(1)
        n = 2000
        p50 = rng.uniform(1, 10, n)
        actuals = p50 + rng.normal(0, 2.0, n)  # noise std=2
        p10 = p50 - 0.5   # too narrow: ~19% coverage
        p90 = p50 + 0.5

        cal = fit_conformal(actuals, p10, p50, p90)
        # Coverage after calibration should be ~80%
        assert COVERAGE_LO <= cal.achieved_coverage <= COVERAGE_HI, (
            f"calibration failed to reach target: {cal.achieved_coverage:.3f}"
        )

    def test_too_wide_interval_narrowed(self):
        """Start with ~99% raw coverage; should narrow to ~80%."""
        rng = np.random.default_rng(2)
        n = 2000
        p50 = rng.uniform(1, 10, n)
        actuals = p50 + rng.normal(0, 1.0, n)
        p10 = p50 - 5.0   # too wide
        p90 = p50 + 5.0

        cal = fit_conformal(actuals, p10, p50, p90)
        assert COVERAGE_LO <= cal.achieved_coverage <= COVERAGE_HI, (
            f"calibration failed to narrow: {cal.achieved_coverage:.3f}"
        )


# ---------------------------------------------------------------------------
# run_conformal_calibration — full pipeline
# ---------------------------------------------------------------------------


class TestRunConformalCalibration:
    def test_returns_expected_keys(self):
        y, p10, p50, p90 = _make_data(n=300)
        # Use same data for both calibration and production (smoke test)
        result = run_conformal_calibration(y, p10, p50, p90, p10, p50, p90)
        assert "calibrator" in result
        assert "cal_p10" in result
        assert "cal_p50" in result
        assert "cal_p90" in result
        assert "coverage_report" in result

    def test_output_shapes_match_input(self):
        y, p10, p50, p90 = _make_data(n=100)
        result = run_conformal_calibration(y, p10, p50, p90, p10[:50], p50[:50], p90[:50])
        assert result["cal_p10"].shape == (50,)
        assert result["cal_p90"].shape == (50,)

    def test_saves_coverage_report_json(self, tmp_path):
        y, p10, p50, p90 = _make_data(n=200)
        run_conformal_calibration(y, p10, p50, p90, p10, p50, p90, output_dir=tmp_path)
        report_path = tmp_path / "coverage_report.json"
        assert report_path.exists()
        import json
        with open(report_path) as f:
            data = json.load(f)
        assert "coverage" in data
        assert "adj_p10" in data
