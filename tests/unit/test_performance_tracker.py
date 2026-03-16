"""Unit tests for src/monitoring/performance_tracker.py — 25+ tests."""

from __future__ import annotations

import polars as pl
import pytest

from src.monitoring.performance_tracker import (
    calibration_monitor,
    cusum_detector,
    segment_performance_check,
)


# ===========================================================================
# TestCusumDetector
# ===========================================================================

class TestCusumDetector:
    def test_flat_mae_no_decay(self):
        # MAE exactly at baseline (below threshold)
        baseline = 10.0
        mae_series = [10.0] * 8
        result = cusum_detector(mae_series, baseline)
        assert result["is_decay"] is False

    def test_increasing_mae_decay(self):
        # MAE escalating well above threshold for 3 consecutive weeks
        baseline = 10.0
        mae_series = [10.0, 10.0, 10.0, 10.0, 12.0, 12.5, 13.0]  # last 3 above 11.5
        result = cusum_detector(mae_series, baseline, threshold_factor=1.15, min_weeks=2)
        assert result["is_decay"] is True

    def test_weeks_above_count(self):
        baseline = 10.0
        # threshold = 10 * 1.15 = 11.5; last 3 are 12, 12.5, 13
        mae_series = [10.0, 10.0, 12.0, 12.5, 13.0]
        result = cusum_detector(mae_series, baseline)
        assert result["weeks_above"] == 3

    def test_empty_series(self):
        result = cusum_detector([], 10.0)
        assert result["is_decay"] is False
        assert result["current_mae"] == 0.0

    def test_single_week_no_decay(self):
        # Only 1 week above threshold but min_weeks=2
        baseline = 10.0
        mae_series = [10.0, 10.0, 12.0]
        result = cusum_detector(mae_series, baseline, min_weeks=2)
        # Last 1 week above → weeks_above=1 < 2
        assert result["is_decay"] is False

    def test_current_mae_is_last(self):
        mae_series = [9.0, 10.0, 11.0, 12.0]
        result = cusum_detector(mae_series, 10.0)
        assert result["current_mae"] == pytest.approx(12.0)

    def test_cusum_value_non_negative_at_end_of_decay(self):
        baseline = 10.0
        mae_series = [12.0] * 4  # all above threshold
        result = cusum_detector(mae_series, baseline)
        assert result["cusum_value"] >= 0.0

    def test_baseline_in_return(self):
        baseline = 7.5
        result = cusum_detector([7.5] * 4, baseline)
        assert result["baseline_mae"] == pytest.approx(7.5)

    def test_threshold_in_return(self):
        baseline = 10.0
        factor = 1.2
        result = cusum_detector([10.0], baseline, threshold_factor=factor)
        assert result["threshold"] == pytest.approx(12.0)

    def test_cusum_resets_after_below_threshold(self):
        # CUSUM resets to 0 when MAE drops below threshold
        baseline = 10.0
        mae_series = [12.0, 12.0, 9.0, 9.0]  # dips below after 2 weeks
        result = cusum_detector(mae_series, baseline, min_weeks=2)
        # After dip, weeks_above should reset
        assert result["weeks_above"] == 0

    def test_min_weeks_respected(self):
        baseline = 10.0
        mae_series = [12.0, 12.0]  # exactly min_weeks=2
        result = cusum_detector(mae_series, baseline, min_weeks=2)
        assert result["is_decay"] is True

    def test_one_week_below_resets_consecutive_count(self):
        baseline = 10.0
        mae_series = [12.0, 9.0, 12.0]  # interrupted → only 1 consecutive at end
        result = cusum_detector(mae_series, baseline, min_weeks=2)
        assert result["is_decay"] is False


# ===========================================================================
# TestSegmentPerformanceCheck
# ===========================================================================

class TestSegmentPerformanceCheck:
    def _make_df(self, segments: list[str], maes: list[float]) -> pl.DataFrame:
        return pl.DataFrame({"segment": segments, "mae": maes})

    def test_no_alert_below_threshold(self):
        df = self._make_df(["FOODS"], [24.0])
        # 24 vs baseline 20 → 20% increase → below 25%
        result = segment_performance_check(df, {"FOODS": 20.0}, ["FOODS"])
        assert result[0]["is_alert"] is False

    def test_alert_above_threshold(self):
        df = self._make_df(["FOODS"], [26.0])
        # 26 vs baseline 20 → 30% increase > 25%
        result = segment_performance_check(df, {"FOODS": 20.0}, ["FOODS"])
        assert result[0]["is_alert"] is True

    def test_returns_all_segments(self):
        df = self._make_df(["A", "B", "C"], [10.0, 20.0, 30.0])
        baseline = {"A": 10.0, "B": 20.0, "C": 30.0}
        result = segment_performance_check(df, baseline, ["A", "B", "C"])
        assert len(result) == 3
        returned_segs = {r["segment"] for r in result}
        assert returned_segs == {"A", "B", "C"}

    def test_pct_increase_correct(self):
        df = self._make_df(["SEG"], [15.0])
        result = segment_performance_check(df, {"SEG": 10.0}, ["SEG"])
        assert result[0]["pct_increase"] == pytest.approx(0.5, rel=1e-6)

    def test_empty_segments(self):
        df = self._make_df([], [])
        result = segment_performance_check(df, {}, [])
        assert result == []

    def test_missing_segment_in_df_defaults_to_zero(self):
        df = self._make_df(["A"], [10.0])
        result = segment_performance_check(df, {"A": 10.0, "MISSING": 5.0}, ["A", "MISSING"])
        seg_map = {r["segment"]: r for r in result}
        # MISSING not in df → current_mae defaults to 0
        assert seg_map["MISSING"]["current_mae"] == 0.0

    def test_result_fields_present(self):
        df = self._make_df(["A"], [5.0])
        result = segment_performance_check(df, {"A": 5.0}, ["A"])
        assert set(result[0].keys()) == {
            "segment", "current_mae", "baseline_mae", "pct_increase", "is_alert"
        }

    def test_exactly_25pct_not_alert(self):
        df = self._make_df(["X"], [12.5])
        # 12.5 vs 10.0 → exactly 25% increase → threshold is > 1.25, so not alert
        result = segment_performance_check(df, {"X": 10.0}, ["X"])
        assert result[0]["is_alert"] is False


# ===========================================================================
# TestCalibrationMonitor
# ===========================================================================

class TestCalibrationMonitor:
    def test_perfect_coverage(self):
        actuals = [5.0, 6.0, 7.0]
        p10 = [0.0, 0.0, 0.0]
        p90 = [10.0, 10.0, 10.0]
        result = calibration_monitor(actuals, p10, p90)
        assert result["coverage"] == pytest.approx(1.0)

    def test_zero_coverage(self):
        actuals = [5.0, 6.0, 7.0]
        p10 = [20.0, 20.0, 20.0]
        p90 = [30.0, 30.0, 30.0]
        result = calibration_monitor(actuals, p10, p90)
        assert result["coverage"] == pytest.approx(0.0)

    def test_target_coverage_in_result(self):
        result = calibration_monitor([1.0], [0.0], [2.0], target_coverage=0.80)
        assert result["target"] == pytest.approx(0.80)

    def test_ok_at_80pct(self):
        # 8 out of 10 inside interval
        actuals = [1.0] * 8 + [100.0, -100.0]
        p10 = [0.0] * 10
        p90 = [2.0] * 10
        result = calibration_monitor(actuals, p10, p90, target_coverage=0.80)
        assert result["is_ok"] is True
        assert result["recommendation"] == "ok"

    def test_too_narrow_recalibrate_wider(self):
        # Only 6 out of 10 inside → 0.60 coverage < 0.70
        actuals = [1.0] * 6 + [100.0] * 4
        p10 = [0.0] * 10
        p90 = [2.0] * 10
        result = calibration_monitor(actuals, p10, p90)
        assert result["recommendation"] == "recalibrate_wider"
        assert result["is_ok"] is False

    def test_too_wide_recalibrate_narrower(self):
        # All 10 inside → 1.0 coverage > 0.90
        actuals = [1.0] * 10
        p10 = [0.0] * 10
        p90 = [2.0] * 10
        result = calibration_monitor(actuals, p10, p90)
        assert result["recommendation"] == "recalibrate_narrower"
        assert result["is_ok"] is False

    def test_n_samples_count(self):
        actuals = list(range(20))
        p10 = [a - 1 for a in actuals]
        p90 = [a + 1 for a in actuals]
        result = calibration_monitor(actuals, p10, p90)
        assert result["n_samples"] == 20

    def test_empty_actuals_no_crash(self):
        result = calibration_monitor([], [], [])
        assert "coverage" in result
        assert result["n_samples"] == 0

    def test_boundary_coverage_70_is_ok(self):
        # Exactly 70% coverage
        actuals = [1.0] * 7 + [100.0] * 3
        p10 = [0.0] * 10
        p90 = [2.0] * 10
        result = calibration_monitor(actuals, p10, p90)
        assert result["is_ok"] is True
