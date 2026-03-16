"""Unit tests for forecast evaluation metrics.

All tests use analytically known values to verify numeric correctness.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from src.evaluation.metrics import (
    bias,
    compute_all_metrics,
    coverage_80,
    mae,
    mean_pinball_loss,
    pinball_loss,
    rmse,
    smape,
    wrmsse,
)


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------


def test_mae_perfect():
    assert mae([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_mae_known():
    # |1-2| + |2-4| + |3-0| = 1+2+3 = 6 → mean = 2.0
    assert mae([1, 2, 3], [2, 4, 0]) == pytest.approx(2.0)


def test_mae_accepts_numpy():
    a = np.array([1.0, 2.0, 3.0])
    p = np.array([2.0, 4.0, 0.0])
    assert mae(a, p) == pytest.approx(2.0)


def test_mae_accepts_polars_series():
    assert mae(pl.Series([1, 2, 3]), pl.Series([2, 4, 0])) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------


def test_rmse_perfect():
    assert rmse([5, 5, 5], [5, 5, 5]) == pytest.approx(0.0)


def test_rmse_known():
    # errors: [1, 2, 3] → MSE = (1+4+9)/3 = 14/3 → RMSE = sqrt(14/3)
    assert rmse([1, 2, 3], [2, 4, 6]) == pytest.approx(math.sqrt(14 / 3))


# ---------------------------------------------------------------------------
# sMAPE
# ---------------------------------------------------------------------------


def test_smape_perfect():
    assert smape([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_smape_known():
    # y=2, ŷ=4: 2*|2-4|/(2+4) = 4/6 = 2/3
    assert smape([2], [4]) == pytest.approx(2 / 3, rel=1e-6)


def test_smape_both_zero():
    """Both y and ŷ = 0 → sMAPE = 0 (not nan/inf)."""
    assert smape([0], [0]) == pytest.approx(0.0)


def test_smape_mixed_zeros():
    # row 0: both 0 → 0; row 1: 2 and 4 → 2/3 → mean = 1/3
    assert smape([0, 2], [0, 4]) == pytest.approx(1 / 3, rel=1e-6)


# ---------------------------------------------------------------------------
# Bias
# ---------------------------------------------------------------------------


def test_bias_zero():
    assert bias([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_bias_over_predicting():
    # ŷ - y = [1, 1, 1] → bias = 1
    assert bias([1, 2, 3], [2, 3, 4]) == pytest.approx(1.0)


def test_bias_under_predicting():
    assert bias([2, 3, 4], [1, 2, 3]) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# WRMSSE
# ---------------------------------------------------------------------------


def test_wrmsse_perfect():
    """Perfect predictions → WRMSSE = 0."""
    y_true = {"s1": [1, 2, 3], "s2": [4, 5, 6]}
    y_pred = {"s1": [1, 2, 3], "s2": [4, 5, 6]}
    train = {"s1": [1] * 10, "s2": [4] * 10}
    assert wrmsse(y_true, y_pred, train) == pytest.approx(0.0)


def test_wrmsse_known_scale():
    """Constant training series → scale = 0 → replaced by 1 → RMSSE = raw RMSE."""
    # train all constant → diffs = 0 → scale = 1 (fallback)
    # errors: 1 each → RMSE = 1 → RMSSE = 1/1 = 1 → WRMSSE = 1
    y_true = {"s1": [2.0]}
    y_pred = {"s1": [3.0]}
    train = {"s1": [5.0, 5.0, 5.0, 5.0]}  # constant → scale = 1 (fallback)
    result = wrmsse(y_true, y_pred, train)
    assert result == pytest.approx(1.0)


def test_wrmsse_empty():
    assert wrmsse({}, {}, {}) == pytest.approx(0.0)


def test_wrmsse_with_weights():
    """Weighted average: series with weight 0 doesn't affect result."""
    y_true = {"s1": [2.0], "s2": [2.0]}
    y_pred = {"s1": [3.0], "s2": [2.0]}  # s1 has error, s2 is perfect
    train = {"s1": [5.0, 5.0], "s2": [5.0, 5.0]}
    # weight all on s2 (perfect) → WRMSSE should be 0
    weights = {"s1": 0.0, "s2": 1.0}
    result = wrmsse(y_true, y_pred, train, weights=weights)
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Coverage@80
# ---------------------------------------------------------------------------


def test_coverage_80_all_inside():
    assert coverage_80([1, 2, 3], [0, 1, 2], [2, 3, 4]) == pytest.approx(1.0)


def test_coverage_80_none_inside():
    assert coverage_80([5, 5, 5], [0, 0, 0], [1, 1, 1]) == pytest.approx(0.0)


def test_coverage_80_half():
    # row 0: 1 in [0, 2] ✓; row 1: 5 not in [0, 3] ✗ → 50%
    assert coverage_80([1, 5], [0, 0], [2, 3]) == pytest.approx(0.5)


def test_coverage_80_boundary_inclusive():
    """Boundary points are included (y == p10 or y == p90)."""
    assert coverage_80([1, 3], [1, 0], [2, 3]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Pinball Loss
# ---------------------------------------------------------------------------


def test_pinball_q50_is_half_mae():
    """At q=0.5 pinball = 0.5 * MAE."""
    y = [1.0, 2.0, 3.0]
    p = [2.0, 4.0, 0.0]
    # MAE = 2.0 → pinball_0.5 = 1.0
    assert pinball_loss(y, p, 0.5) == pytest.approx(0.5 * mae(y, p))


def test_pinball_perfect():
    assert pinball_loss([1, 2, 3], [1, 2, 3], 0.1) == pytest.approx(0.0)


def test_pinball_known_q10_underprediction():
    # y=2, ŷ=1 → err=1 > 0 → loss = 0.1 * 1 = 0.1
    assert pinball_loss([2], [1], 0.10) == pytest.approx(0.10)


def test_pinball_known_q10_overprediction():
    # y=1, ŷ=2 → err=-1 < 0 → loss = (0.1-1)*(-1) = 0.9
    assert pinball_loss([1], [2], 0.10) == pytest.approx(0.90)


def test_mean_pinball_loss_symmetric():
    """If errors are symmetric around zero, p50 loss dominates."""
    # For perfect p50, l50 = 0; l10 and l90 are symmetric
    y = np.array([1.0])
    p50 = np.array([1.0])  # perfect p50
    p10 = np.array([0.5])  # under by 0.5
    p90 = np.array([1.5])  # over by 0.5
    result = mean_pinball_loss(y, p10, p50, p90)
    # l50 = 0; l10 = 0.1*0.5 = 0.05; l90 = (1-0.9)*0.5 = 0.05 → mean = 0.1/3
    assert result == pytest.approx(0.1 / 3, rel=1e-6)


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------


def test_compute_all_metrics_keys():
    result = compute_all_metrics([1, 2, 3], [1, 2, 3], [0, 1, 2], [2, 3, 4])
    assert {"mae", "rmse", "smape", "bias", "coverage_80", "pinball_loss"}.issubset(
        set(result.keys())
    )


def test_compute_all_metrics_without_quantiles():
    result = compute_all_metrics([1, 2, 3], [1, 2, 3])
    assert "mae" in result
    assert "coverage_80" not in result


def test_compute_all_metrics_perfect():
    result = compute_all_metrics([1, 2, 3], [1, 2, 3], [0, 1, 2], [2, 3, 4])
    assert result["mae"] == pytest.approx(0.0)
    assert result["bias"] == pytest.approx(0.0)
    assert result["coverage_80"] == pytest.approx(1.0)
