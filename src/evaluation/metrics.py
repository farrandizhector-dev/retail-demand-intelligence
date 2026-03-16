"""Forecast evaluation metrics for demand forecasting.

Implements all metrics required by the M5 evaluation protocol (spec section 9.6):
- MAE   : Mean Absolute Error
- RMSE  : Root Mean Squared Error
- sMAPE : Symmetric Mean Absolute Percentage Error
- Bias  : Mean signed error (ŷ − y)
- WRMSSE: Weighted Root Mean Squared Scaled Error (official M5 metric)
- Coverage@80: fraction of actuals within [p10, p90] prediction interval
- Pinball Loss: quantile loss averaged over p10/p50/p90

All functions accept plain Python lists, numpy arrays, or Polars Series.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

ArrayLike = Sequence[float] | np.ndarray


def _to_array(x: ArrayLike) -> np.ndarray:
    """Convert any array-like to a float64 numpy array."""
    if hasattr(x, "to_numpy"):  # Polars Series
        return x.to_numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# Point-forecast metrics
# ---------------------------------------------------------------------------


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Error.

    Parameters
    ----------
    y_true, y_pred:
        Arrays of actual and predicted values (same length).

    Returns
    -------
    float
    """
    a = _to_array(y_true)
    p = _to_array(y_pred)
    return float(np.mean(np.abs(a - p)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root Mean Squared Error."""
    a = _to_array(y_true)
    p = _to_array(y_pred)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def smape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Symmetric Mean Absolute Percentage Error.

    sMAPE = mean(2 * |y − ŷ| / (|y| + |ŷ|))

    Zero-denominator rows (both actual and prediction are 0) contribute 0
    to the average.
    """
    a = _to_array(y_true)
    p = _to_array(y_pred)
    denom = np.abs(a) + np.abs(p)
    num = 2.0 * np.abs(a - p)
    safe = np.where(denom == 0.0, 1.0, denom)  # avoid division by zero
    return float(np.mean(np.where(denom == 0.0, 0.0, num / safe)))


def bias(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean signed forecast error  (ŷ − y).

    Positive → model over-predicts on average.
    Negative → model under-predicts on average.
    """
    a = _to_array(y_true)
    p = _to_array(y_pred)
    return float(np.mean(p - a))


# ---------------------------------------------------------------------------
# WRMSSE — Weighted Root Mean Squared Scaled Error (M5 official metric)
# ---------------------------------------------------------------------------


def _series_scale(train_values: np.ndarray) -> float:
    """Compute the naive-forecast scale for one series.

    scale = sqrt(mean( (y_t − y_{t-1})^2 )) for t = 2..T

    Returns 1.0 when the series is too short or has zero variance
    (avoids division by zero).
    """
    if len(train_values) < 2:
        return 1.0
    diffs = np.diff(train_values.astype(np.float64))
    msd = float(np.mean(diffs ** 2))
    return float(np.sqrt(msd)) if msd > 0.0 else 1.0


def wrmsse(
    y_true: dict[str, ArrayLike],
    y_pred: dict[str, ArrayLike],
    train_series: dict[str, ArrayLike],
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted Root Mean Squared Scaled Error.

    Parameters
    ----------
    y_true:
        Dict mapping series_id → array of actuals for the test window.
    y_pred:
        Dict mapping series_id → array of predictions for the test window.
    train_series:
        Dict mapping series_id → full training series (used for scale).
    weights:
        Optional revenue weights (series_id → w_i, sum = 1).
        When None, equal weights are used.

    Returns
    -------
    float
        WRMSSE score (lower is better).
    """
    ids = list(y_true.keys())
    n = len(ids)
    if n == 0:
        return 0.0

    rmsse_values: list[float] = []
    for sid in ids:
        a = _to_array(y_true[sid])
        p = _to_array(y_pred[sid])
        scale = _series_scale(_to_array(train_series[sid]))
        rmsse_i = float(np.sqrt(np.mean((a - p) ** 2))) / scale
        rmsse_values.append(rmsse_i)

    if weights is None:
        w = np.ones(n) / n
    else:
        w = np.array([weights.get(sid, 1.0 / n) for sid in ids], dtype=np.float64)
        total = w.sum()
        w = w / total if total > 0 else np.ones(n) / n

    return float(np.dot(w, rmsse_values))


# ---------------------------------------------------------------------------
# Interval / quantile metrics
# ---------------------------------------------------------------------------


def coverage_80(
    y_true: ArrayLike,
    y_p10: ArrayLike,
    y_p90: ArrayLike,
) -> float:
    """Coverage@80: fraction of actuals inside the [p10, p90] interval.

    Parameters
    ----------
    y_true:
        Actual observed values.
    y_p10, y_p90:
        Lower (10th quantile) and upper (90th quantile) prediction bounds.

    Returns
    -------
    float in [0, 1].
    """
    a = _to_array(y_true)
    lo = _to_array(y_p10)
    hi = _to_array(y_p90)
    inside = (a >= lo) & (a <= hi)
    return float(np.mean(inside))


def pinball_loss(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    quantile: float,
) -> float:
    """Quantile (pinball) loss for a single quantile level.

    L_q(y, ŷ) = max(q*(y − ŷ), (q−1)*(y − ŷ))
               = (y − ŷ) * q  if y >= ŷ
                 (ŷ − y) * (1−q) otherwise

    Parameters
    ----------
    y_true:
        Actual values.
    y_pred:
        Predicted quantile values.
    quantile:
        Quantile level in (0, 1), e.g. 0.10 for p10, 0.90 for p90.

    Returns
    -------
    float — mean pinball loss.
    """
    a = _to_array(y_true)
    p = _to_array(y_pred)
    err = a - p
    loss = np.where(err >= 0, quantile * err, (quantile - 1.0) * err)
    return float(np.mean(loss))


def mean_pinball_loss(
    y_true: ArrayLike,
    y_p10: ArrayLike,
    y_p50: ArrayLike,
    y_p90: ArrayLike,
) -> float:
    """Average pinball loss across quantiles p10, p50, p90."""
    l10 = pinball_loss(y_true, y_p10, 0.10)
    l50 = pinball_loss(y_true, y_p50, 0.50)
    l90 = pinball_loss(y_true, y_p90, 0.90)
    return (l10 + l50 + l90) / 3.0


# ---------------------------------------------------------------------------
# Aggregate helper
# ---------------------------------------------------------------------------


def compute_all_metrics(
    y_true: ArrayLike,
    y_pred_p50: ArrayLike,
    y_pred_p10: ArrayLike | None = None,
    y_pred_p90: ArrayLike | None = None,
) -> dict[str, float]:
    """Compute all scalar metrics for a set of predictions.

    Parameters
    ----------
    y_true:
        Actual values.
    y_pred_p50:
        Point (median) predictions.
    y_pred_p10, y_pred_p90:
        Optional lower/upper quantile predictions.
        When None, coverage_80 and pinball_loss are omitted.

    Returns
    -------
    dict with keys: mae, rmse, smape, bias, (coverage_80, pinball_loss).
    """
    result: dict[str, float] = {
        "mae": mae(y_true, y_pred_p50),
        "rmse": rmse(y_true, y_pred_p50),
        "smape": smape(y_true, y_pred_p50),
        "bias": bias(y_true, y_pred_p50),
    }
    if y_pred_p10 is not None and y_pred_p90 is not None:
        result["coverage_80"] = coverage_80(y_true, y_pred_p10, y_pred_p90)
        result["pinball_loss"] = mean_pinball_loss(y_true, y_pred_p10, y_pred_p50, y_pred_p90)
    return result
