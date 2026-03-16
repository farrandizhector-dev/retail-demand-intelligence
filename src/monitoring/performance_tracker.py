"""Model performance decay detection (spec §11.2).

Three monitors:
  1. cusum_detector         — CUSUM on rolling MAE
  2. segment_performance_check — MAE increase by segment
  3. calibration_monitor    — coverage@80 rolling check
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. CUSUM detector
# ---------------------------------------------------------------------------

def cusum_detector(
    mae_series: list[float],
    baseline_mae: float,
    threshold_factor: float = 1.15,
    min_weeks: int = 2,
) -> dict[str, Any]:
    """CUSUM-based detection of model performance decay.

    Parameters
    ----------
    mae_series:
        Ordered list of weekly MAE values.
    baseline_mae:
        Reference MAE from training/validation.
    threshold_factor:
        Multiplier applied to baseline_mae to define the acceptable ceiling.
    min_weeks:
        Minimum consecutive weeks above threshold to declare decay.

    Returns
    -------
    dict with keys: is_decay, current_mae, baseline_mae, weeks_above,
    cusum_value, threshold.
    """
    threshold = baseline_mae * threshold_factor

    if not mae_series:
        return {
            "is_decay": False,
            "current_mae": 0.0,
            "baseline_mae": float(baseline_mae),
            "weeks_above": 0,
            "cusum_value": 0.0,
            "threshold": float(threshold),
        }

    # Compute CUSUM series
    cusum = 0.0
    cusum_series: list[float] = []
    for mae in mae_series:
        cusum = max(0.0, cusum + (mae - threshold))
        cusum_series.append(cusum)

    # Count consecutive recent weeks above threshold
    weeks_above = 0
    for mae in reversed(mae_series):
        if mae > threshold:
            weeks_above += 1
        else:
            break

    is_decay = bool(weeks_above >= min_weeks)

    return {
        "is_decay": is_decay,
        "current_mae": float(mae_series[-1]),
        "baseline_mae": float(baseline_mae),
        "weeks_above": int(weeks_above),
        "cusum_value": float(cusum_series[-1]),
        "threshold": float(threshold),
    }


# ---------------------------------------------------------------------------
# 2. Segment performance check
# ---------------------------------------------------------------------------

def segment_performance_check(
    metrics_df: pl.DataFrame,
    baseline_metrics: dict[str, float],
    segments: list[str],
) -> list[dict[str, Any]]:
    """Check MAE by segment against baseline values.

    Parameters
    ----------
    metrics_df:
        Polars DataFrame with columns [segment_col, "mae"]. The first non-"mae"
        column is treated as the segment identifier column.
    baseline_metrics:
        Mapping of segment_value → baseline MAE.
    segments:
        List of segment values to evaluate.

    Returns
    -------
    List of dicts: {segment, current_mae, baseline_mae, pct_increase, is_alert}
    for each segment in segments list.
    """
    if not segments:
        return []

    # Detect the segment column (first column that is not "mae")
    seg_col = next(
        (c for c in metrics_df.columns if c != "mae"),
        metrics_df.columns[0] if metrics_df.columns else None,
    )

    # Build lookup: segment → current mae
    if seg_col is not None and "mae" in metrics_df.columns:
        current_lookup: dict[str, float] = {
            str(row[seg_col]): float(row["mae"])
            for row in metrics_df.iter_rows(named=True)
        }
    else:
        current_lookup = {}

    results: list[dict[str, Any]] = []
    for seg in segments:
        current_mae = current_lookup.get(str(seg), 0.0)
        base_mae = float(baseline_metrics.get(str(seg), 0.0))

        if base_mae > 0:
            pct_increase = (current_mae - base_mae) / base_mae
        else:
            pct_increase = 0.0

        is_alert = bool(current_mae > base_mae * 1.25)

        results.append(
            {
                "segment": str(seg),
                "current_mae": float(current_mae),
                "baseline_mae": float(base_mae),
                "pct_increase": float(pct_increase),
                "is_alert": is_alert,
            }
        )

    return results


# ---------------------------------------------------------------------------
# 3. Calibration monitor
# ---------------------------------------------------------------------------

def calibration_monitor(
    actuals: list[float],
    p10: list[float],
    p90: list[float],
    target_coverage: float = 0.80,
) -> dict[str, Any]:
    """Check empirical coverage of prediction intervals.

    Parameters
    ----------
    actuals:
        Observed values.
    p10:
        10th percentile forecasts.
    p90:
        90th percentile forecasts.
    target_coverage:
        Expected fraction of actuals within [p10, p90].

    Returns
    -------
    dict with keys: coverage, target, is_ok, recommendation, n_samples.
    """
    n = len(actuals)

    if n == 0:
        return {
            "coverage": 0.0,
            "target": float(target_coverage),
            "is_ok": False,
            "recommendation": "recalibrate_wider",
            "n_samples": 0,
        }

    covered = sum(
        1 for a, lo, hi in zip(actuals, p10, p90)
        if lo <= a <= hi
    )
    coverage = covered / n

    is_ok = bool(0.70 <= coverage <= 0.90)

    if coverage < 0.70:
        recommendation = "recalibrate_wider"
    elif coverage > 0.90:
        recommendation = "recalibrate_narrower"
    else:
        recommendation = "ok"

    return {
        "coverage": float(coverage),
        "target": float(target_coverage),
        "is_ok": is_ok,
        "recommendation": recommendation,
        "n_samples": int(n),
    }
