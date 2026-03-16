"""Conformal post-hoc calibration of LightGBM quantile forecasts.

Implements conformal quantile regression (spec section 9.3):
- Uses the last backtesting fold (fold 5) as calibration set.
- For each quantile level (p10, p90):
    residual = actual − forecast_q  (for every calibration observation)
  Then:
    adjustment_p10 = quantile(residuals_p10, 0.10)
    adjustment_p90 = quantile(residuals_p90, 0.90)
- Calibrated intervals:
    calibrated_p10 = forecast_p10 + adjustment_p10
    calibrated_p90 = forecast_p90 + adjustment_p90
- Verified against target coverage 0.80 (±0.05 tolerance: 0.75 – 0.85).
  If coverage is outside range, adjustments are widened/narrowed iteratively.
- Monotonicity enforced: p10 ≤ p50 ≤ p90.
- Non-negativity enforced: all values ≥ 0.

Coverage target (spec §9.3): 80 % ± 5 %  (i.e. 75 % – 85 %).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COVERAGE = 0.80
COVERAGE_TOL = 0.05          # ±5 %
COVERAGE_LO = TARGET_COVERAGE - COVERAGE_TOL  # 0.75
COVERAGE_HI = TARGET_COVERAGE + COVERAGE_TOL  # 0.85
MAX_ITER = 50                 # widening iterations
WIDEN_STEP = 0.01             # coverage adjustment step per iteration


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ConformalCalibrator:
    """Calibration artefact produced by ``fit_conformal``.

    Attributes
    ----------
    adj_p10:
        Additive adjustment applied to raw p10 forecasts.
    adj_p90:
        Additive adjustment applied to raw p90 forecasts.
    achieved_coverage:
        Empirical coverage on the calibration set after adjustments.
    n_calibration:
        Number of observations used for fitting.
    coverage_within_target:
        True if 0.75 ≤ achieved_coverage ≤ 0.85.
    meta:
        Additional diagnostics (residual stats, etc.).
    """

    adj_p10: float = 0.0
    adj_p90: float = 0.0
    achieved_coverage: float = 0.0
    n_calibration: int = 0
    coverage_within_target: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_conformal(
    actuals: np.ndarray,
    forecast_p10: np.ndarray,
    forecast_p50: np.ndarray,
    forecast_p90: np.ndarray,
    target_coverage: float = TARGET_COVERAGE,
    coverage_tol: float = COVERAGE_TOL,
    max_iter: int = MAX_ITER,
) -> ConformalCalibrator:
    """Fit conformal calibration adjustments from a calibration fold.

    Parameters
    ----------
    actuals:
        Ground-truth values on the calibration set.
    forecast_p10:
        Raw p10 quantile forecasts on the calibration set.
    forecast_p50:
        Raw p50 (median) forecasts (used for diagnostics only).
    forecast_p90:
        Raw p90 quantile forecasts on the calibration set.
    target_coverage:
        Desired empirical coverage (default 0.80).
    coverage_tol:
        Acceptable deviation from target (default ±0.05).
    max_iter:
        Max widening/narrowing iterations if initial coverage misses target.

    Returns
    -------
    ConformalCalibrator
    """
    actuals = np.asarray(actuals, dtype=float)
    fp10 = np.asarray(forecast_p10, dtype=float)
    fp50 = np.asarray(forecast_p50, dtype=float)
    fp90 = np.asarray(forecast_p90, dtype=float)

    n = len(actuals)
    if n == 0:
        raise ValueError("actuals must be non-empty.")
    if not (len(fp10) == len(fp90) == n):
        raise ValueError("actuals, forecast_p10, forecast_p90 must have the same length.")

    # ── Compute residuals ──────────────────────────────────────────────────
    # For a p10 quantile: residual = actual - forecast_p10
    # Under-coverage of p10 means p10 was too HIGH (actual < p10) → negative residuals
    residuals_p10 = actuals - fp10  # shape (n,)
    residuals_p90 = actuals - fp90  # shape (n,)

    # ── Initial adjustments ────────────────────────────────────────────────
    # For p10: we want ~10 % of actuals below calibrated_p10 → use q(residuals, 0.10)
    # After calibration: calibrated_p10 = fp10 + adj_p10
    # actual ≥ calibrated_p10  ⟺  actual - fp10 ≥ adj_p10
    # i.e. adj_p10 = quantile(residuals_p10, 0.10)  → ~10% will be below
    adj_p10 = float(np.quantile(residuals_p10, 0.10))
    # For p90: calibrated_p90 = fp90 + adj_p90
    # actual ≤ calibrated_p90  ⟺  actual - fp90 ≤ adj_p90
    # i.e. adj_p90 = quantile(residuals_p90, 0.90)  → ~90% will be below
    adj_p90 = float(np.quantile(residuals_p90, 0.90))

    # ── Evaluate coverage ──────────────────────────────────────────────────
    def _coverage(a10: float, a90: float) -> float:
        cal_lo = fp10 + a10
        cal_hi = fp90 + a90
        return float(np.mean((actuals >= cal_lo) & (actuals <= cal_hi)))

    coverage = _coverage(adj_p10, adj_p90)
    lo = target_coverage - coverage_tol
    hi = target_coverage + coverage_tol

    # ── Iterative widening / narrowing ─────────────────────────────────────
    for _ in range(max_iter):
        if lo <= coverage <= hi:
            break
        if coverage < lo:
            # Widen interval: decrease p10 (shift down), increase p90 (shift up)
            adj_p10 -= WIDEN_STEP
            adj_p90 += WIDEN_STEP
        else:  # coverage > hi
            # Narrow interval
            adj_p10 += WIDEN_STEP
            adj_p90 -= WIDEN_STEP
        coverage = _coverage(adj_p10, adj_p90)

    within_target = lo <= coverage <= hi
    if not within_target:
        logger.warning(
            "Conformal calibration: achieved_coverage=%.3f outside [%.2f, %.2f] "
            "after %d iterations.",
            coverage, lo, hi, max_iter,
        )

    # ── Diagnostics ────────────────────────────────────────────────────────
    raw_coverage = float(np.mean((actuals >= fp10) & (actuals <= fp90)))
    bias = float(np.mean(fp50 - actuals))

    logger.info(
        "Conformal fit: n=%d, raw_coverage=%.3f, adj_p10=%.4f, adj_p90=%.4f, "
        "achieved_coverage=%.3f",
        n, raw_coverage, adj_p10, adj_p90, coverage,
    )

    return ConformalCalibrator(
        adj_p10=adj_p10,
        adj_p90=adj_p90,
        achieved_coverage=coverage,
        n_calibration=n,
        coverage_within_target=within_target,
        meta={
            "raw_coverage": raw_coverage,
            "bias_p50": bias,
            "residuals_p10_q50": float(np.quantile(residuals_p10, 0.50)),
            "residuals_p90_q50": float(np.quantile(residuals_p90, 0.50)),
            "target_coverage": target_coverage,
            "coverage_tol": coverage_tol,
        },
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate(
    forecast_p10: np.ndarray,
    forecast_p50: np.ndarray,
    forecast_p90: np.ndarray,
    calibrator: ConformalCalibrator,
    clip_negative: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply conformal adjustments to raw quantile forecasts.

    Parameters
    ----------
    forecast_p10, forecast_p50, forecast_p90:
        Raw quantile forecasts to calibrate.
    calibrator:
        Fitted ``ConformalCalibrator`` from ``fit_conformal``.
    clip_negative:
        If True, clip all outputs to ≥ 0 (default True).

    Returns
    -------
    (cal_p10, cal_p50, cal_p90) — calibrated quantile arrays.
    All arrays have the same dtype (float64) and shape as the inputs.
    Monotonicity (p10 ≤ p50 ≤ p90) and non-negativity are enforced.
    """
    p10 = np.asarray(forecast_p10, dtype=float).copy()
    p50 = np.asarray(forecast_p50, dtype=float).copy()
    p90 = np.asarray(forecast_p90, dtype=float).copy()

    # Apply adjustments
    cal_p10 = p10 + calibrator.adj_p10
    cal_p90 = p90 + calibrator.adj_p90
    cal_p50 = p50  # p50 is not adjusted (only the interval bounds)

    # ── Enforce monotonicity: p10 ≤ p50 ≤ p90 ───────────────────────────
    # 1. p10 must not exceed p50
    cal_p10 = np.minimum(cal_p10, cal_p50)
    # 2. p90 must not be below p50
    cal_p90 = np.maximum(cal_p90, cal_p50)
    # 3. p10 must not exceed p90
    cal_p10 = np.minimum(cal_p10, cal_p90)

    # ── Non-negativity ────────────────────────────────────────────────────
    if clip_negative:
        cal_p10 = np.maximum(0.0, cal_p10)
        cal_p50 = np.maximum(0.0, cal_p50)
        cal_p90 = np.maximum(0.0, cal_p90)

    return cal_p10, cal_p50, cal_p90


# ---------------------------------------------------------------------------
# Coverage evaluation
# ---------------------------------------------------------------------------


def evaluate_coverage(
    actuals: np.ndarray,
    cal_p10: np.ndarray,
    cal_p90: np.ndarray,
) -> dict[str, Any]:
    """Evaluate empirical coverage of calibrated prediction intervals.

    Parameters
    ----------
    actuals:
        Ground-truth values.
    cal_p10:
        Calibrated lower bound (p10).
    cal_p90:
        Calibrated upper bound (p90).

    Returns
    -------
    dict with keys:
        ``coverage``, ``n``, ``within_target``, ``interval_width_mean``,
        ``interval_width_median``, ``pct_below_lo``, ``pct_above_hi``.
    """
    y = np.asarray(actuals, dtype=float)
    lo = np.asarray(cal_p10, dtype=float)
    hi = np.asarray(cal_p90, dtype=float)

    n = len(y)
    inside = (y >= lo) & (y <= hi)
    below = y < lo
    above = y > hi

    coverage = float(inside.mean())
    interval_width = hi - lo

    return {
        "coverage": coverage,
        "n": n,
        "within_target": COVERAGE_LO <= coverage <= COVERAGE_HI,
        "interval_width_mean": float(interval_width.mean()),
        "interval_width_median": float(np.median(interval_width)),
        "pct_below_lo": float(below.mean()),
        "pct_above_hi": float(above.mean()),
    }


# ---------------------------------------------------------------------------
# Full pipeline: fit + calibrate + evaluate
# ---------------------------------------------------------------------------


def run_conformal_calibration(
    calibration_actuals: np.ndarray,
    calibration_p10: np.ndarray,
    calibration_p50: np.ndarray,
    calibration_p90: np.ndarray,
    forecast_p10: np.ndarray,
    forecast_p50: np.ndarray,
    forecast_p90: np.ndarray,
    output_dir: Path | None = None,
    mlflow_run=None,
) -> dict[str, Any]:
    """Fit calibrator on fold-5 data and apply to production forecasts.

    Parameters
    ----------
    calibration_*:
        Arrays from the calibration fold (fold 5 actuals + forecasts).
    forecast_*:
        Production quantile forecasts to calibrate.
    output_dir:
        If provided, saves ``calibration_report.json`` to this directory.
    mlflow_run:
        Active MLflow run for logging coverage metrics.

    Returns
    -------
    dict with keys:
        ``calibrator``, ``cal_p10``, ``cal_p50``, ``cal_p90``,
        ``coverage_report``.
    """
    calibrator = fit_conformal(
        calibration_actuals, calibration_p10, calibration_p50, calibration_p90
    )

    cal_p10, cal_p50, cal_p90 = calibrate(
        forecast_p10, forecast_p50, forecast_p90, calibrator
    )

    coverage_report = evaluate_coverage(calibration_actuals, calibration_p10 + calibrator.adj_p10, calibration_p90 + calibrator.adj_p90)

    coverage_report["adj_p10"] = calibrator.adj_p10
    coverage_report["adj_p90"] = calibrator.adj_p90
    coverage_report["achieved_coverage_calibration"] = calibrator.achieved_coverage
    coverage_report["coverage_within_target"] = calibrator.coverage_within_target
    coverage_report["n_calibration"] = calibrator.n_calibration

    # MLflow logging
    if mlflow_run is not None:
        try:
            import mlflow
            mlflow.log_metrics({
                "conformal_coverage": coverage_report["coverage"],
                "conformal_adj_p10": calibrator.adj_p10,
                "conformal_adj_p90": calibrator.adj_p90,
                "conformal_interval_width_mean": coverage_report["interval_width_mean"],
            })
        except Exception as exc:
            logger.warning("MLflow logging failed: %s", exc)

    # Save coverage report
    if output_dir is not None:
        import json
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "coverage_report.json"
        serialisable = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in coverage_report.items()
        }
        with open(report_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        logger.info("Coverage report saved: %s", report_path)

    return {
        "calibrator": calibrator,
        "cal_p10": cal_p10,
        "cal_p50": cal_p50,
        "cal_p90": cal_p90,
        "coverage_report": coverage_report,
    }
