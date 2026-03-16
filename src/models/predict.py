"""Inference pipeline for the trained LightGBM global model.

Generates point and quantile predictions from a ``TrainedModels`` object,
then applies post-processing:
- Clip predictions to [0, ∞) (demand is non-negative)
- Enforce monotonicity: p10 ≤ p50 ≤ p90 (interpolate if violated)

Output format: fact_forecast_daily — one row per (id, date) in the
forecast horizon.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.models.training import TrainedModels, prepare_xy


# ---------------------------------------------------------------------------
# Monotonicity enforcement
# ---------------------------------------------------------------------------


def enforce_monotonicity(
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ensure p10 ≤ p50 ≤ p90 element-wise.

    Violations are resolved by linear interpolation between the nearest
    valid quantile boundaries.  Non-negativity is enforced after.

    Parameters
    ----------
    p10, p50, p90:
        Arrays of quantile predictions (same length).

    Returns
    -------
    p10, p50, p90 arrays with monotonicity guaranteed.
    """
    # Fix p10 > p50: set p10 = p50
    p10 = np.minimum(p10, p50)
    # Fix p90 < p50: set p90 = p50
    p90 = np.maximum(p90, p50)
    # Non-negativity
    p10 = np.maximum(p10, 0.0)
    p50 = np.maximum(p50, 0.0)
    p90 = np.maximum(p90, 0.0)
    return p10, p50, p90


# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------


def predict(
    models: TrainedModels,
    feature_df: pl.DataFrame,
    id_col: str = "id",
    date_col: str = "date",
) -> pl.DataFrame:
    """Generate predictions for every row in ``feature_df``.

    Typically called with the feature rows for the forecast horizon
    (dates > cutoff_date in the feature store, or future rows built for
    the test window).

    Parameters
    ----------
    models:
        Trained point + quantile models.
    feature_df:
        Feature store rows to predict on.  Must contain the same
        ``feature_cols`` that the model was trained on.
    id_col:
        Series identifier column.
    date_col:
        Date column.

    Returns
    -------
    pl.DataFrame with columns:
        id, date, forecast_p10, forecast_p50, forecast_p90
    """
    if len(feature_df) == 0:
        return pl.DataFrame(schema={
            id_col: pl.Utf8,
            date_col: pl.Date,
            "forecast_p10": pl.Float64,
            "forecast_p50": pl.Float64,
            "forecast_p90": pl.Float64,
        })

    X, _ = prepare_xy(feature_df, models.feature_cols, models.cat_cols)

    # Point forecast (used as p50 if no quantile model for 0.50)
    p50_raw = models.point_model.predict(X.values)

    # Quantile predictions
    q10_model = models.quantile_models.get(0.10)
    q50_model = models.quantile_models.get(0.50)
    q90_model = models.quantile_models.get(0.90)

    p10_raw = q10_model.predict(X.values) if q10_model else p50_raw * 0.5
    p50_raw = q50_model.predict(X.values) if q50_model else p50_raw
    p90_raw = q90_model.predict(X.values) if q90_model else p50_raw * 1.5

    p10, p50, p90 = enforce_monotonicity(p10_raw, p50_raw, p90_raw)

    return pl.DataFrame({
        id_col: feature_df[id_col].to_list(),
        date_col: feature_df[date_col].to_list(),
        "forecast_p10": p10.tolist(),
        "forecast_p50": p50.tolist(),
        "forecast_p90": p90.tolist(),
    }).with_columns(pl.col(date_col).cast(pl.Date))


# ---------------------------------------------------------------------------
# Convenience: generate horizon rows
# ---------------------------------------------------------------------------


def generate_forecast_horizon(
    feature_store_df: pl.DataFrame,
    cutoff_date,
    id_col: str = "id",
    date_col: str = "date",
) -> pl.DataFrame:
    """Return the rows of the feature store that are strictly after cutoff_date.

    These rows have null lag/rolling features (set to null by the feature
    builder) and represent the forecast horizon to be predicted.

    Parameters
    ----------
    feature_store_df:
        Full feature store DataFrame (train + future rows).
    cutoff_date:
        Last training date.

    Returns
    -------
    pl.DataFrame — feature rows for the horizon (date > cutoff_date).
    """
    return feature_store_df.filter(pl.col(date_col) > cutoff_date)
