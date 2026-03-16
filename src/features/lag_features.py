"""Lag features for demand forecasting.

Computes lagged sales values within each SKU-store series (identified by
``id``).  The DataFrame must be sorted by ``(id, date)`` before calling any
function in this module.

LEAKAGE PROTOCOL (rule 4 — section 8.2):
  All lags are strictly backward-looking.  ``lag_N`` at row t is the sales
  value at row (t − N), computed inside the ``over("id")`` window.  When the
  feature store is built for a specific backtesting fold, data after
  ``cutoff_date`` is excluded BEFORE calling these functions, so no future
  information can leak into the features.
"""

from __future__ import annotations

from datetime import date

import polars as pl


# Canonical lag configuration (from configs/features.yaml)
LAG_PERIODS: list[int] = [1, 7, 14, 21, 28, 56, 91, 364]


def add_lag_features(
    df: pl.DataFrame,
    *,
    lags: list[int] = LAG_PERIODS,
    sales_col: str = "sales",
    id_col: str = "id",
    date_col: str = "date",
    cutoff_date: date | None = None,
) -> pl.DataFrame:
    """Add lag_{N} columns to ``df``.

    The DataFrame must be sorted by ``(id_col, date_col)`` to produce
    correct lag values.  If it is not sorted, a copy is sorted internally
    and then the result is reordered to match the original index.

    Parameters
    ----------
    df:
        Long-format sales DataFrame, sorted by ``(id, date)``.
    lags:
        List of lag periods (days) to compute.
    sales_col:
        Name of the target column to lag.
    id_col:
        Column identifying each unique time series (SKU-store).
    date_col:
        Date column (``pl.Date``).
    cutoff_date:
        When not None, only data up to and including this date is used to
        compute lag values.  Rows after ``cutoff_date`` receive ``null``
        lags — they must be predicted, not used.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with ``lag_1``, ``lag_7``, … columns appended.
        Lags that reference before the series start produce ``null``.
    """
    # Optionally filter to cutoff before computing — preserves ordering
    compute_df = df.filter(pl.col(date_col) <= cutoff_date) if cutoff_date else df

    # Ensure sort for correct over() semantics
    compute_df = compute_df.sort([id_col, date_col])

    exprs = [
        pl.col(sales_col)
        .shift(lag)
        .over(id_col)
        .alias(f"lag_{lag}")
        for lag in lags
    ]
    result = compute_df.with_columns(exprs)

    if cutoff_date:
        # Append the future rows back (they will have null lags)
        future = df.filter(pl.col(date_col) > cutoff_date)
        if not future.is_empty():
            lag_dtype = result[f"lag_{lags[0]}"].dtype
            null_lag_cols = [pl.lit(None).cast(lag_dtype).alias(f"lag_{lag}") for lag in lags]
            future = future.with_columns(null_lag_cols)
            result = pl.concat([result, future], how="vertical").sort([id_col, date_col])

    return result
