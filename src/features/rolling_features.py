"""Rolling window statistics for demand forecasting.

All rolling computations are backward-looking (the window includes the
current row and the N-1 preceding rows within each series).

LEAKAGE PROTOCOL (rule 4 — section 8.2):
  When called from a backtesting fold, ``df`` contains only rows up to
  ``cutoff_date``, ensuring the rolling window never touches future demand.

Windows: 7, 14, 28, 56, 91 days.
Statistics per window: mean, std, median, min, max.
Extra features: rolling_zero_pct_28, rolling_cv_28, ratio_mean_7_28,
ratio_mean_28_91 (trend proxies).
"""

from __future__ import annotations

from datetime import date

import polars as pl


ROLLING_WINDOWS: list[int] = [7, 14, 28, 56, 91]


def add_rolling_features(
    df: pl.DataFrame,
    *,
    windows: list[int] = ROLLING_WINDOWS,
    sales_col: str = "sales",
    id_col: str = "id",
    date_col: str = "date",
    cutoff_date: date | None = None,
) -> pl.DataFrame:
    """Add rolling statistics columns to ``df``.

    For each window size W and each statistic S, adds column
    ``rolling_{S}_{W}``.  Also computes extra derived rolling features.

    Parameters
    ----------
    df:
        Long-format sales DataFrame, sorted by ``(id, date)``.
    windows:
        List of rolling window sizes (days) to use.
    sales_col:
        Target column.
    id_col:
        Series identifier column.
    date_col:
        Date column.
    cutoff_date:
        If set, only data on or before this date contributes to rolling
        windows.  Rows after this date are appended with null rolling stats.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with rolling feature columns appended.
    """
    compute_df = df.filter(pl.col(date_col) <= cutoff_date) if cutoff_date else df
    compute_df = compute_df.sort([id_col, date_col])

    exprs: list[pl.Expr] = []

    for w in windows:
        ms = 1  # min_samples=1: first row always gets a value
        exprs += [
            pl.col(sales_col)
            .rolling_mean(window_size=w, min_samples=ms)
            .over(id_col)
            .alias(f"rolling_mean_{w}"),
            pl.col(sales_col)
            .rolling_std(window_size=w, min_samples=ms)
            .over(id_col)
            .alias(f"rolling_std_{w}"),
            pl.col(sales_col)
            .rolling_quantile(0.5, interpolation="nearest", window_size=w, min_samples=ms)
            .over(id_col)
            .alias(f"rolling_median_{w}"),
            pl.col(sales_col)
            .rolling_min(window_size=w, min_samples=ms)
            .over(id_col)
            .alias(f"rolling_min_{w}"),
            pl.col(sales_col)
            .rolling_max(window_size=w, min_samples=ms)
            .over(id_col)
            .alias(f"rolling_max_{w}"),
        ]

    result = compute_df.with_columns(exprs)

    # --- Extra rolling features (always computed with hardcoded windows) ---
    # Ensure required base windows exist even if not in `windows` list
    extra_base_windows = {7, 28, 91} - set(windows)
    if extra_base_windows:
        base_exprs = []
        for w in sorted(extra_base_windows):
            base_exprs += [
                pl.col(sales_col)
                .rolling_mean(window_size=w, min_samples=1)
                .over(id_col)
                .alias(f"rolling_mean_{w}"),
                pl.col(sales_col)
                .rolling_std(window_size=w, min_samples=1)
                .over(id_col)
                .alias(f"rolling_std_{w}"),
            ]
        result = result.with_columns(base_exprs)

    extra_exprs: list[pl.Expr] = []

    # rolling_zero_pct_28: fraction of zeros in last 28 days
    extra_exprs.append(
        pl.col(sales_col)
        .map_elements(
            lambda x: 1.0 if x == 0 else 0.0,
            return_dtype=pl.Float64,
        )
        .rolling_mean(window_size=28, min_samples=1)
        .over(id_col)
        .alias("rolling_zero_pct_28")
    )

    # rolling_cv_28: coefficient of variation over last 28 days
    extra_exprs.append(
        (
            pl.col("rolling_std_28") / pl.col("rolling_mean_28").replace(0.0, None)
        )
        .fill_null(0.0)
        .alias("rolling_cv_28")
    )

    # ratio_mean_7_28: short-term vs medium-term demand (trend signal)
    extra_exprs.append(
        (
            pl.col("rolling_mean_7") / pl.col("rolling_mean_28").replace(0.0, None)
        )
        .fill_null(1.0)
        .alias("ratio_mean_7_28")
    )

    # ratio_mean_28_91: medium-term vs long-term (trend signal)
    extra_exprs.append(
        (
            pl.col("rolling_mean_28") / pl.col("rolling_mean_91").replace(0.0, None)
        )
        .fill_null(1.0)
        .alias("ratio_mean_28_91")
    )

    result = result.with_columns(extra_exprs)

    if cutoff_date:
        future = df.filter(pl.col(date_col) > cutoff_date)
        if not future.is_empty():
            rolling_cols = [
                col for col in result.columns if col not in df.columns
            ]
            null_exprs = [
                pl.lit(None, dtype=pl.Float64).alias(c) for c in rolling_cols
            ]
            future = future.with_columns(null_exprs)
            result = pl.concat([result, future], how="vertical").sort([id_col, date_col])

    return result
