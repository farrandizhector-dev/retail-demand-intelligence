"""Intermittency features for M5 demand forecasting.

The M5 dataset has >50 % zero sales (ADR-003, rule 6).  These features
capture the structure of demand sparsity and are critical for intermittent
and lumpy series.

Features:
- ``pct_zero_last_28d``: rolling fraction of zero days in the last 28 days.
- ``pct_zero_last_91d``: rolling fraction of zero days in the last 91 days.
- ``days_since_last_sale``: calendar days since the last non-zero sale.
- ``demand_intervals_mean``: series-level mean of inter-demand intervals
    (= ADI, computed on all historical data; broadcast to all rows).
- ``non_zero_demand_mean``: series-level mean of non-zero demand values.
- ``burstiness``: (σ_I − μ_I) / (σ_I + μ_I) of inter-demand intervals;
    computed on all historical data, broadcast to each row.
- ``streak_zeros``: number of consecutive zero-sale days ending at each row.

LEAKAGE PROTOCOL (rule 4):
  Rolling features use backward-looking windows (no future data).
  Series-level statistics (demand_intervals_mean, burstiness) are
  pre-computed on the training portion before calling this function.
"""

from __future__ import annotations

from datetime import date

import polars as pl


def add_intermittency_features(
    df: pl.DataFrame,
    *,
    sales_col: str = "sales",
    id_col: str = "id",
    date_col: str = "date",
    cutoff_date: date | None = None,
) -> pl.DataFrame:
    """Add all intermittency features to ``df``.

    Parameters
    ----------
    df:
        Long-format sales DataFrame, sorted by ``(id, date)``.
    sales_col, id_col, date_col:
        Column name overrides.
    cutoff_date:
        If set, only data up to this date is used for rolling features.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with intermittency columns appended.
    """
    compute_df = df.filter(pl.col(date_col) <= cutoff_date) if cutoff_date else df
    compute_df = compute_df.sort([id_col, date_col])

    # --- Rolling zero fractions ---
    is_zero_expr = pl.when(pl.col(sales_col) == 0).then(1.0).otherwise(0.0)

    compute_df = compute_df.with_columns(
        is_zero_expr.alias("_is_zero"),
    )

    compute_df = compute_df.with_columns(
        pl.col("_is_zero")
        .rolling_mean(window_size=28, min_samples=1)
        .over(id_col)
        .alias("pct_zero_last_28d"),
        pl.col("_is_zero")
        .rolling_mean(window_size=91, min_samples=1)
        .over(id_col)
        .alias("pct_zero_last_91d"),
    )

    # --- days_since_last_sale ---
    compute_df = compute_df.with_columns(
        # Record date for non-zero rows; null otherwise
        pl.when(pl.col(sales_col) > 0)
        .then(pl.col(date_col))
        .otherwise(None)
        .forward_fill()
        .over(id_col)
        .alias("_last_sale_date")
    )
    compute_df = compute_df.with_columns(
        (pl.col(date_col) - pl.col("_last_sale_date"))
        .dt.total_days()
        .fill_null(999)     # series never sold: very large gap
        .clip(0, 999)
        .alias("days_since_last_sale")
    )

    # --- streak_zeros: consecutive zeros ending at each row ---
    # Strategy: cumsum of non-zero markers; within each (id, nonzero_group),
    # the streak length is the position within the group.
    compute_df = compute_df.with_columns(
        (pl.col(sales_col) > 0)
        .cum_sum()
        .over(id_col)
        .alias("_nz_cum")
    )
    compute_df = compute_df.with_columns(
        # Row-number within each (id, _nz_cum) group
        pl.int_range(pl.len())
        .over([id_col, "_nz_cum"])
        .alias("_rank_in_group")
    )
    compute_df = compute_df.with_columns(
        pl.when(pl.col(sales_col) == 0)
        .then(pl.col("_rank_in_group"))
        .otherwise(0)
        .alias("streak_zeros")
    )

    # --- Series-level statistics (broadcast to all rows) ---
    series_stats = _compute_series_stats(compute_df, sales_col=sales_col, id_col=id_col)

    compute_df = compute_df.join(series_stats, on=id_col, how="left")

    # Cleanup temp columns
    compute_df = compute_df.drop(["_is_zero", "_last_sale_date", "_nz_cum", "_rank_in_group"])

    if cutoff_date:
        future = df.filter(pl.col(date_col) > cutoff_date)
        if not future.is_empty():
            new_cols = [
                c for c in compute_df.columns if c not in df.columns
            ]
            null_exprs = [pl.lit(None, dtype=pl.Float64).alias(c) for c in new_cols]
            future = future.with_columns(null_exprs)
            compute_df = (
                pl.concat([compute_df, future], how="vertical")
                .sort([id_col, date_col])
            )

    return compute_df


def _compute_series_stats(
    df: pl.DataFrame,
    *,
    sales_col: str = "sales",
    id_col: str = "id",
) -> pl.DataFrame:
    """Compute series-level demand interval statistics.

    Returns a DataFrame with one row per ``id`` containing:
    ``demand_intervals_mean``, ``non_zero_demand_mean``, ``burstiness``.

    Uses Python-level iteration over groups to apply the burstiness function
    to the full group Series (Polars map_elements is element-wise, not group-wise).

    Parameters
    ----------
    df:
        Sorted long-format DataFrame (all historical data, no future rows).
    sales_col, id_col:
        Column name overrides.
    """
    rows: list[dict] = []
    for (id_val,), group_df in df.group_by([id_col], maintain_order=True):
        sales = group_df[sales_col].to_list()
        n = len(sales)
        nonzero_vals = [v for v in sales if v > 0]
        n_nz = len(nonzero_vals)

        adi = n / n_nz if n_nz > 0 else float("inf")
        mean_nz = sum(nonzero_vals) / n_nz if n_nz > 0 else 0.0
        bursty = _burstiness_pure(sales)

        rows.append(
            {
                id_col: id_val,
                "demand_intervals_mean": adi,
                "non_zero_demand_mean": mean_nz,
                "burstiness": bursty,
            }
        )

    if not rows:
        return pl.DataFrame(
            {id_col: [], "demand_intervals_mean": [], "non_zero_demand_mean": [], "burstiness": []}
        )
    return pl.DataFrame(rows)


def _burstiness_pure(sales: list) -> float:
    """Compute burstiness of inter-demand intervals for a list of sales values.

    B = (σ_I − μ_I) / (σ_I + μ_I) where I are inter-arrival intervals.
    Returns 0.0 when insufficient data.
    """
    nonzero_positions = [i for i, v in enumerate(sales) if v > 0]
    if len(nonzero_positions) < 2:
        return 0.0
    intervals = [
        nonzero_positions[k + 1] - nonzero_positions[k]
        for k in range(len(nonzero_positions) - 1)
    ]
    n = len(intervals)
    mu = sum(intervals) / n
    if n < 2:
        return 0.0
    sigma = (sum((x - mu) ** 2 for x in intervals) / n) ** 0.5
    denom = sigma + mu
    return (sigma - mu) / denom if denom != 0 else 0.0
