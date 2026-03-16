"""ABC and XYZ demand classification for item-level analysis.

ABC classification groups items by revenue contribution:
  - A: top 80 % of cumulative revenue  (high-value, few SKUs)
  - B: next 15 % of cumulative revenue (medium-value)
  - C: bottom 5 % of cumulative revenue (low-value, many SKUs)

XYZ classification groups items by demand variability (coefficient of variation):
  - X: CV < 0.50  — stable, predictable demand
  - Y: 0.50 <= CV < 1.00 — moderate variability
  - Z: CV >= 1.00  — highly variable / intermittent demand

Both classifications are at item level (aggregated across all stores).
The output enriches the demand-classification table produced by
:mod:`src.classification.demand_classifier`.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------------
# Thresholds (configurable for V2)
# ---------------------------------------------------------------------------

ABC_A_THRESHOLD: float = 0.80  # top 80 % of revenue → class A
ABC_B_THRESHOLD: float = 0.95  # next 15 % (80–95 %) → class B; remainder → C

XYZ_X_THRESHOLD: float = 0.50   # CV < 0.50  → X
XYZ_Y_THRESHOLD: float = 1.00   # CV < 1.00  → Y; else Z


# ---------------------------------------------------------------------------
# ABC classification
# ---------------------------------------------------------------------------


def compute_abc(
    sales_df: pl.DataFrame,
    prices_df: pl.DataFrame | None = None,
    *,
    item_col: str = "item_id",
    sales_col: str = "sales",
    price_col: str = "sell_price",
    date_col: str = "date",
) -> pl.DataFrame:
    """Compute ABC class for every unique item.

    Revenue proxy = sum(units_sold × sell_price) per item across all stores
    and dates.  If ``prices_df`` is None or a price is unavailable, units
    sold (without price weighting) is used as the revenue proxy.

    Parameters
    ----------
    sales_df:
        Long-format silver sales DataFrame with at least
        ``(item_id, store_id, date, sales)``.
    prices_df:
        Silver daily prices DataFrame with ``(item_id, store_id, date, sell_price)``
        (the output of :func:`src.transform.pipeline.prices_weekly_to_daily`).
        Pass ``None`` to use unit counts as revenue proxy.
    item_col, sales_col, price_col, date_col:
        Column name overrides.

    Returns
    -------
    pl.DataFrame
        Columns: ``item_id, total_revenue_proxy, abc_class``
    """
    if prices_df is not None and not prices_df.is_empty():
        # Join prices onto sales; use sales × price where price available
        joined = sales_df.join(
            prices_df.select([item_col, "store_id", date_col, price_col]),
            on=[item_col, "store_id", date_col],
            how="left",
        )
        revenue_expr = (
            pl.col(sales_col) * pl.col(price_col).fill_null(1.0)
        ).alias("_rev")
    else:
        # Unit count as proxy
        joined = sales_df.clone()
        revenue_expr = pl.col(sales_col).cast(pl.Float64).alias("_rev")

    item_revenue = (
        joined
        .with_columns(revenue_expr)
        .group_by(item_col)
        .agg(pl.col("_rev").sum().alias("total_revenue_proxy"))
        .sort("total_revenue_proxy", descending=True)
    )

    total_rev = item_revenue["total_revenue_proxy"].sum()
    if total_rev == 0:
        item_revenue = item_revenue.with_columns(
            pl.lit("C").alias("abc_class")
        )
        return item_revenue

    # prev_cum_pct = cumulative revenue of items BEFORE this one (Pareto-correct)
    # An item belongs to class A if prev_cum_pct < A_threshold (it contributes
    # toward reaching the boundary, not past it).
    item_revenue = item_revenue.with_columns(
        (
            (pl.col("total_revenue_proxy").cum_sum() - pl.col("total_revenue_proxy"))
            / total_rev
        ).alias("prev_cum_pct")
    )

    item_revenue = item_revenue.with_columns(
        pl.when(pl.col("prev_cum_pct") < ABC_A_THRESHOLD)
        .then(pl.lit("A"))
        .when(pl.col("prev_cum_pct") < ABC_B_THRESHOLD)
        .then(pl.lit("B"))
        .otherwise(pl.lit("C"))
        .alias("abc_class")
    )

    return item_revenue.select([item_col, "total_revenue_proxy", "abc_class"])


# ---------------------------------------------------------------------------
# XYZ classification
# ---------------------------------------------------------------------------


def compute_xyz(
    sales_df: pl.DataFrame,
    *,
    item_col: str = "item_id",
    sales_col: str = "sales",
) -> pl.DataFrame:
    """Compute XYZ class for every unique item.

    XYZ is based on the coefficient of variation (CV = σ / μ) of daily
    sales aggregated across all stores per item.

    Parameters
    ----------
    sales_df:
        Long-format silver sales DataFrame.
    item_col, sales_col:
        Column name overrides.

    Returns
    -------
    pl.DataFrame
        Columns: ``item_id, cv, xyz_class``
    """
    # Aggregate across stores: total units sold per (item_id, date)
    item_daily = (
        sales_df
        .group_by([item_col, "date"])
        .agg(pl.col(sales_col).sum().alias("daily_total"))
    )

    item_stats = (
        item_daily
        .group_by(item_col)
        .agg(
            pl.col("daily_total").mean().alias("_mean"),
            pl.col("daily_total").std(ddof=0).alias("_std"),
        )
    )

    item_stats = item_stats.with_columns(
        pl.when(pl.col("_mean") > 0)
        .then(pl.col("_std") / pl.col("_mean"))
        .otherwise(float("inf"))
        .alias("cv")
    )

    item_stats = item_stats.with_columns(
        pl.when(pl.col("cv") < XYZ_X_THRESHOLD)
        .then(pl.lit("X"))
        .when(pl.col("cv") < XYZ_Y_THRESHOLD)
        .then(pl.lit("Y"))
        .otherwise(pl.lit("Z"))
        .alias("xyz_class")
    )

    return item_stats.select([item_col, "cv", "xyz_class"])


# ---------------------------------------------------------------------------
# Combined: enrich demand classification table
# ---------------------------------------------------------------------------


def enrich_with_abc_xyz(
    classification_df: pl.DataFrame,
    sales_df: pl.DataFrame,
    prices_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Add ``abc_class`` and ``xyz_class`` to the demand classification table.

    Parameters
    ----------
    classification_df:
        Output of :func:`src.classification.demand_classifier.classify_all_series`.
        Must contain ``item_id``.
    sales_df:
        Long-format silver sales.
    prices_df:
        Optional silver daily prices for revenue weighting.

    Returns
    -------
    pl.DataFrame
        ``classification_df`` with ``abc_class``, ``xyz_class``,
        ``total_revenue_proxy``, and ``cv`` columns added.
    """
    abc = compute_abc(sales_df, prices_df)
    xyz = compute_xyz(sales_df)

    result = (
        classification_df
        .join(abc, on="item_id", how="left")
        .join(xyz.select(["item_id", "cv", "xyz_class"]), on="item_id", how="left")
    )

    # Fill items with no revenue (all zero sales) as class C / Z
    result = result.with_columns(
        pl.col("abc_class").fill_null("C"),
        pl.col("xyz_class").fill_null("Z"),
    )
    return result


def save_full_classification(
    df: pl.DataFrame,
    output_path: Path,
) -> None:
    """Write the enriched classification table (ADI/CV² + ABC/XYZ) to Parquet.

    Parameters
    ----------
    df:
        Output of :func:`enrich_with_abc_xyz`.
    output_path:
        Destination path (parent directory created if needed).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="snappy")
