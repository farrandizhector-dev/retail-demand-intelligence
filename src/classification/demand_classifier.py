"""ADI/CV² demand classification for each SKU-store series.

Implements the Syntetos-Boylan-Croston matrix (section 5 of spec):

                    CV² (squared coefficient of variation)
                    Low (< 0.49)         High (>= 0.49)
ADI            ┌─────────────────────┬─────────────────────┐
(avg demand    │     SMOOTH          │     ERRATIC         │
interval)  Low │   ADI<1.32          │   ADI<1.32          │
(< 1.32)       │   CV²<0.49          │   CV²>=0.49         │
               ├─────────────────────┼─────────────────────┤
           High│     INTERMITTENT    │     LUMPY           │
(>= 1.32)      │   ADI>=1.32         │   ADI>=1.32         │
               │   CV²<0.49          │   CV²>=0.49         │
               └─────────────────────┴─────────────────────┘

ADI  = total_periods / non_zero_periods  (average inter-demand interval)
CV²  = (std_nonzero / mean_nonzero)²    (squared coefficient of variation
        computed on non-zero demand values only)
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import polars as pl

# ---------------------------------------------------------------------------
# Constants (from spec, section 5)
# ---------------------------------------------------------------------------

ADI_THRESHOLD: float = 1.32
CV2_THRESHOLD: float = 0.49

DemandClass = Literal["smooth", "erratic", "intermittent", "lumpy"]


# ---------------------------------------------------------------------------
# Single-series helpers (used in tests and scalar computations)
# ---------------------------------------------------------------------------


def compute_adi(sales: list[int] | pl.Series) -> float:
    """Compute Average Demand Interval for one series.

    ADI = total_periods / non_zero_periods.
    Returns ``float('inf')`` if the series is all zeros.

    Parameters
    ----------
    sales:
        Sequence of non-negative integer demand values (one per period).
    """
    values = list(sales) if isinstance(sales, pl.Series) else sales
    total = len(values)
    n_nonzero = sum(1 for v in values if v > 0)
    if n_nonzero == 0:
        return float("inf")
    return total / n_nonzero


def compute_cv2(sales: list[int] | pl.Series) -> float:
    """Compute squared coefficient of variation on non-zero demand values.

    CV² = (σ / μ)² where σ and μ are computed on {sales > 0} only.
    Returns 0.0 if fewer than 2 non-zero observations exist.

    Parameters
    ----------
    sales:
        Sequence of non-negative integer demand values.
    """
    values = list(sales) if isinstance(sales, pl.Series) else sales
    nonzero = [v for v in values if v > 0]
    if len(nonzero) < 2:
        return 0.0
    n = len(nonzero)
    mean = sum(nonzero) / n
    if mean == 0.0:
        return 0.0
    var = sum((v - mean) ** 2 for v in nonzero) / n  # population variance
    return var / (mean ** 2)


def classify_demand(adi: float, cv2: float) -> DemandClass:
    """Map (ADI, CV²) to a demand class label.

    Parameters
    ----------
    adi:
        Average demand interval (>= 1).
    cv2:
        Squared coefficient of variation (>= 0).
    """
    high_adi = adi >= ADI_THRESHOLD
    high_cv2 = cv2 >= CV2_THRESHOLD
    if not high_adi and not high_cv2:
        return "smooth"
    if not high_adi and high_cv2:
        return "erratic"
    if high_adi and not high_cv2:
        return "intermittent"
    return "lumpy"


# ---------------------------------------------------------------------------
# Batch computation (Polars, vectorised over all SKU-store series)
# ---------------------------------------------------------------------------


def classify_all_series(
    sales_df: pl.DataFrame,
    *,
    id_col: str = "id",
    sales_col: str = "sales",
    item_col: str = "item_id",
    store_col: str = "store_id",
    state_col: str = "state_id",
    dept_col: str = "dept_id",
    cat_col: str = "cat_id",
) -> pl.DataFrame:
    """Compute ADI, CV² and demand class for every SKU-store series.

    Parameters
    ----------
    sales_df:
        Long-format silver sales DataFrame.  Must contain at minimum the
        columns ``id``, ``sales``, ``item_id``, ``store_id``, ``state_id``,
        ``dept_id``, ``cat_id``.
    id_col, sales_col, item_col, store_col, state_col, dept_col, cat_col:
        Column name overrides.

    Returns
    -------
    pl.DataFrame
        One row per unique ``id`` with columns:
        ``id, item_id, store_id, state_id, dept_id, cat_id,
          adi, cv_squared, demand_class, avg_daily_demand, pct_zero_days,
          n_periods, n_nonzero_periods``
    """
    meta_cols = [item_col, store_col, state_col, dept_col, cat_col]

    stats = (
        sales_df
        .group_by(id_col)
        .agg(
            *[pl.col(c).first().alias(c) for c in meta_cols],
            # Period counts
            pl.col(sales_col).count().alias("n_periods"),
            (pl.col(sales_col) > 0).sum().alias("n_nonzero_periods"),
            # Summary stats
            pl.col(sales_col).mean().alias("avg_daily_demand"),
            ((pl.col(sales_col) == 0).sum() / pl.col(sales_col).count()).alias("pct_zero_days"),
            # Non-zero stats for CV²
            pl.col(sales_col).filter(pl.col(sales_col) > 0).mean().alias("_mean_nz"),
            pl.col(sales_col).filter(pl.col(sales_col) > 0).std(ddof=0).alias("_std_nz"),
        )
    )

    stats = stats.with_columns(
        # ADI
        (pl.col("n_periods") / pl.col("n_nonzero_periods").replace(0, None))
        .fill_null(float("inf"))
        .alias("adi"),
        # CV² = (std / mean)²
        pl.when(
            pl.col("n_nonzero_periods") >= 2,
        )
        .then(
            (pl.col("_std_nz") / pl.col("_mean_nz").replace(0.0, None)) ** 2
        )
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("cv_squared"),
    )

    # Vectorised classification using Polars conditions
    stats = stats.with_columns(
        pl.when(
            (pl.col("adi") < ADI_THRESHOLD) & (pl.col("cv_squared") < CV2_THRESHOLD)
        )
        .then(pl.lit("smooth"))
        .when(
            (pl.col("adi") < ADI_THRESHOLD) & (pl.col("cv_squared") >= CV2_THRESHOLD)
        )
        .then(pl.lit("erratic"))
        .when(
            (pl.col("adi") >= ADI_THRESHOLD) & (pl.col("cv_squared") < CV2_THRESHOLD)
        )
        .then(pl.lit("intermittent"))
        .otherwise(pl.lit("lumpy"))
        .alias("demand_class"),
    )

    return stats.drop(["_mean_nz", "_std_nz"]).rename({id_col: "id"})


def save_demand_classification(
    classification_df: pl.DataFrame,
    output_path: Path,
) -> None:
    """Write the demand classification table to Parquet.

    Parameters
    ----------
    classification_df:
        Output of :func:`classify_all_series`.
    output_path:
        Destination path (parent directory created if needed).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    classification_df.write_parquet(output_path, compression="snappy")
